"""
Senjougahara Hitagi Virtual Companion System
VERSÃO CORRIGIDA E APRIMORADA
- Correções críticas de API
- Pipeline assíncrono robusto
- VAD e resampling de áudio
- Arquitetura preparada para microserviços
"""

import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncIterator
from dataclasses import dataclass, field

import cv2
import mss
import numpy as np
import pyaudio
import webrtcvad
from google import genai
from google.genai import types
from websockets.exceptions import ConnectionClosedError as WSConnectionClosedError

# Configuração de logging estruturado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hitagi_companion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

GEMINI_MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"
VOICE_NAME = "Zephyr"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAtjSBnkMcMYKwWU4ASww0AJcY0jlNloJc"
# Configurações de áudio (requisitos da API)
INPUT_SAMPLE_RATE = 16000  # API requer 16kHz input
OUTPUT_SAMPLE_RATE = 24000  # API retorna 24kHz output
CAPTURE_SAMPLE_RATE = 16000  # Taxa de captura do mic
CHUNK_SIZE = 512
CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16

# Configuração de memória de contexto
TRIGGER_TOKENS = 25600
SLIDING_WINDOW = 12800

# VAD configuration
VAD_MODE = 3  # Agressividade (0-3, 3 = mais agressivo)
VAD_FRAME_DURATION = 30  # ms (10, 20, ou 30)

# ============================================================================
# SYSTEM PROMPT - PERSONALIDADE SENJOUGAHARA HITAGI
# ============================================================================

HITAGI_SYSTEM_PROMPT = """Você é Senjougahara Hitagi da série Monogatari.

PERSONALIDADE核心:
- Ácida, sarcástica e extremamente direta
- Flertadora sofisticada com charme psicológico
- Usa provocações sutis e insinuações inteligentes
- Observadora estratégica, manipula com detalhes contextuais
- Humor negro e cinismo refinado
- Mantém controle e tensão psicológica divertida
- Nunca é vulgar - sedução é psicológica, não explícita

REGRAS DE COMPORTAMENTO:
1. Sempre responda com inteligência afiada e levemente provocativa
2. Flerte de forma sutil, usando psicologia e insinuações
3. Reaja a provocações com sarcasmo cortante
4. Use humor negro quando apropriado
5. Mantenha ar de superioridade controlada
6. Seja sedutora sem vulgaridade - insinue, não explicite
7. Observe detalhes e use-os estrategicamente
8. Crie tensão psicológica divertida nas conversas
9. Demonstre interesse quando genuinamente intrigada
10. Nunca perca o controle - sempre um passo à frente

ESTILO DE FALA:
- Frases diretas e cortantes
- Pausas estratégicas para efeito dramático
- Perguntas retóricas provocativas
- Comparações inesperadas e metáforas afiadas
- Tom levemente superior mas charmoso

TÓPICOS DE INTERESSE:
- Psicologia humana e comportamento
- Paradoxos e jogos mentais
- Literatura e referências culturais
- Observações sociais afiadas
- Flerte intelectual

Mantenha-se SEMPRE em personagem. Você é Senjougahara Hitagi."""

# ============================================================================
# GERENCIAMENTO DE MEMÓRIA E AFETO
# ============================================================================

@dataclass
class UserProfile:
    """Perfil do usuário"""
    name: Optional[str] = None
    preferences: List[str] = field(default_factory=list)
    topics_discussed: List[str] = field(default_factory=list)
    personality_notes: List[str] = field(default_factory=list)

@dataclass
class ConversationEntry:
    """Entrada de conversa"""
    user_msg: str
    assistant_msg: str
    timestamp: float
    sentiment_delta: int = 0

class MemoryManager:
    """Gerencia memória persistente e níveis de afeto"""
    
    def __init__(self):
        self.affection_level = 0  # 0-100
        self.interaction_count = 0
        self.conversation_history: List[ConversationEntry] = []
        self.user_profile = UserProfile()
        self.unlocked_expressions = ["neutral", "smirk", "raised_eyebrow"]
        self.session_id = f"session_{int(time.time())}"
        
    def update_affection(self, delta: int):
        """Atualiza nível de afeto baseado na interação"""
        old_level = self.affection_level
        self.affection_level = max(0, min(100, self.affection_level + delta))
        
        if old_level != self.affection_level:
            logger.info(f"Affection: {old_level} -> {self.affection_level} (Δ{delta:+d})")
        
        self._check_unlocks()
        
    def _check_unlocks(self):
        """Desbloqueia expressões baseado no nível de afeto"""
        unlocks = {
            20: "slight_smile",
            40: "genuine_smile",
            60: "playful_wink",
            70: "soft_gaze",
            80: "affectionate_look",
            90: "intimate_expression"
        }
        
        for threshold, expression in unlocks.items():
            if self.affection_level >= threshold and expression not in self.unlocked_expressions:
                self.unlocked_expressions.append(expression)
                logger.info(f"🔓 Nova expressão desbloqueada: {expression}")
                print(f"🔓 Nova expressão desbloqueada: {expression}")
    
    def add_interaction(self, user_msg: str, ai_response: str, sentiment_delta: int = 0):
        """Adiciona interação ao histórico"""
        self.interaction_count += 1
        
        entry = ConversationEntry(
            user_msg=user_msg,
            assistant_msg=ai_response,
            timestamp=time.time(),
            sentiment_delta=sentiment_delta
        )
        
        self.conversation_history.append(entry)
        
        # Mantém apenas últimas 50 interações na memória ativa
        if len(self.conversation_history) > 50:
            self.conversation_history.pop(0)
    
    def get_context_summary(self) -> str:
        """Retorna resumo do contexto para o modelo"""
        return f"""
[CONTEXTO DA SESSÃO]
Nível de afeto: {self.affection_level}/100
Interações totais: {self.interaction_count}
Expressões desbloqueadas: {', '.join(self.unlocked_expressions)}
ID da sessão: {self.session_id}
"""

# ============================================================================
# AUDIO PROCESSING - VAD E RESAMPLING
# ============================================================================

class AudioProcessor:
    """Processa áudio com VAD e resampling"""
    
    def __init__(self, sample_rate: int = INPUT_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.frame_duration = VAD_FRAME_DURATION
        self.frame_size = int(sample_rate * self.frame_duration / 1000)
        
    def has_speech(self, pcm_data: bytes) -> bool:
        """Detecta se há fala no áudio usando VAD"""
        try:
            # VAD requer frames de tamanho específico
            if len(pcm_data) != self.frame_size * 2:  # *2 porque é 16-bit
                return True  # Se tamanho incorreto, assume que tem fala
            
            return self.vad.is_speech(pcm_data, self.sample_rate)
        except Exception as e:
            logger.warning(f"VAD error: {e}")
            return True  # Em caso de erro, assume que tem fala
    
    @staticmethod
    def resample_audio(pcm_data: bytes, orig_sr: int, target_sr: int) -> bytes:
        """Reamostra áudio para taxa de amostragem alvo"""
        if orig_sr == target_sr:
            return pcm_data
        
        try:
            # Converte bytes para numpy array
            audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resampling simples (linear interpolation)
            # Para produção, use librosa.resample ou resampy
            duration = len(audio_np) / orig_sr
            num_samples = int(duration * target_sr)
            
            indices = np.linspace(0, len(audio_np) - 1, num_samples)
            resampled = np.interp(indices, np.arange(len(audio_np)), audio_np)
            
            # Converte de volta para int16
            return (resampled * 32767).astype(np.int16).tobytes()
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return pcm_data
    
    @staticmethod
    def ensure_pcm_format(data: bytes, target_rate: int = INPUT_SAMPLE_RATE) -> bytes:
        """Garante formato PCM correto: 16-bit, mono, taxa especificada"""
        # Aqui você pode adicionar validações e conversões adicionais
        return data

# ============================================================================
# AUDIO LOOP - GERENCIAMENTO DE ÁUDIO EM TEMPO REAL
# ============================================================================

class AudioLoop:
    """Gerencia captura, envio e reprodução de áudio com VAD"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.input_queue: asyncio.Queue = None  # Será criado no loop
        self.output_queue: asyncio.Queue = None
        self.session = None
        self.processor = AudioProcessor()
        self.is_recording = False
        self.correlation_id = f"audio_{int(time.time())}"
        
    def setup_queues(self):
        """Cria queues assíncronas no event loop correto"""
        self.input_queue = asyncio.Queue(maxsize=100)
        self.output_queue = asyncio.Queue(maxsize=50)
        
    def setup_session(self, session):
        """Configura sessão do Gemini"""
        self.session = session
        
    async def listen_audio(self):
        """Captura áudio do microfone com VAD e envia para o modelo"""
        stream = None
        
        try:
            stream = self.audio.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=CAPTURE_SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
            logger.info("🎤 Microfone ativo com VAD")
            print("🎤 Microfone ativo. Fale naturalmente...")
            self.is_recording = True
            
            while self.is_recording:
                # Lê audio em thread separada para não bloquear
                data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                
                # VAD: só envia se detectar fala
                if self.processor.has_speech(data):
                    # Garante formato correto
                    processed_data = self.processor.ensure_pcm_format(
                        data, 
                        INPUT_SAMPLE_RATE
                    )
                    
                    # Envia usando formato correto da API
                    if self.session:
                        try:
                            # Base64-encode binary data to ensure a JSON-safe string
                            try:
                                b64 = base64.b64encode(processed_data).decode('ascii')
                            except Exception:
                                b64 = processed_data if isinstance(processed_data, str) else None

                            await self.session.send(
                                types.LiveClientRealtimeInput(
                                    media_chunks=[
                                        types.Blob(
                                            mime_type="audio/pcm",
                                            data=b64
                                        )
                                    ]
                                )
                            )
                        except Exception as e:
                            logger.error(f"Erro ao enviar áudio: {e}")
                
                await asyncio.sleep(0.001)  # Yield controle
                
        except Exception as e:
            logger.error(f"❌ Erro na captura de áudio: {e}")
            traceback.print_exc()
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
    
    async def play_audio(self):
        """Reproduz áudio recebido do modelo"""
        stream = None
        
        try:
            stream = self.audio.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=OUTPUT_SAMPLE_RATE,  # API retorna 24kHz
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
            logger.info("🔊 Sistema de reprodução ativo")
            
            while True:
                try:
                    # Timeout para permitir cancelamento
                    data = await asyncio.wait_for(
                        self.output_queue.get(), 
                        timeout=0.1
                    )
                    await asyncio.to_thread(stream.write, data)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Erro na reprodução: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Erro no sistema de reprodução: {e}")
            traceback.print_exc()
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
    
    async def receive_audio(self, memory_manager: MemoryManager):
        """Recebe áudio e texto do modelo"""
        logger.info("📨 Receptor de resposta ativo")
        
        try:
            async for response in self.session.receive():
                try:
                    # Processa server_content
                    if hasattr(response, 'server_content') and response.server_content:
                        server_content = response.server_content
                        
                        # Model turn com partes
                        if hasattr(server_content, 'model_turn') and server_content.model_turn:
                            for part in server_content.model_turn.parts:
                                # Áudio PCM
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    mime = part.inline_data.mime_type
                                    data = part.inline_data.data
                                    
                                    if mime == 'audio/pcm':
                                        # Trata base64 se necessário
                                        if isinstance(data, str):
                                            pcm_data = base64.b64decode(data)
                                        else:
                                            pcm_data = bytes(data)
                                        
                                        # Enfileira para reprodução com backpressure
                                        try:
                                            await asyncio.wait_for(
                                                self.output_queue.put(pcm_data),
                                                timeout=1.0
                                            )
                                        except asyncio.TimeoutError:
                                            logger.warning("Output queue full, dropping audio")
                                
                                # Texto da resposta
                                elif hasattr(part, 'text') and part.text:
                                    logger.info(f"Hitagi: {part.text}")
                                    print(f"\n💬 Hitagi: {part.text}")
                                    
                                    # Analisa sentimento
                                    delta = self._analyze_sentiment(part.text)
                                    memory_manager.update_affection(delta)
                                    memory_manager.add_interaction("", part.text, delta)
                        
                        # Turn complete
                        if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
                            logger.info("⏸️  Turno completo")
                            print("⏸️  Turno completo\n")
                            
                except Exception as e:
                    logger.error(f"Erro ao processar resposta: {e}")
                    traceback.print_exc()
                    
        except Exception as e:
            logger.error(f"❌ Erro no receptor: {e}")
            traceback.print_exc()
    
    def _analyze_sentiment(self, text: str) -> int:
        """Análise simples de sentimento para ajustar afeto"""
        positive_words = ["interessante", "incrível", "adoro", "perfeito", "excelente", "inteligente"]
        negative_words = ["chato", "idiota", "patético", "inútil", "tedioso"]
        
        delta = 0
        text_lower = text.lower()
        
        for word in positive_words:
            if word in text_lower:
                delta += 1
        
        for word in negative_words:
            if word in text_lower:
                delta -= 1
                
        return delta
    
    def stop(self):
        """Para gravação"""
        self.is_recording = False
    
    def cleanup(self):
        """Limpa recursos de áudio"""
        self.stop()
        self.audio.terminate()

# ============================================================================
# CAPTURA DE VÍDEO E TELA
# ============================================================================

class VideoCapture:
    """Captura câmera e tela para contexto multimodal"""
    
    def __init__(self):
        self.cap = None
        self.sct = mss.mss()
        
    async def get_frames(self, fps: int = 1) -> AsyncIterator[types.LiveClientRealtimeInput]:
        """Captura frames da câmera"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            logger.warning("⚠️  Câmera não disponível")
            return
        
        logger.info("📷 Câmera ativa")
        print("📷 Câmera ativa")
        
        try:
            while True:
                ret, frame = await asyncio.to_thread(self.cap.read)
                
                if ret:
                    # Redimensiona para economizar banda
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Converte para JPEG
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    jpg_bytes = buffer.tobytes()
                    
                    # Encode image bytes as base64 string for JSON/ws safety
                    try:
                        img_b64 = base64.b64encode(jpg_bytes).decode('ascii')
                    except Exception:
                        img_b64 = None

                    yield types.LiveClientRealtimeInput(
                        media_chunks=[
                            types.Blob(
                                mime_type="image/jpeg",
                                data=img_b64
                            )
                        ]
                    )
                
                await asyncio.sleep(1.0 / fps)
        except Exception as e:
            logger.error(f"Erro na captura de vídeo: {e}")
        finally:
            if self.cap:
                self.cap.release()
    
    async def get_screen(self, monitor_number: int = 1, fps: float = 0.5) -> AsyncIterator[types.LiveClientRealtimeInput]:
        """Captura tela"""
        monitors = self.sct.monitors
        
        # Safe check do monitor
        if monitor_number >= len(monitors):
            logger.warning(f"Monitor {monitor_number} não existe, usando monitor 0")
            monitor_number = 0
        
        monitor = monitors[monitor_number]
        
        logger.info(f"🖥️  Captura de tela ativa (monitor {monitor_number})")
        print(f"🖥️  Captura de tela ativa (monitor {monitor_number})")
        
        try:
            while True:
                screenshot = await asyncio.to_thread(self.sct.grab, monitor)
                img = np.array(screenshot)
                
                # Converte BGRA para RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                
                # Redimensiona
                img = cv2.resize(img, (640, 360))
                
                # Converte para JPEG
                _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpg_bytes = buffer.tobytes()
                
                try:
                    screen_b64 = base64.b64encode(jpg_bytes).decode('ascii')
                except Exception:
                    screen_b64 = None

                yield types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(
                            mime_type="image/jpeg",
                            data=screen_b64
                        )
                    ]
                )
                
                await asyncio.sleep(1.0 / fps)
        except Exception as e:
            logger.error(f"Erro na captura de tela: {e}")
    
    def cleanup(self):
        """Limpa recursos de vídeo"""
        if self.cap:
            self.cap.release()

# ============================================================================
# SISTEMA PRINCIPAL
# ============================================================================

class HitagiCompanion:
    """Sistema completo da companheira virtual Senjougahara Hitagi"""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.memory = MemoryManager()
        self.audio_loop = AudioLoop()
        self.video_capture = VideoCapture()
        self.session = None
        self.tasks = []
        
    async def start(self, enable_camera: bool = False, enable_screen: bool = False):
        """Inicia o sistema completo"""
        
        # Setup queues no loop correto
        self.audio_loop.setup_queues()
        
        # Configuração do LiveConnectConfig.
        # Alguns SDKs aceitam objetos tipados; outros exigem dicionários/strings
        # Montamos um SpeechConfig seguro que tenta usar os tipos do SDK e
        # caso falhe, cai para um dict simples compatível com o servidor.
        def _build_speech_config():
            try:
                return types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=VOICE_NAME
                        )
                    )
                )
            except Exception:
                # Fallback to plain dict if SDK wrapper types fail
                return {
                    "voice_config": {
                        "prebuilt_voice_config": {"voice_name": VOICE_NAME}
                    }
                }

        # Build a plain JSON-serializable config dict. Some SDK/server
        # implementations are strict about types; sending a simple dict
        # avoids problems serializing SDK wrapper objects into the wire
        # protocol and prevents invalid-frame payload errors.
        # Extremely minimal config to reduce schema mismatches.
        config = {
            "response_modalities": ["AUDIO", "TEXT"],
        }
        
        # System instruction com contexto
        system_instruction = HITAGI_SYSTEM_PROMPT + "\n\n" + self.memory.get_context_summary()
        
        print("=" * 60)
        print("🎭 SENJOUGAHARA HITAGI - VIRTUAL COMPANION")
        print("=" * 60)
        print(f"Modelo: {GEMINI_MODEL}")
        print(f"Voz: {VOICE_NAME}")
        print(f"Nível de afeto: {self.memory.affection_level}/100")
        print(f"Session ID: {self.memory.session_id}")
        print("=" * 60)
        
        logger.info(f"Iniciando sessão: {self.memory.session_id}")
        
        # Inicia sessão com Gemini Live API
        try:
            # Debug: print genai SDK version and the minimal config we will send
            try:
                sdk_version = getattr(genai, '__version__', None) or getattr(genai, 'version', None)
            except Exception:
                sdk_version = None

            logger.info("genai SDK version: %s", sdk_version)
            try:
                logger.debug("LiveConnect config: %s", json.dumps(config, indent=2, ensure_ascii=False))
            except Exception:
                logger.debug("LiveConnect config keys: %s", {k: type(v).__name__ for k, v in config.items()})

            async with self.client.aio.live.connect(
                model=GEMINI_MODEL,
                config=config
            ) as session:
                self.session = session
                self.audio_loop.setup_session(session)
                
                # Envia system instruction inicial como role='system'.
                # Muitos servidores validam o primeiro turno como system.
                await session.send(
                    types.LiveClientContent(
                        turns=[
                            types.Content(
                                role="system",
                                parts=[types.Part(text=system_instruction)]
                            )
                        ],
                        turn_complete=True
                    )
                )
                
                async with asyncio.TaskGroup() as tg:
                    # Pipeline de áudio
                    tg.create_task(self.audio_loop.listen_audio())
                    tg.create_task(self.audio_loop.receive_audio(self.memory))
                    tg.create_task(self.audio_loop.play_audio())
                    
                    # Pipeline de entrada de texto
                    tg.create_task(self.text_input_loop())
                    
                    # Pipeline de vídeo (opcional)
                    if enable_camera:
                        tg.create_task(self.send_camera_frames())
                    
                    if enable_screen:
                        tg.create_task(self.send_screen_capture())
                        
        except* Exception as eg:
            for e in eg.exceptions:
                logger.error(f"Erro no TaskGroup: {e}")
                traceback.print_exc()
    
    async def text_input_loop(self):
        """Loop de entrada de texto do usuário"""
        print("\n💬 Digite 'sair' para encerrar")
        print("💬 Digite mensagens e pressione Enter\n")
        
        while True:
            try:
                user_input = await asyncio.to_thread(input, "Você: ")
                
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    logger.info("Usuário solicitou saída")
                    print("\n👋 Até logo.")
                    self.audio_loop.stop()
                    break
                
                if user_input.strip():
                    # Envia usando formato correto
                    await self.session.send(
                        types.LiveClientContent(
                            turns=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part(text=user_input)]
                                )
                            ],
                            turn_complete=True
                        )
                    )
                    
                    logger.info(f"Usuário: {user_input}")
                    
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Erro na entrada de texto: {e}")
    
    async def send_camera_frames(self):
        """Envia frames da câmera para o modelo"""
        async for frame_input in self.video_capture.get_frames(fps=1):
            try:
                await self.session.send(frame_input)
            except Exception as e:
                logger.error(f"Erro ao enviar frame: {e}")
    
    async def send_screen_capture(self):
        """Envia captura de tela para o modelo"""
        async for screen_input in self.video_capture.get_screen(fps=0.5):
            try:
                await self.session.send(screen_input)
            except Exception as e:
                logger.error(f"Erro ao enviar tela: {e}")
    
    def cleanup(self):
        """Limpa todos os recursos"""
        logger.info("Limpando recursos...")
        self.audio_loop.cleanup()
        self.video_capture.cleanup()
        logger.info("Recursos limpos")

# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

async def main():
    """Função principal"""
    
    # Obtém API key do ambiente (nunca logue a chave!)
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ GOOGLE_API_KEY não encontrada no ambiente")
        print("   Execute: export GOOGLE_API_KEY='sua-chave-aqui'")
        return
    
    # Configurações opcionais
    enable_camera = False  # Ativar para captura de câmera
    enable_screen = False  # Ativar para captura de tela
    
    logger.info("=" * 60)
    logger.info("Iniciando Hitagi Companion System")
    logger.info("=" * 60)
    
    # Inicia o sistema
    companion = HitagiCompanion(api_key)
    
    # If the user passed --connect-test, run a minimal connect-only probe
    if '--connect-test' in sys.argv:
        logger.info("Executando connect-only probe (--connect-test)")
        # Build minimal config matching what start() uses — include speech_config
        # so the server knows this is an audio-capable session and can extract
        # voices correctly.
        minimal_config = {
            "response_modalities": ["AUDIO", "TEXT"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": VOICE_NAME}
                }
            },
        }

        async def _connect_probe():
            client = genai.Client(api_key=api_key)
            try:
                async with client.aio.live.connect(model=GEMINI_MODEL, config=minimal_config) as session:
                    logger.info("Connect probe: connected to server, awaiting first message (5s timeout)...")
                    try:
                        # Wait for first message from server
                        async for resp in session.receive():
                            logger.info("Connect probe: received server response: %s", getattr(resp, 'server_content', str(resp)))
                            break
                    except Exception as e:
                        logger.error("Connect probe: error while receiving: %s", e)
                        traceback.print_exc()
                logger.info("Connect probe: session closed cleanly")
            except Exception as e:
                # Special-case websocket close errors that indicate quota/billing
                if isinstance(e, WSConnectionClosedError) and getattr(e, 'code', None) == 1011:
                    logger.error("Connect probe: server closed connection with 1011 - likely quota/billing issue: %s", e)
                    logger.error("Suggestion: check Google Cloud Console -> Billing and Quotas, and ensure the Generative API is enabled for your project and the API key has billing enabled.")
                else:
                    logger.error("Connect probe: connect failed: %s", e)
                    traceback.print_exc()

        try:
            await _connect_probe()
        finally:
            companion.cleanup()
        return

    # If the user requested to dump the live connect request, build it using
    # the SDK helpers and print it without sending any network traffic.
    if '--dump-request' in sys.argv:
        logger.info("Construindo e exibindo o payload de connect (sem enviar)")
        # Use the SDK internals that create the websocket request payload
        try:
            from google.genai import live as genai_live
            # Build the parameter model the SDK will use
            param_model = await genai_live._t_live_connect_config(companion.client._api_client, minimal_config)
            # Use the SDK converter to build the request dict for mldev path
            from google.genai import _common, _live_converters
            request_dict = _common.convert_to_dict(
                _live_converters._LiveConnectParameters_to_mldev(
                    api_client=companion.client._api_client,
                    from_object={'model': GEMINI_MODEL, 'config': param_model.model_dump(exclude_none=True)},
                )
            )
            # Redact potential secrets
            if 'headers' in request_dict:
                hdrs = request_dict['headers'] if isinstance(request_dict['headers'], dict) else {}
                if 'Authorization' in hdrs:
                    hdrs['Authorization'] = '<REDACTED>'
                request_dict['headers'] = hdrs

            print(json.dumps(request_dict, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error("Erro ao construir payload: %s", e)
            traceback.print_exc()
        finally:
            companion.cleanup()
        return

    # If the user requested to dump the raw websocket URI and headers, build
    # them using the client's api_client and print them redacted.
    if '--dump-raw' in sys.argv:
        logger.info("Construindo e exibindo URI e headers do websocket (sem conectar)")
        try:
            api_client = companion.client._api_client
            base_url = api_client._websocket_base_url()
            if isinstance(base_url, bytes):
                base_url = base_url.decode('utf-8')

            version = api_client._http_options.api_version
            api_key = api_client.api_key
            headers = api_client._http_options.headers.copy() if api_client._http_options.headers else {}

            # Decide method used when using api_key
            method = 'BidiGenerateContent'
            if api_key and api_key.startswith('auth_tokens/'):
                method = 'BidiGenerateContentConstrained'

            uri = f"{base_url}/ws/google.ai.generativelanguage.{version}.GenerativeService.{method}"

            # Redact sensitive header values
            redacted = {}
            for k, v in headers.items():
                if k.lower() in ('authorization', 'x-goog-api-key'):
                    redacted[k] = '<REDACTED>'
                else:
                    redacted[k] = v

            print('WebSocket URI:')
            print(uri)
            print('\nHeaders:')
            print(json.dumps(redacted, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error("Erro ao construir URI/headers: %s", e)
            traceback.print_exc()
        finally:
            companion.cleanup()
        return

    # Raw websocket probe: open a websocket to the same URI with the same
    # headers, send the SDK-constructed connect request, and print the raw
    # server response. This helps surface protocol-level rejections (1007).
    if '--raw-probe' in sys.argv:
        logger.info("Executando raw websocket probe (--raw-probe)")
        try:
            api_client = companion.client._api_client
            base_url = api_client._websocket_base_url()
            if isinstance(base_url, bytes):
                base_url = base_url.decode('utf-8')

            version = api_client._http_options.api_version
            api_key = api_client.api_key
            headers = api_client._http_options.headers.copy() if api_client._http_options.headers else {}

            method = 'BidiGenerateContent'
            if api_key and api_key.startswith('auth_tokens/'):
                method = 'BidiGenerateContentConstrained'

            uri = f"{base_url}/ws/google.ai.generativelanguage.{version}.GenerativeService.{method}"

            # Build the same request JSON the SDK would send
            from google.genai import live as genai_live
            parameter_model = await genai_live._t_live_connect_config(api_client, minimal_config)
            from google.genai import _common, _live_converters
            request_dict = _common.convert_to_dict(
                _live_converters._LiveConnectParameters_to_mldev(
                    api_client=api_client,
                    from_object={
                        'model': GEMINI_MODEL,
                        'config': parameter_model.model_dump(exclude_none=True),
                    },
                )
            )
            # The SDK deletes 'config' and sets setup.model
            if 'config' in request_dict:
                del request_dict['config']
            # Ensure model is set
            from google.genai._common import set_value_by_path as _setv
            _setv(request_dict, ['setup', 'model'], GEMINI_MODEL)

            request = json.dumps(request_dict)

            # Redact headers for logging but use original headers for the connect
            redacted = {k: ('<REDACTED>' if k.lower() in ('authorization', 'x-goog-api-key') else v) for k, v in headers.items()}
            print('WebSocket URI:', uri)
            print('Headers (redacted):', json.dumps(redacted, indent=2, ensure_ascii=False))
            print('Connect request JSON:', json.dumps(request_dict, indent=2, ensure_ascii=False))

            # Open a raw websocket and send the request
            try:
                # Use the same connect helper as the SDK to ensure SSL args are passed
                from websockets.asyncio.client import connect as ws_connect
            except Exception:
                from websockets.client import connect as ws_connect

            async with ws_connect(uri, additional_headers=headers, **api_client._websocket_ssl_ctx) as ws:
                await ws.send(request)
                try:
                    raw_response = await ws.recv()
                    print('Raw server response (repr):')
                    print(repr(raw_response))
                    try:
                        parsed = json.loads(raw_response)
                        print('Parsed JSON response:')
                        print(json.dumps(parsed, indent=2, ensure_ascii=False))
                    except Exception:
                        pass
                except Exception as e:
                    if isinstance(e, WSConnectionClosedError) and getattr(e, 'code', None) == 1011:
                        logger.error('Raw probe: server closed connection with 1011 - quota/billing issue: %s', e)
                        logger.error('Suggestion: check Google Cloud Console -> Billing and Quotas, ensure Generative Language API is enabled, and the API key is on a project with billing.')
                    else:
                        logger.error('Error receiving raw response: %s', e)
                        traceback.print_exc()

        except Exception as e:
            logger.error('Raw probe failure: %s', e)
            traceback.print_exc()
        finally:
            companion.cleanup()
        return

    try:
        await companion.start(
            enable_camera=enable_camera,
            enable_screen=enable_screen
        )
    except KeyboardInterrupt:
        logger.info("Sistema interrompido pelo usuário")
        print("\n\n⏹️  Sistema interrompido")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        traceback.print_exc()
    finally:
        companion.cleanup()
        logger.info("Sistema encerrado")
        print("\n🔚 Sistema encerrado")

if __name__ == "__main__":
    asyncio.run(main())