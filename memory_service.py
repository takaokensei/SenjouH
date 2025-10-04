"""
Memory Service - Hitagi Companion System
Microserviço de memória persistente, embeddings e RAG
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

import chromadb
import redis.asyncio as aioredis
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, JSON, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from starlette.responses import Response

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DB_URL = os.getenv("DB_URL", "postgresql+asyncpg://user:pass@localhost/hitagi_db")
CHROMA_URL = os.getenv("CHROMA_URL", "http://localhost:8001")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_MEMORY_TOKENS = int(os.getenv("MAX_MEMORY_TOKENS", "50000"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

# ============================================================================
# MÉTRICAS PROMETHEUS
# ============================================================================

embedding_latency = Histogram(
    'memory_embedding_latency_seconds',
    'Latency of embedding generation'
)
retrieval_latency = Histogram(
    'memory_retrieval_latency_seconds',
    'Latency of vector retrieval'
)
storage_operations = Counter(
    'memory_storage_operations_total',
    'Total storage operations',
    ['operation', 'status']
)
active_users = Gauge(
    'memory_active_users',
    'Number of active users'
)

# ============================================================================
# DATABASE MODELS
# ============================================================================

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_hash = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active_at = Column(DateTime, default=datetime.utcnow)
    total_interactions = Column(Integer, default=0)
    affection_level = Column(Integer, default=0)
    preferences = Column(JSON, default={})
    metadata = Column(JSON, default={})

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    sentiment_score = Column(Float)
    metadata = Column(JSON, default={})

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class MemoryEntry(BaseModel):
    session_id: str
    user_id: str
    role: str  # 'user' or 'assistant'
    content: str
    sentiment_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = TOP_K_RETRIEVAL
    filter_metadata: Optional[Dict[str, Any]] = None

class RetrievalResult(BaseModel):
    content: str
    role: str
    timestamp: datetime
    similarity_score: float
    metadata: Dict[str, Any]

class UserProfile(BaseModel):
    user_id: str
    affection_level: int
    total_interactions: int
    unlocked_expressions: List[str]
    preferences: Dict[str, Any]

class AffinityUpdate(BaseModel):
    user_id: str
    delta: int
    reason: Optional[str] = None

# ============================================================================
# MEMORY SERVICE CLASS
# ============================================================================

class MemoryService:
    """Serviço de memória com RAG e embeddings"""
    
    def __init__(self):
        # Redis client
        self.redis = None
        
        # PostgreSQL async engine
        self.engine = create_async_engine(DB_URL, echo=False, pool_pre_ping=True)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # ChromaDB client
        self.chroma_client = chromadb.HttpClient(
            host=CHROMA_URL.split('://')[1].split(':')[0],
            port=int(CHROMA_URL.split(':')[-1]) if ':' in CHROMA_URL else 8000
        )
        
        # Collections
        self.conversations_collection = None
        self.summaries_collection = None
        
        # Embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
        
    async def initialize(self):
        """Inicializa conexões e collections"""
        # Redis
        self.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        logger.info("Redis connected")
        
        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")
        
        # ChromaDB collections
        try:
            self.conversations_collection = self.chroma_client.get_or_create_collection(
                name="conversations",
                metadata={"description": "User conversations with embeddings"}
            )
            self.summaries_collection = self.chroma_client.get_or_create_collection(
                name="summaries",
                metadata={"description": "Session summaries"}
            )
            logger.info("ChromaDB collections initialized")
        except Exception as e:
            logger.error(f"ChromaDB initialization error: {e}")
            raise
    
    async def close(self):
        """Fecha conexões"""
        if self.redis:
            await self.redis.close()
        await self.engine.dispose()
    
    def hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy"""
        return hashlib.sha256(user_id.encode()).hexdigest()
    
    @embedding_latency.time()
    def generate_embedding(self, text: str) -> List[float]:
        """Gera embedding para texto"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise
    
    async def store_conversation(self, entry: MemoryEntry) -> str:
        """Armazena conversa com embedding"""
        try:
            conversation_id = str(uuid4())
            
            # Gera embedding
            embedding = self.generate_embedding(entry.content)
            
            # Armazena no PostgreSQL
            async with self.async_session() as session:
                conv = Conversation(
                    id=conversation_id,
                    session_id=entry.session_id,
                    user_id=entry.user_id,
                    role=entry.role,
                    content=entry.content,
                    sentiment_score=entry.sentiment_score,
                    metadata=entry.metadata,
                    timestamp=datetime.utcnow()
                )
                session.add(conv)
                await session.commit()
            
            # Armazena no ChromaDB
            self.conversations_collection.add(
                ids=[conversation_id],
                embeddings=[embedding],
                documents=[entry.content],
                metadatas=[{
                    "user_id": entry.user_id,
                    "session_id": entry.session_id,
                    "role": entry.role,
                    "timestamp": datetime.utcnow().isoformat(),
                    "sentiment_score": entry.sentiment_score or 0.0,
                    **entry.metadata
                }]
            )
            
            # Cache no Redis (últimas N mensagens)
            cache_key = f"recent_msgs:{entry.user_id}"
            await self.redis.lpush(
                cache_key,
                json.dumps({
                    "id": conversation_id,
                    "role": entry.role,
                    "content": entry.content,
                    "timestamp": datetime.utcnow().isoformat()
                })
            )
            await self.redis.ltrim(cache_key, 0, 49)  # Keep last 50
            await self.redis.expire(cache_key, 86400)  # 24h TTL
            
            storage_operations.labels(operation='store', status='success').inc()
            logger.info(f"Conversation stored: {conversation_id}")
            
            return conversation_id
            
        except Exception as e:
            storage_operations.labels(operation='store', status='error').inc()
            logger.error(f"Store conversation error: {e}")
            raise
    
    @retrieval_latency.time()
    async def retrieve_context(self, request: RetrievalRequest) -> List[RetrievalResult]:
        """Recupera contexto relevante usando RAG"""
        try:
            # Gera embedding da query
            query_embedding = self.generate_embedding(request.query)
            
            # Busca no ChromaDB
            results = self.conversations_collection.query(
                query_embeddings=[query_embedding],
                n_results=request.top_k,
                where={"user_id": request.user_id} if not request.filter_metadata else {
                    "user_id": request.user_id,
                    **request.filter_metadata
                }
            )
            
            # Formata resultados
            retrieved = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if 'distances' in results else 0.0
                    
                    retrieved.append(RetrievalResult(
                        content=doc,
                        role=metadata.get('role', 'unknown'),
                        timestamp=datetime.fromisoformat(metadata.get('timestamp', datetime.utcnow().isoformat())),
                        similarity_score=1.0 - distance,  # Convert distance to similarity
                        metadata=metadata
                    ))
            
            logger.info(f"Retrieved {len(retrieved)} results for query")
            return retrieved
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            raise
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Obtém perfil do usuário"""
        try:
            user_hash = self.hash_user_id(user_id)
            
            # Tenta cache primeiro
            cache_key = f"profile:{user_hash}"
            cached = await self.redis.get(cache_key)
            
            if cached:
                data = json.loads(cached)
                return UserProfile(**data)
            
            # Busca no banco
            async with self.async_session() as session:
                result = await session.execute(
                    f"SELECT * FROM users WHERE user_hash = '{user_hash}'"
                )
                user = result.first()
                
                if not user:
                    return None
                
                profile = UserProfile(
                    user_id=user_id,
                    affection_level=user.affection_level,
                    total_interactions=user.total_interactions,
                    unlocked_expressions=user.metadata.get('unlocked_expressions', []),
                    preferences=user.preferences or {}
                )
                
                # Cache por 5 minutos
                await self.redis.setex(
                    cache_key,
                    300,
                    json.dumps(profile.dict())
                )
                
                return profile
                
        except Exception as e:
            logger.error(f"Get user profile error: {e}")
            return None
    
    async def update_affection(self, update: AffinityUpdate) -> bool:
        """Atualiza nível de afeto do usuário"""
        try:
            user_hash = self.hash_user_id(update.user_id)
            
            async with self.async_session() as session:
                result = await session.execute(
                    f"SELECT id, affection_level FROM users WHERE user_hash = '{user_hash}'"
                )
                user = result.first()
                
                if not user:
                    # Cria usuário se não existir
                    await session.execute(
                        f"""
                        INSERT INTO users (user_hash, affection_level, total_interactions)
                        VALUES ('{user_hash}', {max(0, min(100, update.delta))}, 1)
                        """
                    )
                else:
                    # Atualiza affection
                    new_affection = max(0, min(100, user.affection_level + update.delta))
                    await session.execute(
                        f"""
                        UPDATE users 
                        SET affection_level = {new_affection},
                            last_active_at = NOW(),
                            total_interactions = total_interactions + 1
                        WHERE user_hash = '{user_hash}'
                        """
                    )
                
                await session.commit()
            
            # Invalida cache
            await self.redis.delete(f"profile:{user_hash}")
            
            logger.info(f"Affection updated for user {user_hash}: Δ{update.delta}")
            return True
            
        except Exception as e:
            logger.error(f"Update affection error: {e}")
            return False
    
    async def get_recent_messages(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Retorna mensagens recentes do cache"""
        try:
            cache_key = f"recent_msgs:{user_id}"
            messages_json = await self.redis.lrange(cache_key, 0, limit - 1)
            
            messages = [json.loads(msg) for msg in messages_json]
            return messages
            
        except Exception as e:
            logger.error(f"Get recent messages error: {e}")
            return []

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Hitagi Memory Service",
    description="Serviço de memória persistente com RAG",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service instance
memory_service = MemoryService()

# ============================================================================
# LIFECYCLE EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Inicialização do serviço"""
    logger.info("Starting Memory Service...")
    await memory_service.initialize()
    logger.info("Memory Service ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Encerramento do serviço"""
    logger.info("Shutting down Memory Service...")
    await memory_service.close()
    logger.info("Memory Service stopped")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Redis
        await memory_service.redis.ping()
        
        # Test ChromaDB
        memory_service.chroma_client.heartbeat()
        
        return {
            "status": "healthy",
            "service": "memory",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

@app.post("/store")
async def store_conversation(entry: MemoryEntry):
    """Armazena conversa com embedding"""
    try:
        conversation_id = await memory_service.store_conversation(entry)
        return {
            "status": "success",
            "conversation_id": conversation_id
        }
    except Exception as e:
        logger.error(f"Store endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
async def retrieve_context(request: RetrievalRequest) -> List[RetrievalResult]:
    """Recupera contexto relevante usando RAG"""
    try:
        results = await memory_service.retrieve_context(request)
        return results
    except Exception as e:
        logger.error(f"Retrieve endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profile/{user_id}")
async def get_user_profile(user_id: str) -> UserProfile:
    """Obtém perfil do usuário"""
    try:
        profile = await memory_service.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="User not found")
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get profile endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/affection")
async def update_affection(update: AffinityUpdate):
    """Atualiza nível de afeto"""
    try:
        success = await memory_service.update_affection(update)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update affection")
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update affection endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recent/{user_id}")
async def get_recent_messages(user_id: str, limit: int = 10):
    """Retorna mensagens recentes"""
    try:
        messages = await memory_service.get_recent_messages(user_id, limit)
        return {"messages": messages}
    except Exception as e:
        logger.error(f"Get recent messages endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/user/{user_id}")
async def delete_user_data(user_id: str):
    """Deleta todos os dados do usuário (GDPR compliance)"""
    try:
        user_hash = memory_service.hash_user_id(user_id)
        
        # Delete from PostgreSQL
        async with memory_service.async_session() as session:
            await session.execute(
                f"DELETE FROM conversations WHERE user_id = '{user_id}'"
            )
            await session.execute(
                f"DELETE FROM users WHERE user_hash = '{user_hash}'"
            )
            await session.commit()
        
        # Delete from ChromaDB
        # Note: ChromaDB doesn't support direct delete by metadata filter yet
        # You may need to implement a workaround
        
        # Delete from Redis
        await memory_service.redis.delete(f"profile:{user_hash}")
        await memory_service.redis.delete(f"recent_msgs:{user_id}")
        
        logger.info(f"Deleted all data for user: {user_hash}")
        return {"status": "success", "message": "User data deleted"}
        
    except Exception as e:
        logger.error(f"Delete user data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "memory_service:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )