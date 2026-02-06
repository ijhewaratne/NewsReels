# =============================================================================
# CHROMADB INTEGRATION - Vector Memory for Multi-Agent Swarm
# =============================================================================
"""
ChromaDB integration for semantic memory, content similarity search,
and agent knowledge persistence.

Features:
- Collection management per agent/topic
- Embedding strategies for different content types
- Semantic search and similarity matching
- Content deduplication via embeddings
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

# Try to import chromadb
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not installed. Vector memory will be disabled.")


# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

class EmbeddingModel(str, Enum):
    """Supported embedding models"""
    # OpenAI
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    
    # Sentence Transformers (local)
    ALL_MINILM_L6 = "all-MiniLM-L6-v2"
    ALL_MPNET_BASE = "all-mpnet-base-v2"
    
    # Custom
    CUSTOM = "custom"


EMBEDDING_DIMENSIONS = {
    EmbeddingModel.OPENAI_ADA_002: 1536,
    EmbeddingModel.OPENAI_3_SMALL: 1536,
    EmbeddingModel.OPENAI_3_LARGE: 3072,
    EmbeddingModel.ALL_MINILM_L6: 384,
    EmbeddingModel.ALL_MPNET_BASE: 768,
}


# =============================================================================
# COLLECTION SCHEMAS
# =============================================================================

COLLECTION_SCHEMAS = {
    "news_articles": {
        "description": "Raw news articles and content",
        "metadata_schema": {
            "source": str,
            "published_at": str,
            "topics": list,
            "sentiment": float,
            "credibility": float,
        }
    },
    "debate_arguments": {
        "description": "Agent debate arguments and positions",
        "metadata_schema": {
            "debate_id": str,
            "agent_id": str,
            "agent_role": str,
            "argument_type": str,
            "confidence": float,
        }
    },
    "scripts": {
        "description": "Generated video scripts",
        "metadata_schema": {
            "script_id": str,
            "news_item_id": str,
            "version": int,
            "tone": str,
            "word_count": int,
        }
    },
    "visual_descriptions": {
        "description": "Visual scene descriptions",
        "metadata_schema": {
            "scene_id": str,
            "script_id": str,
            "scene_number": int,
            "visual_style": str,
        }
    },
    "agent_memory": {
        "description": "Agent-specific memories and learnings",
        "metadata_schema": {
            "agent_id": str,
            "memory_type": str,
            "importance": float,
            "created_at": str,
        }
    },
    "published_videos": {
        "description": "Published video content for reference",
        "metadata_schema": {
            "video_id": str,
            "topic": str,
            "published_at": str,
            "performance_score": float,
        }
    },
}


# =============================================================================
# CHROMADB MANAGER
# =============================================================================

class ChromaManager:
    """
    Manager for ChromaDB vector database operations.
    
    Provides:
    - Collection management
    - Document storage with embeddings
    - Semantic search
    - Similarity-based deduplication
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        auth_token: Optional[str] = None,
        embedding_model: EmbeddingModel = EmbeddingModel.ALL_MINILM_L6,
        embedding_function: Optional[Callable] = None
    ):
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.embedding_model = embedding_model
        self.embedding_function = embedding_function
        self._client = None
        self._collections: Dict[str, Any] = {}
        
    async def connect(self):
        """Connect to ChromaDB"""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, vector operations disabled")
            return
        
        try:
            settings = Settings(
                chroma_server_host=self.host,
                chroma_server_http_port=self.port,
                chroma_server_headers={
                    "X-Chroma-Token": self.auth_token
                } if self.auth_token else {}
            )
            
            self._client = chromadb.HttpClient(settings=settings)
            
            # Initialize collections
            await self._initialize_collections()
            
            logger.info(f"Connected to ChromaDB at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    async def _initialize_collections(self):
        """Initialize all collections"""
        for name, schema in COLLECTION_SCHEMAS.items():
            try:
                collection = self._client.get_or_create_collection(
                    name=name,
                    metadata={
                        "description": schema["description"],
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                )
                self._collections[name] = collection
                logger.debug(f"Initialized collection: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize collection {name}: {e}")
    
    # ========================================================================
    # DOCUMENT OPERATIONS
    # ========================================================================
    
    async def add_document(
        self,
        collection_name: str,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        check_duplicate: bool = True,
        similarity_threshold: float = 0.95
    ) -> bool:
        """
        Add a document to a collection.
        
        Args:
            collection_name: Target collection
            document_id: Unique document ID
            content: Document content
            metadata: Optional metadata
            check_duplicate: Whether to check for duplicates
            similarity_threshold: Threshold for duplicate detection
            
        Returns:
            True if added, False if duplicate detected
        """
        if not self._client:
            logger.warning("ChromaDB not connected")
            return False
        
        collection = self._collections.get(collection_name)
        if not collection:
            logger.error(f"Collection not found: {collection_name}")
            return False
        
        # Check for duplicates
        if check_duplicate:
            is_duplicate = await self._check_duplicate(
                collection_name,
                content,
                similarity_threshold
            )
            if is_duplicate:
                logger.info(f"Duplicate detected for document {document_id}")
                return False
        
        # Add metadata
        doc_metadata = metadata or {}
        doc_metadata["added_at"] = datetime.now(timezone.utc).isoformat()
        doc_metadata["content_hash"] = self._compute_hash(content)
        
        try:
            collection.add(
                ids=[document_id],
                documents=[content],
                metadatas=[doc_metadata]
            )
            logger.debug(f"Added document {document_id} to {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    async def get_document(
        self,
        collection_name: str,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        if not self._client:
            return None
        
        collection = self._collections.get(collection_name)
        if not collection:
            return None
        
        try:
            result = collection.get(ids=[document_id])
            if result and result['ids']:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except Exception as e:
            logger.error(f"Failed to get document: {e}")
        
        return None
    
    async def update_document(
        self,
        collection_name: str,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a document"""
        if not self._client:
            return False
        
        collection = self._collections.get(collection_name)
        if not collection:
            return False
        
        try:
            update_data = {}
            if content:
                update_data['documents'] = [content]
            if metadata:
                metadata['updated_at'] = datetime.now(timezone.utc).isoformat()
                update_data['metadatas'] = [metadata]
            
            collection.update(
                ids=[document_id],
                **update_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            return False
    
    async def delete_document(
        self,
        collection_name: str,
        document_id: str
    ) -> bool:
        """Delete a document"""
        if not self._client:
            return False
        
        collection = self._collections.get(collection_name)
        if not collection:
            return False
        
        try:
            collection.delete(ids=[document_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    # ========================================================================
    # SEARCH OPERATIONS
    # ========================================================================
    
    async def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search in a collection.
        
        Returns:
            List of matching documents with similarity scores
        """
        if not self._client:
            return []
        
        collection = self._collections.get(collection_name)
        if not collection:
            return []
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )
            
            documents = []
            for i, doc_id in enumerate(results['ids'][0]):
                documents.append({
                    'id': doc_id,
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def find_similar(
        self,
        collection_name: str,
        content: str,
        threshold: float = 0.85,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Find documents similar to given content"""
        results = await self.search(
            collection_name=collection_name,
            query=content,
            n_results=n_results
        )
        
        # Filter by threshold
        return [r for r in results if r['similarity'] >= threshold]
    
    async def _check_duplicate(
        self,
        collection_name: str,
        content: str,
        threshold: float
    ) -> bool:
        """Check if similar content already exists"""
        similar = await self.find_similar(
            collection_name=collection_name,
            content=content,
            threshold=threshold,
            n_results=1
        )
        return len(similar) > 0
    
    # ========================================================================
    # AGENT MEMORY OPERATIONS
    # ========================================================================
    
    async def store_agent_memory(
        self,
        agent_id: str,
        memory_type: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory for an agent.
        
        Returns:
            Memory ID
        """
        memory_id = f"mem:{agent_id}:{int(datetime.now(timezone.utc).timestamp())}"
        
        memory_metadata = {
            "agent_id": agent_id,
            "memory_type": memory_type,
            "importance": importance,
            **(metadata or {})
        }
        
        await self.add_document(
            collection_name="agent_memory",
            document_id=memory_id,
            content=content,
            metadata=memory_metadata,
            check_duplicate=False
        )
        
        return memory_id
    
    async def recall_agent_memories(
        self,
        agent_id: str,
        query: str,
        memory_type: Optional[str] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Recall agent memories matching query"""
        filter_metadata = {"agent_id": agent_id}
        if memory_type:
            filter_metadata["memory_type"] = memory_type
        
        return await self.search(
            collection_name="agent_memory",
            query=query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )
    
    async def get_important_memories(
        self,
        agent_id: str,
        min_importance: float = 0.7,
        n_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Get agent's most important memories"""
        # Get all memories for agent
        collection = self._collections.get("agent_memory")
        if not collection:
            return []
        
        try:
            results = collection.get(
                where={"agent_id": agent_id}
            )
            
            memories = []
            for i, doc_id in enumerate(results['ids']):
                importance = results['metadatas'][i].get('importance', 0)
                if importance >= min_importance:
                    memories.append({
                        'id': doc_id,
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i],
                        'importance': importance
                    })
            
            # Sort by importance
            memories.sort(key=lambda x: x['importance'], reverse=True)
            return memories[:n_results]
            
        except Exception as e:
            logger.error(f"Failed to get important memories: {e}")
            return []
    
    # ========================================================================
    # NEWS CONTENT OPERATIONS
    # ========================================================================
    
    async def store_news_article(
        self,
        article_id: str,
        title: str,
        content: str,
        source: str,
        published_at: str,
        topics: List[str],
        sentiment: float = 0.0,
        credibility: float = 0.5
    ) -> bool:
        """Store a news article"""
        full_content = f"{title}\n\n{content}"
        
        metadata = {
            "source": source,
            "published_at": published_at,
            "topics": topics,
            "sentiment": sentiment,
            "credibility": credibility,
            "title": title
        }
        
        return await self.add_document(
            collection_name="news_articles",
            document_id=article_id,
            content=full_content,
            metadata=metadata,
            check_duplicate=True,
            similarity_threshold=0.92
        )
    
    async def find_related_news(
        self,
        query: str,
        topics: Optional[List[str]] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Find news articles related to query"""
        filter_metadata = None
        if topics:
            filter_metadata = {"topics": {"$contains": topics}}
        
        return await self.search(
            collection_name="news_articles",
            query=query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )
    
    # ========================================================================
    # SCRIPT OPERATIONS
    # ========================================================================
    
    async def store_script(
        self,
        script_id: str,
        news_item_id: str,
        content: str,
        version: int = 1,
        tone: str = "neutral",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a video script"""
        script_metadata = {
            "script_id": script_id,
            "news_item_id": news_item_id,
            "version": version,
            "tone": tone,
            "word_count": len(content.split()),
            **(metadata or {})
        }
        
        return await self.add_document(
            collection_name="scripts",
            document_id=f"{script_id}:v{version}",
            content=content,
            metadata=script_metadata,
            check_duplicate=False
        )
    
    async def find_similar_scripts(
        self,
        content: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Find scripts similar to given content"""
        return await self.find_similar(
            collection_name="scripts",
            content=content,
            threshold=0.8,
            n_results=n_results
        )
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _compute_hash(self, content: str) -> str:
        """Compute content hash"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        collection = self._collections.get(collection_name)
        if not collection:
            return {}
        
        try:
            count = collection.count()
            return {
                "name": collection_name,
                "document_count": count,
                "description": COLLECTION_SCHEMAS.get(collection_name, {}).get("description", "")
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        stats = {}
        for name in COLLECTION_SCHEMAS.keys():
            stats[name] = await self.get_collection_stats(name)
        return stats


# =============================================================================
# EMBEDDING UTILITIES
# =============================================================================

class EmbeddingProvider:
    """Provider for text embeddings"""
    
    def __init__(
        self,
        model: EmbeddingModel = EmbeddingModel.ALL_MINILM_L6,
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key
        self._embedding_function = None
        
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts"""
        if self.model in [EmbeddingModel.ALL_MINILM_L6, EmbeddingModel.ALL_MPNET_BASE]:
            # Use sentence-transformers (local)
            return await self._get_local_embeddings(texts)
        elif self.model in [EmbeddingModel.OPENAI_ADA_002, 
                           EmbeddingModel.OPENAI_3_SMALL,
                           EmbeddingModel.OPENAI_3_LARGE]:
            # Use OpenAI API
            return await self._get_openai_embeddings(texts)
        else:
            raise ValueError(f"Unsupported embedding model: {self.model}")
    
    async def _get_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using local model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            if self._embedding_function is None:
                self._embedding_function = SentenceTransformer(self.model.value)
            
            embeddings = self._embedding_function.encode(texts)
            return embeddings.tolist()
            
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise
    
    async def _get_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI API"""
        import aiohttp
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json={
                    "model": self.model.value,
                    "input": texts
                }
            ) as response:
                result = await response.json()
                return [item["embedding"] for item in result["data"]]


# Import Enum
from enum import Enum
