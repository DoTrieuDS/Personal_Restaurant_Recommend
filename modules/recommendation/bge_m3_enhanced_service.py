"""
Enhanced BGE-M3 Service for Content-based Scoring
Service để sử dụng BGE-M3 embeddings cho content-based filtering
"""

import os
import time
import torch
import logging
import numpy as np
import pandas as pd
import psutil
from typing import List, Dict, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
from datetime import datetime
import gc
from .model_config import model_manager, ModelConfig
from .data_config import DataConfig


class BGE_M3_EnhancedService:
    """
    Enhanced BGE-M3 Service với pre-computed embeddings và lazy loading
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = model_manager.get_model_config('bge_m3')
        self.model_config = ModelConfig()
        
        # Model paths
        self.model_path = model_manager.get_absolute_path('bge_m3')
        self.embeddings_path = DataConfig.RESTAURANT_EMBEDDINGS
        
        # Lazy loading attributes
        self._model = None
        self._embeddings_df = None
        self._embeddings_matrix = None
        self._business_id_to_idx = {}
        self._document_matrix = None
        self._document_ids = []
        self._is_loaded = False
        
        # Memory monitoring
        self._last_access_time = None
        self._memory_usage_mb = 0
        
        # Cache for query embeddings
        self._cache = OrderedDict()
        self._cache_size = self.config.get('cache_size', 1000)
    
    def _check_memory(self) -> bool:
        """Check if enough memory available"""
        process = psutil.Process()
        current_usage = process.memory_info().rss / (1024 * 1024)
        available = psutil.virtual_memory().available / (1024 * 1024)
        
        if current_usage > self.model_config.MAX_MEMORY_MB * 0.7:
            self.logger.warning(f"⚠️ High memory usage: {current_usage:.1f}MB")
            return False
        
        if available < 500:
            self.logger.warning(f"⚠️ Low system memory: {available:.1f}MB")
            return False
        
        return True
    
    def _load_model(self):
        """Lazy load BGE-M3 model"""
        if self._model is not None:
            self._last_access_time = time.time()
            return
        
        # Check memory
        if not self._check_memory():
            raise MemoryError("Insufficient memory to load BGE-M3 model")
        
        self.logger.info("🔄 Loading BGE-M3 model...")
        start_time = time.time()
        
        try:
            # Use existing OptimizedLocalLoRAService if available
            from .optimized_local_lora_service import OptimizedLocalLoRAService
            
            # Initialize with optimized settings
            self._model = OptimizedLocalLoRAService(
                lora_path=self.model_path,
                device="auto"
            )
            
            self._is_loaded = True
            self._last_access_time = time.time()
            
            load_time = time.time() - start_time
            self.logger.info(f"✅ BGE-M3 model loaded in {load_time:.2f}s")
            
        except ImportError:
            # Fallback to SentenceTransformers
            self.logger.info("🔄 Using SentenceTransformers fallback...")
            
            self._model = SentenceTransformer(
                self.model_path,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Optimize for inference
            self._model.eval()
            self._model.max_seq_length = self.config['max_seq_length']
            
            self._is_loaded = True
            self._last_access_time = time.time()
            
            load_time = time.time() - start_time
            self.logger.info(f"✅ BGE-M3 model loaded (fallback) in {load_time:.2f}s")
    
    def _load_embeddings(self):
        """Load pre-computed embeddings"""
        if self._embeddings_matrix is not None:
            return
        
        self.logger.info("🔄 Loading pre-computed embeddings...")
        start_time = time.time()
        
        try:
            # Load embeddings from parquet
            self._embeddings_df = pd.read_parquet(self.embeddings_path)
            
            # Extract embeddings matrix
            embeddings_list = []
            business_ids = []
            
            for idx, row in self._embeddings_df.iterrows():
                embedding = row['embedding']
                
                # Handle different formats
                if isinstance(embedding, str):
                    embedding = eval(embedding)
                
                embeddings_list.append(np.array(embedding, dtype=np.float32))
                business_ids.append(row['business_id'])
            
            # Create matrix
            self._embeddings_matrix = np.vstack(embeddings_list)
            
            # Create mapping
            self._business_id_to_idx = {bid: idx for idx, bid in enumerate(business_ids)}
            
            load_time = time.time() - start_time
            self._memory_usage_mb = self._embeddings_matrix.nbytes / (1024 * 1024)
            
            self.logger.info(f"✅ Loaded {len(business_ids)} embeddings in {load_time:.2f}s")
            self.logger.info(f"   Memory usage: {self._memory_usage_mb:.1f}MB")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading embeddings: {e}")
            raise
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Returns all loaded restaurant embeddings as a dictionary.
        This is used to prime other services like online learning.
        
        Returns:
            Dict mapping business_id -> embedding_vector
        """
        self._load_embeddings()  # Ensure embeddings are loaded
        
        if self._embeddings_matrix is None or not self._business_id_to_idx:
            self.logger.warning("No embeddings loaded, returning empty dict")
            return {}
        
        # Convert matrix + mapping to Dict[str, np.ndarray] format for Online Learning
        embeddings_dict = {}
        for business_id, idx in self._business_id_to_idx.items():
            embeddings_dict[business_id] = self._embeddings_matrix[idx].copy()
        
        self.logger.info(f"Prepared {len(embeddings_dict)} content embeddings for brain fusion")
        return embeddings_dict
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query text to embedding
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        # Check cache
        if query in self._cache:
            self._last_access_time = time.time()
            return self._cache[query]
        
        # Load model if needed
        self._load_model()
        
        # Encode
        if hasattr(self._model, 'encode_texts'):
            # Using OptimizedLocalLoRAService
            embedding = self._model.encode_texts(query, normalize=True)[0]
        else:
            # Using SentenceTransformers
            embedding = self._model.encode(query, normalize_embeddings=True)
        
        # Update cache
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[query] = embedding
        
        return embedding
    
    def get_content_scores(self, query: str, candidate_ids: List[str]) -> Dict[str, float]:
        """
        Get content-based scores for candidates
        
        Args:
            query: User query
            candidate_ids: List of business IDs
            
        Returns:
            Dictionary mapping business_id to content score
        """
        try:
            # Load embeddings if needed
            self._load_embeddings()
            
            # Encode query
            query_embedding = self.encode_query(query)
            
            # Get candidate embeddings
            scores = {}
            
            for business_id in candidate_ids:
                if business_id in self._business_id_to_idx:
                    idx = self._business_id_to_idx[business_id]
                    candidate_embedding = self._embeddings_matrix[idx]
                    
                    # Compute similarity
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        candidate_embedding.reshape(1, -1)
                    )[0, 0]
                    
                    scores[business_id] = float(similarity)
                else:
                    scores[business_id] = 0.0
            
            return scores
            
        except Exception as e:
            self.logger.error(f"❌ Error computing content scores: {e}")
            return {bid: 0.0 for bid in candidate_ids}
    
    def get_user_preference_embedding(self, liked_restaurant_ids: List[str]) -> Optional[np.ndarray]:
        """
        Create user preference embedding from liked restaurants
        
        Args:
            liked_restaurant_ids: List of liked restaurant IDs
            
        Returns:
            Average embedding of liked restaurants
        """
        try:
            # Load embeddings if needed
            self._load_embeddings()
            
            embeddings = []
            
            for restaurant_id in liked_restaurant_ids:
                if restaurant_id in self._business_id_to_idx:
                    idx = self._business_id_to_idx[restaurant_id]
                    embeddings.append(self._embeddings_matrix[idx])
            
            if embeddings:
                # Return average embedding
                return np.mean(embeddings, axis=0)
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error creating preference embedding: {e}")
            return None
    
    def compute_preference_scores(self, 
                                 user_preference_embedding: np.ndarray,
                                 candidate_ids: List[str]) -> Dict[str, float]:
        """
        Compute scores based on user preference embedding
        
        Args:
            user_preference_embedding: User's preference embedding
            candidate_ids: List of business IDs
            
        Returns:
            Dictionary mapping business_id to preference score
        """
        try:
            scores = {}
            
            for business_id in candidate_ids:
                if business_id in self._business_id_to_idx:
                    idx = self._business_id_to_idx[business_id]
                    candidate_embedding = self._embeddings_matrix[idx]
                    
                    # Compute similarity
                    similarity = cosine_similarity(
                        user_preference_embedding.reshape(1, -1),
                        candidate_embedding.reshape(1, -1)
                    )[0, 0]
                    
                    scores[business_id] = float(similarity)
                else:
                    scores[business_id] = 0.0
            
            return scores
            
        except Exception as e:
            self.logger.error(f"❌ Error computing preference scores: {e}")
            return {bid: 0.0 for bid in candidate_ids}
    
    def unload(self):
        """Unload models and data to free memory"""
        self.logger.info("🗑️ Unloading BGE-M3 service...")
        
        # Unload model
        if hasattr(self._model, 'unload'):
            self._model.unload()
        self._model = None
        
        # Clear embeddings
        self._embeddings_df = None
        self._embeddings_matrix = None
        self._business_id_to_idx = {}
        
        # Clear cache
        self._cache.clear()
        
        self._is_loaded = False
        self._memory_usage_mb = 0
        
        # Force garbage collection
        gc.collect()
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            'model_loaded': self._is_loaded,
            'embeddings_loaded': self._embeddings_matrix is not None,
            'num_embeddings': len(self._business_id_to_idx),
            'memory_usage_mb': self._memory_usage_mb,
            'cache_size': len(self._cache),
            'last_access': datetime.fromtimestamp(self._last_access_time).isoformat() if self._last_access_time else None
        }


def demo_bge_m3_enhanced():
    """Demo enhanced BGE-M3 service"""
    print("DEMO ENHANCED BGE-M3 SERVICE")
    print("=" * 50)
    
    service = BGE_M3_EnhancedService()
    
    # Test encoding
    print("\n1. Testing query encoding...")
    query = "vietnamese pho restaurant"
    embedding = service.encode_query(query)
    print(f"   Query: '{query}'")
    print(f"   Embedding shape: {embedding.shape}")
    
    # Test content scoring
    print("\n2. Testing content scoring...")
    candidate_ids = ["rest001", "rest002", "rest003"]
    scores = service.get_content_scores(query, candidate_ids)
    
    print(f"   Content scores:")
    for bid, score in scores.items():
        print(f"     {bid}: {score:.3f}")
    
    # Test preference embedding
    print("\n3. Testing preference embedding...")
    liked_restaurants = ["rest001", "rest003"]
    pref_embedding = service.get_user_preference_embedding(liked_restaurants)
    
    if pref_embedding is not None:
        print(f"   Created preference embedding from {len(liked_restaurants)} restaurants")
        print(f"   Shape: {pref_embedding.shape}")
        
        # Test preference scoring
        pref_scores = service.compute_preference_scores(pref_embedding, candidate_ids)
        print(f"   Preference scores:")
        for bid, score in pref_scores.items():
            print(f"     {bid}: {score:.3f}")
    
    # Get status
    print("\n4. Service status:")
    status = service.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\nEnhanced BGE-M3 demo completed!")


if __name__ == "__main__":
    demo_bge_m3_enhanced() 