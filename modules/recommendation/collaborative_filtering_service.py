"""
Collaborative Filtering Service
Sử dụng extracted embeddings từ Funk-SVD để tính collaborative scores
"""

import os
import numpy as np
import json
import pickle
import logging
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import time

from .model_config import model_manager

class CollaborativeFilteringService:
    """
    Service tính collaborative filtering scores từ extracted embeddings
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = model_manager.get_model_config('collaborative_filtering')
        
        # Model components
        self._user_embeddings = None
        self._item_embeddings = None
        self._user_biases = None
        self._item_biases = None
        self._global_mean = None
        
        # Mappings
        self._user_to_idx = None
        self._item_to_idx = None
        self._idx_to_user = None
        self._idx_to_item = None
        
        # Model metadata
        self._n_users = 0
        self._n_items = 0
        self._embedding_dim = 0
        
        # Cache for frequently accessed users/items
        self._user_cache = OrderedDict()
        self._item_cache = OrderedDict()
        
        self._is_loaded = False
        self._last_access_time = None
        
    def _load_embeddings(self):
        """Load CF embeddings và mappings"""
        if self._is_loaded:
            self._last_access_time = time.time()
            return
            
        try:
            self.logger.info("Loading Collaborative Filtering embeddings...")
            
            # Load embeddings
            embeddings_path = model_manager.get_absolute_path('cf_embeddings')
            if not os.path.exists(embeddings_path):
                self.logger.warning(f"CF embeddings not found: {embeddings_path}")
                return
                
            data = np.load(embeddings_path)
            
            # Load arrays
            self._user_embeddings = data['user_embeddings']
            self._item_embeddings = data['item_embeddings']
            self._user_biases = data['user_biases']
            self._item_biases = data['item_biases']
            self._global_mean = float(data['global_mean'])
            
            # Convert to float16 nếu config enable
            if self.config.get('use_float16', False):
                self._user_embeddings = self._user_embeddings.astype(np.float16)
                self._item_embeddings = self._item_embeddings.astype(np.float16)
                self._user_biases = self._user_biases.astype(np.float16)
                self._item_biases = self._item_biases.astype(np.float16)
            
            # Model metadata
            self._n_users = self._user_embeddings.shape[0]
            self._n_items = self._item_embeddings.shape[0]
            self._embedding_dim = self._user_embeddings.shape[1]
            
            # Load mappings
            mappings_path = model_manager.get_absolute_path('cf_mappings')
            if os.path.exists(mappings_path):
                mappings_data = np.load(mappings_path, allow_pickle=True)
                
                self._user_to_idx = mappings_data['user_to_idx'].item()
                self._item_to_idx = mappings_data['item_to_idx'].item()
                self._idx_to_user = mappings_data['idx_to_user'].item()
                self._idx_to_item = mappings_data['idx_to_item'].item()
            
            self._is_loaded = True
            self._last_access_time = time.time()
            
            # Calculate memory usage
            user_mem = self._user_embeddings.nbytes / (1024*1024)
            item_mem = self._item_embeddings.nbytes / (1024*1024)
            
            self.logger.info(f"CF embeddings loaded successfully")
            self.logger.info(f"   Users: {self._n_users:,}, Items: {self._n_items:,}")
            self.logger.info(f"   Embedding dim: {self._embedding_dim}")
            self.logger.info(f"   Memory usage: {user_mem + item_mem:.1f} MB")
            
        except Exception as e:
            self.logger.error(f"Error loading CF embeddings: {e}")
            self._is_loaded = False
    
    def get_user_item_score(self, user_id: str, item_id: str) -> float:
        """
        Tính score cho user-item pair
        
        Args:
            user_id: String user ID
            item_id: String item ID (business_id)
            
        Returns:
            Predicted rating score (0-5 scale)
        """
        self._load_embeddings()
        
        if not self._is_loaded:
            return 0.5  # Fallback score
        
        try:
            # Get indices
            user_idx = self._user_to_idx.get(user_id)
            item_idx = self._item_to_idx.get(item_id)
            
            if user_idx is None or item_idx is None:
                return 0.5  # Cold start fallback
            
            # Get embeddings
            user_emb = self._user_embeddings[user_idx]
            item_emb = self._item_embeddings[item_idx]
            
            # Get biases
            user_bias = self._user_biases[user_idx]
            item_bias = self._item_biases[item_idx]
            
            # Calculate predicted rating: global_mean + bu + bi + pu^T * qi
            dot_product = np.dot(user_emb, item_emb)
            predicted_rating = self._global_mean + user_bias + item_bias + dot_product
            
            # Clamp to valid rating range
            predicted_rating = max(1.0, min(5.0, predicted_rating))
            
            return float(predicted_rating)
            
        except Exception as e:
            self.logger.error(f"Error calculating user-item score: {e}")
            return 0.5
    
    def get_user_similarity_scores(self, user_id: str, item_ids: List[str]) -> Dict[str, float]:
        """
        Tính similarity scores cho list items với user
        
        Args:
            user_id: User ID
            item_ids: List of item IDs to score
            
        Returns:
            Dict mapping item_id -> normalized score (0-1)
        """
        self._load_embeddings()
        
        if not self._is_loaded:
            return {item_id: 0.5 for item_id in item_ids}
        
        try:
            scores = {}
            
            # Get user index
            user_idx = self._user_to_idx.get(user_id)
            if user_idx is None:
                # Cold start user - return average scores
                return {item_id: 0.5 for item_id in item_ids}
            
            # Get user embedding
            user_emb = self._user_embeddings[user_idx]
            user_bias = self._user_biases[user_idx]
            
            for item_id in item_ids:
                item_idx = self._item_to_idx.get(item_id)
                
                if item_idx is None:
                    scores[item_id] = 0.5  # Cold start item
                    continue
                
                # Get item embedding
                item_emb = self._item_embeddings[item_idx]
                item_bias = self._item_biases[item_idx]
                
                # Calculate predicted rating
                dot_product = np.dot(user_emb, item_emb)
                predicted_rating = self._global_mean + user_bias + item_bias + dot_product
                
                # Normalize to 0-1 scale (rating 1-5 -> 0-1)
                normalized_score = (predicted_rating - 1.0) / 4.0
                normalized_score = max(0.0, min(1.0, normalized_score))
                
                scores[item_id] = float(normalized_score)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity scores: {e}")
            return {item_id: 0.5 for item_id in item_ids}
    
    def get_user_recommendations(self, user_id: str, exclude_items: List[str] = None, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Get top-K recommendations cho user
        
        Args:
            user_id: User ID
            exclude_items: Items to exclude (already rated)
            top_k: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        self._load_embeddings()
        
        if not self._is_loaded:
            return []
        
        try:
            user_idx = self._user_to_idx.get(user_id)
            if user_idx is None:
                return []  # Cold start user
            
            # Get user embedding và bias
            user_emb = self._user_embeddings[user_idx]
            user_bias = self._user_biases[user_idx]
            
            # Calculate scores for all items
            dot_products = np.dot(self._item_embeddings, user_emb)
            predicted_ratings = self._global_mean + user_bias + self._item_biases + dot_products
            
            # Get top-K items
            exclude_indices = set()
            if exclude_items:
                exclude_indices = {self._item_to_idx.get(item_id) 
                                 for item_id in exclude_items 
                                 if self._item_to_idx.get(item_id) is not None}
            
            # Mask excluded items
            masked_ratings = predicted_ratings.copy()
            for idx in exclude_indices:
                masked_ratings[idx] = -np.inf
            
            # Get top-K indices
            top_indices = np.argsort(masked_ratings)[-top_k:][::-1]
            
            # Convert to (item_id, score) tuples
            recommendations = []
            for idx in top_indices:
                if idx in exclude_indices:
                    continue
                    
                item_id = self._idx_to_item.get(idx)
                if item_id:
                    score = float(predicted_ratings[idx])
                    recommendations.append((item_id, score))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if CF service is available"""
        if not self.config.get('enable', True):
            return False
            
        embeddings_path = model_manager.get_absolute_path('cf_embeddings')
        mappings_path = model_manager.get_absolute_path('cf_mappings')
        
        return os.path.exists(embeddings_path) and os.path.exists(mappings_path)
    
    def get_status(self) -> Dict:
        """Get CF service status"""
        return {
            'available': self.is_available(),
            'loaded': self._is_loaded,
            'n_users': self._n_users,
            'n_items': self._n_items,
            'embedding_dim': self._embedding_dim,
            'global_mean': self._global_mean,
            'last_access': self._last_access_time,
            'config': self.config
        }
    
    def unload(self):
        """Unload CF models để free memory"""
        self._user_embeddings = None
        self._item_embeddings = None
        self._user_biases = None
        self._item_biases = None
        self._user_to_idx = None
        self._item_to_idx = None
        self._idx_to_user = None
        self._idx_to_item = None
        
        self._user_cache.clear()
        self._item_cache.clear()
        
        self._is_loaded = False
        self.logger.info("CF embeddings unloaded")


def demo_cf_service():
    """Demo CF service"""
    print("DEMO COLLABORATIVE FILTERING SERVICE")
    print("=" * 50)
    
    cf_service = CollaborativeFilteringService()
    
    # Check availability
    print(f"Available: {cf_service.is_available()}")
    
    if not cf_service.is_available():
        print("CF service not available")
        return
    
    # Get status
    status = cf_service.get_status()
    print(f"\nStatus:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Test with sample data (if we have mappings)
    cf_service._load_embeddings()
    
    if cf_service._is_loaded and cf_service._user_to_idx:
        # Get first user và item for testing
        sample_user = list(cf_service._user_to_idx.keys())[0]
        sample_items = list(cf_service._item_to_idx.keys())[:5]
        
        print(f"\nTesting with user: {sample_user}")
        print(f"Testing with items: {len(sample_items)} items")
        
        # Test user-item score
        score = cf_service.get_user_item_score(sample_user, sample_items[0])
        print(f"User-item score: {score:.3f}")
        
        # Test similarity scores
        sim_scores = cf_service.get_user_similarity_scores(sample_user, sample_items)
        print(f"Similarity scores: {sim_scores}")
    
    print("CF Service demo completed!")

if __name__ == "__main__":
    demo_cf_service() 