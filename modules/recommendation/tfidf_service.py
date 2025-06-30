"""
TF-IDF Service for Restaurant Recommendation
Service để train và sử dụng TF-IDF cho keyword matching
"""

import os
import pickle
import json
import logging
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import psutil
from datetime import datetime
from .model_config import model_manager, ModelConfig
from .data_config import DataConfig


class TFIDFService:
    """
    TF-IDF Service với lazy loading và memory optimization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = model_manager.get_model_config('tfidf')
        self.model_config = ModelConfig()
        
        # Lazy loading attributes
        self._vectorizer = None
        self._document_matrix = None
        self._document_ids = []
        self._vocabulary = None
        self._is_trained = False
        
        # Memory monitoring
        self._last_access_time = None
        self._memory_usage_mb = 0
        
        # Paths
        self.model_path = model_manager.paths.TFIDF_MODEL_PATH
        self.vocab_path = model_manager.paths.TFIDF_VOCAB_PATH
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def _check_memory(self) -> bool:
        """Check if enough memory available"""
        process = psutil.Process()
        current_usage = process.memory_info().rss / (1024 * 1024)
        available = psutil.virtual_memory().available / (1024 * 1024)
        
        if current_usage > self.model_config.MAX_MEMORY_MB * 0.7:
            self.logger.warning(f"High memory usage: {current_usage:.1f}MB")
            return False
        
        if available < 500:  # Less than 500MB available
            self.logger.warning(f"Low system memory: {available:.1f}MB")
            return False
        
        return True
    
    def _load_vectorizer(self):
        """Lazy load TF-IDF vectorizer"""
        if self._vectorizer is not None:
            self._last_access_time = time.time()
            return
        
        self.logger.info("Loading TF-IDF vectorizer...")
        
        # Check if saved model exists
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                
                self._vectorizer = saved_data['vectorizer']
                self._document_matrix = saved_data['document_matrix']
                self._document_ids = saved_data['document_ids']
                self._is_trained = True
                
                # Load vocabulary
                if os.path.exists(self.vocab_path):
                    with open(self.vocab_path, 'r', encoding='utf-8') as f:
                        self._vocabulary = json.load(f)
                
                self._last_access_time = time.time()
                self._memory_usage_mb = self._estimate_memory_usage()
                
                self.logger.info(f"✅ TF-IDF loaded: {len(self._document_ids)} documents, "
                               f"{self._document_matrix.shape[1]} features, "
                               f"Memory: {self._memory_usage_mb:.1f}MB")
                
            except Exception as e:
                self.logger.error(f"❌ Error loading TF-IDF: {e}")
                self._initialize_new_vectorizer()
        else:
            self._initialize_new_vectorizer()
    
    def _initialize_new_vectorizer(self):
        """Initialize new TF-IDF vectorizer"""
        self._vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            min_df=self.config['min_df'],
            max_df=self.config['max_df'],
            use_idf=self.config['use_idf'],
            sublinear_tf=self.config['sublinear_tf'],
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        self._is_trained = False
        self._last_access_time = time.time()
        self.logger.info("🆕 Initialized new TF-IDF vectorizer")
    
    def train_from_restaurants(self, force_retrain: bool = False) -> bool:
        """
        Train TF-IDF từ restaurant data
        
        Args:
            force_retrain: Force retrain even if model exists
            
        Returns:
            True if successful
        """
        try:
            # Check if already trained
            if self._is_trained and not force_retrain:
                self.logger.info("ℹ️ TF-IDF already trained, skipping...")
                return True
            
            # Check memory
            if not self._check_memory():
                raise MemoryError("Insufficient memory for TF-IDF training")
            
            self.logger.info("🏋️ Training TF-IDF from restaurant data...")
            start_time = time.time()
            
            # Load restaurant data
            restaurant_data_path = DataConfig.get_restaurant_data_path()
            self.logger.info(f"📂 Loading data from: {restaurant_data_path}")
            
            # Load in chunks to save memory
            chunk_size = 5000
            all_texts = []
            all_ids = []
            
            for chunk in pd.read_parquet(restaurant_data_path, chunksize=chunk_size):
                # Filter restaurants only
                restaurants = chunk[chunk['poi_type'] == 'Restaurant'].copy()
                
                # Create text representation
                texts = self._create_text_representation(restaurants)
                ids = restaurants['business_id'].tolist()
                
                all_texts.extend(texts)
                all_ids.extend(ids)
                
                self.logger.info(f"   Processed {len(all_ids)} restaurants...")
                
                # Check memory periodically
                if not self._check_memory():
                    self.logger.warning("⚠️ Memory limit reached, stopping at {len(all_ids)} restaurants")
                    break
            
            # Initialize vectorizer if needed
            if self._vectorizer is None:
                self._initialize_new_vectorizer()
            
            # Fit and transform
            self.logger.info(f"🔄 Fitting TF-IDF on {len(all_texts)} documents...")
            self._document_matrix = self._vectorizer.fit_transform(all_texts)
            self._document_ids = all_ids
            self._is_trained = True
            
            # Extract vocabulary
            self._vocabulary = {
                'terms': list(self._vectorizer.vocabulary_.keys()),
                'idf_scores': self._vectorizer.idf_.tolist() if hasattr(self._vectorizer, 'idf_') else []
            }
            
            # Save model
            self._save_model()
            
            train_time = time.time() - start_time
            self._memory_usage_mb = self._estimate_memory_usage()
            
            self.logger.info(f"✅ TF-IDF training completed in {train_time:.2f}s")
            self.logger.info(f"   Documents: {len(self._document_ids)}")
            self.logger.info(f"   Features: {self._document_matrix.shape[1]}")
            self.logger.info(f"   Memory usage: {self._memory_usage_mb:.1f}MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ TF-IDF training failed: {e}")
            return False
    
    def _create_text_representation(self, restaurants_df: pd.DataFrame) -> List[str]:
        """Create text representation cho TF-IDF"""
        texts = []
        
        for _, row in restaurants_df.iterrows():
            # Combine relevant text fields
            text_parts = []
            
            # Name (important)
            if pd.notna(row.get('name')):
                text_parts.append(row['name'] * 3)  # Weight name higher
            
            # Categories (important)
            if pd.notna(row.get('categories')):
                text_parts.append(row['categories'] * 2)  # Weight categories
            
            # Cuisine types
            if pd.notna(row.get('cuisine_types')):
                cuisine_str = str(row['cuisine_types']).replace('[', '').replace(']', '').replace("'", "")
                text_parts.append(cuisine_str)
            
            # Description
            if pd.notna(row.get('description_text')):
                text_parts.append(str(row['description_text'])[:500])  # Limit length
            
            # City and state
            if pd.notna(row.get('city')):
                text_parts.append(row['city'])
            
            # Join all parts
            text = ' '.join(text_parts)
            texts.append(text)
        
        return texts
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, float]]:
        """
        Search restaurants using TF-IDF
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (document_id, score) tuples
        """
        try:
            # Load vectorizer if needed
            self._load_vectorizer()
            
            if not self._is_trained:
                self.logger.warning("⚠️ TF-IDF not trained, training now...")
                if not self.train_from_restaurants():
                    return []
            
            # Transform query
            query_vector = self._vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self._document_matrix).flatten()
            
            # Get top-k results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include non-zero scores
                    results.append({
                        'business_id': self._document_ids[idx],
                        'tfidf_score': float(similarities[idx])
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ TF-IDF search error: {e}")
            return []
    
    def get_scores_for_candidates(self, query: str, candidate_ids: List[str]) -> Dict[str, float]:
        """
        Get TF-IDF scores for specific candidates
        
        Args:
            query: Search query
            candidate_ids: List of business IDs
            
        Returns:
            Dictionary mapping business_id to score
        """
        try:
            # Load vectorizer if needed
            self._load_vectorizer()
            
            if not self._is_trained:
                return {cid: 0.0 for cid in candidate_ids}
            
            # Transform query
            query_vector = self._vectorizer.transform([query])
            
            # Map document IDs to indices
            id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self._document_ids)}
            
            scores = {}
            for candidate_id in candidate_ids:
                if candidate_id in id_to_idx:
                    idx = id_to_idx[candidate_id]
                    score = cosine_similarity(
                        query_vector, 
                        self._document_matrix[idx:idx+1]
                    )[0, 0]
                    scores[candidate_id] = float(score)
                else:
                    scores[candidate_id] = 0.0
            
            return scores
            
        except Exception as e:
            self.logger.error(f"❌ Error getting TF-IDF scores: {e}")
            return {cid: 0.0 for cid in candidate_ids}
    
    def _save_model(self):
        """Save trained model"""
        try:
            # Save vectorizer and matrix
            save_data = {
                'vectorizer': self._vectorizer,
                'document_matrix': self._document_matrix,
                'document_ids': self._document_ids,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            # Save vocabulary
            with open(self.vocab_path, 'w', encoding='utf-8') as f:
                json.dump(self._vocabulary, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 TF-IDF model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving TF-IDF model: {e}")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        if self._document_matrix is None:
            return 0.0
        
        # Sparse matrix memory estimation
        matrix_memory = (
            self._document_matrix.data.nbytes + 
            self._document_matrix.indices.nbytes + 
            self._document_matrix.indptr.nbytes
        ) / (1024 * 1024)
        
        # Vocabulary memory (rough estimate)
        vocab_memory = len(str(self._vocabulary)) / (1024 * 1024) if self._vocabulary else 0
        
        return matrix_memory + vocab_memory
    
    def unload(self):
        """Unload model to free memory"""
        self.logger.info("🗑️ Unloading TF-IDF model...")
        self._vectorizer = None
        self._document_matrix = None
        self._document_ids = []
        self._vocabulary = None
        self._is_trained = False
        self._memory_usage_mb = 0
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            'loaded': self._vectorizer is not None,
            'trained': self._is_trained,
            'num_documents': len(self._document_ids),
            'num_features': self._document_matrix.shape[1] if self._document_matrix is not None else 0,
            'memory_usage_mb': self._memory_usage_mb,
            'last_access': datetime.fromtimestamp(self._last_access_time).isoformat() if self._last_access_time else None
        }


def demo_tfidf_service():
    """Demo TF-IDF service"""
    print("DEMO TF-IDF SERVICE")
    print("=" * 50)
    
    service = TFIDFService()
    
    # Train if needed
    print("\n1. Training TF-IDF...")
    success = service.train_from_restaurants()
    
    if success:
        # Test search
        print("\n2. Testing search...")
        queries = [
            "vietnamese pho restaurant",
            "italian pizza pasta",
            "seafood fine dining"
        ]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            results = service.search(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['business_id']} - Score: {result['tfidf_score']:.3f}")
        
        # Get status
        print("\n3. Service status:")
        status = service.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
    
    print("\nTF-IDF Service demo completed!")


if __name__ == "__main__":
    demo_tfidf_service() 