"""
Advanced Machine Learning Service - Phase 5 (Memory Optimized)
Deep Personalization, Collaborative Filtering với User Behavior Monitoring tích hợp
Tối ưu memory để tránh OOM
"""

import numpy as np
import pandas as pd
import psutil
import gc
import torch
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from modules.memory.short_term import SessionStore
from modules.domain.schemas import UserProfile, FeedbackAction
from modules.domain.restaurant_schemas import RestaurantProfile, CuisineType, PriceLevel
from .online_learning_service import OnlineLearningService

@dataclass
class ServiceConfig:
    """Service configuration với memory optimization"""
    # Memory constraints
    max_memory_usage_mb: int = 3500  # Increased threshold
    max_user_embeddings: int = 1000  # Limit user embeddings
    max_restaurant_embeddings: int = 2000  # Limit restaurant embeddings
    
    # Processing limits
    batch_size: int = 16  # Reduced batch size
    max_sequence_length: int = 256  # Reduced sequence length
    
    # Cache configuration
    cache_ttl_seconds: int = 300  # 5 minutes
    max_cache_entries: int = 500
    
    # User behavior monitoring
    max_user_activities: int = 50  # Limit per user
    behavior_retention_hours: int = 24
    
    # Performance optimization
    enable_fp16: bool = True
    lazy_loading: bool = True
    sequential_loading: bool = True
    memory_monitoring: bool = True

@dataclass
class UserBehaviorEvent:
    """User behavior event for monitoring"""
    event_type: str  # 'search', 'click', 'like', 'view'
    user_id: str
    restaurant_id: Optional[str] = None
    data: Dict = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AdvancedMLService:
    """
    Advanced Machine Learning Service với User Behavior Monitoring tích hợp
    Features:
    - Deep User Profiling với memory constraints
    - Collaborative Filtering with Matrix Factorization (optimized)
    - Content-based Filtering với TF-IDF
    - User Behavior Monitoring tích hợp
    - Online Learning từ Like/Dislike feedback (PHASE 2 - STEP 5)
    - Memory optimization và monitoring
    - CUDA OOM exception handling
    """
    
    def __init__(self, memory: SessionStore, config: ServiceConfig = None):
        self.memory = memory
        self.config = config or ServiceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor(self.config.max_memory_usage_mb)
        
        # ML Models với memory constraints
        self.user_item_matrix = None
        self.svd_model = None
        self.tfidf_vectorizer = None
        self.restaurant_features = None
        
        # Learning parameters (reduced for memory)
        self.learning_rate = 0.01
        self.regularization = 0.1
        self.n_factors = 32  # Reduced from 50
        
        # Exploration/Exploitation parameters
        self.epsilon = 0.1
        self.temperature = 1.0
        
        # Limited embeddings với LRU eviction
        self.user_embeddings = LimitedDict(max_size=self.config.max_user_embeddings)
        self.restaurant_embeddings = LimitedDict(max_size=self.config.max_restaurant_embeddings)
        
        # User Behavior Monitoring (tích hợp từ RealtimeService)
        self.user_behavior_streams = defaultdict(lambda: deque(maxlen=self.config.max_user_activities))
        self.behavior_cache = LimitedDict(max_size=self.config.max_cache_entries)
        
        # PHASE 2 - STEP 5: Online Learning Service với Brain Fusion
        self.online_learning_service = self._initialize_online_learning_with_content_embeddings(memory)
        
        # Performance tracking
        self.model_performance = {
            'accuracy': 0.0,
            'coverage': 0.0,
            'diversity': 0.0,
            'novelty': 0.0,
            'learning_performance': 0.0,  # New metric for online learning
            'last_updated': datetime.now()
        }
        
        # Memory-safe initialization
        if self.config.sequential_loading:
            self._initialize_models_sequential()
        else:
            self._initialize_models()
    
    def _initialize_models_sequential(self):
        """Initialize models sequentially to avoid memory spikes"""
        try:
            self.logger.info("Sequential model initialization starting...")
            
            # Check memory before each step
            if not self.memory_monitor.check_memory_available():
                raise MemoryError("Insufficient memory for model initialization")
            
            # Step 1: Initialize basic models
            self._initialize_basic_models()
            gc.collect()  # Force garbage collection
            
            # Step 2: Load existing models if available
            if not self.config.lazy_loading:
                try:
                    self._load_trained_models()
                except:
                    self.logger.info("No existing models found, will train new ones")
            
            self.logger.info("Sequential model initialization completed")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            self._fallback_initialization()
    
    def _initialize_models(self):
        """Standard model initialization với memory checks"""
        try:
            if not self.memory_monitor.check_memory_available():
                self.logger.warning("Low memory detected, switching to sequential loading")
                self._initialize_models_sequential()
                return
            
            self._initialize_basic_models()
            
            try:
                self._load_trained_models()
                self.logger.info("Loaded existing ML models")
            except:
                self._train_initial_models()
                self.logger.info("Initialized new ML models")
                
        except Exception as e:
            self.logger.error(f"Model initialization error: {e}")
            self._fallback_initialization()
    
    def _initialize_basic_models(self):
        """Initialize basic model components"""
        # Initialize service instances with lazy loading
        self._tfidf_service = None
        self._bge_m3_service = None
        self._t5_rerank_service = None
        self._t5_message_service = None
        self._cf_service = None  # Collaborative Filtering service
        
        # Initialize SVD model với reduced dimensions (kept for compatibility)
        self.svd_model = TruncatedSVD(n_components=self.n_factors, random_state=42)
        
        # Old TF-IDF kept for backward compatibility
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,  # Reduced from 1000
            stop_words='english',
            max_df=0.8,
            min_df=2
        )
        
        self.logger.info("Basic models initialized with memory optimization")
    
    def _fallback_initialization(self):
        """Fallback initialization với minimal features"""
        self.logger.warning("Using fallback initialization mode")
        
        # Minimal model setup
        self.svd_model = None
        self.tfidf_vectorizer = None
        self.user_embeddings.clear()
        self.restaurant_embeddings.clear()
        
        # Clear service instances
        self._tfidf_service = None
        self._bge_m3_service = None
        self._t5_rerank_service = None
        self._t5_message_service = None
        
        gc.collect()
    
    def _initialize_online_learning_with_content_embeddings(self, memory: SessionStore) -> OnlineLearningService:
        """
        Initialize OnlineLearningService với Content Brain embeddings (Brain Fusion)
        
        Thực hiện knowledge transfer từ Content Brain (BGE-M3) sang Learning Brain
        để giải quyết cold start problem và tăng hiệu quả personalization
        """
        try:
            self.logger.info("Initializing Brain Fusion: Content Brain -> Learning Brain")
            
            # Get content embeddings từ BGE-M3 service (Content Brain)
            content_embeddings = self._get_content_embeddings_for_fusion()
            
            if content_embeddings and len(content_embeddings) > 0:
                self.logger.info(f"Brain Fusion: Transferring {len(content_embeddings)} content embeddings to Learning Brain")
                
                # Initialize OnlineLearningService với content embeddings
                online_service = OnlineLearningService(memory, initial_embeddings=content_embeddings)
                
                self.logger.info("Brain Fusion completed successfully! Learning Brain now has semantic understanding.")
                return online_service
            else:
                self.logger.warning("No content embeddings available for Brain Fusion, using standard initialization")
                return OnlineLearningService(memory)
                
        except Exception as e:
            self.logger.error(f"Brain Fusion failed: {e}, falling back to standard initialization")
            return OnlineLearningService(memory)
    
    def _get_content_embeddings_for_fusion(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get content embeddings từ BGE-M3 service cho Brain Fusion
        
        Returns:
            Dict mapping business_id -> embedding vector
        """
        try:
            # Initialize basic models first to ensure all service attributes exist
            if not hasattr(self, '_bge_m3_service'):
                self._initialize_basic_models()
            
            # Ensure BGE-M3 service is loaded
            self._ensure_bge_m3_service()
            
            if self._bge_m3_service is None:
                self.logger.warning("BGE-M3 service not available for Brain Fusion")
                return None
            
            # Get all embeddings từ Content Brain
            self.logger.info("Fetching content embeddings from BGE-M3 service...")
            content_embeddings = self._bge_m3_service.get_all_embeddings()
            
            if content_embeddings and len(content_embeddings) > 0:
                self.logger.info(f"Retrieved {len(content_embeddings)} content embeddings for Brain Fusion")
                
                # Log embedding dimension for verification
                first_embedding = next(iter(content_embeddings.values()))
                self.logger.info(f"Content embedding dimension: {first_embedding.shape[0]}")
                
                return content_embeddings
            else:
                self.logger.warning("No content embeddings retrieved from BGE-M3 service")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting content embeddings for Brain Fusion: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    # ==================== LAZY LOADING SERVICES ====================
    
    def _ensure_tfidf_service(self):
        """Lazy load TF-IDF service"""
        if self._tfidf_service is None:
            from .tfidf_service import TFIDFService
            self._tfidf_service = TFIDFService()
            self.logger.info("TF-IDF service initialized")
    
    def _ensure_bge_m3_service(self):
        """Lazy load BGE-M3 service"""
        if self._bge_m3_service is None:
            from .bge_m3_enhanced_service import BGE_M3_EnhancedService
            self._bge_m3_service = BGE_M3_EnhancedService()
            self.logger.info("BGE-M3 service initialized")
    
    def _ensure_t5_rerank_service(self):
        """Lazy load T5 reranking service"""
        if self._t5_rerank_service is None:
            from .t5_reranking_service import T5RerankingService
            self._t5_rerank_service = T5RerankingService()
            self.logger.info("T5 reranking service initialized")
    
    def _ensure_t5_message_service(self):
        """Lazy load T5 message service"""
        if self._t5_message_service is None:
            from .t5_message_service import T5MessageService
            self._t5_message_service = T5MessageService()
            self.logger.info("T5 message service initialized")
    
    def _ensure_cf_service(self):
        """Lazy load Collaborative Filtering service"""
        if self._cf_service is None:
            from .collaborative_filtering_service import CollaborativeFilteringService
            self._cf_service = CollaborativeFilteringService()
            self.logger.info("Collaborative Filtering service initialized")
    
    def _load_trained_models(self):
        """Load existing trained models (placeholder implementation)"""
        # Placeholder - in production này sẽ load từ disk
        self.logger.info("Loading trained models (placeholder)")
        # Intentionally raise để trigger training
        raise FileNotFoundError("No trained models found")
    
    def _train_initial_models(self):
        """Train initial models with minimal data (placeholder implementation)"""
        self.logger.info("Training initial models (placeholder)")
        # Placeholder - in production này sẽ train models với real data
        pass
    
    # ==================== USER BEHAVIOR MONITORING (tích hợp) ====================
    
    def track_user_behavior(self, event: UserBehaviorEvent):
        """Track user behavior event (thay thế RealtimeService)"""
        try:
            # Memory check before adding
            if not self.memory_monitor.check_memory_available():
                self._cleanup_old_behavior_data()
            
            # Add to user's behavior stream
            self.user_behavior_streams[event.user_id].append(event)
            
            # Update behavior patterns
            self._update_user_behavior_patterns(event)
            
            # PHASE 2 - STEP 5: Process feedback signals for online learning
            if event.event_type in ['like', 'dislike'] and event.restaurant_id:
                self._process_online_learning_signal(event)
            
            self.logger.debug(f"Tracked behavior: {event.event_type} by {event.user_id}")
            
        except Exception as e:
            self.logger.error(f"Behavior tracking error: {e}")
    
    def _update_user_behavior_patterns(self, event: UserBehaviorEvent):
        """Update user behavior patterns for real-time learning"""
        user_id = event.user_id
        
        # Invalidate cache for this user
        cache_keys_to_remove = [k for k in self.behavior_cache.keys() if user_id in k]
        for key in cache_keys_to_remove:
            self.behavior_cache.pop(key, None)
        
        # Update user embeddings if significant behavior
        if event.event_type in ['like', 'visit'] and event.restaurant_id:
            self._update_user_embedding_realtime(user_id, event.restaurant_id, event.event_type)
    
    def get_user_behavior_insights(self, user_id: str) -> Dict:
        """Get user behavior insights"""
        try:
            cache_key = f"behavior_insights:{user_id}"
            
            # Check cache first
            if cache_key in self.behavior_cache:
                return self.behavior_cache[cache_key]
            
            behaviors = list(self.user_behavior_streams.get(user_id, []))
            
            if not behaviors:
                return {'insights': 'insufficient_data', 'patterns': {}}
            
            # Analyze patterns
            insights = self._analyze_behavior_patterns(behaviors)
            
            # Cache results
            self.behavior_cache[cache_key] = insights
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Behavior insights error: {e}")
            return {'error': str(e)}
    
    def _analyze_behavior_patterns(self, behaviors: List[UserBehaviorEvent]) -> Dict:
        """Analyze user behavior patterns"""
        search_count = sum(1 for b in behaviors if b.event_type == 'search')
        click_count = sum(1 for b in behaviors if b.event_type == 'click')
        like_count = sum(1 for b in behaviors if b.event_type == 'like')
        
        ctr = click_count / max(search_count, 1)
        engagement = like_count / max(click_count, 1)
        
        return {
            'activity_level': 'high' if len(behaviors) > 20 else 'medium' if len(behaviors) > 10 else 'low',
            'click_through_rate': ctr,
            'engagement_rate': engagement,
            'preferred_times': self._get_preferred_times(behaviors),
            'cuisine_preferences': self._extract_cuisine_preferences(behaviors)
        }
    
    def _cleanup_old_behavior_data(self):
        """Cleanup old behavior data to free memory"""
        cutoff_time = datetime.now() - timedelta(hours=self.config.behavior_retention_hours)
        
        for user_id in list(self.user_behavior_streams.keys()):
            # Remove old events
            user_stream = self.user_behavior_streams[user_id]
            while user_stream and user_stream[0].timestamp < cutoff_time:
                user_stream.popleft()
            
            # Remove empty streams
            if not user_stream:
                del self.user_behavior_streams[user_id]
        
        # Clear old cache entries
        self.behavior_cache.clear()
        gc.collect()
        
        self.logger.info("Cleaned up old behavior data")
    
    # ==================== MEMORY-OPTIMIZED RECOMMENDATIONS ====================
    
    def get_deep_personalized_recommendations(self, 
                                            user_id: str, 
                                            city: str,
                                            candidates: List[Dict],
                                            exploration_factor: float = 0.1) -> List[Dict]:
        """
        Get deep personalized recommendations với memory optimization
        PHASE 2 - STEP 5: Enhanced với online learning
        """
        try:
            # Memory check before processing
            if not self.memory_monitor.check_memory_available():
                self.logger.warning("Low memory, using simplified recommendations")
                return self._get_simplified_recommendations(user_id, candidates)
            
            start_time = datetime.now()
            
            # Track search behavior
            query = ""
            for candidate in candidates[:1]:  # Get query from first candidate if available
                if 'query' in candidate:
                    query = candidate['query']
                    break
            
            self.track_user_behavior(UserBehaviorEvent(
                'search', user_id, data={'city': city, 'candidate_count': len(candidates), 'query': query}
            ))
            
            # Limit candidates for memory efficiency
            max_candidates = min(len(candidates), 50)  # Process max 50 candidates
            candidates = candidates[:max_candidates]
            
            # 1. Collaborative Filtering Scores (với memory optimization)
            collab_scores = self._get_collaborative_scores_optimized(user_id, candidates)
            
            # 2. Content-based Scores (với reduced features)
            content_scores = self._get_content_based_scores_optimized(user_id, candidates)
            
            # 3. Behavior-based Scores (thay thế deep learning)
            behavior_scores = self._get_behavior_based_scores(user_id, candidates)
            
            # 4. Context-aware Scores (simplified)
            context_scores = self._get_context_aware_scores_optimized(user_id, city, candidates)
            
            # PHASE 2 - STEP 5: Online Learning Scores
            online_learning_scores = self._get_online_learning_scores(user_id, candidates)
            
            # 5. Memory-efficient ensemble scoring with online learning
            final_scores = self._ensemble_scoring_with_online_learning(
                candidates, collab_scores, content_scores, 
                behavior_scores, context_scores, online_learning_scores, user_id
            )
            
            # 6. Apply exploration if needed
            if exploration_factor > 0:
                final_scores = self._apply_exploration_optimized(final_scores, exploration_factor)
            
            # Sort and return
            for i, candidate in enumerate(candidates):
                candidate['ml_score'] = final_scores[i]
                candidate['ml_components'] = {
                    'collaborative': collab_scores[i],
                    'content_based': content_scores[i],
                    'behavior_based': behavior_scores[i],
                    'context_aware': context_scores[i],
                    'online_learning': online_learning_scores[i]  # New component
                }
            
            candidates.sort(key=lambda x: x['ml_score'], reverse=True)
            
            # 7. Apply T5 reranking to top candidates (optional)
            if len(candidates) > 5 and self._should_use_t5_reranking():
                candidates = self._apply_t5_reranking(candidates, query or "restaurant recommendation", user_id)
            
            # 8. Generate personalized messages for top restaurants
            if self._should_generate_messages():
                self._add_personalized_messages(candidates[:5], user_id, city)
            
            process_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Deep personalization with online learning completed in {process_time:.3f}s")
            
            return candidates
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"CUDA OOM Error: {e}")
            torch.cuda.empty_cache()  # Clear CUDA cache
            gc.collect()
            return self._get_simplified_recommendations(user_id, candidates)
            
        except Exception as e:
            self.logger.error(f"Recommendation error: {e}")
            return self._get_simplified_recommendations(user_id, candidates)
    
    def _get_simplified_recommendations(self, user_id: str, candidates: List[Dict]) -> List[Dict]:
        """Simplified recommendations for low memory situations"""
        # Simple scoring based on ratings and user behavior
        user_behaviors = self.user_behavior_streams.get(user_id, [])
        liked_restaurants = {b.restaurant_id for b in user_behaviors if b.event_type == 'like'}
        
        for candidate in candidates:
            # Enhanced simple scoring
            metadata = candidate.get('metadata', {})
            stars = metadata.get('stars', 3.0)
            review_count = metadata.get('review_count', 0)
            
            # Base score từ quality (0.5-0.9 range)
            base_score = 0.5 + (stars - 1) * 0.1  # 1★→0.5, 5★→0.9
            
            # Popularity boost
            popularity_boost = min(0.1, review_count / 1000)  # Up to +0.1
            
            # Behavior boost
            behavior_boost = 0.0
            if candidate.get('business_id') in liked_restaurants:
                behavior_boost = 0.15 # Increased boost for liked items
            
            final_score = min(0.95, base_score + popularity_boost + behavior_boost)
            
            candidate['ml_score'] = final_score
            candidate['ml_components'] = {
                'simplified': True,
                'base_score': base_score,
                'popularity_boost': popularity_boost,
                'behavior_boost': behavior_boost
            }
        
        candidates.sort(key=lambda x: x['ml_score'], reverse=True)
        return candidates
    
    # ==================== MEMORY-OPTIMIZED HELPER METHODS ====================
    
    def _get_collaborative_scores_optimized(self, user_id: str, candidates: List[Dict]) -> List[float]:
        """Collaborative filtering using real Funk-SVD embeddings"""
        try:
            # Ensure CF service is loaded
            self._ensure_cf_service()
            
            # Check if CF service is available
            if not self._cf_service.is_available():
                self.logger.warning("CF service not available, using fallback")
                return self._get_collaborative_scores_fallback(user_id, candidates)
            
            # Extract business IDs from candidates
            business_ids = [candidate.get('business_id', '') for candidate in candidates]
            
            # Get collaborative scores from CF service
            cf_scores = self._cf_service.get_user_similarity_scores(user_id, business_ids)
            
            # Convert to list in same order as candidates
            scores = []
            for candidate in candidates:
                business_id = candidate.get('business_id', '')
                score = cf_scores.get(business_id, 0.5)  # Default to 0.5 if not found
                scores.append(score)
            
            self.logger.debug(f"CF scores for user {user_id}: min={min(scores):.3f}, max={max(scores):.3f}")
            return scores
            
        except Exception as e:
            self.logger.error(f"CF scoring error: {e}")
            return self._get_collaborative_scores_fallback(user_id, candidates)
    
    def _get_collaborative_scores_fallback(self, user_id: str, candidates: List[Dict]) -> List[float]:
        """Fallback collaborative scoring using behavior data"""
        try:
            # Use behavior data instead of full matrix factorization
            user_behaviors = self.user_behavior_streams.get(user_id, [])
            liked_restaurants = {b.restaurant_id for b in user_behaviors if b.event_type == 'like'}
            
            scores = []
            for candidate in candidates:
                # Simple collaborative score based on user behavior
                restaurant_id = candidate.get('business_id', '')
                if restaurant_id in liked_restaurants:
                    scores.append(0.9)  # High score for previously liked
                else:
                    # Base score on restaurant quality (stars và reviews)
                    metadata = candidate.get('metadata', {})
                    stars = metadata.get('stars', 3.0)
                    review_count = metadata.get('review_count', 0)
                    
                    # Convert stars (1-5) to score (0.4-0.8)
                    quality_score = 0.4 + (stars - 1) * 0.1  # 1★→0.4, 5★→0.8
                    
                    # Boost for popularity
                    popularity_boost = min(0.1, review_count / 1000)  # Up to +0.1 for 1000+ reviews
                    
                    final_score = min(0.85, quality_score + popularity_boost)
                    scores.append(final_score)
            
            return scores
        except Exception as e:
            self.logger.error(f"Fallback CF scoring error: {e}")
            return [0.6] * len(candidates)  # Higher default score
    
    def _get_content_based_scores_optimized(self, user_id: str, candidates: List[Dict]) -> List[float]:
        """Memory-optimized content-based filtering using BGE-M3 and TF-IDF"""
        try:
            # Get query if available
            query = ""
            behaviors = self.user_behavior_streams.get(user_id, [])
            for behavior in reversed(behaviors):
                if behavior.event_type == 'search' and behavior.data:
                    query = behavior.data.get('query', '')
                    break
            
            # Initialize scores based on restaurant quality
            scores = []
            for candidate in candidates:
                metadata = candidate.get('metadata', {})
                stars = metadata.get('stars', 3.0)
                # Base score from stars (0.6-0.9 range)
                base_score = 0.5 + (stars - 1) * 0.1  # 1★→0.5, 5★→0.9
                scores.append(base_score)
            
            # 1. BGE-M3 content scores (if query available)
            if query:
                self._ensure_bge_m3_service()
                candidate_ids = [c.get('business_id', '') for c in candidates]
                bge_scores = self._bge_m3_service.get_content_scores(query, candidate_ids)
                
                # Merge BGE-M3 scores
                for i, candidate in enumerate(candidates):
                    business_id = candidate.get('business_id', '')
                    if business_id in bge_scores:
                        scores[i] = 0.7 * bge_scores[business_id] + 0.3 * scores[i]
            
            # 2. TF-IDF scores as supplement
            if query:
                self._ensure_tfidf_service()
                candidate_ids = [c.get('business_id', '') for c in candidates]
                tfidf_scores = self._tfidf_service.get_scores_for_candidates(query, candidate_ids)
                
                # Merge TF-IDF scores (lower weight)
                for i, candidate in enumerate(candidates):
                    business_id = candidate.get('business_id', '')
                    if business_id in tfidf_scores and tfidf_scores[business_id] > 0:
                        scores[i] = 0.8 * scores[i] + 0.2 * tfidf_scores[business_id]
            
            # 3. User preference embedding boost (if available)
            liked_restaurants = [b.restaurant_id for b in behaviors if b.event_type == 'like' and b.restaurant_id]
            if liked_restaurants and self._bge_m3_service:
                pref_embedding = self._bge_m3_service.get_user_preference_embedding(liked_restaurants[-10:])  # Last 10
                if pref_embedding is not None:
                    candidate_ids = [c.get('business_id', '') for c in candidates]
                    pref_scores = self._bge_m3_service.compute_preference_scores(pref_embedding, candidate_ids)
                    
                    # Merge preference scores
                    for i, candidate in enumerate(candidates):
                        business_id = candidate.get('business_id', '')
                        if business_id in pref_scores:
                            scores[i] = 0.6 * scores[i] + 0.4 * pref_scores[business_id]
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Content-based scoring error: {e}")
            return [0.5] * len(candidates)
    
    def _get_behavior_based_scores(self, user_id: str, candidates: List[Dict]) -> List[float]:
        """Score based on user behavior patterns"""
        try:
            behaviors = list(self.user_behavior_streams.get(user_id, []))
            
            scores = []
            for candidate in candidates:
                # Base score từ restaurant quality
                metadata = candidate.get('metadata', {})
                stars = metadata.get('stars', 3.0)
                base_score = 0.5 + (stars - 1) * 0.08  # 1★→0.5, 5★→0.82
                
                # Check for similar restaurants in user behavior
                restaurant_id = candidate.get('business_id', '')
                categories = candidate.get('metadata', {}).get('categories', '').lower()
                
                # Boost for user behavior patterns
                behavior_boost = 0.0
                for behavior in behaviors[-10:]:  # Last 10 behaviors
                    if behavior.event_type == 'like':
                        behavior_boost += 0.05  # Each like adds 0.05
                    elif behavior.event_type == 'search':
                        behavior_boost += 0.02  # Each search adds 0.02
                
                final_score = min(0.9, base_score + behavior_boost)
                scores.append(final_score)
            
            return scores
        except Exception as e:
            self.logger.error(f"Behavior scoring error: {e}")
            return [0.65] * len(candidates)  # Higher default
    
    def _get_context_aware_scores_optimized(self, user_id: str, city: str, candidates: List[Dict]) -> List[float]:
        """Simplified context-aware scoring"""
        current_hour = datetime.now().hour
        
        scores = []
        for candidate in candidates:
            # Base score từ restaurant quality
            metadata = candidate.get('metadata', {})
            stars = metadata.get('stars', 3.0)
            review_count = metadata.get('review_count', 0)
            
            # Base score (0.6-0.85)
            base_score = 0.6 + (stars - 1) * 0.0625  # 1★→0.6, 5★→0.85
            
            # Context adjustments
            context_boost = 0.0
            
            # Time-based adjustments
            categories = metadata.get('categories', '').lower()
            if 11 <= current_hour <= 14:  # Lunch time
                if any(keyword in categories for keyword in ['lunch', 'cafe', 'fast food']):
                    context_boost += 0.05
            elif 17 <= current_hour <= 21:  # Dinner time
                if any(keyword in categories for keyword in ['dinner', 'fine dining', 'restaurant']):
                    context_boost += 0.05
            
            # Popularity boost
            if review_count >= 100:
                context_boost += 0.03
            if review_count >= 500:
                context_boost += 0.02
            
            final_score = min(0.9, base_score + context_boost)
            scores.append(final_score)
        
        return scores
    
    def _get_online_learning_scores(self, user_id: str, candidates: List[Dict]) -> List[float]:
        """
        PHASE 2 - STEP 5: Get personalized scores từ online learning service
        """
        try:
            # Extract restaurant IDs
            restaurant_ids = [candidate.get('business_id', '') for candidate in candidates]
            
            # Get personalized scores from online learning
            online_scores = self.online_learning_service.get_personalized_scores(user_id, restaurant_ids)
            
            # Convert to list in same order as candidates
            scores = []
            for candidate in candidates:
                business_id = candidate.get('business_id', '')
                score = online_scores.get(business_id, 0.5)  # Default to neutral
                scores.append(score)
            
            self.logger.debug(f"Online learning scores for user {user_id}: min={min(scores):.3f}, max={max(scores):.3f}")
            return scores
            
        except Exception as e:
            self.logger.error(f"Error getting online learning scores: {e}")
            return [0.5] * len(candidates)  # Default neutral scores
    
    def _ensemble_scoring_with_online_learning(self, candidates: List[Dict], 
                                             collab_scores: List[float],
                                             content_scores: List[float],
                                             behavior_scores: List[float],
                                             context_scores: List[float],
                                             online_learning_scores: List[float],
                                             user_id: str) -> List[float]:
        """
        Enhanced ensemble scoring với online learning component
        """
        
        # Get user learning insights để adjust weights
        learning_insights = self.online_learning_service.get_learning_insights(user_id)
        personalization_confidence = learning_insights.get('personalization_confidence', 0.0)
        
        # Adaptive weights based on learning maturity
        if personalization_confidence < 0.1:
            # Cold start: rely more on content and context
            weights = {
                'collaborative': 0.15,
                'content_based': 0.35,
                'behavior_based': 0.15,
                'context_aware': 0.25,
                'online_learning': 0.1  # Low weight for new users
            }
        elif personalization_confidence < 0.5:
            # Learning phase: balanced approach
            weights = {
                'collaborative': 0.2,
                'content_based': 0.3,
                'behavior_based': 0.2,
                'context_aware': 0.15,
                'online_learning': 0.15  # Increasing weight
            }
        else:
            # Mature user: emphasize learned preferences
            weights = {
                'collaborative': 0.2,
                'content_based': 0.25,
                'behavior_based': 0.15,
                'context_aware': 0.15,
                'online_learning': 0.25  # Higher weight for mature users
            }
        
        final_scores = []
        for i in range(len(candidates)):
            # Calculate weighted ensemble
            ensemble_score = (
                weights['collaborative'] * collab_scores[i] +
                weights['content_based'] * content_scores[i] +
                weights['behavior_based'] * behavior_scores[i] +
                weights['context_aware'] * context_scores[i] +
                weights['online_learning'] * online_learning_scores[i]
            )
            
            # Apply quality boost để đảm bảo good restaurants có scores cao
            metadata = candidates[i].get('metadata', {})
            stars = metadata.get('stars', 3.0)
            review_count = metadata.get('review_count', 0)
            
            # Quality multiplier (0.9-1.15 range)
            quality_multiplier = 0.9 + (stars - 1) * 0.0625  # 1★→0.9x, 5★→1.15x
            
            # Popularity boost 
            if review_count >= 100:
                quality_multiplier += 0.05
            if review_count >= 500:
                quality_multiplier += 0.05
            
            final_score = min(0.95, ensemble_score * quality_multiplier)
            final_scores.append(final_score)
        
        return final_scores
    
    def _apply_exploration_optimized(self, scores: List[float], exploration_factor: float) -> List[float]:
        """Enhanced exploration with stronger effects"""
        if exploration_factor <= 0:
            return scores
        
        # Enhanced exploration với multiple strategies
        explored_scores = []
        
        for i, score in enumerate(scores):
            if exploration_factor < 0.3:  # Safe choices (0.0 - 0.3)
                # Conservative: Small random noise + slight preference for high scores
                noise = np.random.normal(0, 0.02, 1)[0]
                boost = 0.1 if score > 0.4 else 0.0  # Boost high-confidence recommendations
                new_score = score + (noise * exploration_factor) + (boost * exploration_factor)
                
            elif exploration_factor < 0.7:  # Balanced (0.3 - 0.7)
                # Medium exploration: Moderate noise + some diversity boost
                noise = np.random.normal(0, 0.08, 1)[0]
                diversity_boost = 0.1 if i > 2 else 0.0  # Boost lower-ranked items
                new_score = score + (noise * exploration_factor) + (diversity_boost * exploration_factor)
                
            else:  # Adventurous (0.7 - 1.0)
                # High exploration: Large noise + strong diversity preference
                noise = np.random.normal(0, 0.15, 1)[0]
                adventure_boost = 0.2 if i > 4 else 0.0  # Strong boost for lower-ranked items
                new_score = score + (noise * exploration_factor) + (adventure_boost * exploration_factor)
            
            explored_scores.append(max(0.01, min(0.99, new_score)))  # Wider range
        
        # Log exploration effect for debugging
        score_changes = [abs(new - old) for new, old in zip(explored_scores, scores)]
        avg_change = sum(score_changes) / len(score_changes)
        self.logger.info(f"Exploration factor {exploration_factor:.1f} caused avg score change: {avg_change:.4f}")
        
        return explored_scores
    
    # ==================== HELPER METHODS ====================
    
    def _get_preferred_times(self, behaviors: List[UserBehaviorEvent]) -> List[int]:
        """Extract preferred times from behavior"""
        hours = [b.timestamp.hour for b in behaviors if b.timestamp]
        return list(set(hours))
    
    def _extract_cuisine_preferences(self, behaviors: List[UserBehaviorEvent]) -> Dict[str, float]:
        """Extract cuisine preferences from behavior"""
        # Simple mock implementation
        return {'italian': 0.8, 'american': 0.6, 'asian': 0.7}
    
    def _update_user_embedding_realtime(self, user_id: str, restaurant_id: str, event_type: str):
        """Update user embedding based on real-time behavior"""
        # Simplified embedding update
        if user_id not in self.user_embeddings:
            self.user_embeddings[user_id] = np.random.normal(0, 0.1, 32)  # Small embedding
        
        # Adjust embedding slightly based on feedback
        if event_type == 'like':
            self.user_embeddings[user_id] *= 1.01  # Small positive adjustment
    
    # ==================== MEMORY MONITORING ====================
    
    def get_memory_status(self) -> Dict:
        """Get current memory status"""
        return self.memory_monitor.get_status()
    
    def force_cleanup(self):
        """Force cleanup to free memory"""
        # Clear large data structures
        self.user_embeddings.clear()
        self.restaurant_embeddings.clear()
        self.behavior_cache.clear()
        
        # Cleanup old behavior data
        self._cleanup_old_behavior_data()
        
        # Unload services if loaded
        if self._tfidf_service:
            self._tfidf_service.unload()
            self._tfidf_service = None
        
        if self._bge_m3_service:
            self._bge_m3_service.unload()
            self._bge_m3_service = None
        
        if self._t5_rerank_service:
            self._t5_rerank_service.unload()
            self._t5_rerank_service = None
        
        if self._t5_message_service:
            self._t5_message_service.unload()
            self._t5_message_service = None
        
        if self._cf_service:
            self._cf_service.unload()
            self._cf_service = None
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Forced cleanup completed")
    
    # ==================== T5 INTEGRATION METHODS ====================
    
    def _should_use_t5_reranking(self) -> bool:
        """Check if T5 reranking should be used"""
        # Use T5 reranking if memory is sufficient
        memory_status = self.memory_monitor.get_status()
        return memory_status['usage_percentage'] < 70 and memory_status['available_mb'] > 1000
    
    def _should_generate_messages(self) -> bool:
        """Check if personalized messages should be generated"""
        # Generate messages if memory permits
        memory_status = self.memory_monitor.get_status()
        return memory_status['usage_percentage'] < 80 and memory_status['available_mb'] > 800
    
    def _apply_t5_reranking(self, candidates: List[Dict], query: str, user_id: str) -> List[Dict]:
        """Apply T5 reranking to candidates"""
        try:
            self._ensure_t5_rerank_service()
            
            # Convert to RerankCandidate format
            from .t5_reranking_service import RerankCandidate
            
            rerank_candidates = []
            for candidate in candidates[:20]:  # Rerank top 20
                rerank_candidates.append(RerankCandidate(
                    business_id=candidate.get('business_id', ''),
                    name=candidate.get('metadata', {}).get('name', 'Unknown'),
                    categories=candidate.get('metadata', {}).get('categories', ''),
                    initial_score=candidate.get('ml_score', 0.5),
                    metadata=candidate.get('metadata', {})
                ))
            
            # Get reranked results
            reranked = self._t5_rerank_service.rerank_restaurants(query, rerank_candidates, top_k=20)
            
            # Update candidates with T5 scores
            reranked_map = {r['business_id']: r for r in reranked}
            
            for candidate in candidates:
                business_id = candidate.get('business_id', '')
                if business_id in reranked_map:
                    reranked_data = reranked_map[business_id]
                    candidate['t5_score'] = reranked_data['t5_score']
                    candidate['ml_score'] = reranked_data['combined_score']
                    candidate['ml_components']['t5_rerank'] = reranked_data['t5_score']
            
            # Re-sort by new scores
            candidates.sort(key=lambda x: x.get('ml_score', 0), reverse=True)
            
            self.logger.info("T5 reranking applied successfully")
            return candidates
            
        except Exception as e:
            self.logger.error(f"T5 reranking failed: {e}")
            return candidates
    
    def _add_personalized_messages(self, candidates: List[Dict], user_id: str, city: str):
        """Add personalized messages to top candidates"""
        try:
            self._ensure_t5_message_service()
            
            # Get user profile
            user_profile = self._get_user_profile_summary(user_id)
            
            from .t5_message_service import MessageGenerationRequest
            
            for candidate in candidates:
                try:
                    request = MessageGenerationRequest(
                        user_profile=user_profile,
                        restaurant_info={
                            'business_id': candidate.get('business_id', ''),
                            'name': candidate.get('metadata', {}).get('name', 'Restaurant'),
                            'cuisine_types': candidate.get('metadata', {}).get('cuisine_types', ['cuisine']),
                            'stars': candidate.get('metadata', {}).get('stars', 0),
                            'price_level': self._estimate_price_level(candidate.get('metadata', {}))
                        },
                        context={
                            'city': city,
                            'time_of_day': 'lunch' if 11 <= datetime.now().hour <= 14 else 'dinner'
                        }
                    )
                    
                    message = self._t5_message_service.generate_message(request)
                    candidate['personalized_message'] = message
                    
                except Exception as e:
                    self.logger.warning(f"Message generation failed for {candidate.get('business_id')}: {e}")
                    candidate['personalized_message'] = None
            
            self.logger.info("Personalized messages added")
            
        except Exception as e:
            self.logger.error(f"Message generation failed: {e}")
    
    def _get_user_profile_summary(self, user_id: str) -> Dict:
        """Get user profile summary for message generation"""
        behaviors = self.user_behavior_streams.get(user_id, [])
        
        # Extract preferences from behaviors
        liked_categories = []
        for behavior in behaviors:
            if behavior.event_type == 'like' and behavior.data:
                categories = behavior.data.get('categories', '').lower().split(',')
                liked_categories.extend([c.strip() for c in categories])
        
        # Simple preference extraction
        cuisine_counts = {}
        for cat in liked_categories:
            if 'vietnamese' in cat:
                cuisine_counts['vietnamese'] = cuisine_counts.get('vietnamese', 0) + 1
            elif 'italian' in cat:
                cuisine_counts['italian'] = cuisine_counts.get('italian', 0) + 1
            elif 'chinese' in cat:
                cuisine_counts['chinese'] = cuisine_counts.get('chinese', 0) + 1
        
        # Top cuisines
        top_cuisines = sorted(cuisine_counts.keys(), key=lambda x: cuisine_counts[x], reverse=True)[:3]
        
        return {
            'preferences': {
                'cuisine_types': top_cuisines if top_cuisines else ['international'],
                'price_sensitivity': 'medium'  # Default
            }
        }
    
    def _estimate_price_level(self, metadata: Dict) -> str:
        """Estimate price level from metadata"""
        stars = metadata.get('stars', 0)
        if stars <= 2.5:
            return 'budget'
        elif stars <= 3.5:
            return 'moderate'
        elif stars <= 4.5:
            return 'expensive'
        else:
            return 'luxury'
    
    def _process_online_learning_signal(self, event: UserBehaviorEvent):
        """
        Process behavior event for online learning (PHASE 2 - STEP 5)
        Chỉ xử lý Like/Dislike signals
        """
        try:
            if event.event_type not in ['like', 'dislike']:
                return
            
            # Extract context from event data
            context = {
                'city': event.data.get('city') if event.data else None,
                'query': event.data.get('query') if event.data else None,
                'timestamp': event.timestamp.isoformat() if event.timestamp else None
            }
            
            # Send signal to online learning service
            learning_result = self.online_learning_service.process_feedback_signal(
                user_id=event.user_id,
                restaurant_id=event.restaurant_id,
                feedback_type=event.event_type,
                context=context
            )
            
            if learning_result['success']:
                # Update performance tracking
                self.model_performance['learning_performance'] = learning_result.get('learning_impact', 0)
                self.model_performance['last_updated'] = datetime.now()
                
                self.logger.info(f"Online learning processed: {event.event_type} signal with impact {learning_result['learning_impact']:.3f}")
            else:
                self.logger.warning(f"Online learning failed: {learning_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"Error processing online learning signal: {e}")
    
    # ==================== SIMPLIFIED API ====================
    
    def get_ml_insights(self, user_id: str) -> Dict:
        """Get simplified ML insights for user"""
        try:
            behavior_insights = self.get_user_behavior_insights(user_id)
            
            # PHASE 2 - STEP 5: Add online learning insights
            learning_insights = self.online_learning_service.get_learning_insights(user_id)
            
            insights = {
                'user_profile_strength': min(len(self.user_behavior_streams.get(user_id, [])) / 20, 1.0),
                'behavior_insights': behavior_insights,
                'learning_insights': learning_insights,  # New section
                'memory_usage': self.memory_monitor.get_usage_mb(),
                'recommendations_served': len(self.behavior_cache.data),
                'last_activity': datetime.now().isoformat(),
                'online_learning_stats': self.online_learning_service.get_system_stats()  # New stats
            }
            
            return insights
        except Exception as e:
            self.logger.error(f"ML insights error: {e}")
            return {'error': str(e)}
    
    def process_explicit_feedback(self, 
                                user_id: str, 
                                restaurant_id: str, 
                                feedback_type: str,
                                context: Optional[Dict] = None) -> Dict:
        """
        PHASE 2 - STEP 5: Process explicit feedback (Like/Dislike) for online learning
        
        Args:
            user_id: User ID
            restaurant_id: Restaurant business ID
            feedback_type: 'like' or 'dislike'
            context: Additional context
            
        Returns:
            Processing result with learning impact
        """
        try:
            # Create behavior event
            behavior_event = UserBehaviorEvent(
                event_type=feedback_type,
                user_id=user_id,
                restaurant_id=restaurant_id,
                data=context
            )
            
            # Track behavior (this will trigger online learning)
            self.track_user_behavior(behavior_event)
            
            # Get learning insights for response
            learning_insights = self.online_learning_service.get_learning_insights(user_id)
            
            result = {
                'success': True,
                'feedback_processed': True,
                'learning_stage': learning_insights.get('learning_stage', 'unknown'),
                'personalization_confidence': learning_insights.get('personalization_confidence', 0.0),
                'total_feedback': learning_insights.get('total_feedback', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Processed explicit feedback: {feedback_type} from {user_id} for {restaurant_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing explicit feedback: {e}")
            return {
                'success': False,
                'error': str(e),
                'feedback_processed': False
            }
    
    def get_online_learning_status(self) -> Dict:
        """
        PHASE 2 - STEP 5: Get status của online learning system
        """
        try:
            return {
                'service_active': True,
                'learning_stats': self.online_learning_service.get_system_stats(),
                'performance_metrics': self.model_performance,
                'memory_status': self.memory_monitor.get_status()
            }
        except Exception as e:
            self.logger.error(f"Error getting online learning status: {e}")
            return {'error': str(e)}


class LimitedDict:
    """Dictionary với size limit và LRU eviction"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data = {}
        self.access_order = deque()
    
    def __getitem__(self, key):
        if key in self.data:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.data[key]
        raise KeyError(key)
    
    def __setitem__(self, key, value):
        if key in self.data:
            # Update existing
            self.access_order.remove(key)
        elif len(self.data) >= self.max_size:
            # Evict least recently used
            oldest = self.access_order.popleft()
            del self.data[oldest]
        
        self.data[key] = value
        self.access_order.append(key)
    
    def __contains__(self, key):
        return key in self.data
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def pop(self, key, default=None):
        if key in self.data:
            self.access_order.remove(key)
            return self.data.pop(key)
        return default
    
    def clear(self):
        self.data.clear()
        self.access_order.clear()
    
    def keys(self):
        return self.data.keys()


class MemoryMonitor:
    """Memory monitoring class"""
    
    def __init__(self, max_usage_mb: int):
        self.max_usage_mb = max_usage_mb
        self.logger = logging.getLogger(__name__)
    
    def get_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def get_available_mb(self) -> float:
        """Get available memory in MB"""
        return psutil.virtual_memory().available / (1024 * 1024)
    
    def check_memory_available(self) -> bool:
        """Check if enough memory is available"""
        current_usage = self.get_usage_mb()
        available = self.get_available_mb()
        
        if current_usage > self.max_usage_mb:
            self.logger.warning(f"Memory usage {current_usage:.1f}MB exceeds limit {self.max_usage_mb}MB")
            return False
        
        if available < 500:  # Less than 500MB available
            self.logger.warning(f"Low system memory: {available:.1f}MB available")
            return False
        
        return True
    
    def get_status(self) -> Dict:
        """Get memory status"""
        return {
            'current_usage_mb': self.get_usage_mb(),
            'max_usage_mb': self.max_usage_mb,
            'available_mb': self.get_available_mb(),
            'usage_percentage': (self.get_usage_mb() / self.max_usage_mb) * 100,
            'memory_ok': self.check_memory_available()
        }


def demo_advanced_ml_service():
    """Demo Advanced ML Service với memory optimization"""
    print("DEMO ADVANCED ML SERVICE - MEMORY OPTIMIZED")
    print("=" * 60)
    
    from modules.memory.short_term import SessionStore
    from shared.settings import Settings
    
    settings = Settings()
    memory = SessionStore(settings)
    
    # Create optimized config
    config = ServiceConfig(
        max_memory_usage_mb=3500,
        batch_size=16,
        max_sequence_length=256
    )
    
    ml_service = AdvancedMLService(memory, config)
    
    # Test memory status
    print("\n1. Memory Status Check...")
    memory_status = ml_service.get_memory_status()
    print(f"   Current usage: {memory_status['current_usage_mb']:.1f}MB")
    print(f"   Usage: {memory_status['usage_percentage']:.1f}%")
    print(f"   Memory OK: {memory_status['memory_ok']}")
    
    # Test behavior tracking
    print("\n2. Testing User Behavior Monitoring...")
    
    # Track some behaviors
    behaviors = [
        UserBehaviorEvent('search', 'user123', data={'query': 'pizza'}),
        UserBehaviorEvent('click', 'user123', restaurant_id='rest_001'),
        UserBehaviorEvent('like', 'user123', restaurant_id='rest_001')
    ]
    
    for behavior in behaviors:
        ml_service.track_user_behavior(behavior)
    
    # Get behavior insights
    insights = ml_service.get_user_behavior_insights('user123')
    print(f"   Activity level: {insights.get('activity_level', 'unknown')}")
    print(f"   CTR: {insights.get('click_through_rate', 0):.2%}")
    
    # Test recommendations
    print("\n3. Testing Memory-Optimized Recommendations...")
    
    candidates = [
        {
            'business_id': 'rest_001',
            'metadata': {
                'name': 'Pizza Palace',
                'categories': 'Italian, Pizza',
                'stars': 4.5,
                'price_level': 2
            }
        },
        {
            'business_id': 'rest_002',
            'metadata': {
                'name': 'Burger King',
                'categories': 'American, Fast Food',
                'stars': 3.8,
                'price_level': 1
            }
        }
    ]
    
    recommendations = ml_service.get_deep_personalized_recommendations(
        user_id="user123",
        city="Philadelphia",
        candidates=candidates.copy(),
        exploration_factor=0.1
    )
    
    for rec in recommendations:
        print(f"   {rec['metadata']['name']} - Score: {rec['ml_score']:.3f}")
    
    # Test ML insights
    print("\n4. Testing ML Insights...")
    ml_insights = ml_service.get_ml_insights("user123")
    print(f"   Profile strength: {ml_insights.get('user_profile_strength', 0):.2%}")
    print(f"   Memory usage: {ml_insights.get('memory_usage', 0):.1f}MB")
    
    print("\nAdvanced ML Service demo completed!")

if __name__ == "__main__":
    demo_advanced_ml_service() 