"""
Online Learning Service - PHASE 2 STEP 5
Real-time learning từ Like/Dislike feedback signals
Cập nhật user và restaurant embeddings theo thời gian thực
"""

import numpy as np
import logging
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import threading
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

from modules.memory.short_term import SessionStore

@dataclass
class LearningSignal:
    """Signal để học từ user feedback"""
    user_id: str
    restaurant_id: str
    signal_type: str  # 'like' hoặc 'dislike'
    signal_strength: float  # 1.0 cho like, -1.0 cho dislike
    timestamp: datetime
    context: Dict = None

@dataclass
class UserEmbeddingUpdate:
    """Cập nhật embedding cho user"""
    user_id: str
    old_embedding: np.ndarray
    new_embedding: np.ndarray
    update_reason: str
    update_strength: float
    timestamp: datetime

@dataclass
class RestaurantEmbeddingUpdate:
    """Cập nhật embedding cho restaurant"""
    restaurant_id: str
    old_embedding: np.ndarray
    new_embedding: np.ndarray
    update_reason: str
    update_strength: float
    timestamp: datetime

class OnlineLearningService:
    """
    Service để học trực tuyến từ Like/Dislike feedback
    
    Features:
    - Real-time user embedding updates
    - Real-time restaurant embedding updates  
    - Collaborative filtering learning
    - Exploration vs exploitation balance
    - Adaptive learning rates
    - Cold start handling
    """
    
    FEEDBACK_COOLDOWN_SECONDS = 60 # Ignore identical feedback within 60 seconds

    def __init__(self, memory: SessionStore, initial_embeddings: Optional[Dict[str, np.ndarray]] = None):
        self.memory = memory
        self.logger = logging.getLogger(__name__)
        
        # Learning parameters
        self.learning_rate = 0.01  # Base learning rate
        self.adaptive_lr = True    # Adaptive learning rate
        self.exploration_rate = 0.1  # Exploration vs exploitation
        
        # Embedding dimensions
        self.user_embedding_dim = 64
        self.restaurant_embedding_dim = 64
        
        # Online embeddings (sẽ được cập nhật real-time)
        self.user_embeddings = {}  # user_id -> embedding
        
        # Prime restaurant embeddings with rich content embeddings if available
        self.restaurant_embeddings = initial_embeddings.copy() if initial_embeddings is not None else {}
        if initial_embeddings is not None and len(initial_embeddings) > 0:
            # Infer embedding dimension from initial embeddings
            first_embedding = next(iter(initial_embeddings.values()))
            self.restaurant_embedding_dim = first_embedding.shape[0]
            # Match user embedding dimension
            self.user_embedding_dim = self.restaurant_embedding_dim
            self.logger.info(f"✅ Primed Online Learning with {len(self.restaurant_embeddings)} content embeddings. Dim set to {self.restaurant_embedding_dim}.")

        # Learning history
        self.learning_signals = deque(maxlen=10000)  # Recent signals
        self.user_learning_history = defaultdict(list)  # user_id -> [signals]
        self.restaurant_learning_history = defaultdict(list)  # restaurant_id -> [signals]
        
        # Last feedback timestamp to prevent spam
        self.last_feedback = {}
        
        # Performance tracking
        self.learning_stats = {
            'total_signals_processed': 0,
            'user_embeddings_updated': 0,
            'restaurant_embeddings_updated': 0,
            'last_update_time': None,
            'learning_rate_adjustments': 0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load existing embeddings nếu có
        self._load_existing_embeddings()
        
        self.logger.info("✅ OnlineLearningService initialized")
    
    def process_feedback_signal(self, 
                               user_id: str, 
                               restaurant_id: str, 
                               feedback_type: str,
                               context: Optional[Dict] = None) -> Dict:
        """
        Xử lý feedback signal và cập nhật embeddings real-time
        
        Args:
            user_id: ID của user
            restaurant_id: ID của restaurant
            feedback_type: 'like' hoặc 'dislike'
            context: Additional context (city, query, etc.)
            
        Returns:
            Dictionary với kết quả learning update
        """
        try:
            start_time = time.time()
            
            # IDEMPOTENCY CHECK: Ignore duplicate signals within the cooldown period
            last_signal_time = self.last_feedback.get((user_id, restaurant_id))
            if last_signal_time and (datetime.now() - last_signal_time).total_seconds() < self.FEEDBACK_COOLDOWN_SECONDS:
                self.logger.info(f"Ignoring duplicate '{feedback_type}' signal for {user_id} -> {restaurant_id} within cooldown.")
                return {
                    'success': True,
                    'signal_processed': False,
                    'reason': f'Duplicate signal within {self.FEEDBACK_COOLDOWN_SECONDS}s cooldown'
                }

            # Validate feedback type
            if feedback_type not in ['like', 'dislike']:
                raise ValueError(f"Invalid feedback type: {feedback_type}")
            
            # Create learning signal
            signal_strength = 1.0 if feedback_type == 'like' else -1.0
            signal = LearningSignal(
                user_id=user_id,
                restaurant_id=restaurant_id,
                signal_type=feedback_type,
                signal_strength=signal_strength,
                timestamp=datetime.now(),
                context=context or {}
            )
            
            # Process signal
            with self.lock:
                # Update last feedback timestamp
                self.last_feedback[(user_id, restaurant_id)] = signal.timestamp

                # Store signal
                self.learning_signals.append(signal)
                self.user_learning_history[user_id].append(signal)
                self.restaurant_learning_history[restaurant_id].append(signal)
                
                # Update embeddings
                user_update = self._update_user_embedding(signal)
                restaurant_update = self._update_restaurant_embedding(signal)
                
                # Update stats
                self.learning_stats['total_signals_processed'] += 1
                self.learning_stats['last_update_time'] = datetime.now()
            
            # Calculate learning impact
            learning_impact = self._calculate_learning_impact(signal, user_update, restaurant_update)
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'signal_processed': True,
                'user_embedding_updated': user_update is not None,
                'restaurant_embedding_updated': restaurant_update is not None,
                'learning_impact': learning_impact,
                'processing_time': processing_time,
                'timestamp': signal.timestamp.isoformat()
            }
            
            self.logger.info(f"Processed {feedback_type} signal: {user_id} -> {restaurant_id} (impact: {learning_impact:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing feedback signal: {e}")
            return {
                'success': False,
                'error': str(e),
                'signal_processed': False
            }
    
    def _update_user_embedding(self, signal: LearningSignal) -> Optional[UserEmbeddingUpdate]:
        """Cập nhật user embedding dựa trên signal"""
        try:
            user_id = signal.user_id
            restaurant_id = signal.restaurant_id
            
            # Get hoặc create user embedding
            if user_id not in self.user_embeddings:
                self.user_embeddings[user_id] = self._initialize_user_embedding(user_id)
            
            old_embedding = self.user_embeddings[user_id].copy()
            
            # Get restaurant embedding for gradient calculation
            restaurant_embedding = self._get_restaurant_embedding(restaurant_id)
            
            # Calculate gradient cho user embedding
            gradient = self._calculate_user_gradient(
                user_embedding=old_embedding,
                restaurant_embedding=restaurant_embedding,
                signal_strength=signal.signal_strength
            )
            
            # Adaptive learning rate
            current_lr = self._get_adaptive_learning_rate(user_id, 'user')
            
            # Update embedding
            new_embedding = old_embedding + current_lr * gradient
            
            # Normalize embedding
            new_embedding = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)
            
            # Store updated embedding
            self.user_embeddings[user_id] = new_embedding
            
            # Create update record
            update = UserEmbeddingUpdate(
                user_id=user_id,
                old_embedding=old_embedding,
                new_embedding=new_embedding,
                update_reason=f"{signal.signal_type}_feedback",
                update_strength=np.linalg.norm(gradient) * current_lr,
                timestamp=signal.timestamp
            )
            
            self.learning_stats['user_embeddings_updated'] += 1
            
            return update
            
        except Exception as e:
            self.logger.error(f"Error updating user embedding: {e}")
            return None
    
    def _update_restaurant_embedding(self, signal: LearningSignal) -> Optional[RestaurantEmbeddingUpdate]:
        """Cập nhật restaurant embedding dựa trên signal"""
        try:
            user_id = signal.user_id
            restaurant_id = signal.restaurant_id
            
            # Get hoặc create restaurant embedding
            if restaurant_id not in self.restaurant_embeddings:
                self.restaurant_embeddings[restaurant_id] = self._initialize_restaurant_embedding(restaurant_id)
            
            old_embedding = self.restaurant_embeddings[restaurant_id].copy()
            
            # Get user embedding for gradient calculation
            user_embedding = self._get_user_embedding(user_id)
            
            # Calculate gradient cho restaurant embedding
            gradient = self._calculate_restaurant_gradient(
                user_embedding=user_embedding,
                restaurant_embedding=old_embedding,
                signal_strength=signal.signal_strength
            )
            
            # Adaptive learning rate
            current_lr = self._get_adaptive_learning_rate(restaurant_id, 'restaurant')
            
            # Update embedding
            new_embedding = old_embedding + current_lr * gradient
            
            # Normalize embedding
            new_embedding = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)
            
            # Store updated embedding
            self.restaurant_embeddings[restaurant_id] = new_embedding
            
            # Create update record
            update = RestaurantEmbeddingUpdate(
                restaurant_id=restaurant_id,
                old_embedding=old_embedding,
                new_embedding=new_embedding,
                update_reason=f"{signal.signal_type}_feedback",
                update_strength=np.linalg.norm(gradient) * current_lr,
                timestamp=signal.timestamp
            )
            
            self.learning_stats['restaurant_embeddings_updated'] += 1
            
            return update
            
        except Exception as e:
            self.logger.error(f"Error updating restaurant embedding: {e}")
            return None
    
    def _calculate_user_gradient(self, 
                                user_embedding: np.ndarray, 
                                restaurant_embedding: np.ndarray,
                                signal_strength: float) -> np.ndarray:
        """
        Calculate gradient cho user embedding update
        Sử dụng collaborative filtering loss function
        """
        # Predicted preference score
        predicted_score = np.dot(user_embedding, restaurant_embedding)
        
        # Target score từ signal
        target_score = signal_strength  # 1.0 for like, -1.0 for dislike
        
        # Error
        error = target_score - predicted_score
        
        # Gradient cho user embedding (collaborative filtering)
        gradient = error * restaurant_embedding
        
        # Add regularization
        regularization = 0.01 * user_embedding
        gradient = gradient - regularization
        
        return gradient
    
    def _calculate_restaurant_gradient(self,
                                     user_embedding: np.ndarray,
                                     restaurant_embedding: np.ndarray,
                                     signal_strength: float) -> np.ndarray:
        """
        Calculate gradient cho restaurant embedding update
        """
        # Predicted preference score
        predicted_score = np.dot(user_embedding, restaurant_embedding)
        
        # Target score từ signal
        target_score = signal_strength
        
        # Error
        error = target_score - predicted_score
        
        # Gradient cho restaurant embedding
        gradient = error * user_embedding
        
        # Add regularization
        regularization = 0.01 * restaurant_embedding
        gradient = gradient - regularization
        
        return gradient
    
    def _get_adaptive_learning_rate(self, entity_id: str, entity_type: str) -> float:
        """
        Get adaptive learning rate cho entity (user hoặc restaurant)
        Learning rate giảm theo số lần update
        """
        if not self.adaptive_lr:
            return self.learning_rate
        
        # Count số lần entity đã được update
        if entity_type == 'user':
            update_count = len(self.user_learning_history[entity_id])
        else:
            update_count = len(self.restaurant_learning_history[entity_id])
        
        # Adaptive learning rate: lr / (1 + update_count * decay_rate)
        decay_rate = 0.001
        adaptive_lr = self.learning_rate / (1 + update_count * decay_rate)
        
        # Minimum learning rate
        min_lr = 0.001
        adaptive_lr = max(adaptive_lr, min_lr)
        
        return adaptive_lr
    
    def _initialize_user_embedding(self, user_id: str) -> np.ndarray:
        """Initialize embedding cho new user"""
        # Random initialization với small values
        embedding = np.random.normal(0, 0.1, self.user_embedding_dim)
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        self.logger.debug(f"Initialized embedding for new user: {user_id}")
        return embedding
    
    def _initialize_restaurant_embedding(self, restaurant_id: str) -> np.ndarray:
        """Initialize embedding cho new restaurant"""
        # Random initialization với small values
        embedding = np.random.normal(0, 0.1, self.restaurant_embedding_dim)
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        self.logger.debug(f"Initialized embedding for new restaurant: {restaurant_id}")
        return embedding
    
    def _get_user_embedding(self, user_id: str) -> np.ndarray:
        """Get user embedding, initialize nếu chưa có"""
        if user_id not in self.user_embeddings:
            self.user_embeddings[user_id] = self._initialize_user_embedding(user_id)
        return self.user_embeddings[user_id]
    
    def _get_restaurant_embedding(self, restaurant_id: str) -> np.ndarray:
        """Get restaurant embedding, initialize nếu chưa có"""
        if restaurant_id not in self.restaurant_embeddings:
            self.restaurant_embeddings[restaurant_id] = self._initialize_restaurant_embedding(restaurant_id)
        return self.restaurant_embeddings[restaurant_id]
    
    def _calculate_learning_impact(self, 
                                  signal: LearningSignal,
                                  user_update: Optional[UserEmbeddingUpdate],
                                  restaurant_update: Optional[RestaurantEmbeddingUpdate]) -> float:
        """Calculate overall impact của learning update"""
        impact = 0.0
        
        if user_update:
            impact += user_update.update_strength
        
        if restaurant_update:
            impact += restaurant_update.update_strength
        
        # Scale impact
        impact = min(impact, 1.0)  # Cap at 1.0
        
        return impact
    
    def get_personalized_scores(self, 
                               user_id: str, 
                               restaurant_ids: List[str]) -> Dict[str, float]:
        """
        Get personalized scores cho restaurants dựa trên learned embeddings
        
        Args:
            user_id: User ID
            restaurant_ids: List of restaurant IDs
            
        Returns:
            Dictionary mapping restaurant_id -> personalized_score
        """
        try:
            user_embedding = self._get_user_embedding(user_id)
            scores = {}
            
            for restaurant_id in restaurant_ids:
                restaurant_embedding = self._get_restaurant_embedding(restaurant_id)
                
                # Calculate similarity score
                similarity = np.dot(user_embedding, restaurant_embedding)
                
                # Apply exploration boost cho restaurants chưa có feedback
                if not self._user_has_feedback_for_restaurant(user_id, restaurant_id):
                    exploration_boost = self.exploration_rate * np.random.uniform(0, 0.2)
                    similarity += exploration_boost
                
                # Normalize to [0, 1] range
                normalized_score = (similarity + 1) / 2
                
                scores[restaurant_id] = float(normalized_score)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error calculating personalized scores: {e}")
            return {rid: 0.5 for rid in restaurant_ids}  # Default neutral scores
    
    def _user_has_feedback_for_restaurant(self, user_id: str, restaurant_id: str) -> bool:
        """Check if user đã có feedback cho restaurant"""
        user_signals = self.user_learning_history.get(user_id, [])
        return any(signal.restaurant_id == restaurant_id for signal in user_signals)
    
    def get_learning_insights(self, user_id: str) -> Dict:
        """Get insights về learning progress cho user"""
        try:
            user_signals = self.user_learning_history.get(user_id, [])
            
            if not user_signals:
                return {
                    'total_feedback': 0,
                    'learning_stage': 'cold_start',
                    'personalization_confidence': 0.0
                }
            
            # Analyze signals
            like_count = sum(1 for s in user_signals if s.signal_type == 'like')
            dislike_count = sum(1 for s in user_signals if s.signal_type == 'dislike')
            total_feedback = len(user_signals)
            
            # Learning stage
            if total_feedback < 5:
                learning_stage = 'cold_start'
            elif total_feedback < 20:
                learning_stage = 'learning'
            else:
                learning_stage = 'mature'
            
            # Personalization confidence
            confidence = min(total_feedback / 20.0, 1.0)  # Max confidence at 20 feedbacks
            
            # Recent activity
            recent_signals = [s for s in user_signals if s.timestamp > datetime.now() - timedelta(days=7)]
            
            insights = {
                'total_feedback': total_feedback,
                'like_count': like_count,
                'dislike_count': dislike_count,
                'like_ratio': like_count / max(total_feedback, 1),
                'learning_stage': learning_stage,
                'personalization_confidence': confidence,
                'recent_activity': len(recent_signals),
                'embedding_dimension': self.user_embedding_dim,
                'last_feedback_time': user_signals[-1].timestamp.isoformat() if user_signals else None
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting learning insights: {e}")
            return {'error': str(e)}
    
    def save_embeddings(self, filepath: str = "models/online_embeddings.pkl"):
        """Save learned embeddings to disk"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            save_data = {
                'user_embeddings': self.user_embeddings,
                'restaurant_embeddings': self.restaurant_embeddings,
                'learning_stats': self.learning_stats,
                'timestamp': datetime.now().isoformat(),
                'embedding_dimensions': {
                    'user': self.user_embedding_dim,
                    'restaurant': self.restaurant_embedding_dim
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            self.logger.info(f"Saved {len(self.user_embeddings)} user and {len(self.restaurant_embeddings)} restaurant embeddings to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")
    
    def _load_existing_embeddings(self, filepath: str = "models/online_embeddings.pkl"):
        """Load existing embeddings from disk"""
        try:
            if not os.path.exists(filepath):
                self.logger.info("No existing embeddings found, starting fresh")
                return
            
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.user_embeddings = save_data.get('user_embeddings', {})
            self.restaurant_embeddings = save_data.get('restaurant_embeddings', {})
            self.learning_stats = save_data.get('learning_stats', self.learning_stats)
            
            self.logger.info(f"Loaded {len(self.user_embeddings)} user and {len(self.restaurant_embeddings)} restaurant embeddings")
            
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'learning_stats': self.learning_stats,
            'embeddings_count': {
                'users': len(self.user_embeddings),
                'restaurants': len(self.restaurant_embeddings)
            },
            'signals_count': {
                'total': len(self.learning_signals),
                'recent_24h': len([s for s in self.learning_signals 
                                  if s.timestamp > datetime.now() - timedelta(days=1)])
            },
            'learning_parameters': {
                'learning_rate': self.learning_rate,
                'adaptive_lr': self.adaptive_lr,
                'exploration_rate': self.exploration_rate,
                'embedding_dimensions': {
                    'user': self.user_embedding_dim,
                    'restaurant': self.restaurant_embedding_dim
                }
            }
        }


def demo_online_learning():
    """Demo Online Learning Service"""
    print("DEMO ONLINE LEARNING SERVICE")
    print("=" * 50)
    
    from modules.memory.short_term import SessionStore
    from shared.settings import Settings
    
    # Initialize
    settings = Settings()
    memory = SessionStore(settings)
    learning_service = OnlineLearningService(memory)
    
    # Test users and restaurants
    test_data = [
        ("user_001", "restaurant_001", "like"),
        ("user_001", "restaurant_002", "dislike"),
        ("user_001", "restaurant_003", "like"),
        ("user_002", "restaurant_001", "like"),
        ("user_002", "restaurant_002", "like"),
        ("user_003", "restaurant_001", "dislike"),
    ]
    
    print(f"\n1. Processing feedback signals...")
    for user_id, restaurant_id, feedback_type in test_data:
        result = learning_service.process_feedback_signal(
            user_id=user_id,
            restaurant_id=restaurant_id,
            feedback_type=feedback_type,
            context={'city': 'Ho Chi Minh City'}
        )
        
        print(f"   {feedback_type}: {user_id} -> {restaurant_id} (impact: {result.get('learning_impact', 0):.3f})")
    
    # Test personalized scores
    print(f"\n2. Testing personalized scores...")
    test_restaurants = ["restaurant_001", "restaurant_002", "restaurant_003", "restaurant_004"]
    
    for user_id in ["user_001", "user_002", "user_003"]:
        scores = learning_service.get_personalized_scores(user_id, test_restaurants)
        print(f"   {user_id}:")
        for restaurant_id, score in scores.items():
            print(f"     {restaurant_id}: {score:.3f}")
    
    # Get learning insights
    print(f"\n3. Learning insights...")
    for user_id in ["user_001", "user_002"]:
        insights = learning_service.get_learning_insights(user_id)
        print(f"   {user_id}: {insights['total_feedback']} feedback, {insights['learning_stage']} stage, {insights['personalization_confidence']:.2f} confidence")
    
    # System stats
    print(f"\n4. System statistics...")
    stats = learning_service.get_system_stats()
    print(f"   Users with embeddings: {stats['embeddings_count']['users']}")
    print(f"   Restaurants with embeddings: {stats['embeddings_count']['restaurants']}")
    print(f"   Total signals processed: {stats['learning_stats']['total_signals_processed']}")
    print(f"   User embeddings updated: {stats['learning_stats']['user_embeddings_updated']}")
    print(f"   Restaurant embeddings updated: {stats['learning_stats']['restaurant_embeddings_updated']}")
    
    print(f"\nOnline Learning Service demo completed!")

if __name__ == "__main__":
    demo_online_learning() 