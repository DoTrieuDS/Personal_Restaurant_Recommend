"""
Feedback Learning Service
Xử lý feedback để cải thiện recommendation algorithm theo thời gian
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from modules.domain.schemas import (
    FeedbackAction, UserInteraction, LearningSignal,
    FeedbackType, UserProfile, RestaurantPreferences, DiningStyle, CuisinePreference
)
from modules.memory.short_term import SessionStore
from .user_profile_service import UserProfileService

class FeedbackLearningService:
    """
    Service xử lý feedback để cải thiện recommendation algorithm
    Implements collaborative filtering và content-based learning
    """
    
    def __init__(self, memory: SessionStore, user_profile_service: UserProfileService):
        self.memory = memory
        self.user_profile_service = user_profile_service
        self.logger = logging.getLogger(__name__)
        
        # Learning caches
        self.poi_similarity_cache = {}
        self.user_similarity_cache = {}
        self.feedback_matrix = defaultdict(dict)  # user_id -> {poi_id: score}
        
        # Learning parameters
        self.min_feedback_for_learning = 3  # Minimum feedback để bắt đầu learn
        self.similarity_threshold = 0.5
        self.decay_factor = 0.95  # Older feedback có weight thấp hơn
        
        # Load existing feedback data
        self._load_feedback_history()
    
    def process_feedback_batch(self, feedbacks: List[FeedbackAction]) -> Dict:
        """
        Xử lý batch feedback để update learning models
        """
        start_time = datetime.now()
        
        # 1. Process individual feedbacks
        learning_signals = []
        for feedback in feedbacks:
            signal = self.user_profile_service.record_feedback(feedback)
            learning_signals.append(signal)
            
            # Update feedback matrix
            self._update_feedback_matrix(feedback)
        
        # 2. Update similarity matrices
        affected_users = list(set(f.user_id for f in feedbacks))
        affected_pois = list(set(f.business_id for f in feedbacks))
        
        self._update_user_similarities(affected_users)
        self._update_poi_similarities(affected_pois)
        
        # 3. Generate insights
        insights = self._generate_learning_insights(feedbacks, learning_signals)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'processed_feedbacks': len(feedbacks),
            'learning_signals': len(learning_signals),
            'affected_users': len(affected_users),
            'affected_pois': len(affected_pois),
            'processing_time': processing_time,
            'insights': insights,
            'timestamp': datetime.now()
        }
        
        self.logger.info(f"Processed {len(feedbacks)} feedbacks in {processing_time:.2f}s")
        return result
    
    def get_collaborative_boost(self, user_id: str, poi_id: str) -> float:
        """
        Tính collaborative filtering boost cho user-POI pair
        
        Returns:
            Float từ -1.0 đến 1.0 (collaborative boost)
        """
        if user_id not in self.feedback_matrix:
            return 0.0  # New user, no collaborative data
        
        # Find similar users
        similar_users = self._find_similar_users(user_id, limit=10)
        
        if not similar_users:
            return 0.0
        
        # Calculate weighted average of similar users' feedback for this POI
        weighted_scores = []
        total_weight = 0
        
        for similar_user_id, similarity in similar_users:
            if poi_id in self.feedback_matrix[similar_user_id]:
                score = self.feedback_matrix[similar_user_id][poi_id]
                weight = similarity
                
                weighted_scores.append(score * weight)
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        avg_score = sum(weighted_scores) / total_weight
        
        # Normalize to [-1, 1] range
        # Assuming feedback scores are in [-2, 2] range
        collaborative_boost = max(-1.0, min(1.0, avg_score / 2.0))
        
        self.logger.debug(f"Collaborative boost for user {user_id}, POI {poi_id}: {collaborative_boost:.3f}")
        return collaborative_boost
    
    def get_content_based_boost(self, user_id: str, poi_metadata: Dict) -> float:
        """
        Tính content-based boost dựa trên POI features và user history
        
        Returns:
            Float từ -1.0 đến 1.0 (content boost)
        """
        if user_id not in self.feedback_matrix:
            return 0.0
        
        user_feedback = self.feedback_matrix[user_id]
        
        if len(user_feedback) < self.min_feedback_for_learning:
            return 0.0
        
        # Find POIs with similar features that user liked
        similar_pois = self._find_similar_pois(poi_metadata, limit=20)
        
        positive_signals = 0
        negative_signals = 0
        total_weight = 0
        
        for similar_poi_id, similarity in similar_pois:
            if similar_poi_id in user_feedback:
                score = user_feedback[similar_poi_id]
                weight = similarity
                
                if score > 0:
                    positive_signals += score * weight
                else:
                    negative_signals += abs(score) * weight
                
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Calculate net preference
        net_preference = (positive_signals - negative_signals) / total_weight
        content_boost = max(-1.0, min(1.0, net_preference / 2.0))
        
        self.logger.debug(f"Content-based boost for user {user_id}: {content_boost:.3f}")
        return content_boost
    
    def get_temporal_boost(self, user_id: str, poi_id: str) -> float:
        """
        Tính temporal boost dựa trên recent trends
        
        Returns:
            Float từ -1.0 đến 1.0 (temporal boost)
        """
        # Get recent feedback for this POI (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_feedback = self._get_recent_poi_feedback(poi_id, recent_cutoff)
        
        if len(recent_feedback) < 3:
            return 0.0
        
        # Calculate trend
        positive_recent = sum(1 for f in recent_feedback if self._feedback_to_score(f) > 0)
        total_recent = len(recent_feedback)
        recent_ratio = positive_recent / total_recent
        
        # Compare with historical ratio
        all_feedback = self._get_all_poi_feedback(poi_id)
        if len(all_feedback) < 10:
            return 0.0
        
        positive_all = sum(1 for f in all_feedback if self._feedback_to_score(f) > 0)
        total_all = len(all_feedback)
        historical_ratio = positive_all / total_all
        
        # Temporal boost based on trend
        trend_change = recent_ratio - historical_ratio
        temporal_boost = max(-0.5, min(0.5, trend_change * 2))  # Scale to [-0.5, 0.5]
        
        self.logger.debug(f"Temporal boost for POI {poi_id}: {temporal_boost:.3f}")
        return temporal_boost
    
    def get_diversity_penalty(self, user_id: str, candidate_pois: List[Dict]) -> Dict[str, float]:
        """
        Tính diversity penalty để tránh recommendation quá đồng nhất
        
        Returns:
            Dict mapping poi_id -> penalty_score (0 to -1)
        """
        penalties = {}
        
        # Get user's recent recommendations/interactions
        recent_interactions = self._get_user_recent_interactions(user_id, days=7)
        
        if not recent_interactions:
            return {poi['business_id']: 0.0 for poi in candidate_pois}
        
        # Extract categories from recent interactions
        recent_categories = []
        for interaction in recent_interactions:
            poi_data = self._get_poi_metadata(interaction['business_id'])
            if poi_data:
                categories = poi_data.get('categories', '').lower().split(', ')
                recent_categories.extend(categories)
        
        category_counts = defaultdict(int)
        for cat in recent_categories:
            category_counts[cat] += 1
        
        # Calculate penalty for each candidate POI
        for poi in candidate_pois:
            poi_id = poi['business_id']
            poi_categories = poi.get('categories', '').lower().split(', ')
            
            # Calculate overlap with recent categories
            overlap_penalty = 0.0
            for cat in poi_categories:
                if cat in category_counts:
                    # More penalty for more frequent categories
                    overlap_penalty += category_counts[cat] * 0.1
            
            # Cap penalty at -1.0
            penalties[poi_id] = max(-1.0, -overlap_penalty)
        
        return penalties
    
    def get_popularity_boost(self, poi_id: str, time_window_days: int = 30) -> float:
        """
        Tính popularity boost dựa trên recent feedback volume
        
        Returns:
            Float từ 0.0 đến 0.5 (popularity boost)
        """
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_feedback = self._get_recent_poi_feedback(poi_id, cutoff_date)
        
        if not recent_feedback:
            return 0.0
        
        # Calculate feedback volume boost
        volume_boost = min(0.3, len(recent_feedback) / 100.0)  # Cap at 0.3 for 100+ feedbacks
        
        # Calculate sentiment boost
        positive_feedback = sum(1 for f in recent_feedback if self._feedback_to_score(f) > 0)
        sentiment_ratio = positive_feedback / len(recent_feedback)
        sentiment_boost = (sentiment_ratio - 0.5) * 0.4  # [-0.2, 0.2] range
        
        popularity_boost = max(0.0, volume_boost + sentiment_boost)
        
        self.logger.debug(f"Popularity boost for POI {poi_id}: {popularity_boost:.3f}")
        return popularity_boost
    
    def update_recommendation_scores(self, 
                                   user_id: str, 
                                   candidate_recommendations: List[Dict],
                                   use_collaborative: bool = True,
                                   use_content_based: bool = True,
                                   use_temporal: bool = True,
                                   use_diversity: bool = True) -> List[Dict]:
        """
        Cập nhật recommendation scores với learning-based adjustments
        """
        if not candidate_recommendations:
            return candidate_recommendations
        
        # Get diversity penalties if enabled
        diversity_penalties = {}
        if use_diversity:
            diversity_penalties = self.get_diversity_penalty(user_id, candidate_recommendations)
        
        # Update each recommendation
        for rec in candidate_recommendations:
            poi_id = rec['business_id']
            poi_metadata = rec.get('poi_info', {})
            
            # Original score
            original_score = rec.get('reranked_score', rec.get('ranking_score', 0.5))
            
            # Calculate boosts
            total_boost = 0.0
            boost_details = {}
            
            if use_collaborative:
                collab_boost = self.get_collaborative_boost(user_id, poi_id)
                total_boost += collab_boost * 0.3  # 30% weight
                boost_details['collaborative'] = collab_boost
            
            if use_content_based:
                content_boost = self.get_content_based_boost(user_id, poi_metadata)
                total_boost += content_boost * 0.3  # 30% weight
                boost_details['content_based'] = content_boost
            
            if use_temporal:
                temporal_boost = self.get_temporal_boost(user_id, poi_id)
                total_boost += temporal_boost * 0.2  # 20% weight
                boost_details['temporal'] = temporal_boost
            
            # Popularity boost (independent of user)
            popularity_boost = self.get_popularity_boost(poi_id)
            total_boost += popularity_boost * 0.2  # 20% weight
            boost_details['popularity'] = popularity_boost
            
            # Diversity penalty
            if use_diversity and poi_id in diversity_penalties:
                diversity_penalty = diversity_penalties[poi_id]
                total_boost += diversity_penalty * 0.3  # 30% weight for diversity
                boost_details['diversity'] = diversity_penalty
            
            # Update score
            adjusted_score = original_score + total_boost
            adjusted_score = max(0.0, min(1.0, adjusted_score))  # Clamp to [0, 1]
            
            # Update recommendation
            rec['learning_adjusted_score'] = adjusted_score
            rec['learning_boost_details'] = boost_details
            rec['total_learning_boost'] = total_boost
        
        # Re-sort by adjusted scores
        candidate_recommendations.sort(key=lambda x: x['learning_adjusted_score'], reverse=True)
        
        # Update ranks
        for i, rec in enumerate(candidate_recommendations):
            rec['learning_adjusted_rank'] = i + 1
        
        self.logger.info(f"Updated scores for {len(candidate_recommendations)} recommendations using learning")
        return candidate_recommendations
    
    # ========================================
    # RESTAURANT-SPECIFIC METHODS
    # ========================================
    
    def get_restaurant_collaborative_boost(self, user_id: str, restaurant_id: str, city: str) -> float:
        """
        Tính collaborative boost cho restaurant trong specific city
        
        Args:
            user_id: User identifier
            restaurant_id: Restaurant business ID
            city: City context
            
        Returns:
            Collaborative boost từ -1.0 đến 1.0
        """
        try:
            # Find similar users trong same city
            similar_users = self._find_similar_users_in_city(user_id, city, limit=10)
            
            if not similar_users:
                return 0.0
            
            # Get weighted feedback from similar users for this restaurant
            weighted_scores = []
            total_weight = 0.0
            
            for similar_user_id, similarity in similar_users:
                if similar_user_id in self.feedback_matrix:
                    user_feedback = self.feedback_matrix[similar_user_id]
                    
                    if restaurant_id in user_feedback:
                        feedback_score = user_feedback[restaurant_id]
                        weight = similarity * 0.8  # Reduce weight for restaurant-specific context
                        
                        weighted_scores.append(feedback_score * weight)
                        total_weight += weight
            
            if total_weight == 0:
                return 0.0
            
            # Calculate weighted average
            avg_score = sum(weighted_scores) / total_weight
            
            # Normalize to [-1, 1] and apply city boost
            city_boost = self._get_city_context_boost(city)
            collaborative_boost = (avg_score / 2.0) * (1.0 + city_boost * 0.2)
            
            return max(-1.0, min(1.0, collaborative_boost))
            
        except Exception as e:
            self.logger.error(f"Error calculating restaurant collaborative boost: {e}")
            return 0.0
    
    def get_cuisine_content_boost(self, user_id: str, restaurant_metadata: Dict) -> float:
        """
        Tính content-based boost dựa trên cuisine patterns
        
        Args:
            user_id: User identifier
            restaurant_metadata: Restaurant metadata
            
        Returns:
            Content boost từ -1.0 đến 1.0
        """
        try:
            if user_id not in self.feedback_matrix:
                return 0.0
            
            user_feedback = self.feedback_matrix[user_id]
            
            if len(user_feedback) < self.min_feedback_for_learning:
                return 0.0  # Need minimum feedback for content analysis
            
            # Extract restaurant cuisines
            from modules.domain.restaurant_schemas import extract_cuisine_types_from_categories
            restaurant_cuisines = extract_cuisine_types_from_categories(
                restaurant_metadata.get('categories', '')
            )
            
            if not restaurant_cuisines:
                return 0.0
            
            # Analyze user's cuisine preferences from feedback history
            cuisine_scores = self._analyze_user_cuisine_preferences(user_id)
            
            # Calculate boost based on cuisine matching
            content_boost = 0.0
            for cuisine in restaurant_cuisines:
                cuisine_score = cuisine_scores.get(cuisine.value, 0.0)
                content_boost += cuisine_score * 0.3  # Each cuisine contributes up to 0.3
            
            # Apply restaurant quality factor
            quality_factor = self._calculate_restaurant_quality_factor(restaurant_metadata)
            content_boost *= quality_factor
            
            return max(-1.0, min(1.0, content_boost))
            
        except Exception as e:
            self.logger.error(f"Error calculating cuisine content boost: {e}")
            return 0.0
    
    def update_restaurant_recommendation_scores(self,
                                              user_id: str,
                                              restaurant_candidates: List[Dict],
                                              city: str) -> List[Dict]:
        """
        Update restaurant recommendation scores với city-aware learning
        
        Args:
            user_id: User identifier
            restaurant_candidates: List of restaurant candidates
            city: City context
            
        Returns:
            Updated candidates với learning-adjusted scores
        """
        if not restaurant_candidates:
            return restaurant_candidates
        
        try:
            updated_candidates = []
            
            # Get city-specific diversity penalties
            diversity_penalties = self._get_restaurant_diversity_penalties(user_id, restaurant_candidates, city)
            
            for candidate in restaurant_candidates:
                restaurant_id = candidate['business_id']
                restaurant_metadata = candidate.get('poi_info', {})
                
                # Original scores
                original_score = candidate.get('reranked_score', candidate.get('ranking_score', 0.5))
                
                # Restaurant-specific boosts
                total_boost = 0.0
                boost_details = {}
                
                # 1. Collaborative boost (city-aware)
                collaborative_boost = self.get_restaurant_collaborative_boost(user_id, restaurant_id, city)
                total_boost += collaborative_boost * 0.35  # 35% weight
                boost_details['collaborative'] = collaborative_boost
                
                # 2. Cuisine content boost
                cuisine_boost = self.get_cuisine_content_boost(user_id, restaurant_metadata)
                total_boost += cuisine_boost * 0.25  # 25% weight
                boost_details['cuisine_content'] = cuisine_boost
                
                # 3. Restaurant popularity boost (city-specific)
                popularity_boost = self._get_city_restaurant_popularity_boost(restaurant_id, city)
                total_boost += popularity_boost * 0.15  # 15% weight
                boost_details['city_popularity'] = popularity_boost
                
                # 4. Temporal trends boost
                temporal_boost = self._get_restaurant_temporal_boost(restaurant_id, city)
                total_boost += temporal_boost * 0.10  # 10% weight
                boost_details['temporal'] = temporal_boost
                
                # 5. Diversity penalty
                diversity_penalty = diversity_penalties.get(restaurant_id, 0.0)
                total_boost += diversity_penalty * 0.15  # 15% weight
                boost_details['diversity'] = diversity_penalty
                
                # Calculate final adjusted score
                adjusted_score = original_score + total_boost
                adjusted_score = max(0.0, min(1.0, adjusted_score))
                
                # Update candidate
                candidate['restaurant_learning_boost'] = total_boost
                candidate['restaurant_boost_details'] = boost_details
                candidate['restaurant_adjusted_score'] = adjusted_score
                
                updated_candidates.append(candidate)
            
            # Sort by adjusted scores
            updated_candidates.sort(key=lambda x: x.get('restaurant_adjusted_score', x.get('reranked_score', 0)), reverse=True)
            
            # Update ranks
            for i, candidate in enumerate(updated_candidates):
                candidate['restaurant_adjusted_rank'] = i + 1
            
            self.logger.info(f"Updated restaurant scores for {len(updated_candidates)} candidates in {city}")
            return updated_candidates
            
        except Exception as e:
            self.logger.error(f"Error updating restaurant recommendation scores: {e}")
            return restaurant_candidates
    
    def _find_similar_users_in_city(self, user_id: str, city: str, limit: int = 10) -> List[tuple]:
        """Find similar users with feedback in the same city"""
        if user_id not in self.feedback_matrix:
            return []
        
        user_feedback = self.feedback_matrix[user_id]
        city_similar_users = []
        
        for other_user_id, other_feedback in self.feedback_matrix.items():
            if other_user_id == user_id:
                continue
            
            # Check if other user has restaurant experience in this city
            has_city_experience = self._user_has_city_restaurant_experience(other_user_id, city)
            
            if not has_city_experience:
                continue
            
            # Calculate similarity based on restaurant preferences
            similarity = self._calculate_restaurant_preference_similarity(user_feedback, other_feedback)
            
            if similarity > 0.4:  # Threshold for restaurant similarity
                city_similar_users.append((other_user_id, similarity))
        
        # Sort by similarity
        city_similar_users.sort(key=lambda x: x[1], reverse=True)
        return city_similar_users[:limit]
    
    def _analyze_user_cuisine_preferences(self, user_id: str) -> Dict[str, float]:
        """Analyze user's cuisine preferences from feedback history"""
        if user_id not in self.feedback_matrix:
            return {}
        
        user_feedback = self.feedback_matrix[user_id]
        cuisine_scores = {}
        
        # Simplified: would need restaurant metadata to map business_ids to cuisines
        # For now, return empty dict
        return cuisine_scores
    
    def _calculate_restaurant_quality_factor(self, restaurant_metadata: Dict) -> float:
        """Calculate quality factor cho restaurant"""
        stars = restaurant_metadata.get('stars', 0)
        review_count = restaurant_metadata.get('review_count', 0)
        
        # Quality factor based on rating and review count
        rating_factor = min(1.0, stars / 5.0)
        popularity_factor = min(1.0, review_count / 100.0)  # Normalize to 100 reviews
        
        return (rating_factor * 0.7) + (popularity_factor * 0.3)
    
    def _get_restaurant_diversity_penalties(self, user_id: str, candidates: List[Dict], city: str) -> Dict[str, float]:
        """Calculate diversity penalties for restaurant recommendations"""
        penalties = {}
        
        # Get user's recent restaurant visits in this city
        recent_visits = self._get_user_recent_restaurant_visits(user_id, city, days=30)
        
        if not recent_visits:
            return {candidate['business_id']: 0.0 for candidate in candidates}
        
        # Extract cuisines from recent visits
        recent_cuisines = []
        for visit in recent_visits:
            # Would need to lookup restaurant metadata
            # For now, simplified
            pass
        
        # Calculate penalties based on cuisine repetition
        for candidate in candidates:
            restaurant_id = candidate['business_id']
            restaurant_metadata = candidate.get('poi_info', {})
            
            # Simple penalty for recently visited restaurants
            if restaurant_id in [visit['restaurant_id'] for visit in recent_visits]:
                penalties[restaurant_id] = -0.3
            else:
                penalties[restaurant_id] = 0.0
        
        return penalties
    
    def _get_city_restaurant_popularity_boost(self, restaurant_id: str, city: str) -> float:
        """Get popularity boost for restaurant in specific city"""
        # Get recent feedback for this restaurant in this city
        city_feedback = self._get_restaurant_city_feedback(restaurant_id, city, days=30)
        
        if len(city_feedback) < 3:
            return 0.0
        
        # Calculate popularity boost
        positive_feedback = sum(1 for f in city_feedback if self._feedback_to_score(f) > 0)
        popularity_ratio = positive_feedback / len(city_feedback)
        
        # Boost based on local popularity
        return (popularity_ratio - 0.5) * 0.3  # -0.15 to 0.15 range
    
    def _get_restaurant_temporal_boost(self, restaurant_id: str, city: str) -> float:
        """Get temporal trends boost for restaurant"""
        recent_feedback = self._get_restaurant_city_feedback(restaurant_id, city, days=14)
        older_feedback = self._get_restaurant_city_feedback(restaurant_id, city, days=90)
        
        if len(recent_feedback) < 2 or len(older_feedback) < 5:
            return 0.0
        
        # Calculate trend
        recent_score = sum(self._feedback_to_score(f) for f in recent_feedback) / len(recent_feedback)
        historical_score = sum(self._feedback_to_score(f) for f in older_feedback) / len(older_feedback)
        
        trend_change = recent_score - historical_score
        return max(-0.2, min(0.2, trend_change * 0.5))  # Scale to [-0.2, 0.2]
    
    def _user_has_city_restaurant_experience(self, user_id: str, city: str) -> bool:
        """Check if user has restaurant experience in city"""
        # Simplified: check if user has any feedback
        # In reality, would need to check city-specific restaurant feedback
        return user_id in self.feedback_matrix and len(self.feedback_matrix[user_id]) > 0
    
    def _calculate_restaurant_preference_similarity(self, feedback1: Dict, feedback2: Dict) -> float:
        """Calculate similarity between two users' restaurant preferences"""
        # Find common restaurants
        common_restaurants = set(feedback1.keys()) & set(feedback2.keys())
        
        if len(common_restaurants) < 2:
            return 0.0
        
        # Calculate correlation on common restaurants
        scores1 = [feedback1[rid] for rid in common_restaurants]
        scores2 = [feedback2[rid] for rid in common_restaurants]
        
        return self._pearson_correlation(scores1, scores2)
    
    def _get_user_recent_restaurant_visits(self, user_id: str, city: str, days: int = 30) -> List[Dict]:
        """Get user's recent restaurant visits in city"""
        # Simplified: return empty list
        # In reality, would query user's restaurant visit history
        return []
    
    def _get_restaurant_city_feedback(self, restaurant_id: str, city: str, days: int = 30) -> List:
        """Get feedback for restaurant in specific city"""
        # Simplified: use general restaurant feedback
        return self._get_recent_poi_feedback(restaurant_id, datetime.now() - timedelta(days=days))
    
    def _get_city_context_boost(self, city: str) -> float:
        """Get boost factor based on city context"""
        # Cities with more data get higher confidence
        city_popularity = {
            "ho chi minh city": 0.3,
            "hanoi": 0.3,
            "da nang": 0.2,
            "nha trang": 0.1,
            "can tho": 0.1
        }
        
        return city_popularity.get(city.lower(), 0.0)
    
    # ========================================
    # PRIVATE HELPER METHODS
    # ========================================
    
    def _load_feedback_history(self):
        """Load existing feedback data vào memory"""
        # Simplified - trong production sẽ load từ database
        self.logger.info("Loading feedback history...")
    
    def _update_feedback_matrix(self, feedback: FeedbackAction):
        """Update feedback matrix với new feedback"""
        score = self._feedback_to_score(feedback)
        self.feedback_matrix[feedback.user_id][feedback.business_id] = score
    
    def _feedback_to_score(self, feedback: FeedbackAction) -> float:
        """Convert feedback thành numeric score"""
        if feedback.feedback_type == FeedbackType.RATE and feedback.feedback_value:
            # Rating 1-5 -> score -2 to 2
            return (feedback.feedback_value - 3.0) * (2.0 / 2.0)
        
        # Other feedback types
        score_mapping = {
            FeedbackType.LIKE: 1.5,
            FeedbackType.DISLIKE: -1.5,
            FeedbackType.BOOKMARK: 1.0,
            FeedbackType.VISIT: 2.0,
            FeedbackType.SKIP: -0.5,
            FeedbackType.SHARE: 0.8
        }
        
        return score_mapping.get(feedback.feedback_type, 0.0)
    
    def _find_similar_users(self, user_id: str, limit: int = 10) -> List[Tuple[str, float]]:
        """Tìm users tương tự dựa trên feedback patterns"""
        if user_id not in self.feedback_matrix:
            return []
        
        user_feedback = self.feedback_matrix[user_id]
        similarities = []
        
        for other_user_id, other_feedback in self.feedback_matrix.items():
            if other_user_id == user_id:
                continue
            
            # Calculate Pearson correlation
            similarity = self._calculate_user_similarity(user_feedback, other_feedback)
            
            if similarity > self.similarity_threshold:
                similarities.append((other_user_id, similarity))
        
        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def _calculate_user_similarity(self, feedback1: Dict, feedback2: Dict) -> float:
        """Tính similarity giữa 2 users dựa trên feedback"""
        # Find common POIs
        common_pois = set(feedback1.keys()) & set(feedback2.keys())
        
        if len(common_pois) < 2:
            return 0.0
        
        # Calculate Pearson correlation
        scores1 = [feedback1[poi] for poi in common_pois]
        scores2 = [feedback2[poi] for poi in common_pois]
        
        return self._pearson_correlation(scores1, scores2)
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Tính Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) == 0:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _find_similar_pois(self, poi_metadata: Dict, limit: int = 20) -> List[Tuple[str, float]]:
        """Tìm POIs tương tự dựa trên features"""
        # Simplified similarity based on categories
        target_categories = set(poi_metadata.get('categories', '').lower().split(', '))
        similarities = []
        
        # In production, would use actual POI database
        # For now, return empty list
        return similarities
    
    def _get_recent_poi_feedback(self, poi_id: str, cutoff_date: datetime) -> List[FeedbackAction]:
        """Lấy recent feedback cho POI"""
        # Simplified - tìm trong cache
        recent_feedback = []
        for user_feedback in self.user_profile_service.feedback_cache.values():
            if (user_feedback.business_id == poi_id and 
                user_feedback.timestamp >= cutoff_date):
                recent_feedback.append(user_feedback)
        
        return recent_feedback
    
    def _get_all_poi_feedback(self, poi_id: str) -> List[FeedbackAction]:
        """Lấy tất cả feedback cho POI"""
        all_feedback = []
        for user_feedback in self.user_profile_service.feedback_cache.values():
            if user_feedback.business_id == poi_id:
                all_feedback.append(user_feedback)
        
        return all_feedback
    
    def _get_user_recent_interactions(self, user_id: str, days: int = 7) -> List[Dict]:
        """Lấy recent interactions của user"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_interactions = []
        
        for interaction in self.user_profile_service.interaction_cache.values():
            if (interaction.user_id == user_id and 
                interaction.timestamp >= cutoff_date):
                recent_interactions.append({
                    'business_id': interaction.business_id,
                    'interaction_type': interaction.interaction_type,
                    'timestamp': interaction.timestamp
                })
        
        return recent_interactions
    
    def _get_poi_metadata(self, poi_id: str) -> Optional[Dict]:
        """Lấy metadata của POI"""
        # In production, would lookup from POI database
        # For now, return None
        return None
    
    def _update_user_similarities(self, user_ids: List[str]):
        """Update user similarity cache cho affected users"""
        # Simplified - trong production sẽ expensive operation
        pass
    
    def _update_poi_similarities(self, poi_ids: List[str]):
        """Update POI similarity cache cho affected POIs"""
        # Simplified - trong production sẽ expensive operation
        pass
    
    def _generate_learning_insights(self, 
                                  feedbacks: List[FeedbackAction], 
                                  signals: List[LearningSignal]) -> Dict:
        """Generate insights từ feedback processing"""
        insights = {
            'feedback_distribution': defaultdict(int),
            'top_liked_categories': [],
            'emerging_trends': [],
            'user_behavior_patterns': {}
        }
        
        # Feedback distribution
        for feedback in feedbacks:
            insights['feedback_distribution'][feedback.feedback_type.value] += 1
        
        # Convert defaultdict to regular dict for JSON serialization
        insights['feedback_distribution'] = dict(insights['feedback_distribution'])
        
        return insights


def demo_feedback_learning():
    """Demo Feedback Learning Service"""
    print("DEMO FEEDBACK LEARNING SERVICE")
    print("=" * 50)
    
    # Setup services
    from modules.memory.short_term import SessionStore
    from shared.settings import Settings
    
    settings = Settings()
    memory = SessionStore(settings)
    user_service = UserProfileService(memory)
    learning_service = FeedbackLearningService(memory, user_service)
    
    # Create test user with profile
    user_id = "test_learner_001"
    profile = user_service.get_or_create_profile(user_id)
    
    # Update preferences
    preferences = RestaurantPreferences(
        dining_styles=[DiningStyle.CASUAL],
        preferred_cuisines=[CuisinePreference.VIETNAMESE]
    )
    user_service.update_profile(user_id, restaurant_preferences=preferences)
    
    # Generate some feedback
    print(f"\n1. Generating sample feedback...")
    from modules.domain.schemas import FeedbackAction, FeedbackType
    
    feedbacks = [
        FeedbackAction(
            user_id=user_id,
            business_id="vietnamese_restaurant_001",
            feedback_type=FeedbackType.LIKE,
            search_query="Vietnamese restaurant",
            destination="Ho Chi Minh City"
        ),
        FeedbackAction(
            user_id=user_id,
            business_id="temple_001",
            feedback_type=FeedbackType.VISIT,
            search_query="Cultural sites",
            destination="Ho Chi Minh City"
        ),
        FeedbackAction(
            user_id=user_id,
            business_id="fastfood_001",
            feedback_type=FeedbackType.DISLIKE,
            search_query="Quick food",
            destination="Ho Chi Minh City"
        )
    ]
    
    # Process feedback batch
    result = learning_service.process_feedback_batch(feedbacks)
    print(f"   Processed: {result['processed_feedbacks']} feedbacks")
    print(f"   Learning signals: {result['learning_signals']}")
    print(f"   Processing time: {result['processing_time']:.3f}s")
    
    # Test boosts
    print(f"\n2. Testing learning-based boosts...")
    
    # Test collaborative boost
    collab_boost = learning_service.get_collaborative_boost(user_id, "vietnamese_restaurant_002")
    print(f"   Collaborative boost: {collab_boost:.3f}")
    
    # Test content-based boost
    poi_metadata = {
        "name": "Phở Sài Gòn",
        "categories": "Vietnamese, Restaurants, Noodles",
        "poi_type": "restaurant",
        "stars": 4.3
    }
    content_boost = learning_service.get_content_based_boost(user_id, poi_metadata)
    print(f"   Content-based boost: {content_boost:.3f}")
    
    # Test temporal boost
    temporal_boost = learning_service.get_temporal_boost(user_id, "vietnamese_restaurant_001")
    print(f"   Temporal boost: {temporal_boost:.3f}")
    
    # Test popularity boost
    popularity_boost = learning_service.get_popularity_boost("vietnamese_restaurant_001")
    print(f"   Popularity boost: {popularity_boost:.3f}")
    
    # Test recommendation score updates
    print(f"\n3. Testing recommendation score updates...")
    
    mock_recommendations = [
        {
            'business_id': 'vietnamese_restaurant_002',
            'reranked_score': 0.75,
            'poi_info': {
                'name': 'Phở Sài Gòn',
                'categories': 'Vietnamese, Restaurants, Noodles',
                'poi_type': 'restaurant',
                'stars': 4.3
            }
        },
        {
            'business_id': 'temple_002',
            'reranked_score': 0.70,
            'poi_info': {
                'name': 'Jade Emperor Pagoda',
                'categories': 'Temples, Cultural, Tourist Attractions',
                'poi_type': 'attraction',
                'stars': 4.5
            }
        }
    ]
    
    updated_recommendations = learning_service.update_recommendation_scores(
        user_id, mock_recommendations
    )
    
    for i, rec in enumerate(updated_recommendations):
        print(f"   {i+1}. {rec['poi_info']['name']}")
        print(f"      Original: {rec['reranked_score']:.3f} -> Adjusted: {rec['learning_adjusted_score']:.3f}")
        print(f"      Total boost: {rec['total_learning_boost']:.3f}")
    
    print(f"\nFeedback Learning Service demo completed!")

if __name__ == "__main__":
    demo_feedback_learning() 