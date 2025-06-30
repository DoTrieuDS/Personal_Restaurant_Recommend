"""
User Profile Service
Quản lý user profiles, preferences và learning từ user behavior
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from modules.domain.schemas import (
    UserProfile, UserDemographics, RestaurantPreferences, 
    FeedbackAction, UserInteraction, LearningSignal,
    PersonalizationMetrics, DiningStyle, CuisinePreference
)
from modules.memory.short_term import SessionStore
import hashlib

class UserProfileService:
    """
    Service quản lý User Profiles và Personalization Learning
    """
    
    def __init__(self, memory: SessionStore):
        self.memory = memory
        self.logger = logging.getLogger(__name__)
        
        # In-memory cache cho profiles (production nên dùng Redis/DB)
        self.profile_cache = {}
        self.feedback_cache = {}
        self.interaction_cache = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.feedback_weights = {
            "like": 1.0,
            "dislike": -1.0,
            "bookmark": 0.8,
            "visit": 1.0,
            "rate": 1.0,  # Will be scaled by actual rating
            "skip": -0.3,
            "share": 0.6
        }
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """
        Lấy hoặc tạo user profile mới
        """
        # Kiểm tra cache trước
        if user_id in self.profile_cache:
            return self.profile_cache[user_id]
        
        # Kiểm tra trong memory store
        profile_data = self.memory.get(f"profile:{user_id}")
        
        if profile_data:
            profile = UserProfile(**profile_data)
            self.profile_cache[user_id] = profile
            return profile
        
        # Tạo profile mới
        new_profile = UserProfile(
            user_id=user_id,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Lưu vào memory store
        self.save_profile(new_profile)
        
        self.logger.info(f"✅ Created new user profile for: {user_id}")
        return new_profile
    
    def save_profile(self, profile: UserProfile):
        """
        Lưu user profile
        """
        profile.last_updated = datetime.now()
        profile.profile_completeness = self._calculate_completeness(profile)
        
        # Cache
        self.profile_cache[profile.user_id] = profile
        
        # Persist to memory store
        self.memory.set(f"profile:{profile.user_id}", profile.dict())
        
        self.logger.debug(f"💾 Saved profile for user: {profile.user_id}")
    
    def update_profile(self, user_id: str, 
                      demographics: Optional[UserDemographics] = None,
                      restaurant_preferences: Optional[RestaurantPreferences] = None) -> UserProfile:
        """
        Cập nhật user profile
        """
        profile = self.get_or_create_profile(user_id)
        
        if demographics:
            profile.demographics = demographics
        
        if restaurant_preferences:
            profile.restaurant_preferences = restaurant_preferences
        
        self.save_profile(profile)
        
        self.logger.info(f"🔄 Updated profile for user: {user_id}")
        return profile
    
    def record_feedback(self, feedback: FeedbackAction) -> LearningSignal:
        """
        Ghi nhận user feedback và tạo learning signal
        """
        # Lưu feedback
        feedback_id = self._generate_feedback_id(feedback)
        self.feedback_cache[feedback_id] = feedback
        self.memory.set(f"feedback:{feedback_id}", feedback.dict())
        
        # Tạo learning signal
        signal = self._process_feedback_to_signal(feedback)
        
        # Cập nhật user profile dựa trên signal
        self._update_profile_from_signal(feedback.user_id, signal)
        
        self.logger.info(f"📝 Recorded feedback: {feedback.feedback_type} for POI {feedback.business_id}")
        return signal
    
    def record_interaction(self, interaction: UserInteraction):
        """
        Ghi nhận user interaction (view, click, dwell time, etc.)
        """
        interaction_id = self._generate_interaction_id(interaction)
        self.interaction_cache[interaction_id] = interaction
        self.memory.set(f"interaction:{interaction_id}", interaction.dict())
        
        # Process implicit feedback nếu có
        if self._should_create_implicit_signal(interaction):
            signal = self._process_interaction_to_signal(interaction)
            self._update_profile_from_signal(interaction.user_id, signal)
        
        self.logger.debug(f"📊 Recorded interaction: {interaction.interaction_type} for POI {interaction.business_id}")
    
    def get_personalized_boost(self, user_id: str, poi_metadata: Dict) -> float:
        """
        Tính toán boost score cho POI dựa trên user preferences
        
        Returns:
            Float từ -1.0 đến 1.0 (boost factor)
        """
        profile = self.get_or_create_profile(user_id)
        
        if profile.profile_completeness < 0.1:
            return 0.0  # Không đủ data để personalize
        
        boost_score = 0.0
        
        # 1. Cuisine preferences
        boost_score += self._calculate_cuisine_boost(profile, poi_metadata)
        
        # 2. Dining style preferences  
        boost_score += self._calculate_dining_boost(profile, poi_metadata)
        
        # 3. Budget preferences
        boost_score += self._calculate_budget_boost(profile, poi_metadata)
        
        # 4. Historical preferences (từ feedback)
        boost_score += self._calculate_historical_boost(user_id, poi_metadata)
        
        # Normalize về [-1, 1]
        boost_score = max(-1.0, min(1.0, boost_score))
        
        self.logger.debug(f"🎯 Personalization boost for user {user_id}: {boost_score:.3f}")
        return boost_score
    
    def get_personalization_metrics(self, user_id: str) -> PersonalizationMetrics:
        """
        Lấy metrics về personalization performance
        """
        profile = self.get_or_create_profile(user_id)
        
        # Tính toán metrics từ cached data
        user_feedbacks = self._get_user_feedbacks(user_id)
        user_interactions = self._get_user_interactions(user_id)
        
        # Calculate metrics
        avg_rating = self._calculate_avg_feedback_score(user_feedbacks)
        click_rate = self._calculate_click_through_rate(user_interactions)
        preference_confidence = self._calculate_preference_confidence(profile)
        
        metrics = PersonalizationMetrics(
            user_id=user_id,
            avg_recommendation_score=avg_rating,
            click_through_rate=click_rate,
            feedback_score=avg_rating,
            preference_confidence=preference_confidence,
            profile_completeness=profile.profile_completeness,
            last_updated=datetime.now(),
            total_recommendations=profile.total_searches,
            total_interactions=len(user_interactions),
            total_feedback_events=len(user_feedbacks)
        )
        
        return metrics
    
    # ========================================
    # PRIVATE HELPER METHODS
    # ========================================
    
    def _calculate_completeness(self, profile: UserProfile) -> float:
        """Tính toán profile completeness score"""
        score = 0.0
        total_fields = 0
        
        # Demographics (weight: 0.2)
        if profile.demographics:
            demo_score = 0
            demo_fields = ['age_range', 'location', 'income_level']
            for field in demo_fields:
                if getattr(profile.demographics, field):
                    demo_score += 1
            score += (demo_score / len(demo_fields)) * 0.2
        
        # Restaurant preferences (weight: 0.8)
        prefs = profile.restaurant_preferences
        pref_score = 0
        
        if prefs.preferred_cuisines:
            pref_score += 0.3
        if prefs.dining_styles:
            pref_score += 0.3
        if prefs.budget_range:
            pref_score += 0.2
        if prefs.group_size_preference:
            pref_score += 0.2
        
        score += pref_score * 0.8
        
        return min(1.0, score)
    
    def _generate_feedback_id(self, feedback: FeedbackAction) -> str:
        """Generate unique feedback ID"""
        data = f"{feedback.user_id}:{feedback.business_id}:{feedback.timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    def _generate_interaction_id(self, interaction: UserInteraction) -> str:
        """Generate unique interaction ID"""
        data = f"{interaction.user_id}:{interaction.business_id}:{interaction.timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    def _process_feedback_to_signal(self, feedback: FeedbackAction) -> LearningSignal:
        """Chuyển đổi feedback thành learning signal"""
        
        # Tính signal strength dựa trên feedback type
        base_strength = abs(self.feedback_weights.get(feedback.feedback_type.value, 0.0))
        
        # Adjust strength based on feedback value (for ratings)
        if feedback.feedback_type.value == "rate" and feedback.feedback_value:
            # Rating 1-5 -> signal strength
            normalized_rating = (feedback.feedback_value - 1) / 4  # 0-1
            base_strength = normalized_rating
        
        # Determine signal type
        weight = self.feedback_weights.get(feedback.feedback_type.value, 0.0)
        if feedback.feedback_type.value == "rate" and feedback.feedback_value:
            signal_type = "positive" if feedback.feedback_value >= 3.5 else "negative"
        else:
            signal_type = "positive" if weight > 0 else "negative" if weight < 0 else "neutral"
        
        signal = LearningSignal(
            user_id=feedback.user_id,
            business_id=feedback.business_id,
            signal_type=signal_type,
            signal_strength=base_strength,
            source_feedbacks=[self._generate_feedback_id(feedback)]
        )
        
        return signal
    
    def _should_create_implicit_signal(self, interaction: UserInteraction) -> bool:
        """Kiểm tra xem có nên tạo implicit learning signal từ interaction không"""
        # Tạo signal cho các interaction có ý nghĩa
        meaningful_interactions = ["click", "dwell_time", "bookmark"]
        return interaction.interaction_type in meaningful_interactions
    
    def _process_interaction_to_signal(self, interaction: UserInteraction) -> LearningSignal:
        """Chuyển đổi interaction thành learning signal"""
        
        signal_strength = 0.3  # Default weak signal
        signal_type = "neutral"
        
        if interaction.interaction_type == "click":
            signal_strength = 0.4
            signal_type = "positive"
        elif interaction.interaction_type == "dwell_time" and interaction.interaction_value:
            # Dwell time > 30s = positive signal
            if interaction.interaction_value > 30:
                signal_strength = min(0.8, interaction.interaction_value / 120)  # Cap at 2 minutes
                signal_type = "positive"
        
        signal = LearningSignal(
            user_id=interaction.user_id,
            business_id=interaction.business_id,
            signal_type=signal_type,
            signal_strength=signal_strength,
            source_interactions=[self._generate_interaction_id(interaction)]
        )
        
        return signal
    
    def _update_profile_from_signal(self, user_id: str, signal: LearningSignal):
        """Cập nhật user profile dựa trên learning signal"""
        # Simplified learning - trong thực tế sẽ phức tạp hơn
        profile = self.get_or_create_profile(user_id)
        
        # Increment activity counters
        if signal.signal_type == "positive":
            # Có thể infer preferences từ POI metadata
            pass
        
        # Update activity timestamp
        profile.last_active = datetime.now()
        
        self.save_profile(profile)
    
    def _calculate_cuisine_boost(self, profile: UserProfile, poi_metadata: Dict) -> float:
        """Tính boost dựa trên cuisine preferences"""
        if not profile.restaurant_preferences.preferred_cuisines:
            return 0.0
        
        categories = poi_metadata.get('categories', '').lower()
        boost = 0.0
        
        for cuisine in profile.restaurant_preferences.preferred_cuisines:
            if cuisine.value in categories:
                boost += 0.3
        
        return min(0.5, boost)
    
    def _calculate_dining_boost(self, profile: UserProfile, poi_metadata: Dict) -> float:
        """Tính boost dựa trên dining style preferences"""
        if not profile.restaurant_preferences.dining_styles:
            return 0.0
        
        categories = poi_metadata.get('categories', '').lower()
        poi_type = poi_metadata.get('poi_type', '').lower()
        boost = 0.0
        
        for dining_style in profile.restaurant_preferences.dining_styles:
            if dining_style.value in categories or dining_style.value in poi_type:
                boost += 0.2
        
        return min(0.4, boost)
    
    def _calculate_budget_boost(self, profile: UserProfile, poi_metadata: Dict) -> float:
        """Tính boost dựa trên budget preferences"""
        # Simplified - trong thực tế cần pricing data
        return 0.0
    
    def _calculate_historical_boost(self, user_id: str, poi_metadata: Dict) -> float:
        """Tính boost dựa trên lịch sử feedback"""
        # Simplified - tìm similar POIs mà user đã like
        similar_feedbacks = self._find_similar_poi_feedbacks(user_id, poi_metadata)
        
        if not similar_feedbacks:
            return 0.0
        
        positive_signals = sum(1 for f in similar_feedbacks if f.feedback_type.value in ["like", "visit", "bookmark"])
        total_signals = len(similar_feedbacks)
        
        if total_signals == 0:
            return 0.0
        
        positive_ratio = positive_signals / total_signals
        return (positive_ratio - 0.5) * 0.4  # -0.2 to 0.2 range
    
    def _find_similar_poi_feedbacks(self, user_id: str, poi_metadata: Dict) -> List[FeedbackAction]:
        """Tìm feedback của user cho các POI tương tự"""
        # Simplified implementation
        user_feedbacks = self._get_user_feedbacks(user_id)
        
        # Filter by similar categories (simplified)
        poi_categories = set(poi_metadata.get('categories', '').lower().split(', '))
        similar_feedbacks = []
        
        for feedback in user_feedbacks:
            # Would need to lookup POI metadata for comparison
            # For now, return empty list
            pass
        
        return similar_feedbacks
    
    def _get_user_feedbacks(self, user_id: str) -> List[FeedbackAction]:
        """Lấy tất cả feedback của user"""
        feedbacks = []
        for feedback_id, feedback in self.feedback_cache.items():
            if feedback.user_id == user_id:
                feedbacks.append(feedback)
        return feedbacks
    
    def _get_user_interactions(self, user_id: str) -> List[UserInteraction]:
        """Lấy tất cả interactions của user"""
        interactions = []
        for interaction_id, interaction in self.interaction_cache.items():
            if interaction.user_id == user_id:
                interactions.append(interaction)
        return interactions
    
    def _calculate_avg_feedback_score(self, feedbacks: List[FeedbackAction]) -> float:
        """Tính average feedback score"""
        if not feedbacks:
            return 0.0
        
        scores = []
        for feedback in feedbacks:
            if feedback.feedback_value:
                scores.append(feedback.feedback_value)
            else:
                # Convert feedback type to numeric score
                weight = self.feedback_weights.get(feedback.feedback_type.value, 0.0)
                scores.append((weight + 1) * 2.5)  # Convert to 1-5 scale
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_click_through_rate(self, interactions: List[UserInteraction]) -> float:
        """Tính click-through rate"""
        if not interactions:
            return 0.0
        
        clicks = sum(1 for i in interactions if i.interaction_type == "click")
        views = sum(1 for i in interactions if i.interaction_type == "view")
        
        return clicks / max(views, 1)
    
    def _calculate_preference_confidence(self, profile: UserProfile) -> float:
        """Tính confidence trong learned preferences"""
        # Base confidence từ profile completeness
        base_confidence = profile.profile_completeness
        
        # Boost từ activity level
        activity_boost = min(0.3, profile.total_searches / 50.0)  # Up to 0.3 boost for 50+ searches
        
        return min(1.0, base_confidence + activity_boost)

    # ========================================
    # RESTAURANT-SPECIFIC METHODS
    # ========================================
    
    def get_restaurant_profile(self, user_id: str) -> Optional['RestaurantProfile']:
        """
        Lấy restaurant-specific profile cho user
        
        Args:
            user_id: User identifier
            
        Returns:
            RestaurantProfile if exists, None otherwise
        """
        try:
            from modules.domain.restaurant_schemas import RestaurantProfile, calculate_restaurant_profile_completeness
            
            cache_key = f"restaurant_profile:{user_id}"
            cached_profile = self.memory.get(cache_key)
            
            if cached_profile:
                return RestaurantProfile(**cached_profile)
            
            # Create new restaurant profile from general profile
            general_profile = self.get_or_create_profile(user_id)
            restaurant_profile = self._convert_to_restaurant_profile(general_profile)
            
            # Save to cache
            self.memory.set(cache_key, restaurant_profile.dict(), ttl_seconds=3600)  # 1 hour cache
            
            return restaurant_profile
            
        except Exception as e:
            self.logger.error(f"❌ Error getting restaurant profile for {user_id}: {e}")
            return None
    
    def update_restaurant_profile(self, 
                                user_id: str, 
                                restaurant_preferences: Optional['RestaurantPreferences'] = None,
                                city_history: Optional[Dict[str, 'CityRestaurantHistory']] = None) -> Optional['RestaurantProfile']:
        """
        Update restaurant-specific profile
        
        Args:
            user_id: User identifier
            restaurant_preferences: Restaurant preferences to update
            city_history: City-specific restaurant history
            
        Returns:
            Updated RestaurantProfile
        """
        try:
            from modules.domain.restaurant_schemas import RestaurantProfile, calculate_restaurant_profile_completeness
            
            # Get existing profile
            profile = self.get_restaurant_profile(user_id)
            if not profile:
                profile = RestaurantProfile(user_id=user_id)
            
            # Update preferences
            if restaurant_preferences:
                profile.preferences = restaurant_preferences
            
            # Update city history
            if city_history:
                profile.city_history.update(city_history)
            
            # Update metadata
            profile.last_updated = datetime.now()
            profile.profile_completeness = calculate_restaurant_profile_completeness(profile)
            
            # Save updated profile
            cache_key = f"restaurant_profile:{user_id}"
            self.memory.set(cache_key, profile.dict(), ttl_seconds=3600)
            
            self.logger.info(f"✅ Updated restaurant profile for {user_id} (completeness: {profile.profile_completeness:.2f})")
            return profile
            
        except Exception as e:
            self.logger.error(f"❌ Error updating restaurant profile for {user_id}: {e}")
            return None
    
    def get_restaurant_personalization_boost(self, 
                                           user_id: str, 
                                           restaurant_metadata: Dict,
                                           city: str) -> float:
        """
        Tính personalization boost cho restaurant recommendation
        
        Args:
            user_id: User identifier
            restaurant_metadata: Restaurant metadata từ search results
            city: Current city context
            
        Returns:
            Boost score từ -1.0 đến 1.0
        """
        try:
            restaurant_profile = self.get_restaurant_profile(user_id)
            if not restaurant_profile:
                return 0.0
            
            total_boost = 0.0
            
            # 1. Cuisine preference boost
            cuisine_boost = self._calculate_restaurant_cuisine_boost(restaurant_profile, restaurant_metadata)
            total_boost += cuisine_boost * 0.4  # 40% weight
            
            # 2. Price level boost
            price_boost = self._calculate_restaurant_price_boost(restaurant_profile, restaurant_metadata)
            total_boost += price_boost * 0.3  # 30% weight
            
            # 3. City-specific history boost
            city_boost = self._calculate_city_history_boost(restaurant_profile, restaurant_metadata, city)
            total_boost += city_boost * 0.2  # 20% weight
            
            # 4. Dining style boost
            style_boost = self._calculate_dining_style_boost(restaurant_profile, restaurant_metadata)
            total_boost += style_boost * 0.1  # 10% weight
            
            # Clamp to [-1, 1] range
            return max(-1.0, min(1.0, total_boost))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating restaurant boost for {user_id}: {e}")
            return 0.0
    
    def analyze_restaurant_preferences(self, user_id: str, city: str) -> Dict:
        """
        Analyze user's restaurant preferences trong specific city
        
        Args:
            user_id: User identifier
            city: City to analyze
            
        Returns:
            Dictionary with preference analysis
        """
        try:
            restaurant_profile = self.get_restaurant_profile(user_id)
            if not restaurant_profile:
                return {}
            
            city_lower = city.lower()
            city_history = restaurant_profile.city_history.get(city_lower)
            
            analysis = {
                'city': city,
                'total_visits': city_history.total_visits if city_history else 0,
                'favorite_cuisines': city_history.favorite_cuisines if city_history else [],
                'average_price_level': city_history.average_price_level if city_history else None,
                'cuisine_preferences': restaurant_profile.preferences.cuisine_types,
                'price_preferences': restaurant_profile.preferences.price_levels,
                'dining_style_preferences': restaurant_profile.preferences.dining_styles,
                'novelty_seeking': restaurant_profile.learning_data.novelty_seeking,
                'price_sensitivity': restaurant_profile.learning_data.price_sensitivity
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing restaurant preferences for {user_id} in {city}: {e}")
            return {}
    
    def record_restaurant_visit(self, 
                              user_id: str, 
                              business_id: str, 
                              city: str,
                              liked: bool = True) -> bool:
        """
        Record restaurant visit to update city history
        
        Args:
            user_id: User identifier
            business_id: Restaurant business ID
            city: City where restaurant is located
            liked: Whether user liked the restaurant
            
        Returns:
            True if successfully recorded
        """
        try:
            from modules.domain.restaurant_schemas import CityRestaurantHistory
            
            restaurant_profile = self.get_restaurant_profile(user_id)
            if not restaurant_profile:
                return False
            
            city_lower = city.lower()
            
            # Get or create city history
            if city_lower not in restaurant_profile.city_history:
                restaurant_profile.city_history[city_lower] = CityRestaurantHistory(
                    city=city,
                    first_visit_date=datetime.now()
                )
            
            city_history = restaurant_profile.city_history[city_lower]
            
            # Update visit history
            if business_id not in city_history.visited_restaurants:
                city_history.visited_restaurants.append(business_id)
                city_history.total_visits += 1
            
            if liked and business_id not in city_history.liked_restaurants:
                city_history.liked_restaurants.append(business_id)
            elif not liked and business_id not in city_history.disliked_restaurants:
                city_history.disliked_restaurants.append(business_id)
            
            # Update timestamps
            city_history.last_visit_date = datetime.now()
            
            # Update learning data
            restaurant_profile.learning_data.total_restaurant_visits += 1
            if city not in restaurant_profile.learning_data.cities_explored:
                restaurant_profile.learning_data.cities_explored.append(city)
            
            # Save updated profile
            self.update_restaurant_profile(user_id, city_history={city_lower: city_history})
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error recording restaurant visit for {user_id}: {e}")
            return False
    
    def _convert_to_restaurant_profile(self, general_profile: UserProfile) -> 'RestaurantProfile':
        """Convert general profile to restaurant-specific profile"""
        from modules.domain.restaurant_schemas import (
            RestaurantProfile, RestaurantPreferences, RestaurantLearningData,
            CuisineType, PriceLevel, DiningStyle
        )
        
        # Convert cuisine preferences
        cuisine_types = []
        for cuisine in general_profile.restaurant_preferences.preferred_cuisines:
            if cuisine.value in [c.value for c in CuisineType]:
                cuisine_types.append(CuisineType(cuisine.value))
        
        # Use dining styles for price estimation
        price_levels = []
        for dining_style in general_profile.restaurant_preferences.dining_styles:
            if dining_style.value == "fine_dining":
                price_levels.append(PriceLevel.LUXURY)
            elif dining_style.value in ["fast_food", "takeaway"]:
                price_levels.append(PriceLevel.BUDGET)
        
        if not price_levels:  # Default to moderate
            price_levels.append(PriceLevel.MODERATE)
        
        # Create restaurant preferences
        restaurant_preferences = RestaurantPreferences(
            cuisine_types=cuisine_types,
            price_levels=price_levels,
            group_size_preference=general_profile.restaurant_preferences.group_size_preference or 2
        )
        
        # Create learning data
        learning_data = RestaurantLearningData(
            total_restaurant_visits=general_profile.total_restaurant_visits,
            cities_explored=general_profile.favorite_cities[:5]  # Top 5 cities
        )
        
        return RestaurantProfile(
            user_id=general_profile.user_id,
            preferences=restaurant_preferences,
            learning_data=learning_data,
            created_at=general_profile.created_at,
            last_updated=general_profile.last_updated
        )
    
    def _calculate_restaurant_cuisine_boost(self, profile: 'RestaurantProfile', restaurant_metadata: Dict) -> float:
        """Calculate boost dựa trên cuisine preferences"""
        if not profile.preferences.cuisine_types:
            return 0.0
        
        from modules.domain.restaurant_schemas import extract_cuisine_types_from_categories
        
        categories = restaurant_metadata.get('categories', '')
        restaurant_cuisines = extract_cuisine_types_from_categories(categories)
        
        boost = 0.0
        for cuisine in profile.preferences.cuisine_types:
            if cuisine in restaurant_cuisines:
                boost += 0.3  # Each matching cuisine adds 0.3
        
        # Check learning data for evolved preferences
        cuisine_scores = profile.learning_data.cuisine_preference_scores
        for cuisine in restaurant_cuisines:
            cuisine_score = cuisine_scores.get(cuisine.value, 0.0)
            boost += cuisine_score * 0.2  # Learned preferences contribute less
        
        return min(1.0, boost)
    
    def _calculate_restaurant_price_boost(self, profile: 'RestaurantProfile', restaurant_metadata: Dict) -> float:
        """Calculate boost dựa trên price preferences"""
        if not profile.preferences.price_levels:
            return 0.0
        
        # Estimate restaurant price level từ rating
        stars = restaurant_metadata.get('stars', 0)
        if stars <= 2.5:
            restaurant_price = 'budget'
        elif stars <= 3.5:
            restaurant_price = 'moderate'
        elif stars <= 4.5:
            restaurant_price = 'expensive'
        else:
            restaurant_price = 'luxury'
        
        # Check if matches user preferences
        for price_level in profile.preferences.price_levels:
            if price_level.value == restaurant_price:
                # Consider price sensitivity
                sensitivity = profile.learning_data.price_sensitivity
                return 0.3 * (1.0 - sensitivity * 0.5)  # Less boost if very price sensitive
        
        return 0.0
    
    def _calculate_city_history_boost(self, profile: 'RestaurantProfile', restaurant_metadata: Dict, city: str) -> float:
        """Calculate boost dựa trên city-specific history"""
        city_lower = city.lower()
        city_history = profile.city_history.get(city_lower)
        
        if not city_history:
            return 0.0  # No history in this city
        
        business_id = restaurant_metadata.get('business_id', '')
        
        # Negative boost if user disliked this restaurant before
        if business_id in city_history.disliked_restaurants:
            return -0.5
        
        # Positive boost if user liked this restaurant before
        if business_id in city_history.liked_restaurants:
            return 0.4
        
        # Small boost for restaurants in preferred categories
        categories = restaurant_metadata.get('categories', '').lower()
        for fav_cuisine in city_history.favorite_cuisines:
            if fav_cuisine.lower() in categories:
                return 0.2
        
        return 0.0
    
    def _calculate_dining_style_boost(self, profile: 'RestaurantProfile', restaurant_metadata: Dict) -> float:
        """Calculate boost dựa trên dining style preferences"""
        if not profile.preferences.dining_styles:
            return 0.0
        
        categories = restaurant_metadata.get('categories', '').lower()
        
        # Simple mapping from categories to dining styles
        for dining_style in profile.preferences.dining_styles:
            if dining_style.value in categories:
                return 0.2
        
        return 0.0


def demo_user_profile_service():
    """Demo User Profile Service"""
    print("👤 DEMO USER PROFILE SERVICE")
    print("=" * 50)
    
    # Mock memory store
    from modules.memory.short_term import SessionStore
    from shared.settings import Settings
    
    settings = Settings()
    memory = SessionStore(settings)
    service = UserProfileService(memory)
    
    # Test user
    user_id = "test_user_001"
    
    # 1. Create/get profile
    print(f"\n1️⃣ Creating profile for user: {user_id}")
    profile = service.get_or_create_profile(user_id)
    print(f"   Profile completeness: {profile.profile_completeness:.2f}")
    
    # 2. Update preferences
    print(f"\n2️⃣ Updating restaurant preferences...")
    from modules.domain.schemas import RestaurantPreferences, DiningStyle, CuisinePreference
    
    preferences = RestaurantPreferences(
        dining_styles=[DiningStyle.CASUAL, DiningStyle.FAMILY_STYLE],
        preferred_cuisines=[CuisinePreference.VIETNAMESE, CuisinePreference.STREET_FOOD],
        budget_range={"min": 20, "max": 100},
        group_size_preference=4
    )
    
    updated_profile = service.update_profile(user_id, restaurant_preferences=preferences)
    print(f"   Updated completeness: {updated_profile.profile_completeness:.2f}")
    
    # 3. Record feedback
    print(f"\n3️⃣ Recording user feedback...")
    from modules.domain.schemas import FeedbackAction, FeedbackType
    
    feedback = FeedbackAction(
        user_id=user_id,
        business_id="vietnamese_restaurant_001",
        feedback_type=FeedbackType.LIKE,
        search_query="Vietnamese restaurant",
        destination="Ho Chi Minh City"
    )
    
    signal = service.record_feedback(feedback)
    print(f"   Created learning signal: {signal.signal_type} (strength: {signal.signal_strength:.2f})")
    
    # 4. Test personalization boost
    print(f"\n4️⃣ Testing personalization boost...")
    poi_metadata = {
        "name": "Phở Hà Nội",
        "categories": "Vietnamese, Restaurants, Noodles",
        "poi_type": "restaurant",
        "stars": 4.2
    }
    
    boost = service.get_personalized_boost(user_id, poi_metadata)
    print(f"   Personalization boost: {boost:.3f}")
    
    # 5. Get metrics
    print(f"\n5️⃣ Personalization metrics...")
    metrics = service.get_personalization_metrics(user_id)
    print(f"   Profile confidence: {metrics.preference_confidence:.2f}")
    print(f"   Total interactions: {metrics.total_interactions}")
    print(f"   Total feedback events: {metrics.total_feedback_events}")
    
    print(f"\n✅ User Profile Service demo completed!")

if __name__ == "__main__":
    demo_user_profile_service() 