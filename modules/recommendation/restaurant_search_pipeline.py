"""
Restaurant Search Pipeline
Specialized pipeline cho restaurant recommendations với city filtering và personalization
"""

import logging
from typing import List, Dict, Optional
import time
from datetime import datetime

from .optimized_local_lora_service import OptimizedLocalLoRAService
from .city_filter_service import CityFilterService
from .user_profile_service import UserProfileService
from .feedback_learning import FeedbackLearningService
from modules.domain.restaurant_schemas import (
    RestaurantSearchRequest, RestaurantRecommendation, RestaurantSearchResponse,
    CuisineType, PriceLevel, DiningStyle, extract_cuisine_types_from_categories
)

class RestaurantSearchPipeline:
    """
    Specialized pipeline cho restaurant search với:
    1. City-based filtering
    2. Semantic search với BGE-M3
    3. Restaurant-specific personalization
    4. Learning-based improvements
    """
    
    def __init__(self, 
                 user_profile_service: Optional[UserProfileService] = None,
                 feedback_learning_service: Optional[FeedbackLearningService] = None):
        """
        Initialize restaurant search pipeline
        
        Args:
            user_profile_service: User profile service for personalization
            feedback_learning_service: Feedback learning service
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize core services
        self.embedding_service = OptimizedLocalLoRAService()
        self.city_filter_service = CityFilterService()
        
        # Personalization services (optional)
        self.user_profile_service = user_profile_service
        self.feedback_learning_service = feedback_learning_service
        
        self.logger.info("✅ Restaurant Search Pipeline initialized")
    
    def search_restaurants(self, request: RestaurantSearchRequest) -> RestaurantSearchResponse:
        """
        Main restaurant search method
        
        Args:
            request: RestaurantSearchRequest with search parameters
            
        Returns:
            RestaurantSearchResponse with ranked restaurant recommendations
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting restaurant search: '{request.user_query}' in {request.city}")
            
            # 1. Validate city
            if not self.city_filter_service.is_city_supported(request.city):
                return self._create_error_response(
                    request, 
                    f"City '{request.city}' is not supported",
                    time.time() - start_time
                )
            
            # 2. Get semantic search results
            semantic_results = self._get_semantic_candidates(request)
            
            if not semantic_results:
                return self._create_empty_response(request, time.time() - start_time)
            
            # 3. Apply city filtering
            city_filtered_results = self._apply_city_filtering(semantic_results, request.city)
            
            # 4. Apply additional filters
            filtered_results = self._apply_restaurant_filters(city_filtered_results, request)
            
            # 5. Apply personalization (if user_id provided)
            personalized_results = self._apply_personalization(filtered_results, request)
            
            # 6. Convert to RestaurantRecommendation objects
            recommendations = self._convert_to_recommendations(personalized_results, request)
            
            # 7. Final ranking and limiting
            final_recommendations = recommendations[:request.num_results]
            
            search_time = time.time() - start_time
            
            # 8. Create response
            response = RestaurantSearchResponse(
                search_request=request,
                restaurants=final_recommendations,
                total_found=len(semantic_results),
                search_time=search_time,
                personalization_applied=bool(request.user_id and request.use_personalization),
                city_restaurant_stats=self.city_filter_service.get_city_stats(request.city),
                search_suggestions=self._generate_search_suggestions(request)
            )
            
            self.logger.info(f"Restaurant search completed: {len(final_recommendations)} results in {search_time:.3f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"Restaurant search failed: {e}")
            return self._create_error_response(request, str(e), time.time() - start_time)
    
    def _get_semantic_candidates(self, request: RestaurantSearchRequest) -> List[Dict]:
        """Get initial candidates từ BGE-M3 semantic search"""
        try:
            # Enhance query với context
            enhanced_query = self._enhance_search_query(request)
            
            # Perform semantic search
            search_results = self.embedding_service.search_similar(
                query=enhanced_query,
                k=min(request.num_results * 5, 100),  # Get more candidates than needed
                return_embeddings=False
            )
            
            return search_results.get('poi_results', [])
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def _enhance_search_query(self, request: RestaurantSearchRequest) -> str:
        """Enhance user query với context information"""
        enhanced_parts = [request.user_query]
        
        # Add meal type context
        if request.meal_type:
            enhanced_parts.append(f"{request.meal_type} restaurant")
        
        # Add cuisine filter context
        if request.cuisine_filter:
            cuisine_names = [cuisine.value for cuisine in request.cuisine_filter]
            enhanced_parts.append(" ".join(cuisine_names))
        
        # Add occasion context
        if request.occasion:
            enhanced_parts.append(f"{request.occasion} dining")
        
        return " ".join(enhanced_parts)
    
    def _apply_city_filtering(self, results: List[Dict], city: str) -> List[Dict]:
        """Apply city-based filtering"""
        return self.city_filter_service.filter_faiss_results_by_city(results, city)
    
    def _apply_restaurant_filters(self, results: List[Dict], request: RestaurantSearchRequest) -> List[Dict]:
        """Apply restaurant-specific filters"""
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # Extract restaurant info
            categories = metadata.get('categories', '').lower()
            stars = metadata.get('stars', 0)
            
            # Price level filtering
            if request.price_filter:
                restaurant_price_level = self._estimate_price_level(stars)
                if restaurant_price_level not in request.price_filter:
                    continue
            
            # Cuisine type filtering
            if request.cuisine_filter:
                restaurant_cuisines = extract_cuisine_types_from_categories(categories)
                if not any(cuisine in request.cuisine_filter for cuisine in restaurant_cuisines):
                    continue
            
            # Dining style filtering
            if request.dining_style_filter:
                restaurant_dining_style = self._estimate_dining_style(categories)
                if restaurant_dining_style not in request.dining_style_filter:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _apply_personalization(self, results: List[Dict], request: RestaurantSearchRequest) -> List[Dict]:
        """Apply personalization nếu có user_id"""
        if not request.user_id or not request.use_personalization:
            return results
        
        if not self.user_profile_service:
            self.logger.warning("User profile service not available for personalization")
            return results
        
        try:
            personalized_results = []
            
            for result in results:
                # Get base score
                base_score = result.get('similarity_score', 0.5)
                
                # Apply user preference boost
                restaurant_metadata = result.get('metadata', {})
                personalization_boost = self.user_profile_service.get_restaurant_personalization_boost(
                    user_id=request.user_id,
                    restaurant_metadata=restaurant_metadata,
                    city=request.city
                )
                
                # Apply learning boost if available
                learning_boost = 0.0
                if self.feedback_learning_service:
                    business_id = result.get('business_id', '')
                    learning_boost = self.feedback_learning_service.get_restaurant_collaborative_boost(
                        user_id=request.user_id,
                        restaurant_id=business_id,
                        city=request.city
                    )
                
                # Calculate final personalized score
                personalized_score = base_score + (personalization_boost * 0.3) + (learning_boost * 0.2)
                
                # Update result
                result['personalization_boost'] = personalization_boost
                result['learning_boost'] = learning_boost
                result['personalized_score'] = personalized_score
                
                personalized_results.append(result)
            
            # Sort by personalized score
            personalized_results.sort(key=lambda x: x.get('personalized_score', x.get('similarity_score', 0)), reverse=True)
            
            return personalized_results
            
        except Exception as e:
            self.logger.error(f"Personalization failed: {e}")
            return results
    
    def _convert_to_recommendations(self, results: List[Dict], request: RestaurantSearchRequest) -> List[RestaurantRecommendation]:
        """Convert search results to RestaurantRecommendation objects"""
        recommendations = []
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            
            # Extract restaurant details
            cuisine_types = extract_cuisine_types_from_categories(metadata.get('categories', ''))
            price_level = self._estimate_price_level(metadata.get('stars', 0))
            dining_style = self._estimate_dining_style(metadata.get('categories', ''))
            
            # Calculate scores
            similarity_score = result.get('similarity_score', 0.0)
            personalization_score = result.get('personalization_boost', 0.0)
            final_score = result.get('personalized_score', similarity_score)
            
            # Generate recommendation reasons
            why_recommended = self._generate_recommendation_reasons(result, request)
            user_match_factors = self._generate_match_factors(result, request)
            
            recommendation = RestaurantRecommendation(
                business_id=result.get('business_id', ''),
                name=metadata.get('name', ''),
                city=metadata.get('city', request.city),
                address=metadata.get('address', ''),
                cuisine_types=[cuisine.value for cuisine in cuisine_types],
                price_level=price_level,
                dining_style=dining_style,
                stars=metadata.get('stars', 0.0),
                review_count=metadata.get('review_count', 0),
                similarity_score=similarity_score,
                personalization_score=personalization_score,
                final_score=final_score,
                rank=i,
                why_recommended=why_recommended,
                user_match_factors=user_match_factors,
                recommended_for=request.meal_type,
                recommendation_confidence=min(1.0, final_score),
                is_new_discovery=self._is_new_discovery(result, request)
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _estimate_price_level(self, stars: float) -> PriceLevel:
        """Estimate price level từ rating"""
        if stars <= 2.5:
            return PriceLevel.BUDGET
        elif stars <= 3.5:
            return PriceLevel.MODERATE
        elif stars <= 4.5:
            return PriceLevel.EXPENSIVE
        else:
            return PriceLevel.LUXURY
    
    def _estimate_dining_style(self, categories: str) -> DiningStyle:
        """Estimate dining style từ categories"""
        categories_lower = categories.lower()
        
        if 'fine dining' in categories_lower:
            return DiningStyle.FINE_DINING
        elif 'street food' in categories_lower or 'food truck' in categories_lower:
            return DiningStyle.STREET_FOOD
        elif 'fast food' in categories_lower:
            return DiningStyle.FAST_FOOD
        elif 'coffee' in categories_lower or 'cafe' in categories_lower:
            return DiningStyle.COFFEE_SHOP
        elif 'bar' in categories_lower or 'pub' in categories_lower:
            return DiningStyle.BAR_PUB
        elif 'buffet' in categories_lower:
            return DiningStyle.BUFFET
        else:
            return DiningStyle.CASUAL
    
    def _generate_recommendation_reasons(self, result: Dict, request: RestaurantSearchRequest) -> List[str]:
        """Generate reasons why this restaurant was recommended"""
        reasons = []
        
        metadata = result.get('metadata', {})
        
        # High rating reason
        stars = metadata.get('stars', 0)
        if stars >= 4.5:
            reasons.append("Highly rated restaurant")
        elif stars >= 4.0:
            reasons.append("Well-rated restaurant")
        
        # Popular reason
        review_count = metadata.get('review_count', 0)
        if review_count >= 100:
            reasons.append("Popular with many reviews")
        
        # Query matching reason
        if result.get('similarity_score', 0) >= 0.8:
            reasons.append("Excellent match for your search")
        elif result.get('similarity_score', 0) >= 0.6:
            reasons.append("Good match for your search")
        
        # Personalization reason
        if result.get('personalization_boost', 0) > 0:
            reasons.append("Matches your preferences")
        
        # Cuisine matching
        categories = metadata.get('categories', '').lower()
        if request.cuisine_filter:
            for cuisine in request.cuisine_filter:
                if cuisine.value in categories:
                    reasons.append(f"Serves {cuisine.value} cuisine")
                    break
        
        return reasons
    
    def _generate_match_factors(self, result: Dict, request: RestaurantSearchRequest) -> List[str]:
        """Generate specific matching factors"""
        factors = []
        
        metadata = result.get('metadata', {})
        categories = metadata.get('categories', '').lower()
        
        # Cuisine matching
        if request.cuisine_filter:
            for cuisine in request.cuisine_filter:
                if cuisine.value in categories:
                    factors.append(f"cuisine_{cuisine.value}")
        
        # Price matching
        if request.price_filter:
            estimated_price = self._estimate_price_level(metadata.get('stars', 0))
            if estimated_price in request.price_filter:
                factors.append(f"price_{estimated_price.value}")
        
        # Context matching
        if request.meal_type:
            factors.append(f"meal_{request.meal_type}")
        
        if request.occasion:
            factors.append(f"occasion_{request.occasion}")
        
        return factors
    
    def _is_new_discovery(self, result: Dict, request: RestaurantSearchRequest) -> bool:
        """Check if this is a new discovery for the user"""
        if not request.user_id or not self.user_profile_service:
            return False
        
        try:
            # Get user's restaurant history for this city
            profile = self.user_profile_service.get_or_create_profile(request.user_id)
            city_history = profile.city_history.get(request.city.lower())
            
            if not city_history:
                return True  # First time in this city
            
            business_id = result.get('business_id', '')
            return business_id not in city_history.visited_restaurants
            
        except:
            return False
    
    def _generate_search_suggestions(self, request: RestaurantSearchRequest) -> List[str]:
        """Generate alternative search suggestions"""
        suggestions = []
        
        # Get popular cuisines in this city
        city_cuisines = self.city_filter_service.get_city_cuisine_preferences(request.city)
        top_cuisines = sorted(city_cuisines.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for cuisine, _ in top_cuisines:
            if cuisine not in request.user_query.lower():
                suggestions.append(f"{cuisine} restaurants in {request.city}")
        
        # Add meal type suggestions
        if not request.meal_type:
            meal_types = ["breakfast", "lunch", "dinner"]
            for meal in meal_types:
                suggestions.append(f"{request.user_query} for {meal}")
        
        return suggestions[:3]
    
    def _create_error_response(self, request: RestaurantSearchRequest, error_message: str, search_time: float) -> RestaurantSearchResponse:
        """Create error response"""
        return RestaurantSearchResponse(
            search_request=request,
            restaurants=[],
            total_found=0,
            search_time=search_time,
            success=False,
            message=error_message
        )
    
    def _create_empty_response(self, request: RestaurantSearchRequest, search_time: float) -> RestaurantSearchResponse:
        """Create empty response when no results found"""
        return RestaurantSearchResponse(
            search_request=request,
            restaurants=[],
            total_found=0,
            search_time=search_time,
            message="No restaurants found matching your criteria",
            search_suggestions=self._generate_search_suggestions(request)
        )

def demo_restaurant_search_pipeline():
    """Demo RestaurantSearchPipeline functionality"""
    print("DEMO RESTAURANT SEARCH PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RestaurantSearchPipeline()
    
    # Test searches
    test_requests = [
        RestaurantSearchRequest(
            city="Ho Chi Minh City",
            user_query="phở ngon",
            num_results=5
        ),
        RestaurantSearchRequest(
            city="Hanoi",
            user_query="fine dining restaurant",
            cuisine_filter=[CuisineType.VIETNAMESE, CuisineType.FRENCH],
            price_filter=[PriceLevel.EXPENSIVE, PriceLevel.LUXURY],
            num_results=3
        ),
        RestaurantSearchRequest(
            city="Da Nang",
            user_query="seafood restaurant near beach",
            meal_type="dinner",
            occasion="date",
            num_results=4
        )
    ]
    
    for i, request in enumerate(test_requests, 1):
        print(f"\nTest Search {i}:")
        print(f"   City: {request.city}")
        print(f"   Query: '{request.user_query}'")
        
        # Perform search
        response = pipeline.search_restaurants(request)
        
        print(f"   Results: {len(response.restaurants)} restaurants in {response.search_time:.3f}s")
        print(f"   Total found: {response.total_found}")
        
        # Show top results
        for j, restaurant in enumerate(response.restaurants[:3], 1):
            print(f"      {j}. {restaurant.name} - {restaurant.stars}⭐")
            print(f"         Cuisine: {', '.join(restaurant.cuisine_types)}")
            print(f"         Score: {restaurant.final_score:.3f}")
            if restaurant.why_recommended:
                print(f"         Why: {restaurant.why_recommended[0]}")
    
    print(f"\nRestaurant Search Pipeline demo completed!")

if __name__ == "__main__":
    # Handle import khi chạy trực tiếp
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Import lại với absolute path
    from modules.recommendation.optimized_local_lora_service import OptimizedLocalLoRAService
    from modules.recommendation.city_filter_service import CityFilterService
    from modules.domain.restaurant_schemas import (
        RestaurantSearchRequest, CuisineType, PriceLevel
    )
    
    demo_restaurant_search_pipeline() 