"""
FastAPI Router cho Restaurant Recommendation Endpoints
Restaurant-focused API với city filtering và personalization
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Optional
import logging
from datetime import datetime

from modules.domain.restaurant_schemas import (
    RestaurantSearchRequest, RestaurantSearchResponse, RestaurantRecommendation,
    RestaurantFeedbackAction, RestaurantInteraction, RestaurantProfile,
    CuisineType, PriceLevel, DiningStyle, SupportedCity
)

from .restaurant_search_pipeline import RestaurantSearchPipeline
from .city_filter_service import CityFilterService
from .user_profile_service import UserProfileService
from .feedback_learning import FeedbackLearningService

from dependency_injector.wiring import inject, Provide
from shared.kernel import Container

# Initialize router
router = APIRouter(tags=["restaurant"], prefix="/restaurant")

# ========================================
# RESTAURANT SEARCH ENDPOINTS
# ========================================

@router.post("/search", response_model=RestaurantSearchResponse)
@inject
async def search_restaurants(
    request: RestaurantSearchRequest,
    user_profile_service = Depends(Provide[Container.user_profile_service]),
    feedback_learning_service = Depends(Provide[Container.feedback_learning_service])
):
    """
    Main restaurant search endpoint với city filtering và personalization
    
    Request Body:
    ```json
    {
        "city": "Ho Chi Minh City",
        "user_query": "nhà hàng phở ngon",
        "user_id": "user123",
        "price_filter": ["budget", "moderate"],
        "cuisine_filter": ["vietnamese"],
        "meal_type": "lunch",
        "num_results": 10,
        "use_personalization": true
    }
    ```
    """
    try:
        # Initialize pipeline với personalization services
        pipeline = RestaurantSearchPipeline(
            user_profile_service=user_profile_service,
            feedback_learning_service=feedback_learning_service
        )
        
        # Perform search
        response = pipeline.search_restaurants(request)
        
        return response
        
    except Exception as e:
        logging.error(f"Restaurant search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Restaurant search failed: {str(e)}")

@router.get("/search")
async def search_restaurants_get(
    city: str = Query(..., description="City to search in"),
    query: str = Query(..., description="Restaurant search query"),
    user_id: Optional[str] = Query(None, description="User ID for personalization"),
    price_filter: Optional[List[str]] = Query(None, description="Price level filters"),
    cuisine_filter: Optional[List[str]] = Query(None, description="Cuisine type filters"),
    meal_type: Optional[str] = Query(None, description="Meal type (breakfast, lunch, dinner)"),
    num_results: int = Query(10, description="Number of results"),
    use_personalization: bool = Query(True, description="Enable personalization")
):
    """
    GET version of restaurant search (for easy testing)
    """
    # Convert query parameters to request object
    request = RestaurantSearchRequest(
        city=city,
        user_query=query,
        user_id=user_id,
        price_filter=[PriceLevel(p) for p in price_filter] if price_filter else None,
        cuisine_filter=[CuisineType(c) for c in cuisine_filter] if cuisine_filter else None,
        meal_type=meal_type,
        num_results=num_results,
        use_personalization=use_personalization
    )
    
    return await search_restaurants(request)

# ========================================
# CITY MANAGEMENT ENDPOINTS
# ========================================

@router.get("/cities", response_model=List[SupportedCity])
async def get_supported_cities():
    """
    Lấy danh sách cities được support cho restaurant search
    """
    try:
        city_service = CityFilterService()
        cities = city_service.get_supported_cities()
        
        return cities
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get supported cities: {str(e)}")

@router.get("/cities/search")
async def search_cities(
    query: str = Query(..., description="City search query"),
    limit: int = Query(5, description="Maximum results")
):
    """
    Search cities by name với fuzzy matching
    """
    try:
        city_service = CityFilterService()
        results = city_service.search_cities(query, limit)
        
        return {
            "query": query,
            "cities": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"City search failed: {str(e)}")

@router.get("/cities/{city}/stats")
async def get_city_restaurant_stats(city: str):
    """
    Lấy restaurant statistics cho specific city
    """
    try:
        city_service = CityFilterService()
        
        if not city_service.is_city_supported(city):
            raise HTTPException(status_code=404, detail=f"City '{city}' is not supported")
        
        stats = city_service.get_city_stats(city)
        cuisine_preferences = city_service.get_city_cuisine_preferences(city)
        
        return {
            "city": city,
            "stats": stats,
            "cuisine_preferences": cuisine_preferences,
            "is_supported": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get city stats: {str(e)}")

# ========================================
# USER PROFILE ENDPOINTS
# ========================================

@router.get("/profile/{user_id}", response_model=Optional[RestaurantProfile])
@inject
async def get_user_restaurant_profile(
    user_id: str,
    user_profile_service = Depends(Provide[Container.user_profile_service])
):
    """
    Lấy restaurant profile của user
    """
    try:
        profile = user_profile_service.get_restaurant_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail=f"Restaurant profile not found for user {user_id}")
        
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get restaurant profile: {str(e)}")

@router.post("/profile/{user_id}/update")
@inject
async def update_user_restaurant_profile(
    user_id: str,
    restaurant_preferences: Dict,
    user_profile_service = Depends(Provide[Container.user_profile_service])
):
    """
    Cập nhật restaurant preferences của user
    """
    try:
        from modules.domain.restaurant_schemas import RestaurantPreferences
        
        # Convert dict to RestaurantPreferences
        preferences = RestaurantPreferences(**restaurant_preferences)
        
        updated_profile = user_profile_service.update_restaurant_profile(
            user_id=user_id,
            restaurant_preferences=preferences
        )
        
        if not updated_profile:
            raise HTTPException(status_code=400, detail="Failed to update restaurant profile")
        
        return {
            "user_id": user_id,
            "profile_completeness": updated_profile.profile_completeness,
            "last_updated": updated_profile.last_updated,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update restaurant profile: {str(e)}")

@router.get("/profile/{user_id}/analysis/{city}")
@inject
async def analyze_user_restaurant_preferences(
    user_id: str,
    city: str,
    user_profile_service = Depends(Provide[Container.user_profile_service])
):
    """
    Analyze user's restaurant preferences trong specific city
    """
    try:
        analysis = user_profile_service.analyze_restaurant_preferences(user_id, city)
        
        return {
            "user_id": user_id,
            "city": city,
            "analysis": analysis,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze preferences: {str(e)}")

# ========================================
# FEEDBACK & LEARNING ENDPOINTS
# ========================================

@router.post("/feedback")
@inject
async def submit_restaurant_feedback(
    feedback: RestaurantFeedbackAction,
    user_profile_service = Depends(Provide[Container.user_profile_service])
):
    """
    Submit feedback cho restaurant
    
    Request Body:
    ```json
    {
        "user_id": "user123",
        "business_id": "restaurant456",
        "feedback_type": "like",
        "rating": 4.5,
        "city": "Ho Chi Minh City",
        "meal_type": "lunch",
        "liked_aspects": ["food", "service"],
        "search_query": "phở ngon"
    }
    ```
    """
    try:
        from modules.domain.schemas import FeedbackAction, FeedbackType
        
        # Convert restaurant feedback to general feedback
        general_feedback = FeedbackAction(
            user_id=feedback.user_id,
            business_id=feedback.business_id,
            feedback_type=FeedbackType(feedback.feedback_type),
            feedback_value=feedback.rating,
            search_query=feedback.search_query,
            destination=feedback.city
        )
        
        # Record feedback
        signal = user_profile_service.record_feedback(general_feedback)
        
        # Record restaurant visit
        liked = feedback.feedback_type in ["like", "visit", "rate"] and (not feedback.rating or feedback.rating >= 3.5)
        user_profile_service.record_restaurant_visit(
            user_id=feedback.user_id,
            business_id=feedback.business_id,
            city=feedback.city,
            liked=liked
        )
        
        return {
            "feedback_recorded": True,
            "signal_strength": signal.signal_strength,
            "signal_type": signal.signal_type,
            "timestamp": feedback.timestamp
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

@router.post("/interaction")
@inject
async def record_restaurant_interaction(
    interaction: RestaurantInteraction,
    user_profile_service = Depends(Provide[Container.user_profile_service])
):
    """
    Record user interaction với restaurant recommendation
    """
    try:
        from modules.domain.schemas import UserInteraction
        
        # Convert to general interaction
        general_interaction = UserInteraction(
            user_id=interaction.user_id,
            business_id=interaction.business_id,
            interaction_type=interaction.interaction_type,
            interaction_value=interaction.dwell_time_seconds,
            search_context=interaction.search_context,
            rank_position=interaction.rank_in_results
        )
        
        user_profile_service.record_interaction(general_interaction)
        
        return {
            "interaction_recorded": True,
            "timestamp": interaction.timestamp
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record interaction: {str(e)}")

# ========================================
# RECOMMENDATION ANALYSIS ENDPOINTS
# ========================================

@router.get("/recommendations/{user_id}/explain")
@inject
async def explain_restaurant_recommendations(
    user_id: str,
    restaurant_id: str = Query(..., description="Restaurant business ID"),
    city: str = Query(..., description="City context"),
    user_profile_service = Depends(Provide[Container.user_profile_service]),
    feedback_learning_service = Depends(Provide[Container.feedback_learning_service])
):
    """
    Explain tại sao restaurant được recommend cho user
    """
    try:
        # Get personalization boost breakdown
        boost = user_profile_service.get_restaurant_personalization_boost(
            user_id=user_id,
            restaurant_metadata={"business_id": restaurant_id},
            city=city
        )
        
        # Get collaborative boost
        collaborative_boost = feedback_learning_service.get_restaurant_collaborative_boost(
            user_id=user_id,
            restaurant_id=restaurant_id,
            city=city
        )
        
        # Get user preferences analysis
        preferences_analysis = user_profile_service.analyze_restaurant_preferences(user_id, city)
        
        return {
            "user_id": user_id,
            "restaurant_id": restaurant_id,
            "city": city,
            "personalization_boost": boost,
            "collaborative_boost": collaborative_boost,
            "user_preferences": preferences_analysis,
            "explanation": {
                "personalization_factors": "Based on your cuisine preferences and price sensitivity",
                "collaborative_factors": "Based on similar users' preferences in this city",
                "city_factors": f"Considering your history in {city}"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to explain recommendations: {str(e)}")

# ========================================
# HEALTH & INFO ENDPOINTS
# ========================================

@router.get("/health")
async def restaurant_service_health():
    """
    Health check cho restaurant service
    """
    try:
        city_service = CityFilterService()
        cities = city_service.get_supported_cities()
        
        return {
            "status": "healthy",
            "service": "Restaurant Recommendation Service",
            "supported_cities": len(cities),
            "features": [
                "City-based filtering",
                "Semantic search with BGE-M3",
                "User personalization",
                "Collaborative filtering",
                "Restaurant feedback learning"
            ],
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now()
        }

@router.get("/info")
async def restaurant_service_info():
    """
    Thông tin chi tiết về restaurant service
    """
    try:
        city_service = CityFilterService()
        cities = city_service.get_supported_cities()
        
        city_stats = {}
        for city in cities[:5]:  # Top 5 cities
            stats = city_service.get_city_stats(city.city_name)
            if stats:
                city_stats[city.city_name] = stats
        
        return {
            "service_name": "Restaurant Recommendation Service",
            "version": "1.0",
            "features": {
                "city_filtering": "Filter restaurants by Vietnamese cities",
                "semantic_search": "BGE-M3 embeddings for semantic matching",
                "personalization": "User preference-based boosting",
                "collaborative_filtering": "City-aware collaborative recommendations",
                "feedback_learning": "Real-time learning from user feedback"
            },
            "supported_cities": len(cities),
            "city_statistics": city_stats,
            "cuisine_types": [cuisine.value for cuisine in CuisineType],
            "price_levels": [price.value for price in PriceLevel],
            "dining_styles": [style.value for style in DiningStyle]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service info: {str(e)}")

def demo_restaurant_router():
    """Demo restaurant router endpoints"""
    print("DEMO RESTAURANT ROUTER")
    print("=" * 50)
    
    print(f"\nAvailable Endpoints:")
    print(f"   POST /restaurant/search - Main restaurant search")
    print(f"   GET  /restaurant/search - GET version for testing")
    print(f"   GET  /restaurant/cities - Get supported cities")
    print(f"   GET  /restaurant/cities/search - Search cities")
    print(f"   GET  /restaurant/cities/{{city}}/stats - City statistics")
    print(f"   GET  /restaurant/profile/{{user_id}} - User restaurant profile")
    print(f"   POST /restaurant/profile/{{user_id}}/update - Update profile")
    print(f"   GET  /restaurant/profile/{{user_id}}/analysis/{{city}} - Preference analysis")
    print(f"   POST /restaurant/feedback - Submit feedback")
    print(f"   POST /restaurant/interaction - Record interaction")
    print(f"   GET  /restaurant/recommendations/{{user_id}}/explain - Explain recommendations")
    print(f"   GET  /restaurant/health - Health check")
    print(f"   GET  /restaurant/info - Service information")
    
    print(f"\nKey Features:")
    print(f"   City-based restaurant filtering")
    print(f"   Semantic search with BGE-M3")
    print(f"   User personalization & learning")
    print(f"   Collaborative filtering")
    print(f"   Restaurant feedback system")
    print(f"   Explainable recommendations")
    
    print(f"\nRestaurant Router demo completed!")

if __name__ == "__main__":
    demo_restaurant_router() 