"""
Restaurant-Specific Schemas cho Travel Recommendation System
Focus on restaurant preferences, search, và personalization
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

# ========================================
# RESTAURANT ENUMS
# ========================================

class CuisineType(str, Enum):
    """Enum cho cuisine types cụ thể cho Việt Nam"""
    VIETNAMESE = "vietnamese"
    NORTHERN_VIETNAMESE = "northern_vietnamese"  # Phở, Bún chả
    CENTRAL_VIETNAMESE = "central_vietnamese"    # Bún bò Huế, Mì Quảng
    SOUTHERN_VIETNAMESE = "southern_vietnamese"  # Bánh mì, Cơm tấm
    
    ASIAN = "asian"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    KOREAN = "korean"
    THAI = "thai"
    
    WESTERN = "western"
    ITALIAN = "italian"
    FRENCH = "french"
    AMERICAN = "american"
    
    SEAFOOD = "seafood"
    VEGETARIAN = "vegetarian"
    STREET_FOOD = "street_food"
    FINE_DINING = "fine_dining"
    LOCAL_SPECIALTY = "local_specialty"

class PriceLevel(str, Enum):
    """Enum cho price levels"""
    BUDGET = "budget"          # < 100k VND
    MODERATE = "moderate"      # 100k - 300k VND
    EXPENSIVE = "expensive"    # 300k - 500k VND
    LUXURY = "luxury"         # > 500k VND

class DiningStyle(str, Enum):
    """Enum cho dining styles"""
    CASUAL = "casual"
    FINE_DINING = "fine_dining"
    STREET_FOOD = "street_food"
    FAST_FOOD = "fast_food"
    COFFEE_SHOP = "coffee_shop"
    BAR_PUB = "bar_pub"
    BUFFET = "buffet"
    FOOD_COURT = "food_court"

class AtmosphereType(str, Enum):
    """Enum cho atmosphere preferences"""
    QUIET = "quiet"
    LIVELY = "lively"
    ROMANTIC = "romantic"
    FAMILY_FRIENDLY = "family_friendly"
    BUSINESS = "business"
    TRENDY = "trendy"
    TRADITIONAL = "traditional"
    OUTDOOR = "outdoor"

# ========================================
# RESTAURANT PREFERENCES
# ========================================

class RestaurantPreferences(BaseModel):
    """User preferences cho restaurants"""
    
    # Core preferences
    cuisine_types: List[CuisineType] = Field(default_factory=list)
    price_levels: List[PriceLevel] = Field(default_factory=list)
    dining_styles: List[DiningStyle] = Field(default_factory=list)
    
    # Dietary needs
    dietary_needs: List[str] = Field(default_factory=list)  # vegetarian, halal, vegan, gluten-free
    
    # Atmosphere preferences
    atmosphere: List[AtmosphereType] = Field(default_factory=list)
    
    # Specific preferences
    spice_tolerance: Optional[str] = None  # "low", "medium", "high", "very_high"
    group_size_preference: Optional[int] = 2
    occasion_preferences: List[str] = Field(default_factory=list)  # date, family, business, friends
    
    # Time preferences
    meal_time_preferences: Dict[str, bool] = Field(default_factory=lambda: {
        "breakfast": True,
        "lunch": True,
        "dinner": True,
        "late_night": False
    })
    
    # Location preferences
    prefers_local_chains: Optional[bool] = None
    prefers_international_chains: Optional[bool] = None
    max_distance_km: Optional[float] = None

class RestaurantLearningData(BaseModel):
    """Learning data cho restaurant recommendations"""
    
    # Preference evolution
    cuisine_preference_scores: Dict[str, float] = Field(default_factory=dict)
    price_sensitivity: float = 0.5  # 0.0 = price-insensitive, 1.0 = very price-sensitive
    novelty_seeking: float = 0.3    # 0.0 = conservative, 1.0 = adventure seeker
    
    # Interaction patterns
    average_rating_given: Optional[float] = None
    total_restaurant_visits: int = 0
    cities_explored: List[str] = Field(default_factory=list)
    
    # Temporal patterns
    last_preference_update: datetime = Field(default_factory=datetime.now)
    preference_stability: float = 0.5  # How stable are user's preferences

class CityRestaurantHistory(BaseModel):
    """Restaurant history cho specific city"""
    
    city: str
    
    # Visit history
    visited_restaurants: List[str] = Field(default_factory=list)  # business_ids
    liked_restaurants: List[str] = Field(default_factory=list)
    disliked_restaurants: List[str] = Field(default_factory=list)
    bookmarked_restaurants: List[str] = Field(default_factory=list)
    
    # Preference evolution trong city này
    preference_evolution: Dict[str, float] = Field(default_factory=dict)  # cuisine -> preference_score
    
    # Statistics
    total_visits: int = 0
    favorite_cuisines: List[str] = Field(default_factory=list)
    average_price_level: Optional[str] = None
    
    # Context data
    first_visit_date: Optional[datetime] = None
    last_visit_date: Optional[datetime] = None
    visit_frequency: float = 0.0  # visits per month

class RestaurantProfile(BaseModel):
    """Complete restaurant profile cho user"""
    
    user_id: str = Field(..., description="Unique user identifier")
    
    # Core preferences
    preferences: RestaurantPreferences = Field(default_factory=RestaurantPreferences)
    
    # City-specific histories
    city_history: Dict[str, CityRestaurantHistory] = Field(default_factory=dict)
    
    # Learning data
    learning_data: RestaurantLearningData = Field(default_factory=RestaurantLearningData)
    
    # Profile metadata
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    profile_completeness: float = Field(default=0.0, ge=0.0, le=1.0)

# ========================================
# RESTAURANT SEARCH REQUESTS
# ========================================

class RestaurantSearchRequest(BaseModel):
    """Request model cho restaurant search"""
    
    # Required fields
    city: str = Field(..., description="City to search in")
    user_query: str = Field(..., description="User's restaurant query")
    
    # Optional user identification
    user_id: Optional[str] = None
    
    # Filters
    price_filter: Optional[List[PriceLevel]] = None
    cuisine_filter: Optional[List[CuisineType]] = None
    dining_style_filter: Optional[List[DiningStyle]] = None
    
    # Context
    meal_type: Optional[str] = None  # breakfast, lunch, dinner
    occasion: Optional[str] = None   # date, family, business, friends
    group_size: Optional[int] = 2
    
    # Personalization
    use_personalization: bool = True
    exploration_factor: float = Field(default=0.1, ge=0.0, le=1.0)  # Balance exploitation vs exploration
    
    # Search parameters
    num_results: int = Field(default=10, ge=1, le=50)
    include_details: bool = True

class RestaurantRecommendation(BaseModel):
    """Single restaurant recommendation với full context"""
    
    # Basic info
    business_id: str
    name: str
    city: str
    address: Optional[str] = None
    
    # Restaurant details
    cuisine_types: List[str] = Field(default_factory=list)
    price_level: Optional[PriceLevel] = None
    dining_style: Optional[DiningStyle] = None
    
    # Quality metrics
    stars: float
    review_count: int
    
    # Recommendation scores
    similarity_score: float  # BGE-M3 semantic similarity
    personalization_score: float  # User preference matching
    final_score: float      # Combined final score
    rank: int
    
    # Personalization details
    why_recommended: List[str] = Field(default_factory=list)  # Reasons for recommendation
    user_match_factors: List[str] = Field(default_factory=list)  # Matching factors
    
    # Context
    recommended_for: Optional[str] = None  # meal_type, occasion
    distance_km: Optional[float] = None
    
    # Metadata
    recommendation_confidence: float = 0.5
    is_new_discovery: bool = False  # New restaurant for this user

class RestaurantSearchResponse(BaseModel):
    """Response model cho restaurant search"""
    
    # Request context
    search_request: RestaurantSearchRequest
    
    # Results
    restaurants: List[RestaurantRecommendation]
    
    # Search metadata
    total_found: int
    search_time: float
    personalization_applied: bool = False
    
    # Success status
    success: bool = True
    message: str = "Restaurant search completed successfully"
    
    # Additional insights
    city_restaurant_stats: Optional[Dict] = None  # Total restaurants in city, etc.
    search_suggestions: List[str] = Field(default_factory=list)  # Alternative search terms

# ========================================
# RESTAURANT FEEDBACK MODELS
# ========================================

class RestaurantFeedbackAction(BaseModel):
    """Feedback action specifically for restaurants"""
    
    user_id: str
    business_id: str
    
    # Feedback details
    feedback_type: str  # like, dislike, visit, bookmark, rate
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)  # 1-5 star rating
    
    # Context
    city: str
    meal_type: Optional[str] = None
    occasion: Optional[str] = None
    group_size: Optional[int] = None
    
    # Specific restaurant feedback
    liked_aspects: List[str] = Field(default_factory=list)  # food, service, atmosphere, price
    disliked_aspects: List[str] = Field(default_factory=list)
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    search_query: Optional[str] = None  # Original search query

class RestaurantInteraction(BaseModel):
    """User interaction với restaurant recommendation"""
    
    user_id: str
    business_id: str
    interaction_type: str  # view, click, dwell, scroll_past
    
    # Context
    city: str
    search_context: Dict = Field(default_factory=dict)
    rank_in_results: Optional[int] = None
    
    # Timing
    dwell_time_seconds: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# ========================================
# CITY MANAGEMENT
# ========================================

class SupportedCity(BaseModel):
    """Model cho supported cities"""
    
    city_name: str
    country: str = "Vietnam"
    
    # Restaurant statistics
    total_restaurants: int
    cuisine_distribution: Dict[str, int] = Field(default_factory=dict)
    price_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Geographic info
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Status
    is_active: bool = True
    last_updated: datetime = Field(default_factory=datetime.now)

# ========================================
# UTILITY FUNCTIONS
# ========================================

def calculate_restaurant_profile_completeness(profile: RestaurantProfile) -> float:
    """Calculate completeness score cho restaurant profile"""
    
    score = 0.0
    total_factors = 10.0
    
    # Basic preferences (3 points)
    if profile.preferences.cuisine_types:
        score += 1.0
    if profile.preferences.price_levels:
        score += 1.0
    if profile.preferences.dining_styles:
        score += 1.0
    
    # Dietary and atmosphere (2 points)
    if profile.preferences.dietary_needs:
        score += 0.5
    if profile.preferences.atmosphere:
        score += 0.5
    if profile.preferences.spice_tolerance:
        score += 0.5
    if profile.preferences.occasion_preferences:
        score += 0.5
    
    # Learning data (2 points)
    if profile.learning_data.cuisine_preference_scores:
        score += 1.0
    if profile.learning_data.total_restaurant_visits > 0:
        score += 1.0
    
    # City history (3 points)
    if profile.city_history:
        score += 1.0
        # Bonus for multiple cities
        if len(profile.city_history) > 1:
            score += 0.5
        # Bonus for substantial history
        total_visits = sum(city.total_visits for city in profile.city_history.values())
        if total_visits > 5:
            score += 1.0
        elif total_visits > 0:
            score += 0.5
    
    return min(1.0, score / total_factors)

def extract_cuisine_types_from_categories(categories: str) -> List[CuisineType]:
    """Extract cuisine types từ restaurant categories string"""
    
    categories_lower = categories.lower()
    extracted_cuisines = []
    
    # Mapping categories to cuisine types
    cuisine_mapping = {
        'vietnamese': CuisineType.VIETNAMESE,
        'pho': CuisineType.NORTHERN_VIETNAMESE,
        'bun bo hue': CuisineType.CENTRAL_VIETNAMESE,
        'banh mi': CuisineType.SOUTHERN_VIETNAMESE,
        'chinese': CuisineType.CHINESE,
        'japanese': CuisineType.JAPANESE,
        'korean': CuisineType.KOREAN,
        'thai': CuisineType.THAI,
        'italian': CuisineType.ITALIAN,
        'french': CuisineType.FRENCH,
        'american': CuisineType.AMERICAN,
        'seafood': CuisineType.SEAFOOD,
        'vegetarian': CuisineType.VEGETARIAN,
        'street food': CuisineType.STREET_FOOD,
        'fine dining': CuisineType.FINE_DINING
    }
    
    for category_keyword, cuisine_type in cuisine_mapping.items():
        if category_keyword in categories_lower:
            extracted_cuisines.append(cuisine_type)
    
    return list(set(extracted_cuisines))  # Remove duplicates 