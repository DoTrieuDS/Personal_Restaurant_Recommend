"""
Core schemas cho Travel Recommendation System
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

class POICoordinates(BaseModel):
    """Tọa độ địa lý của POI"""
    latitude: float
    longitude: float

class POITimingInfo(BaseModel):
    """Thông tin thời gian hoạt động"""
    hours: Optional[str] = None  # "Monday 9:00-17:00|Tuesday 9:00-17:00"
    peak_times: Optional[List[str]] = None  # ["18:00-20:00", "12:00-14:00"]
    average_visit_duration: Optional[int] = None  # minutes

class POIPricing(BaseModel):
    """Thông tin giá cả"""
    price_level: Optional[int] = None  # 1-4 ($ to $$$$)
    average_cost: Optional[float] = None  # USD
    cost_per_person: Optional[float] = None

class SearchRequest(BaseModel):
    """Model cho search request"""
    query: str
    destination: Optional[str] = None
    preferences: Optional[str] = None
    num_results: int = 10
    categories: Optional[List[str]] = None

class RecommendationMetrics(BaseModel):
    """Metrics về performance của recommendation"""
    total_time: float
    retrieval_time: float
    reranking_time: float
    num_candidates_retrieved: int
    num_final_results: int
    timestamp: datetime

# ========================================
# USER PROFILE SCHEMAS
# ========================================

class CuisinePreference(str, Enum):
    """Enum cho cuisine preferences"""
    VIETNAMESE = "vietnamese"
    ASIAN = "asian"
    WESTERN = "western"
    SEAFOOD = "seafood"
    VEGETARIAN = "vegetarian"
    STREET_FOOD = "street_food"
    FINE_DINING = "fine_dining"
    LOCAL_SPECIALTY = "local_specialty"

class DiningStyle(str, Enum):
    """Enum cho dining style preferences"""
    CASUAL = "casual"
    FINE_DINING = "fine_dining"
    FAST_FOOD = "fast_food"
    BUFFET = "buffet"
    TAKEAWAY = "takeaway"
    FAMILY_STYLE = "family_style"
    ROMANTIC = "romantic"
    BUSINESS_MEETING = "business_meeting"

class UserDemographics(BaseModel):
    """User demographics information"""
    age_range: Optional[str] = None  # "18-25", "26-35", etc.
    gender: Optional[str] = None
    location: Optional[str] = None  # Home location
    occupation: Optional[str] = None
    income_level: Optional[str] = None  # "low", "medium", "high"

class RestaurantPreferences(BaseModel):
    """User restaurant preferences"""
    preferred_cuisines: List[CuisinePreference] = Field(default_factory=list)
    dining_styles: List[DiningStyle] = Field(default_factory=list)
    budget_range: Optional[Dict[str, float]] = None  # {"min": 20, "max": 200}
    group_size_preference: Optional[int] = None
    
    # Specific preferences
    likes_crowded_places: Optional[bool] = None
    prefers_authentic_experience: Optional[bool] = None
    vegetarian_friendly: Optional[bool] = None
    spice_tolerance: Optional[str] = None  # "mild", "medium", "spicy"
    preferred_meal_times: Optional[List[str]] = None  # ["lunch", "dinner"]

class UserProfile(BaseModel):
    """Complete user profile for restaurant recommendations"""
    user_id: str = Field(..., description="Unique user identifier")
    username: Optional[str] = None
    email: Optional[str] = None
    
    # Demographics
    demographics: Optional[UserDemographics] = None
    
    # Restaurant preferences
    restaurant_preferences: RestaurantPreferences = Field(default_factory=RestaurantPreferences)
    
    # History tracking
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    
    # Activity metrics
    total_searches: int = 0
    total_restaurant_visits: int = 0
    favorite_cities: List[str] = Field(default_factory=list)
    
    # Profile completeness
    profile_completeness: float = Field(default=0.0, ge=0.0, le=1.0)

# ========================================
# FEEDBACK LEARNING SCHEMAS
# ========================================

class FeedbackType(str, Enum):
    """Các loại feedback từ user"""
    LIKE = "like"
    DISLIKE = "dislike"
    BOOKMARK = "bookmark"
    VISIT = "visit"
    RATE = "rate"
    COMMENT = "comment"
    SKIP = "skip"
    SHARE = "share"

class FeedbackAction(BaseModel):
    """Individual feedback action from user"""
    user_id: str
    business_id: str
    recommendation_id: Optional[str] = None  # ID của recommendation session
    
    feedback_type: FeedbackType
    feedback_value: Optional[float] = None  # Rating value nếu là rate (1-5)
    feedback_text: Optional[str] = None  # Comment text nếu có
    
    # Context
    search_query: Optional[str] = None
    destination: Optional[str] = None
    rank_in_results: Optional[int] = None  # Vị trí trong kết quả search
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None

class UserInteraction(BaseModel):
    """User interaction with POI recommendations"""
    user_id: str
    business_id: str
    interaction_type: str  # "view", "click", "dwell_time", "scroll_past"
    interaction_value: Optional[float] = None  # dwell time in seconds, scroll depth, etc.
    
    # Context when interaction happened
    search_context: Optional[Dict] = None
    recommendation_score: Optional[float] = None
    rank_position: Optional[int] = None
    
    timestamp: datetime = Field(default_factory=datetime.now)

class LearningSignal(BaseModel):
    """Processed learning signal from user behavior"""
    user_id: str
    business_id: str
    signal_type: str  # "positive", "negative", "neutral"
    signal_strength: float = Field(..., ge=0.0, le=1.0)  # 0-1 confidence
    
    # What we learned
    learned_preferences: Dict = Field(default_factory=dict)
    preference_updates: Dict = Field(default_factory=dict)
    
    # Source of signal
    source_feedbacks: List[str] = Field(default_factory=list)  # feedback IDs
    source_interactions: List[str] = Field(default_factory=list)  # interaction IDs
    
    created_at: datetime = Field(default_factory=datetime.now)
    processed: bool = False

class PersonalizationMetrics(BaseModel):
    """Metrics về personalization performance"""
    user_id: str
    
    # Performance metrics
    avg_recommendation_score: float
    click_through_rate: float
    feedback_score: float  # Average rating/feedback
    
    # Learning metrics
    preference_confidence: float  # Confidence in learned preferences
    profile_completeness: float
    last_updated: datetime
    
    # Historical data
    total_recommendations: int
    total_interactions: int
    total_feedback_events: int

# ========================================
# RESTAURANT-FOCUSED REQUEST SCHEMAS
# ========================================

class RestaurantRecommendationRequest(BaseModel):
    """Request model cho restaurant recommendations"""
    user_id: str
    city: str
    user_query: str  # "tôi muốn ăn bún bò huế cay cay"
    
    # Context
    meal_type: Optional[str] = None  # "breakfast", "lunch", "dinner"
    occasion: Optional[str] = None   # "date", "family", "business"
    group_size: Optional[int] = 2
    
    # Preferences
    price_range: Optional[str] = None  # "budget", "moderate", "expensive"
    cuisine_preference: Optional[List[str]] = None
    
    # System parameters
    num_results: int = 10
    use_personalization: bool = True

class SubmitFeedbackRequest(BaseModel):
    """Request model cho feedback submission"""
    user_id: str
    business_id: str
    feedback_type: FeedbackType
    feedback_value: Optional[float] = None
    feedback_text: Optional[str] = None
    recommendation_context: Optional[Dict] = None

class UserProfileUpdateRequest(BaseModel):
    """Request model cho profile updates"""
    user_id: str
    demographics: Optional[UserDemographics] = None
    restaurant_preferences: Optional[RestaurantPreferences] = None

class PersonalizedRecommendationRequest(BaseModel):
    """Request model cho personalized recommendations"""
    user_id: str
    destination: str
    user_query: Optional[str] = None
    num_results: int = 10
    use_personalization: bool = True
    exploration_factor: float = Field(default=0.1, ge=0.0, le=1.0)  # Balance exploitation vs exploration