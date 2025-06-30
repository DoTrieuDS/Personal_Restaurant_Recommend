"""
City Filter Service
Geographic filtering service cho restaurant recommendations
"""

import logging
from typing import List, Dict, Optional, Set
from collections import defaultdict
import pandas as pd
from modules.domain.restaurant_schemas import SupportedCity

class CityFilterService:
    """
    Service để filter restaurants theo city và manage supported cities
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Cache cho city data
        self.city_restaurant_cache = {}  # city -> set of business_ids
        self.supported_cities_cache = {}  # city -> SupportedCity
        self.city_stats_cache = {}       # city -> statistics
        
        # Vietnamese city mapping để handle variations
        self.city_name_mapping = {
            # Standard mappings
            "ho chi minh city": ["ho chi minh city", "saigon", "sai gon", "hcmc", "tphcm", "tp.hcm"],
            "hanoi": ["hanoi", "ha noi", "hà nội"],
            "da nang": ["da nang", "danang", "đà nẵng"],
            "nha trang": ["nha trang", "nhatrang"],
            "can tho": ["can tho", "cần thơ", "cantho"],
            "vung tau": ["vung tau", "vũng tàu", "vungtau"],
            "hue": ["hue", "huế"],
            "hai phong": ["hai phong", "haiphong", "hải phòng"],
            "phu quoc": ["phu quoc", "phú quốc", "phuquoc"],
            "da lat": ["da lat", "dalat", "đà lạt"]
        }
        
        # Initialize city data
        self._initialize_city_data()
    
    def _initialize_city_data(self):
        """Initialize city data từ restaurant database"""
        try:
            from .data_config import DataConfig
            
            # Load metadata từ restaurant database using DataConfig
            try:
                metadata_file = DataConfig.get_restaurant_data_path(prefer_improved=True)
                df = pd.read_parquet(metadata_file)
                self._build_city_caches(df)
                self.logger.info(f"Initialized city data for {len(self.supported_cities_cache)} cities")
            except Exception as e:
                self.logger.warning(f"Could not load restaurant metadata: {e}")
                self._create_default_city_data()
                
        except Exception as e:
            self.logger.error(f"Error initializing city data: {e}")
    
    def _build_city_caches(self, df: pd.DataFrame):
        """Build city caches từ restaurant dataframe"""
        
        # Filter restaurants only
        restaurant_df = df[df['poi_type'] == 'Restaurant'].copy()
        
        # Group by city
        city_groups = restaurant_df.groupby('city')
        
        for city, group in city_groups:
            if len(group) < 5:  # Skip cities với ít hơn 5 restaurants
                continue
                
            city_lower = city.lower()
            business_ids = set(group['business_id'].tolist())
            
            # Build restaurant cache
            self.city_restaurant_cache[city_lower] = business_ids
            
            # Build cuisine distribution
            cuisine_dist = defaultdict(int)
            for categories in group['categories'].dropna():
                for cuisine in categories.split(', '):
                    cuisine_dist[cuisine.lower()] += 1
            
            # Build price distribution
            price_dist = defaultdict(int)
            for stars in group['stars'].dropna():
                if stars <= 2.5:
                    price_dist['budget'] += 1
                elif stars <= 3.5:
                    price_dist['moderate'] += 1
                elif stars <= 4.5:
                    price_dist['expensive'] += 1
                else:
                    price_dist['luxury'] += 1
            
            # Create SupportedCity object
            supported_city = SupportedCity(
                city_name=city,
                total_restaurants=len(business_ids),
                cuisine_distribution=dict(cuisine_dist),
                price_distribution=dict(price_dist),
                latitude=group['latitude'].mean() if not group['latitude'].isna().all() else None,
                longitude=group['longitude'].mean() if not group['longitude'].isna().all() else None
            )
            
            self.supported_cities_cache[city_lower] = supported_city
            
            # Cache statistics
            self.city_stats_cache[city_lower] = {
                'total_restaurants': len(business_ids),
                'avg_rating': group['stars'].mean(),
                'total_reviews': group['review_count'].sum(),
                'top_cuisines': list(dict(sorted(cuisine_dist.items(), key=lambda x: x[1], reverse=True)[:5]).keys())
            }
    
    def _create_default_city_data(self):
        """Tạo default city data nếu không load được từ database"""
        default_cities = [
            {"name": "Ho Chi Minh City", "restaurants": 1000},
            {"name": "Hanoi", "restaurants": 800},
            {"name": "Da Nang", "restaurants": 300},
            {"name": "Nha Trang", "restaurants": 200},
            {"name": "Can Tho", "restaurants": 150}
        ]
        
        for city_info in default_cities:
            city_lower = city_info["name"].lower()
            
            self.supported_cities_cache[city_lower] = SupportedCity(
                city_name=city_info["name"],
                total_restaurants=city_info["restaurants"],
                cuisine_distribution={"vietnamese": 60, "asian": 20, "western": 20},
                price_distribution={"budget": 40, "moderate": 40, "expensive": 20}
            )
    
    def get_restaurants_by_city(self, city: str) -> Set[str]:
        """
        Lấy danh sách restaurant business_ids cho city
        
        Args:
            city: Tên city
            
        Returns:
            Set of business_ids in that city
        """
        normalized_city = self._normalize_city_name(city)
        
        if normalized_city in self.city_restaurant_cache:
            return self.city_restaurant_cache[normalized_city]
        
        self.logger.warning(f"No restaurants found for city: {city}")
        return set()
    
    def filter_faiss_results_by_city(self, results: List[Dict], city: str) -> List[Dict]:
        """
        Filter FAISS search results theo city
        
        Args:
            results: List of FAISS search results
            city: Target city
            
        Returns:
            Filtered results containing only restaurants in specified city
        """
        valid_business_ids = self.get_restaurants_by_city(city)
        
        if not valid_business_ids:
            return []
        
        filtered_results = []
        for result in results:
            business_id = result.get('business_id')
            if business_id in valid_business_ids:
                filtered_results.append(result)
        
        self.logger.info(f"Filtered {len(results)} results to {len(filtered_results)} restaurants in {city}")
        return filtered_results
    
    def get_supported_cities(self) -> List[SupportedCity]:
        """
        Lấy danh sách supported cities
        
        Returns:
            List of SupportedCity objects
        """
        return list(self.supported_cities_cache.values())
    
    def get_city_stats(self, city: str) -> Optional[Dict]:
        """
        Lấy statistics cho specific city
        
        Args:
            city: City name
            
        Returns:
            Dictionary containing city statistics
        """
        normalized_city = self._normalize_city_name(city)
        return self.city_stats_cache.get(normalized_city)
    
    def is_city_supported(self, city: str) -> bool:
        """
        Kiểm tra xem city có được support không
        
        Args:
            city: City name
            
        Returns:
            True if city is supported
        """
        normalized_city = self._normalize_city_name(city)
        return normalized_city in self.supported_cities_cache
    
    def get_nearby_cities(self, city: str, max_distance_km: float = 100) -> List[str]:
        """
        Tìm cities gần đó (placeholder for future geographic search)
        
        Args:
            city: Reference city
            max_distance_km: Maximum distance
            
        Returns:
            List of nearby city names
        """
        # Simplified implementation - just return popular alternatives
        city_lower = city.lower()
        
        nearby_mapping = {
            "ho chi minh city": ["Vung Tau", "Can Tho"],
            "hanoi": ["Hai Phong"],
            "da nang": ["Hue"],
            "nha trang": ["Da Lat"]
        }
        
        return nearby_mapping.get(city_lower, [])
    
    def _normalize_city_name(self, city: str) -> str:
        """
        Normalize city name để handle variations
        
        Args:
            city: Raw city name
            
        Returns:
            Normalized city name
        """
        city_lower = city.lower().strip()
        
        # Check direct mapping
        for standard_name, variations in self.city_name_mapping.items():
            if city_lower in variations:
                return standard_name
        
        # Return as-is if no mapping found
        return city_lower
    
    def search_cities(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search cities by name với fuzzy matching
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching cities với statistics
        """
        query_lower = query.lower()
        matches = []
        
        for city_name, city_obj in self.supported_cities_cache.items():
            # Check if query matches city name or variations
            if query_lower in city_name or any(query_lower in variation for variation in self.city_name_mapping.get(city_name, [])):
                matches.append({
                    'city_name': city_obj.city_name,
                    'total_restaurants': city_obj.total_restaurants,
                    'top_cuisines': list(city_obj.cuisine_distribution.keys())[:3],
                    'match_score': 1.0 if query_lower == city_name else 0.8
                })
        
        # Sort by match score and restaurant count
        matches.sort(key=lambda x: (x['match_score'], x['total_restaurants']), reverse=True)
        return matches[:limit]
    
    def get_city_cuisine_preferences(self, city: str) -> Dict[str, float]:
        """
        Lấy cuisine preferences cho city (popular cuisines in that city)
        
        Args:
            city: City name
            
        Returns:
            Dictionary mapping cuisine -> popularity_score
        """
        normalized_city = self._normalize_city_name(city)
        city_obj = self.supported_cities_cache.get(normalized_city)
        
        if not city_obj:
            return {}
        
        # Convert counts to percentages
        total_restaurants = city_obj.total_restaurants
        cuisine_preferences = {}
        
        for cuisine, count in city_obj.cuisine_distribution.items():
            cuisine_preferences[cuisine] = count / total_restaurants
        
        return cuisine_preferences
    
    def refresh_city_data(self):
        """Refresh city data từ database"""
        self.logger.info("Refreshing city data...")
        self.city_restaurant_cache.clear()
        self.supported_cities_cache.clear()
        self.city_stats_cache.clear()
        self._initialize_city_data()

def demo_city_filter_service():
    """Demo CityFilterService functionality"""
    print("DEMO CITY FILTER SERVICE")
    print("=" * 50)
    
    # Initialize service
    city_service = CityFilterService()
    
    # Test supported cities
    print(f"\n1. Supported Cities:")
    supported_cities = city_service.get_supported_cities()
    for city in supported_cities[:5]:  # Show first 5
        print(f"   {city.city_name}: {city.total_restaurants} restaurants")
    
    # Test city search
    print(f"\n2. City Search:")
    search_results = city_service.search_cities("ho chi", limit=3)
    for result in search_results:
        print(f"   {result['city_name']}: {result['total_restaurants']} restaurants (Score: {result['match_score']})")
    
    # Test city normalization
    print(f"\n3. City Normalization:")
    test_variations = ["Saigon", "HCMC", "Ho Chi Minh City", "ha noi"]
    for variation in test_variations:
        normalized = city_service._normalize_city_name(variation)
        print(f"   '{variation}' -> '{normalized}'")
    
    # Test restaurant filtering
    print(f"\n4. Restaurant Filtering:")
    mock_results = [
        {"business_id": "rest_001", "name": "Test Restaurant 1"},
        {"business_id": "rest_002", "name": "Test Restaurant 2"}
    ]
    
    for test_city in ["Ho Chi Minh City", "Hanoi"]:
        restaurant_count = len(city_service.get_restaurants_by_city(test_city))
        print(f"   {test_city}: {restaurant_count} restaurants available")
    
    # Test city statistics
    print(f"\n5. City Statistics:")
    for test_city in ["ho chi minh city", "hanoi"]:
        stats = city_service.get_city_stats(test_city)
        if stats:
            print(f"   {test_city.title()}:")
            print(f"      Total restaurants: {stats['total_restaurants']}")
            print(f"      Top cuisines: {stats.get('top_cuisines', [])[:3]}")
    
    print(f"\nCity Filter Service demo completed!")

if __name__ == "__main__":
    demo_city_filter_service() 