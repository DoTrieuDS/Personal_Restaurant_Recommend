"""
Data Configuration - Centralized Data Paths
Quản lý đường dẫn đến các file data sau cleanup và refactor
"""

import os
from pathlib import Path

class DataConfig:
    """
    Centralized configuration cho data paths
    """
    
    # Base data directory
    BASE_DATA_DIR = "modules/recommendation/data"
    
    # Restaurant data paths
    RESTAURANTS_DIR = os.path.join(BASE_DATA_DIR, "restaurants")
    MAIN_RESTAURANT_DATA = os.path.join(RESTAURANTS_DIR, "pois_with_improved_cuisines.parquet")
    RESTAURANT_EMBEDDINGS = os.path.join(RESTAURANTS_DIR, "poi_embeddings_structured.parquet")
    
    # Fallback data (legacy support)
    FALLBACK_RESTAURANT_DATA = os.path.join(BASE_DATA_DIR, "pois_with_cleaned_descriptions_updated.parquet")
    
    # FAISS indices paths
    INDICES_DIR = os.path.join(BASE_DATA_DIR, "indices")
    
    # All POIs indices
    ALL_POIS_INDEX_DIR = os.path.join(INDICES_DIR, "all_pois")
    ALL_POIS_FAISS_INDEX = os.path.join(ALL_POIS_INDEX_DIR, "faiss_index.bin")
    ALL_POIS_MAPPINGS = os.path.join(ALL_POIS_INDEX_DIR, "mappings.pkl")
    
    # Restaurant-only indices
    RESTAURANT_INDEX_DIR = os.path.join(INDICES_DIR, "restaurants")
    RESTAURANT_FAISS_INDEX = os.path.join(RESTAURANT_INDEX_DIR, "faiss_index.bin")
    RESTAURANT_MAPPINGS = os.path.join(RESTAURANT_INDEX_DIR, "mappings.pkl")
    
    @classmethod
    def get_restaurant_data_path(cls, prefer_improved: bool = True) -> str:
        """
        Get path to restaurant data file
        
        Args:
            prefer_improved: If True, prefer improved cuisines file
            
        Returns:
            Path to restaurant data file
        """
        if prefer_improved and os.path.exists(cls.MAIN_RESTAURANT_DATA):
            return cls.MAIN_RESTAURANT_DATA
        elif os.path.exists(cls.FALLBACK_RESTAURANT_DATA):
            return cls.FALLBACK_RESTAURANT_DATA
        else:
            raise FileNotFoundError("No restaurant data file found")
    
    @classmethod
    def get_faiss_index_paths(cls, restaurant_only: bool = True) -> tuple:
        """
        Get FAISS index and mappings paths
        
        Args:
            restaurant_only: If True, get restaurant-only indices
            
        Returns:
            Tuple of (index_path, mappings_path)
        """
        if restaurant_only:
            return cls.RESTAURANT_FAISS_INDEX, cls.RESTAURANT_MAPPINGS
        else:
            return cls.ALL_POIS_FAISS_INDEX, cls.ALL_POIS_MAPPINGS
    
    @classmethod
    def verify_all_paths(cls) -> dict:
        """
        Verify all configured paths exist
        
        Returns:
            Dictionary with path verification results
        """
        paths_to_check = {
            'main_restaurant_data': cls.MAIN_RESTAURANT_DATA,
            'restaurant_embeddings': cls.RESTAURANT_EMBEDDINGS,
            'fallback_restaurant_data': cls.FALLBACK_RESTAURANT_DATA,
            'all_pois_faiss_index': cls.ALL_POIS_FAISS_INDEX,
            'all_pois_mappings': cls.ALL_POIS_MAPPINGS,
            'restaurant_faiss_index': cls.RESTAURANT_FAISS_INDEX,
            'restaurant_mappings': cls.RESTAURANT_MAPPINGS
        }
        
        results = {}
        for name, path in paths_to_check.items():
            results[name] = {
                'path': path,
                'exists': os.path.exists(path),
                'size_mb': round(os.path.getsize(path) / (1024 * 1024), 1) if os.path.exists(path) else 0
            }
        
        return results
    
    @classmethod
    def print_data_structure(cls):
        """Print current data structure"""
        print("DATA STRUCTURE AFTER CLEANUP")
        print("=" * 50)
        
        verification = cls.verify_all_paths()
        
        print(f"Base Directory: {cls.BASE_DATA_DIR}")
        print(f"")
        print(f"Restaurant Data:")
        print(f"   Main: {verification['main_restaurant_data']['path']} ({verification['main_restaurant_data']['size_mb']}MB)")
        print(f"   Fallback: {verification['fallback_restaurant_data']['path']} ({verification['fallback_restaurant_data']['size_mb']}MB)")
        print(f"   Embeddings: {verification['restaurant_embeddings']['path']} ({verification['restaurant_embeddings']['size_mb']}MB)")
        
        print(f"")
        print(f"FAISS Indices:")
        print(f"   All POIs Index: {verification['all_pois_faiss_index']['path']} ({verification['all_pois_faiss_index']['size_mb']}MB)")
        print(f"   All POIs Mappings: {verification['all_pois_mappings']['path']} ({verification['all_pois_mappings']['size_mb']}MB)")
        print(f"   Restaurant Index: {verification['restaurant_faiss_index']['path']} ({verification['restaurant_faiss_index']['size_mb']}MB)")
        print(f"   Restaurant Mappings: {verification['restaurant_mappings']['path']} ({verification['restaurant_mappings']['size_mb']}MB)")
        
        # Calculate total size
        total_size = sum(item['size_mb'] for item in verification.values())
        print(f"")
        print(f"Total Size: {total_size:.1f}MB")
        
        # Check for any missing files
        missing_files = [name for name, info in verification.items() if not info['exists']]
        if missing_files:
            print(f"")
            print(f"Missing Files: {missing_files}")
        else:
            print(f"")
            print(f"All files verified successfully!")

# Legacy path support for backward compatibility
class LegacyPaths:
    """Legacy paths for backward compatibility"""
    
    # Old paths before cleanup
    OLD_FAISS_DB = "modules/recommendation/faiss_db"
    OLD_FAISS_DB_RESTAURANTS = "modules/recommendation/faiss_db_restaurants"
    OLD_DATA_DIR = "modules/recommendation/data"
    
    @classmethod
    def get_legacy_path_mapping(cls) -> dict:
        """Get mapping from old paths to new paths"""
        return {
            # FAISS DB mappings
            os.path.join(cls.OLD_FAISS_DB, "faiss_index.bin"): DataConfig.ALL_POIS_FAISS_INDEX,
            os.path.join(cls.OLD_FAISS_DB, "mappings.pkl"): DataConfig.ALL_POIS_MAPPINGS,
            os.path.join(cls.OLD_FAISS_DB_RESTAURANTS, "faiss_index.bin"): DataConfig.RESTAURANT_FAISS_INDEX,
            os.path.join(cls.OLD_FAISS_DB_RESTAURANTS, "mappings.pkl"): DataConfig.RESTAURANT_MAPPINGS,
            
            # Data file mappings
            os.path.join(cls.OLD_DATA_DIR, "pois_with_improved_cuisines.parquet"): DataConfig.MAIN_RESTAURANT_DATA,
            os.path.join(cls.OLD_DATA_DIR, "poi_embeddings_structured.parquet"): DataConfig.RESTAURANT_EMBEDDINGS
        }

def demo_data_config():
    """Demo data configuration"""
    print("DATA CONFIGURATION DEMO")
    print("=" * 40)
    
    # Print current structure
    DataConfig.print_data_structure()
    
    # Test path retrieval
    print(f"\nTESTING PATH RETRIEVAL:")
    try:
        restaurant_data = DataConfig.get_restaurant_data_path()
        print(f"   Restaurant data: {restaurant_data}")
        
        faiss_index, mappings = DataConfig.get_faiss_index_paths(restaurant_only=True)
        print(f"   FAISS index: {faiss_index}")
        print(f"   Mappings: {mappings}")
        
        print(f"   All paths retrieved successfully!")
        
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    demo_data_config() 