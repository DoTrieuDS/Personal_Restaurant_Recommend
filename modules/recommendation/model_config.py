"""
Model Configuration - Centralized Model Paths and Settings
Quản lý đường dẫn và cấu hình cho các ML models
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelPaths:
    """Centralized model paths"""
    # BGE-M3 paths
    BGE_M3_LOCAL_LORA = "modules/recommendation/BGE-M3_embedding"
    BGE_M3_BASE_MODEL = "BAAI/bge-m3"
    
    # T5 model paths
    T5_RERANKING_CHECKPOINT = "modules/recommendation/T5_reranking/t5_rerank_base_final"
    T5_MESSAGE_GEN_CHECKPOINT = "modules/recommendation/T5_personalized_message_generation/model"
    
    # TF-IDF paths
    TFIDF_MODEL_PATH = "modules/recommendation/models/tfidf_vectorizer.pkl"
    TFIDF_VOCAB_PATH = "modules/recommendation/models/tfidf_vocabulary.json"
    
    # Collaborative Filtering paths
    CF_EMBEDDINGS_PATH = "modules/recommendation/models/cf_embeddings.npz"
    CF_MAPPINGS_PATH = "modules/recommendation/models/cf_mappings.npz"
    CF_SUMMARY_PATH = "modules/recommendation/models/cf_summary.json"


@dataclass
class ModelConfig:
    """Configuration for model optimization và memory management"""
    
    # Memory constraints
    MAX_MEMORY_MB = 6000  # Increased to allow T5 models to load
    MEMORY_WARNING_THRESHOLD = 0.8  # Warn at 80% usage
    
    # BGE-M3 settings
    BGE_M3_BATCH_SIZE = 16
    BGE_M3_MAX_SEQ_LENGTH = 256
    BGE_M3_ENABLE_FP16 = True
    BGE_M3_CACHE_SIZE = 1000
    
    # TF-IDF settings
    TFIDF_MAX_FEATURES = 5000
    TFIDF_MIN_DF = 2
    TFIDF_MAX_DF = 0.8
    TFIDF_USE_IDF = True
    TFIDF_SUBLINEAR_TF = True
    
    # T5 Reranking settings
    T5_RERANK_BATCH_SIZE = 8
    T5_RERANK_MAX_LENGTH = 512
    T5_RERANK_TOP_K = 50
    T5_RERANK_ENABLE_FP16 = True
    T5_RERANK_USE_CACHE = True
    
    # T5 Message Generation settings
    T5_MESSAGE_BATCH_SIZE = 4
    T5_MESSAGE_MAX_INPUT_LENGTH = 256
    T5_MESSAGE_MAX_OUTPUT_LENGTH = 128
    T5_MESSAGE_TEMPERATURE = 0.8
    T5_MESSAGE_TOP_P = 0.9
    T5_MESSAGE_ENABLE_FP16 = True
    
    # Lazy loading settings
    LAZY_LOADING_ENABLED = True
    MODEL_TIMEOUT_SECONDS = 300  # Unload models after 5 minutes of inactivity
    
    # Cache settings
    EMBEDDING_CACHE_SIZE = 10000
    RERANKING_CACHE_SIZE = 5000
    MESSAGE_CACHE_SIZE = 1000
    
    # Collaborative Filtering settings
    CF_ENABLE = True
    CF_MAX_USERS_CACHE = 5000  # Cache top active users
    CF_MAX_ITEMS_CACHE = 10000  # Cache top items
    CF_USE_FLOAT16 = True  # Use float16 to save memory
    CF_SIMILARITY_THRESHOLD = 0.1  # Min similarity score


class ModelManager:
    """
    Centralized model management với lazy loading và memory optimization
    """
    
    def __init__(self):
        self.paths = ModelPaths()
        self.config = ModelConfig()
        self._loaded_models = {}
        self._model_last_access = {}
    
    def verify_model_paths(self) -> Dict[str, bool]:
        """Verify tất cả model paths exist"""
        verification = {
            'bge_m3_lora': os.path.exists(self.paths.BGE_M3_LOCAL_LORA),
            't5_reranking': os.path.exists(self.paths.T5_RERANKING_CHECKPOINT),
            't5_message_gen': os.path.exists(self.paths.T5_MESSAGE_GEN_CHECKPOINT),
            'tfidf_model': os.path.exists(self.paths.TFIDF_MODEL_PATH),
            'cf_embeddings': os.path.exists(self.paths.CF_EMBEDDINGS_PATH),
            'cf_mappings': os.path.exists(self.paths.CF_MAPPINGS_PATH),
            'cf_summary': os.path.exists(self.paths.CF_SUMMARY_PATH),
        }
        return verification
    
    def get_absolute_path(self, model_name: str) -> str:
        """Get absolute path cho model"""
        path_map = {
            'bge_m3': self.paths.BGE_M3_LOCAL_LORA,
            't5_rerank': self.paths.T5_RERANKING_CHECKPOINT,
            't5_message': self.paths.T5_MESSAGE_GEN_CHECKPOINT,
            'tfidf': self.paths.TFIDF_MODEL_PATH,
            'cf_embeddings': self.paths.CF_EMBEDDINGS_PATH,
            'cf_mappings': self.paths.CF_MAPPINGS_PATH,
            'cf_summary': self.paths.CF_SUMMARY_PATH
        }
        
        relative_path = path_map.get(model_name)
        if relative_path:
            return os.path.abspath(relative_path)
        return None
    
    def get_model_config(self, model_name: str) -> Dict:
        """Get configuration cho specific model"""
        if model_name == 'bge_m3':
            return {
                'batch_size': self.config.BGE_M3_BATCH_SIZE,
                'max_seq_length': self.config.BGE_M3_MAX_SEQ_LENGTH,
                'enable_fp16': self.config.BGE_M3_ENABLE_FP16,
                'cache_size': self.config.BGE_M3_CACHE_SIZE
            }
        elif model_name == 't5_rerank':
            return {
                'batch_size': self.config.T5_RERANK_BATCH_SIZE,
                'max_length': self.config.T5_RERANK_MAX_LENGTH,
                'top_k': self.config.T5_RERANK_TOP_K,
                'enable_fp16': self.config.T5_RERANK_ENABLE_FP16,
                'use_cache': self.config.T5_RERANK_USE_CACHE
            }
        elif model_name == 't5_message':
            return {
                'batch_size': self.config.T5_MESSAGE_BATCH_SIZE,
                'max_input_length': self.config.T5_MESSAGE_MAX_INPUT_LENGTH,
                'max_output_length': self.config.T5_MESSAGE_MAX_OUTPUT_LENGTH,
                'temperature': self.config.T5_MESSAGE_TEMPERATURE,
                'top_p': self.config.T5_MESSAGE_TOP_P,
                'enable_fp16': self.config.T5_MESSAGE_ENABLE_FP16
            }
        elif model_name == 'tfidf':
            return {
                'max_features': self.config.TFIDF_MAX_FEATURES,
                'min_df': self.config.TFIDF_MIN_DF,
                'max_df': self.config.TFIDF_MAX_DF,
                'use_idf': self.config.TFIDF_USE_IDF,
                'sublinear_tf': self.config.TFIDF_SUBLINEAR_TF
            }
        elif model_name == 'collaborative_filtering':
            return {
                'enable': self.config.CF_ENABLE,
                'max_users_cache': self.config.CF_MAX_USERS_CACHE,
                'max_items_cache': self.config.CF_MAX_ITEMS_CACHE,
                'use_float16': self.config.CF_USE_FLOAT16,
                'similarity_threshold': self.config.CF_SIMILARITY_THRESHOLD
            }
        return {}


# Global instance
model_manager = ModelManager()


def verify_all_models():
    """Verify all model paths và print status"""
    print("VERIFYING MODEL PATHS")
    print("=" * 50)
    
    verification = model_manager.verify_model_paths()
    
    for model, exists in verification.items():
        status = "OK" if exists else "MISSING"
        path = model_manager.get_absolute_path(model.replace('_model', '').replace('_lora', ''))
        print(f"[{status}] {model}: {path}")
    
    print("\nMODEL CONFIGURATIONS")
    print("=" * 50)
    
    for model in ['bge_m3', 't5_rerank', 't5_message', 'tfidf']:
        config = model_manager.get_model_config(model)
        print(f"\n{model.upper()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    return all(verification.values())


if __name__ == "__main__":
    all_models_exist = verify_all_models()
    print(f"\nAll models verified: {'OK' if all_models_exist else 'FAIL'}") 