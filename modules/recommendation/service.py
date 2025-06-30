# modules/recommendation/service.py
from typing import List, Dict, Optional
from modules.memory.short_term import SessionStore
from .recommendation_pipeline_optimized import OptimizedTravelRecommendationPipeline
import logging
from datetime import datetime
import time
import os


class RecommendationService:
    """
    Recommendation Service: quản lý việc tạo và truy cập các recommendation
    thông qua OptimizedTravelRecommendationPipeline và caching kết quả với SessionStore
    """
    
    def __init__(self, memory: SessionStore):
        self.memory = memory
        self.pipeline = None
        self.pipeline_ready = False
        self.logger = logging.getLogger(__name__)
        self.cache = {}  # Cache cho kết quả recommendation
    
    def initialize_pipeline(self):
        """Khởi tạo pipeline nếu chưa được khởi tạo"""
        if self.pipeline is None:
            self.logger.info("🚀 Đang khởi tạo Optimized Recommendation Pipeline với Local LoRA...")
            start_time = time.time()
            
            try:
                # Sử dụng OptimizedTravelRecommendationPipeline
                self.pipeline = OptimizedTravelRecommendationPipeline(
                    lora_path="modules/recommendation/BGE-M3_embedding",
                    faiss_db_path="modules/recommendation/faiss_db_restaurants",
                    device="auto",
                    enable_mixed_precision=True,
                    batch_size=32
                )
                
                self.pipeline_ready = True
                load_time = time.time() - start_time
                self.logger.info(f"✅ Optimized Pipeline với Local LoRA đã khởi tạo thành công sau {load_time:.2f}s")
            except Exception as e:
                self.logger.error(f"❌ Lỗi khởi tạo optimized pipeline: {e}")
                raise
        return self.pipeline
    
    def get_recommendations(self, 
                           user_query: str, 
                           num_candidates: int = 50, 
                           num_results: int = 10) -> Dict:
        """
        Lấy recommendations dựa trên user query
        Sử dụng cache nếu có
        """
        # Tạo cache key
        cache_key = f"query:{user_query}:candidates:{num_candidates}:results:{num_results}"
        
        # Kiểm tra cache trước
        if cache_key in self.cache:
            self.logger.info(f"🔄 Sử dụng cached result cho query: {user_query}")
            return self.cache[cache_key]
            
        # Ensure pipeline is initialized
        self.initialize_pipeline()
        
        # Get recommendations
        start_time = time.time()
        results = self.pipeline.get_recommendations(
            user_query=user_query,
            num_candidates=num_candidates,
            num_final_results=num_results
        )
        
        process_time = time.time() - start_time
        self.logger.info(f"✅ Recommendation cho '{user_query}' hoàn thành trong {process_time:.2f}s")
        
        # Cache kết quả
        self.cache[cache_key] = results
        
        return results
    
    def get_pipeline_info(self) -> Dict:
        """Lấy thông tin về pipeline components"""
        self.initialize_pipeline()
        return self.pipeline.get_service_status()