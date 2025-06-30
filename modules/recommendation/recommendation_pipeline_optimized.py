"""
Optimized Travel Recommendation Pipeline
Sử dụng OptimizedLocalLoRAService với Local LoRA checkpoint
"""

import logging
from typing import List, Dict, Optional, Union
import time
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import optimized service
from .optimized_local_lora_service import OptimizedLocalLoRAService

class OptimizedTravelRecommendationPipeline:
    """
    Optimized Pipeline cho Travel Recommendation:
    User Query → Optimized BGE-M3 Retrieval → Mock Reranking → Final Recommendations
    
    Đặc điểm:
    - Sử dụng OptimizedLocalLoRAService
    - Hoạt động hoàn toàn với Local LoRA checkpoint
    - Không cần download base model
    - Tối ưu performance và memory
    """
    
    def __init__(self, 
                 lora_path: str = "modules/recommendation/BGE-M3_embedding",
                 faiss_db_path: str = "modules/recommendation/faiss_db_restaurants",
                 device: str = "auto",
                 enable_mixed_precision: bool = True,
                 batch_size: int = 32,
                 llama_model_path: Optional[str] = None):
        """
        Khởi tạo Optimized Recommendation Pipeline
        
        Args:
            lora_path: Đường dẫn Local LoRA checkpoint
            faiss_db_path: Đường dẫn FAISS database
            device: Device ('auto', 'cpu', 'cuda')
            enable_mixed_precision: Enable FP16 trên GPU
            batch_size: Batch size tối ưu
            llama_model_path: Đường dẫn Llama-3 model (optional)
        """
        self.logger = self._setup_logger()
        
        # Initialize Optimized BGE-M3 Embedding Service
        self.logger.info("🚀 Initializing Optimized Local LoRA BGE-M3 Service...")
        
        try:
            self.embedding_service = OptimizedLocalLoRAService(
                lora_path=lora_path,
                faiss_db_path=faiss_db_path,
                device=device,
                enable_mixed_precision=enable_mixed_precision,
                batch_size=batch_size
            )
            
            # Get service info
            service_info = self.embedding_service.get_performance_info()
            self.logger.info(f"✅ Embedding service initialized:")
            self.logger.info(f"   Device: {service_info['model']['device']}")
            self.logger.info(f"   Mixed precision: {service_info['model']['mixed_precision']}")
            self.logger.info(f"   Vector DB: {service_info['vector_db']['total_vectors']} vectors")
            self.logger.info(f"   Batch size: {service_info['config']['default_batch_size']}")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing embedding service: {e}")
            raise
        
        # Initialize Llama-3 Reranking Service (placeholder)
        self.llama_model_path = llama_model_path
        self.reranking_service = None  # Sẽ implement sau
        
        self.logger.info("✅ Optimized Recommendation Pipeline initialized successfully!")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("OptimizedPipeline")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def get_initial_candidates(self, 
                             user_query: str, 
                             candidate_count: int = 50) -> Dict:
        """
        Bước 1: Lấy initial candidates từ Optimized BGE-M3 + FAISS
        
        Args:
            user_query: Query từ user
            candidate_count: Số lượng candidates để lấy
            
        Returns:
            Dictionary với candidates và metadata
        """
        try:
            start_time = time.time()
            
            # Sử dụng optimized service để search
            search_results = self.embedding_service.search_similar(
                query=user_query,
                k=candidate_count,
                return_embeddings=True
            )
            
            retrieval_time = time.time() - start_time
            
            # Format candidates cho reranking phase
            candidates = {
                'user_query': user_query,
                'query_embedding': search_results.get('query_embedding', []),
                'candidate_count': search_results['total_results'],
                'candidates': []
            }
            
            # Convert POI results to candidate format
            for poi in search_results['poi_results']:
                candidate = {
                    'business_id': poi.get('business_id', ''),
                    'ranking_score': poi.get('similarity_score', 0.0),  # BGE-M3 similarity
                    'poi_info': {
                        'name': poi.get('metadata', {}).get('name', ''),
                        'city': poi.get('metadata', {}).get('city', ''),
                        'state': poi.get('metadata', {}).get('state', ''),
                        'stars': poi.get('metadata', {}).get('stars', 0),
                        'review_count': poi.get('metadata', {}).get('review_count', 0),
                        'categories': poi.get('metadata', {}).get('categories', ''),
                        'poi_type': poi.get('metadata', {}).get('poi_type', ''),
                        'description': poi.get('metadata', {}).get('cleaned_description', ''),
                        'coordinates': {
                            'latitude': poi.get('metadata', {}).get('latitude', 0),
                            'longitude': poi.get('metadata', {}).get('longitude', 0)
                        }
                    }
                }
                candidates['candidates'].append(candidate)
            
            # Add timing info
            candidates['retrieval_info'] = {
                'method': 'Optimized BGE-M3 + FAISS',
                'retrieval_time': retrieval_time,
                'encoding_time': search_results.get('encoding_time', 0),
                'search_time': search_results.get('search_time', 0),
                'total_time': search_results.get('total_time', 0),
                'timestamp': time.time()
            }
            
            self.logger.info(f"🔍 Retrieved {candidates['candidate_count']} candidates in {retrieval_time:.3f}s")
            self.logger.info(f"   Encoding: {search_results.get('encoding_time', 0):.3f}s")
            self.logger.info(f"   Search: {search_results.get('search_time', 0):.3f}s")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"❌ Error in retrieval phase: {e}")
            raise
    
    def rerank_candidates(self, candidates: Dict) -> Dict:
        """
        Bước 2: Rerank candidates sử dụng Llama-3 (mock implementation)
        
        Args:
            candidates: Candidates từ retrieval phase
            
        Returns:
            Reranked candidates với scores mới
        """
        try:
            start_time = time.time()
            
            user_query = candidates['user_query']
            candidate_list = candidates['candidates']
            
            # Mock reranking logic (sẽ thay bằng Llama-3)
            reranked_candidates = self._mock_llama_reranking(user_query, candidate_list)
            
            reranking_time = time.time() - start_time
            
            # Format kết quả
            reranked_results = {
                'user_query': user_query,
                'reranking_info': {
                    'method': 'Llama-3 (Mock)',
                    'reranking_time': reranking_time,
                    'timestamp': time.time(),
                    'candidates_processed': len(candidate_list)
                },
                'retrieval_info': candidates.get('retrieval_info', {}),
                'final_candidates': reranked_candidates
            }
            
            self.logger.info(f"🎯 Reranked {len(candidate_list)} candidates in {reranking_time:.3f}s")
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in reranking phase: {e}")
            raise
    
    def _mock_llama_reranking(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Mock implementation cho Llama-3 reranking
        TODO: Replace với real Llama-3 implementation
        """
        import re
        
        # Simple scoring based on query keywords
        query_lower = query.lower()
        query_keywords = re.findall(r'\w+', query_lower)
        
        for candidate in candidates:
            poi_info = candidate['poi_info']
            
            # Calculate relevance score based on text matching
            text_fields = [
                poi_info.get('name', '').lower(),
                poi_info.get('categories', '').lower(),
                poi_info.get('description', '').lower()
            ]
            
            text_content = ' '.join(text_fields)
            
            # Count keyword matches
            keyword_score = 0
            for keyword in query_keywords:
                if keyword in text_content:
                    keyword_score += 1
            
            # Normalize keyword score
            keyword_score = keyword_score / len(query_keywords) if query_keywords else 0
            
            # Combine with original BGE-M3 score
            original_score = candidate['ranking_score']
            
            # Weight: 70% BGE-M3, 30% keyword matching
            final_score = 0.7 * original_score + 0.3 * keyword_score
            
            # Add quality factors
            stars = poi_info.get('stars', 0)
            review_count = poi_info.get('review_count', 0)
            
            # Boost score for high-rated places
            if stars >= 4.0:
                final_score *= 1.1
            if review_count >= 100:
                final_score *= 1.05
            
            # Update candidate
            candidate['llama_score'] = final_score
            candidate['keyword_score'] = keyword_score
            candidate['quality_boost'] = {
                'stars': stars,
                'review_count': review_count
            }
        
        # Sort by final score
        candidates.sort(key=lambda x: x['llama_score'], reverse=True)
        
        return candidates
    
    def get_recommendations(self, 
                          user_query: str,
                          num_candidates: int = 50,
                          num_final_results: int = 10) -> Dict:
        """
        Main method: Lấy recommendations hoàn chỉnh
        
        Args:
            user_query: Query từ user
            num_candidates: Số candidates từ retrieval
            num_final_results: Số kết quả cuối cùng
            
        Returns:
            Dictionary với recommendations và metadata
        """
        try:
            pipeline_start = time.time()
            
            self.logger.info(f"🎯 Starting recommendation pipeline for: '{user_query}'")
            
            # Step 1: Get initial candidates
            candidates = self.get_initial_candidates(user_query, num_candidates)
            
            # Step 2: Rerank candidates
            reranked_results = self.rerank_candidates(candidates)
            
            # Step 3: Format final results
            final_candidates = reranked_results['final_candidates'][:num_final_results]
            
            pipeline_time = time.time() - pipeline_start
            
            # Compile final results
            recommendations = {
                'query': user_query,
                'total_pipeline_time': pipeline_time,
                'pipeline_info': {
                    'retrieval': reranked_results.get('retrieval_info', {}),
                    'reranking': reranked_results.get('reranking_info', {}),
                    'num_candidates': len(candidates.get('candidates', [])),
                    'num_final_results': len(final_candidates)
                },
                'recommendations': final_candidates
            }
            
            self.logger.info(f"✅ Pipeline completed in {pipeline_time:.3f}s")
            self.logger.info(f"   Retrieved: {candidates.get('candidate_count', 0)} candidates")
            self.logger.info(f"   Final results: {len(final_candidates)}")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error in recommendation pipeline: {e}")
            raise
    
    def get_service_status(self) -> Dict:
        """
        Lấy trạng thái của các services
        """
        try:
            embedding_info = self.embedding_service.get_performance_info()
            
            status = {
                'embedding_service': {
                    'status': 'active',
                    'model_loaded': embedding_info['model']['loaded'],
                    'device': embedding_info['model']['device'],
                    'vector_db_loaded': embedding_info['vector_db']['loaded'],
                    'total_vectors': embedding_info['vector_db']['total_vectors']
                },
                'reranking_service': {
                    'status': 'mock',
                    'model_loaded': False,
                    'note': 'Using mock Llama-3 implementation'
                },
                'pipeline_ready': True
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ Error getting service status: {e}")
            return {
                'embedding_service': {'status': 'error'},
                'reranking_service': {'status': 'error'},
                'pipeline_ready': False,
                'error': str(e)
            }

def test_optimized_pipeline():
    """Test function cho Optimized Pipeline"""
    print("🧪 TESTING OPTIMIZED TRAVEL RECOMMENDATION PIPELINE")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = OptimizedTravelRecommendationPipeline()
        
        # Get service status
        status = pipeline.get_service_status()
        print(f"\n📊 Service Status:")
        print(f"   Embedding service: {status['embedding_service']['status']}")
        print(f"   Model loaded: {status['embedding_service']['model_loaded']}")
        print(f"   Device: {status['embedding_service']['device']}")
        print(f"   Vector DB: {status['embedding_service']['total_vectors']} vectors")
        print(f"   Pipeline ready: {status['pipeline_ready']}")
        
        # Test recommendations
        test_queries = [
            "nhà hàng phở ngon ở Hà Nội",
            "quán cà phê view đẹp",
            "khách sạn 4 sao gần biển"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            recommendations = pipeline.get_recommendations(
                user_query=query,
                num_candidates=20,
                num_final_results=3
            )
            
            print(f"   Total time: {recommendations['total_pipeline_time']:.3f}s")
            print(f"   Candidates: {recommendations['pipeline_info']['num_candidates']}")
            print(f"   Results: {recommendations['pipeline_info']['num_final_results']}")
            
            print(f"   Top results:")
            for i, rec in enumerate(recommendations['recommendations'][:3]):
                poi_info = rec['poi_info']
                print(f"     {i+1}. {poi_info['name']} - {poi_info['city']}")
                print(f"        Score: {rec['llama_score']:.3f} | Stars: {poi_info['stars']}")
        
        print("\n✅ All pipeline tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_optimized_pipeline() 