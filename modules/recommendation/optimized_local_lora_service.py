#!/usr/bin/env python3
"""
🚀 OPTIMIZED LOCAL LORA BGE-M3 SERVICE (Memory Optimized)
Service tối ưu cho Local LoRA Adapter loading với memory constraints

Đặc điểm:
- CHỈ load từ local LoRA checkpoint
- KHÔNG download base model từ BAAI/bge-m3
- Memory optimization với batch_size=16, max_sequence_length=256
- Increased memory threshold 3500MB
- Proper CUDA OOM exception handling
- Tích hợp FAISS vector database
- Support cả CPU và GPU với memory monitoring
"""

import os
import sys
import time
import logging
import torch
import numpy as np
import psutil
import gc
from typing import List, Dict, Union, Optional, Tuple
import json
from pathlib import Path
from dataclasses import dataclass

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .model_config import ModelConfig

@dataclass
class OptimizedConfig:
    """Optimized configuration for memory efficiency"""
    # Memory settings
    max_memory_threshold_mb: int = ModelConfig.MAX_MEMORY_MB  # Use centralized config
    batch_size: int = 16  # Reduced from 32
    max_sequence_length: int = 256  # Reduced from 384
    
    # Model settings
    enable_fp16: bool = True
    use_fast_tokenizer: bool = True
    enable_mixed_precision: bool = True
    
    # Cache settings
    max_cache_size: int = 100
    cache_ttl_seconds: int = 300
    
    # CUDA settings
    cuda_memory_fraction: float = 0.8
    enable_cuda_optimization: bool = True

class OptimizedLocalLoRAService:
    """
    Service tối ưu cho Local LoRA BGE-M3 embedding với memory optimization
    """
    
    def __init__(self, 
                 lora_path: str = "modules/recommendation/BGE-M3_embedding",
                 faiss_db_path: str = "modules/recommendation/data/indices/restaurants",
                 device: str = "auto",
                 config: Optional[OptimizedConfig] = None):
        """
        Khởi tạo Optimized Local LoRA Service với memory constraints
        """
        self.logger = self._setup_logger()
        self.config = config or OptimizedConfig()
        self.lora_path = os.path.abspath(lora_path)
        self.faiss_db_path = faiss_db_path
        
        # Initialize device với memory checks
        self.device = self._determine_device_optimized(device)
        self.logger.info(f"Using device: {self.device}")
        
        # Memory monitoring
        self.memory_monitor = self._init_memory_monitor()
        
        # Verify LoRA checkpoint
        self._verify_local_checkpoint()
        
        # Initialize components với memory safety
        self.model = None
        self.vector_db = None
        self._load_components_safe()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("OptimizedLoRA")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _init_memory_monitor(self) -> Dict:
        """Initialize memory monitoring"""
        return {
            'initial_usage': self._get_memory_usage_mb(),
            'peak_usage': 0,
            'threshold_mb': self.config.max_memory_threshold_mb
        }
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _check_memory_available(self) -> bool:
        """Check if memory is within limits"""
        current_usage = self._get_memory_usage_mb()
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        
        # Update peak usage
        self.memory_monitor['peak_usage'] = max(self.memory_monitor['peak_usage'], current_usage)
        
        if current_usage > self.config.max_memory_threshold_mb:
            self.logger.warning(f"Memory usage {current_usage:.1f}MB exceeds threshold {self.config.max_memory_threshold_mb}MB")
            return False
        
        if available_memory < 500:  # Less than 500MB available
            self.logger.warning(f"Low system memory: {available_memory:.1f}MB available")
            return False
        
        return True
    
    def _determine_device_optimized(self, device: str) -> str:
        """Xác định device tối ưu với memory checks"""
        if device == "auto":
            if torch.cuda.is_available():
                try:
                    # Check CUDA memory
                    cuda_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if cuda_memory_gb >= 4.0:  # Cần ít nhất 4GB VRAM
                        # Set CUDA memory fraction
                        if self.config.enable_cuda_optimization:
                            torch.cuda.set_per_process_memory_fraction(self.config.cuda_memory_fraction)
                        return "cuda"
                    else:
                        self.logger.warning(f"CUDA available but only {cuda_memory_gb:.1f}GB VRAM, using CPU")
                        return "cpu"
                except Exception as e:
                    self.logger.warning(f"CUDA check failed: {e}, using CPU")
                    return "cpu"
            else:
                return "cpu"
        return device
    
    def _verify_local_checkpoint(self):
        """Verify Local LoRA checkpoint integrity"""
        if not os.path.exists(self.lora_path):
            raise FileNotFoundError(f"❌ LoRA path không tồn tại: {self.lora_path}")
        
        required_files = [
            "adapter_config.json",
            "adapter_model.bin", 
            "tokenizer.json",
            "sentence_bert_config.json"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(self.lora_path, file)):
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(f"❌ Missing LoRA files: {missing_files}")
        
        self.logger.info(f"✅ LoRA checkpoint verified: {self.lora_path}")
    
    def _load_components_safe(self):
        """Load components với memory safety"""
        try:
            # Check memory before loading
            if not self._check_memory_available():
                raise MemoryError("Insufficient memory for component loading")
            
            # Load SentenceTransformers model
            self._load_sentence_transformer_optimized()
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load FAISS database
            self._load_vector_database_optimized()
            
        except Exception as e:
            self.logger.error(f"❌ Component loading failed: {e}")
            self._cleanup_on_error()
            raise
    
    def _load_sentence_transformer_optimized(self):
        """Load SentenceTransformers model với memory optimization"""
        self.logger.info("🔄 Loading SentenceTransformers model (optimized)...")
        
        try:
            # Set offline mode
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            from sentence_transformers import SentenceTransformer
            
            start_time = time.time()
            
            # Memory check before loading
            if not self._check_memory_available():
                raise MemoryError("Insufficient memory for model loading")
            
            # Load model từ local path với optimization settings
            self.model = SentenceTransformer(self.lora_path, device=self.device)
            
            # Apply optimizations
            self.model.eval()  # Set to evaluation mode
            
            # Set max sequence length
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.config.max_sequence_length
            
            # Mixed precision cho GPU
            if self.device == 'cuda' and self.config.enable_mixed_precision:
                try:
                    self.model.half()
                    self.logger.info("✅ Enabled FP16 mixed precision")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not enable FP16: {e}")
            
            load_time = time.time() - start_time
            
            self.logger.info(f"✅ Model loaded successfully!")
            self.logger.info(f"   Load time: {load_time:.3f}s")
            self.logger.info(f"   Device: {self.device}")
            self.logger.info(f"   Max sequence length: {self.config.max_sequence_length}")
            self.logger.info(f"   Batch size: {self.config.batch_size}")
            self.logger.info(f"   Mixed precision: {self.config.enable_mixed_precision and self.device == 'cuda'}")
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"🔥 CUDA OOM Error during model loading: {e}")
            self._handle_cuda_oom()
            raise
        except Exception as e:
            self.logger.error(f"❌ Error loading SentenceTransformers: {e}")
            raise
    
    def _handle_cuda_oom(self):
        """Handle CUDA out of memory errors"""
        self.logger.warning("🔄 Handling CUDA OOM - clearing cache and switching to CPU")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Switch to CPU
        self.device = "cpu"
        self.logger.info("🔄 Switched to CPU mode due to CUDA OOM")
    
    def _load_vector_database_optimized(self):
        """Load FAISS vector database với memory optimization"""
        try:
            from modules.recommendation.faiss_vectorDB import FAISSVectorDB
            
            # Check memory before loading
            if not self._check_memory_available():
                self.logger.warning("⚠️ Low memory, skipping vector database")
                return
            
            # Find FAISS database path
            possible_paths = [
                self.faiss_db_path,
                os.path.join(os.getcwd(), self.faiss_db_path),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), self.faiss_db_path)
            ]
            
            valid_path = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, "faiss_index.bin")):
                    valid_path = path
                    break
            
            if not valid_path:
                raise FileNotFoundError(f"❌ FAISS database not found: {self.faiss_db_path}")
            
            self.vector_db = FAISSVectorDB()
            self.vector_db.load_database(valid_path)
            
            self.logger.info(f"✅ FAISS database loaded: {self.vector_db.index.ntotal} vectors")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading FAISS database: {e}")
            self.vector_db = None  # Continue without vector DB
    
    def _cleanup_on_error(self):
        """Cleanup on error"""
        self.model = None
        self.vector_db = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def encode_texts(self, 
                    texts: Union[str, List[str]], 
                    batch_size: Optional[int] = None,
                    normalize: bool = True,
                    show_progress: bool = False) -> np.ndarray:
        """
        Encode texts thành embeddings với memory optimization
        """
        if self.model is None:
            raise ValueError("❌ Model chưa được load!")
        
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = batch_size or self.config.batch_size
        
        try:
            # Memory check before processing
            if not self._check_memory_available():
                self.logger.warning("⚠️ Low memory detected, reducing batch size")
                batch_size = max(1, batch_size // 2)
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize,
                    show_progress_bar=show_progress,
                    convert_to_tensor=False  # Return numpy
                )
            
            return embeddings
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"🔥 CUDA OOM Error during encoding: {e}")
            self._handle_cuda_oom()
            
            # Retry with smaller batch size on CPU
            batch_size = max(1, batch_size // 4)
            self.logger.info(f"🔄 Retrying encoding with batch size {batch_size} on CPU")
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=normalize,
                    show_progress_bar=show_progress,
                    convert_to_tensor=False
                )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Error encoding texts: {e}")
            raise
    
    def search_similar(self, 
                      query: str, 
                      k: int = 10,
                      return_embeddings: bool = False) -> Dict:
        """
        Tìm kiếm POIs tương tự với query - memory optimized
        """
        if self.model is None:
            raise ValueError("❌ Model chưa được load!")
        
        if self.vector_db is None:
            self.logger.warning("⚠️ Vector database not available")
            return {
                'query': query,
                'total_results': 0,
                'poi_results': [],
                'error': 'Vector database not loaded'
            }
        
        try:
            start_time = time.time()
            
            # Encode query với memory check
            query_embedding = self.encode_texts(query, normalize=True)
            encoding_time = time.time() - start_time
            
            # Search trong FAISS
            search_start = time.time()
            results = self.vector_db.search(query_embedding[0], k=k)
            search_time = time.time() - search_start
            
            # Format results
            formatted_results = {
                'query': query,
                'total_results': len(results),
                'encoding_time': encoding_time,
                'search_time': search_time,
                'total_time': encoding_time + search_time,
                'poi_results': results,
                'memory_usage_mb': self._get_memory_usage_mb()
            }
            
            if return_embeddings:
                formatted_results['query_embedding'] = query_embedding[0].tolist()
            
            self.logger.info(f"🔍 Search completed: {len(results)} results in {formatted_results['total_time']:.3f}s")
            
            return formatted_results
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"🔥 CUDA OOM Error during search: {e}")
            self._handle_cuda_oom()
            
            # Retry search
            return self.search_similar(query, k, return_embeddings)
            
        except Exception as e:
            self.logger.error(f"❌ Error searching: {e}")
            raise
    
    def batch_encode_for_indexing(self, 
                                 texts: List[str],
                                 batch_size: Optional[int] = None,
                                 save_path: Optional[str] = None) -> np.ndarray:
        """
        Batch encoding với memory optimization
        """
        batch_size = batch_size or self.config.batch_size
        
        # Reduce batch size for large datasets
        if len(texts) > 1000:
            batch_size = max(8, batch_size // 2)
            self.logger.info(f"🔄 Large dataset detected, reducing batch size to {batch_size}")
        
        self.logger.info(f"🔄 Batch encoding {len(texts)} texts...")
        
        try:
            embeddings = self.encode_texts(
                texts,
                batch_size=batch_size,
                normalize=True,
                show_progress=True
            )
            
            if save_path:
                np.save(save_path, embeddings)
                self.logger.info(f"💾 Embeddings saved to: {save_path}")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Batch encoding error: {e}")
            raise
    
    def get_performance_info(self) -> Dict:
        """Lấy thông tin performance và memory"""
        memory_usage = self._get_memory_usage_mb()
        
        info = {
            "model": {
                "loaded": self.model is not None,
                "device": self.device,
                "mixed_precision": self.config.enable_mixed_precision and self.device == 'cuda',
                "max_seq_length": self.config.max_sequence_length,
                "batch_size": self.config.batch_size
            },
            "vector_db": {
                "loaded": self.vector_db is not None,
                "total_vectors": self.vector_db.index.ntotal if self.vector_db else 0,
                "dimension": self.vector_db.index.d if self.vector_db else 0
            },
            "memory": {
                "current_usage_mb": memory_usage,
                "peak_usage_mb": self.memory_monitor['peak_usage'],
                "threshold_mb": self.config.max_memory_threshold_mb,
                "usage_percentage": (memory_usage / self.config.max_memory_threshold_mb) * 100,
                "memory_ok": self._check_memory_available()
            },
            "config": {
                "lora_path": self.lora_path,
                "optimized_settings": {
                    "batch_size": self.config.batch_size,
                    "max_sequence_length": self.config.max_sequence_length,
                    "memory_threshold_mb": self.config.max_memory_threshold_mb,
                    "fp16_enabled": self.config.enable_fp16
                }
            }
        }
        
        return info
    
    def force_cleanup(self):
        """Force cleanup để free memory"""
        self.logger.info("🧹 Force cleanup starting...")
        
        # Clear model if needed
        if self.model is not None and not self._check_memory_available():
            self.model = None
            self.logger.info("🗑️ Model cleared to free memory")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        current_usage = self._get_memory_usage_mb()
        self.logger.info(f"🧹 Cleanup completed - Memory usage: {current_usage:.1f}MB")
    
    def get_memory_status(self) -> Dict:
        """Get detailed memory status"""
        return {
            'current_usage_mb': self._get_memory_usage_mb(),
            'peak_usage_mb': self.memory_monitor['peak_usage'],
            'threshold_mb': self.config.max_memory_threshold_mb,
            'available_system_mb': psutil.virtual_memory().available / (1024 * 1024),
            'memory_ok': self._check_memory_available(),
            'cuda_available': torch.cuda.is_available(),
            'cuda_memory_allocated': torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        }
    
    def benchmark_encoding_optimized(self, num_texts: int = 50) -> Dict:
        """
        Benchmark encoding performance với memory monitoring
        """
        self.logger.info(f"🏃‍♂️ Running optimized encoding benchmark ({num_texts} texts)...")
        
        # Generate test texts
        test_texts = [
            f"tìm nhà hàng phở ngon số {i} ở Hà Nội" 
            for i in range(num_texts)
        ]
        
        results = {}
        
        # Test different batch sizes (smaller range for optimization)
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            if batch_size > num_texts:
                continue
            
            try:
                start_time = time.time()
                memory_before = self._get_memory_usage_mb()
                
                embeddings = self.encode_texts(
                    test_texts,
                    batch_size=batch_size,
                    normalize=True,
                    show_progress=False
                )
                
                encoding_time = time.time() - start_time
                memory_after = self._get_memory_usage_mb()
                texts_per_second = num_texts / encoding_time
                
                results[f"batch_{batch_size}"] = {
                    "encoding_time": encoding_time,
                    "texts_per_second": texts_per_second,
                    "batch_size": batch_size,
                    "memory_used_mb": memory_after - memory_before,
                    "memory_ok": self._check_memory_available()
                }
                
                self.logger.info(f"   Batch {batch_size}: {texts_per_second:.1f} texts/sec, Memory: {memory_after - memory_before:.1f}MB")
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for batch {batch_size}: {e}")
                results[f"batch_{batch_size}"] = {"error": str(e)}
        
        # Find best batch size
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            best_batch = max(valid_results.items(), key=lambda x: x[1]['texts_per_second'])
            results["best_performance"] = {
                "batch_size": best_batch[1]["batch_size"],
                "texts_per_second": best_batch[1]["texts_per_second"],
                "memory_efficiency": best_batch[1]["memory_used_mb"]
            }
            
            self.logger.info(f"🏆 Best performance: batch_{best_batch[1]['batch_size']} ({best_batch[1]['texts_per_second']:.1f} texts/sec)")
        
        return results


def test_optimized_service():
    """Test function cho Optimized Service với memory monitoring"""
    print("🧪 TESTING OPTIMIZED LOCAL LORA SERVICE (MEMORY OPTIMIZED)")
    print("=" * 60)
    
    try:
        # Create optimized config
        config = OptimizedConfig(
            max_memory_threshold_mb=3500,
            batch_size=16,
            max_sequence_length=256,
            enable_mixed_precision=True
        )
        
        # Initialize service
        service = OptimizedLocalLoRAService(config=config)
        
        # Get performance info
        info = service.get_performance_info()
        print(f"\n📊 Service Info:")
        print(f"   Model loaded: {info['model']['loaded']}")
        print(f"   Device: {info['model']['device']}")
        print(f"   Mixed precision: {info['model']['mixed_precision']}")
        print(f"   Batch size: {info['model']['batch_size']}")
        print(f"   Max seq length: {info['model']['max_seq_length']}")
        print(f"   Vector DB: {info['vector_db']['total_vectors']} vectors")
        
        # Memory status
        memory_status = service.get_memory_status()
        print(f"\n💾 Memory Status:")
        print(f"   Current usage: {memory_status['current_usage_mb']:.1f}MB")
        print(f"   Threshold: {memory_status['threshold_mb']}MB")
        print(f"   Memory OK: {memory_status['memory_ok']}")
        print(f"   System available: {memory_status['available_system_mb']:.1f}MB")
        
        # Test encoding
        test_text = "tìm nhà hàng phở ngon ở Hà Nội"
        print(f"\n🔤 Test Encoding: '{test_text}'")
        
        start_time = time.time()
        embedding = service.encode_texts(test_text)
        encoding_time = time.time() - start_time
        
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Encoding time: {encoding_time:.3f}s")
        
        # Test search
        print(f"\n🔍 Test Search:")
        results = service.search_similar(test_text, k=3)
        
        print(f"   Results found: {results['total_results']}")
        print(f"   Total time: {results['total_time']:.3f}s")
        print(f"   Memory usage: {results['memory_usage_mb']:.1f}MB")
        
        for i, poi in enumerate(results['poi_results'][:3]):
            metadata = poi.get('metadata', {})
            print(f"   {i+1}. {metadata.get('name', 'N/A')} - {metadata.get('city', 'N/A')}")
            print(f"      Score: {poi.get('similarity_score', 0):.3f}")
        
        # Optimized benchmark
        print(f"\n🏃‍♂️ Running optimized benchmark...")
        benchmark = service.benchmark_encoding_optimized(num_texts=30)
        
        if "best_performance" in benchmark:
            best = benchmark["best_performance"]
            print(f"   Best: batch_{best['batch_size']} ({best['texts_per_second']:.1f} texts/sec)")
            print(f"   Memory efficiency: {best['memory_efficiency']:.1f}MB")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_optimized_service() 