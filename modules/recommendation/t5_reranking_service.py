"""
T5 Reranking Service for Restaurant Recommendation
Service để rerank restaurants sử dụng fine-tuned T5 model
"""

import os
import time
import torch
import logging
import psutil
from typing import List, Dict, Optional, Tuple, Union
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dataclasses import dataclass
import numpy as np
from collections import OrderedDict
from datetime import datetime
import gc
from .model_config import model_manager, ModelConfig


@dataclass
class RerankCandidate:
    """Candidate cho reranking"""
    business_id: str
    name: str
    categories: str
    initial_score: float
    metadata: Dict = None


class T5RerankingService:
    """
    T5 Reranking Service với lazy loading và memory optimization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = model_manager.get_model_config('t5_rerank')
        self.model_config = ModelConfig()
        
        # Model paths
        self.model_path = model_manager.get_absolute_path('t5_rerank')
        
        # Lazy loading attributes
        self._model = None
        self._tokenizer = None
        self._device = None
        self._is_loaded = False
        
        # Memory monitoring
        self._last_access_time = None
        self._memory_usage_mb = 0
        
        # Cache for reranking results
        self._cache = OrderedDict()
        self._cache_size = self.config.get('cache_size', 1000)
    
    def _check_memory(self) -> bool:
        """Check if enough memory available"""
        process = psutil.Process()
        current_usage = process.memory_info().rss / (1024 * 1024)
        available = psutil.virtual_memory().available / (1024 * 1024)
        
        # More conservative for T5 model
        if current_usage > self.model_config.MAX_MEMORY_MB * 0.8:
            self.logger.warning(f"High memory usage: {current_usage:.1f}MB")
            return False
        
        if available < 1000:  # Need at least 1GB available
            self.logger.warning(f"Low system memory: {available:.1f}MB")
            return False
        
        return True
    
    def _determine_device(self) -> str:
        """Determine best device cho model"""
        if torch.cuda.is_available():
            # Check CUDA memory
            cuda_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if cuda_memory >= 4.0:  # Need at least 4GB VRAM
                return "cuda"
            else:
                self.logger.warning(f"CUDA available but only {cuda_memory:.1f}GB VRAM, using CPU")
        return "cpu"
    
    def _load_model(self):
        """Lazy load T5 model"""
        if self._model is not None:
            self._last_access_time = time.time()
            return
        
        # Check memory
        if not self._check_memory():
            raise MemoryError("Insufficient memory to load T5 model")
        
        self.logger.info("Loading T5 reranking model...")
        start_time = time.time()
        
        try:
            # Determine device
            self._device = self._determine_device()
            self.logger.info(f"Using device: {self._device}")
            
            # Load tokenizer
            self._tokenizer = T5Tokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            # Load model with optimization
            self._model = T5ForConditionalGeneration.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype=torch.float16 if self.config['enable_fp16'] and self._device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move to device
            self._model = self._model.to(self._device)
            self._model.eval()
            
            # Disable gradient computation
            for param in self._model.parameters():
                param.requires_grad = False
            
            self._is_loaded = True
            self._last_access_time = time.time()
            
            load_time = time.time() - start_time
            self._estimate_memory_usage()
            
            self.logger.info(f"T5 model loaded in {load_time:.2f}s")
            self.logger.info(f"   Device: {self._device}")
            self.logger.info(f"   Memory usage: {self._memory_usage_mb:.1f}MB")
            
        except Exception as e:
            self.logger.error(f"Error loading T5 model: {e}")
            self._cleanup()
            raise
    
    def rerank_restaurants(self, 
                          query: str, 
                          candidates: List[RerankCandidate],
                          top_k: Optional[int] = None) -> List[Dict]:
        """
        Rerank restaurants using T5
        
        Args:
            query: User query
            candidates: List of candidates to rerank
            top_k: Return only top-k results
            
        Returns:
            Reranked candidates with scores
        """
        try:
            # Load model if needed
            self._load_model()
            
            if not candidates:
                return []
            
            # Limit candidates to process
            max_candidates = self.config.get('top_k', 50)
            candidates_to_process = candidates[:max_candidates]
            
            # Check cache
            cache_key = self._create_cache_key(query, candidates_to_process)
            if cache_key in self._cache:
                self.logger.info("Using cached reranking results")
                return self._cache[cache_key][:top_k] if top_k else self._cache[cache_key]
            
            self.logger.info(f"Reranking {len(candidates_to_process)} candidates...")
            start_time = time.time()
            
            # Batch processing
            batch_size = self.config['batch_size']
            all_scores = []
            
            for i in range(0, len(candidates_to_process), batch_size):
                batch = candidates_to_process[i:i + batch_size]
                batch_scores = self._score_batch(query, batch)
                all_scores.extend(batch_scores)
                
                # Check memory periodically
                if i > 0 and i % (batch_size * 4) == 0:
                    if not self._check_memory():
                        self.logger.warning("Memory pressure detected, stopping early")
                        break
            
            # Combine scores with candidates
            results = []
            for candidate, t5_score in zip(candidates_to_process[:len(all_scores)], all_scores):
                # Combine T5 score with initial score
                combined_score = 0.7 * t5_score + 0.3 * candidate.initial_score
                
                results.append({
                    'business_id': candidate.business_id,
                    'name': candidate.name,
                    'categories': candidate.categories,
                    'initial_score': candidate.initial_score,
                    't5_score': t5_score,
                    'combined_score': combined_score,
                    'metadata': candidate.metadata
                })
            
            # Sort by combined score
            results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            rerank_time = time.time() - start_time
            self.logger.info(f"Reranking completed in {rerank_time:.2f}s")
            
            # Update cache
            self._update_cache(cache_key, results)
            
            return results[:top_k] if top_k else results
            
        except Exception as e:
            self.logger.error(f"Reranking error: {e}")
            # Return original order on error
            return [
                {
                    'business_id': c.business_id,
                    'name': c.name,
                    'categories': c.categories,
                    'initial_score': c.initial_score,
                    't5_score': 0.0,
                    'combined_score': c.initial_score,
                    'metadata': c.metadata
                }
                for c in candidates[:top_k if top_k else len(candidates)]
            ]
    
    def _score_batch(self, query: str, batch: List[RerankCandidate]) -> List[float]:
        """Score a batch of candidates"""
        try:
            # Format inputs cho T5
            inputs = []
            for candidate in batch:
                # Format: "Query: [query] Restaurant: [name] Categories: [categories]"
                input_text = f"Query: {query} Restaurant: {candidate.name} Categories: {candidate.categories}"
                inputs.append(input_text)
            
            # Tokenize
            encoded = self._tokenizer(
                inputs,
                max_length=self.config['max_length'],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(self._device)
            attention_mask = encoded['attention_mask'].to(self._device)
            
            # Generate scores
            with torch.no_grad():
                # For reranking, we use the model to generate relevance scores
                # This is a simplified approach - in production might use different strategy
                outputs = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=10,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                
                # Extract scores (simplified - using generation probability as proxy)
                scores = []
                for i in range(len(batch)):
                    # Use the average log probability as score
                    if hasattr(outputs, 'scores') and outputs.scores:
                        # Get first token score as proxy for relevance
                        score_tensor = outputs.scores[0][i]
                        score = torch.softmax(score_tensor, dim=-1).max().item()
                    else:
                        score = 0.5  # Default score
                    scores.append(score)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Batch scoring error: {e}")
            # Return default scores on error
            return [0.5] * len(batch)
    
    def _create_cache_key(self, query: str, candidates: List[RerankCandidate]) -> str:
        """Create cache key for results"""
        # Simple hash of query and candidate IDs
        candidate_ids = [c.business_id for c in candidates[:10]]  # Use first 10 for key
        key_string = f"{query}:{':'.join(candidate_ids)}"
        return str(hash(key_string))
    
    def _update_cache(self, key: str, results: List[Dict]):
        """Update cache with LRU eviction"""
        # Remove oldest if cache full
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = results
    
    def _estimate_memory_usage(self):
        """Estimate model memory usage"""
        if self._model is None:
            self._memory_usage_mb = 0
            return
        
        # Rough estimation
        param_memory = sum(p.numel() * p.element_size() for p in self._model.parameters()) / (1024 * 1024)
        self._memory_usage_mb = param_memory * 1.5  # Add buffer for activations
    
    def _cleanup(self):
        """Cleanup model to free memory"""
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._memory_usage_mb = 0
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def unload(self):
        """Unload model to free memory"""
        self.logger.info("Unloading T5 reranking model...")
        self._cleanup()
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            'loaded': self._is_loaded,
            'device': self._device if self._device else 'not_loaded',
            'memory_usage_mb': self._memory_usage_mb,
            'cache_size': len(self._cache),
            'last_access': datetime.fromtimestamp(self._last_access_time).isoformat() if self._last_access_time else None,
            'model_path': self.model_path
        }


def demo_t5_reranking():
    """Demo T5 reranking service"""
    print("DEMO T5 RERANKING SERVICE")
    print("=" * 50)
    
    service = T5RerankingService()
    
    # Create test candidates
    candidates = [
        RerankCandidate(
            business_id="rest001",
            name="Pho Saigon",
            categories="Vietnamese, Noodles, Restaurants",
            initial_score=0.8
        ),
        RerankCandidate(
            business_id="rest002",
            name="Pizza Palace",
            categories="Italian, Pizza, Restaurants",
            initial_score=0.7
        ),
        RerankCandidate(
            business_id="rest003",
            name="Ocean Seafood",
            categories="Seafood, Fine Dining, Restaurants",
            initial_score=0.75
        )
    ]
    
    # Test reranking
    query = "vietnamese pho restaurant"
    print(f"\nQuery: '{query}'")
    print(f"Candidates: {len(candidates)}")
    
    # Rerank
    results = service.rerank_restaurants(query, candidates)
    
    print("\nReranking Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['name']}")
        print(f"   Initial score: {result['initial_score']:.3f}")
        print(f"   T5 score: {result['t5_score']:.3f}")
        print(f"   Combined score: {result['combined_score']:.3f}")
    
    # Get status
    print("\nService Status:")
    status = service.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\nT5 Reranking demo completed!")


if __name__ == "__main__":
    demo_t5_reranking() 