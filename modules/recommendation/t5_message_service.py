"""
T5 Personalized Message Generation Service
Service để generate personalized messages cho restaurant recommendations
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
class MessageGenerationRequest:
    """Request cho message generation"""
    user_profile: Dict  # User preferences, history, etc.
    restaurant_info: Dict  # Restaurant details
    context: Optional[Dict] = None  # Additional context (occasion, time, etc.)


class T5MessageService:
    """
    T5 Message Generation Service với lazy loading và memory optimization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = model_manager.get_model_config('t5_message')
        self.model_config = ModelConfig()
        
        # Model paths - using the checkpoint path you provided
        self.model_path = "modules/recommendation/T5_personalized_message_generation/model"
        
        # Lazy loading attributes
        self._model = None
        self._tokenizer = None
        self._device = None
        self._is_loaded = False
        
        # Memory monitoring
        self._last_access_time = None
        self._memory_usage_mb = 0
        
        # Cache for generated messages
        self._cache = OrderedDict()
        self._cache_size = self.config.get('cache_size', 500)
    
    def _check_memory(self) -> bool:
        """Check if enough memory available"""
        process = psutil.Process()
        current_usage = process.memory_info().rss / (1024 * 1024)
        available = psutil.virtual_memory().available / (1024 * 1024)
        
        # Conservative for T5 model
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
            raise MemoryError("Insufficient memory to load T5 message model")
        
        self.logger.info("Loading T5 message generation model...")
        start_time = time.time()
        
        try:
            # Check if model path exists
            if not os.path.exists(self.model_path):
                # Try default path
                self.model_path = model_manager.get_absolute_path('t5_message')
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"T5 message model not found at {self.model_path}")
            
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
            
            self.logger.info(f"T5 message model loaded in {load_time:.2f}s")
            self.logger.info(f"   Device: {self._device}")
            self.logger.info(f"   Memory usage: {self._memory_usage_mb:.1f}MB")
            
        except Exception as e:
            self.logger.error(f"Error loading T5 message model: {e}")
            self._cleanup()
            # Return simple fallback mode
            self._is_loaded = False
    
    def generate_message(self, request: MessageGenerationRequest) -> str:
        """
        Generate personalized message for restaurant recommendation
        
        Args:
            request: MessageGenerationRequest with user and restaurant info
            
        Returns:
            Generated personalized message
        """
        try:
            # Check cache first
            cache_key = self._create_cache_key(request)
            if cache_key in self._cache:
                self.logger.info("Using cached message")
                return self._cache[cache_key]
            
            # Try to load model
            self._load_model()
            
            # If model failed to load, use fallback
            if not self._is_loaded:
                return self._generate_fallback_message(request)
            
            # Generate message with T5
            self.logger.info("Generating personalized message...")
            start_time = time.time()
            
            # Format input
            input_text = self._format_input(request)
            
            # Tokenize
            inputs = self._tokenizer(
                input_text,
                max_length=self.config['max_input_length'],
                truncation=True,
                return_tensors="pt"
            ).to(self._device)
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=self.config['max_output_length'],
                    temperature=self.config['temperature'],
                    top_p=self.config['top_p'],
                    do_sample=True,
                    num_return_sequences=1
                )
            
            # Decode
            message = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process
            message = self._post_process_message(message, request)
            
            gen_time = time.time() - start_time
            self.logger.info(f"Message generated in {gen_time:.2f}s")
            
            # Update cache
            self._update_cache(cache_key, message)
            
            return message
            
        except Exception as e:
            self.logger.error(f"Message generation error: {e}")
            return self._generate_fallback_message(request)
    
    def generate_batch_messages(self, requests: List[MessageGenerationRequest]) -> List[str]:
        """
        Generate messages for multiple restaurants
        
        Args:
            requests: List of message generation requests
            
        Returns:
            List of generated messages
        """
        messages = []
        
        # Process in batches
        batch_size = self.config['batch_size']
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            
            # Check memory before each batch
            if not self._check_memory():
                self.logger.warning("Memory pressure, using fallback for remaining")
                # Use fallback for remaining
                for req in batch:
                    messages.append(self._generate_fallback_message(req))
                continue
            
            # Generate messages for batch
            for req in batch:
                message = self.generate_message(req)
                messages.append(message)
        
        return messages
    
    def _format_input(self, request: MessageGenerationRequest) -> str:
        """Format input cho T5 model"""
        # Extract user info
        user_info = request.user_profile
        user_preferences = user_info.get('preferences', {})
        
        # Extract restaurant info
        restaurant = request.restaurant_info
        
        # Format structured input
        input_parts = []
        
        # User preferences
        if user_preferences.get('cuisine_types'):
            input_parts.append(f"user_cuisine:{','.join(user_preferences['cuisine_types'])}")
        
        if user_preferences.get('price_sensitivity'):
            input_parts.append(f"price_sensitivity:{user_preferences['price_sensitivity']}")
        
        # Restaurant info
        input_parts.append(f"restaurant:{restaurant.get('name', 'Restaurant')}")
        input_parts.append(f"cuisine:{restaurant.get('cuisine_types', 'International')}")
        input_parts.append(f"rating:{restaurant.get('stars', 0)}stars")
        input_parts.append(f"price:{restaurant.get('price_level', 'moderate')}")
        
        # Context
        if request.context:
            if request.context.get('occasion'):
                input_parts.append(f"occasion:{request.context['occasion']}")
            if request.context.get('time_of_day'):
                input_parts.append(f"time:{request.context['time_of_day']}")
        
        # Join with special separator
        input_text = " | ".join(input_parts)
        
        # Add prompt prefix
        input_text = f"Generate personalized recommendation: {input_text}"
        
        return input_text
    
    def _post_process_message(self, message: str, request: MessageGenerationRequest) -> str:
        """Post-process generated message"""
        # Ensure restaurant name is included
        restaurant_name = request.restaurant_info.get('name', 'this restaurant')
        
        if restaurant_name not in message:
            # Try to naturally include restaurant name
            message = message.replace("this restaurant", restaurant_name)
            message = message.replace("This restaurant", restaurant_name)
        
        # Ensure proper formatting
        message = message.strip()
        if not message.endswith('.') and not message.endswith('!'):
            message += '.'
        
        # Capitalize first letter
        if message:
            message = message[0].upper() + message[1:]
        
        return message
    
    def _generate_fallback_message(self, request: MessageGenerationRequest) -> str:
        """Generate fallback message when model unavailable"""
        restaurant = request.restaurant_info
        name = restaurant.get('name', 'This restaurant')
        stars = restaurant.get('stars', 0)
        cuisine = restaurant.get('cuisine_types', ['cuisine'])[0] if isinstance(restaurant.get('cuisine_types'), list) else 'cuisine'
        
        # Generate template-based message
        templates = [
            f"{name} is highly recommended with {stars} stars, perfect for {cuisine} lovers.",
            f"You'll love {name} - a top-rated {cuisine} restaurant with excellent reviews.",
            f"{name} offers exceptional {cuisine} dining with a {stars}-star rating.",
            f"Based on your preferences, {name} is an excellent choice for {cuisine} cuisine."
        ]
        
        # Select template based on hash for consistency
        template_idx = hash(name) % len(templates)
        message = templates[template_idx]
        
        # Add context if available
        if request.context and request.context.get('occasion'):
            occasion = request.context['occasion']
            message += f" Perfect for {occasion}."
        
        return message
    
    def _create_cache_key(self, request: MessageGenerationRequest) -> str:
        """Create cache key for message"""
        # Create key from user preferences and restaurant ID
        user_key = str(hash(str(request.user_profile.get('preferences', {}))))
        restaurant_key = request.restaurant_info.get('business_id', 'unknown')
        context_key = str(hash(str(request.context))) if request.context else 'no_context'
        
        return f"{user_key}:{restaurant_key}:{context_key}"
    
    def _update_cache(self, key: str, message: str):
        """Update cache with LRU eviction"""
        # Remove oldest if cache full
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = message
    
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
        self.logger.info("Unloading T5 message generation model...")
        self._cleanup()
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            'loaded': self._is_loaded,
            'device': self._device if self._device else 'not_loaded',
            'memory_usage_mb': self._memory_usage_mb,
            'cache_size': len(self._cache),
            'last_access': datetime.fromtimestamp(self._last_access_time).isoformat() if self._last_access_time else None,
            'model_path': self.model_path,
            'fallback_mode': not self._is_loaded
        }


def demo_t5_message_generation():
    """Demo T5 message generation service"""
    print("DEMO T5 MESSAGE GENERATION SERVICE")
    print("=" * 50)
    
    service = T5MessageService()
    
    # Create test requests
    requests = [
        MessageGenerationRequest(
            user_profile={
                'preferences': {
                    'cuisine_types': ['vietnamese', 'asian'],
                    'price_sensitivity': 'medium'
                }
            },
            restaurant_info={
                'business_id': 'rest001',
                'name': 'Pho Saigon',
                'cuisine_types': ['vietnamese'],
                'stars': 4.5,
                'price_level': 'moderate'
            },
            context={'occasion': 'casual lunch'}
        ),
        MessageGenerationRequest(
            user_profile={
                'preferences': {
                    'cuisine_types': ['italian'],
                    'price_sensitivity': 'low'
                }
            },
            restaurant_info={
                'business_id': 'rest002',
                'name': 'Bella Italia',
                'cuisine_types': ['italian'],
                'stars': 4.8,
                'price_level': 'expensive'
            },
            context={'occasion': 'romantic dinner'}
        )
    ]
    
    # Generate messages
    print("\nGenerating personalized messages...")
    
    for i, request in enumerate(requests, 1):
        print(f"\n{i}. Restaurant: {request.restaurant_info['name']}")
        print(f"   User preferences: {request.user_profile['preferences']}")
        print(f"   Context: {request.context}")
        
        message = service.generate_message(request)
        print(f"   Message: {message}")
    
    # Test batch generation
    print("\nTesting batch generation...")
    messages = service.generate_batch_messages(requests)
    print(f"   Generated {len(messages)} messages")
    
    # Get status
    print("\nService Status:")
    status = service.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\nT5 Message Generation demo completed!")


if __name__ == "__main__":
    demo_t5_message_generation() 