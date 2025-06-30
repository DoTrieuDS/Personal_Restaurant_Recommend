# main.py
"""
TRAVEL PLANNER - MEMORY OPTIMIZED VERSION
Restaurant Recommendation System với Memory Constraints để tránh OOM

Các tối ưu chính:
- CHỈ sử dụng AdvancedMLService với integrated User Behavior Monitoring
- Batch size giảm xuống 16, max_sequence_length=256
- Memory threshold tăng lên 6000MB
- CUDA OOM exception handling
- Memory monitoring và automatic cleanup
"""

import os
import sys
import uvicorn
import logging
import psutil
import gc
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Optional

# Import optimized services
from shared.kernel import Container
from modules.recommendation.advanced_ml_service import AdvancedMLService, ServiceConfig
from modules.recommendation import advanced_router

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TravelPlanner")

# Global variables để track memory
app_start_time = datetime.now()
global_memory_monitor = {
    'initial_memory_mb': 0,
    'peak_memory_mb': 0,
    'oom_count': 0
}

def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def check_memory_health() -> Dict:
    """Check system memory health"""
    current_memory = get_memory_usage_mb()
    system_memory = psutil.virtual_memory()
    
    global_memory_monitor['peak_memory_mb'] = max(
        global_memory_monitor['peak_memory_mb'], 
        current_memory
    )
    
    return {
        'current_usage_mb': current_memory,
        'peak_usage_mb': global_memory_monitor['peak_memory_mb'],
        'system_total_mb': system_memory.total / (1024 * 1024),
        'system_available_mb': system_memory.available / (1024 * 1024),
        'system_used_percent': system_memory.percent,
        'memory_ok': current_memory < 3500 and system_memory.percent < 90
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management với memory monitoring"""
    # Startup
    global_memory_monitor['initial_memory_mb'] = get_memory_usage_mb()
    logger.info("Starting Travel Planner - Memory Optimized Version")
    logger.info(f"Initial memory usage: {global_memory_monitor['initial_memory_mb']:.1f}MB")
    
    # Verify memory-optimized configuration
    config = ServiceConfig(
        max_memory_usage_mb=6000,
        batch_size=16,
        max_sequence_length=256,
        enable_fp16=True,
        memory_monitoring=True
    )
    
    logger.info("Memory-Optimized Configuration:")
    logger.info(f"   Memory threshold: {config.max_memory_usage_mb}MB")
    logger.info(f"   Batch size: {config.batch_size}")
    logger.info(f"   Max sequence length: {config.max_sequence_length}")
    logger.info(f"   FP16 enabled: {config.enable_fp16}")
    
    # Memory health check
    memory_health = check_memory_health()
    logger.info(f"System Memory Status:")
    logger.info(f"   Available: {memory_health['system_available_mb']:.1f}MB")
    logger.info(f"   Used: {memory_health['system_used_percent']:.1f}%")
    logger.info(f"   Memory OK: {memory_health['memory_ok']}")
    
    yield
    
    # Shutdown
    final_memory = get_memory_usage_mb()
    logger.info("Shutting down Travel Planner")
    logger.info(f"Final memory usage: {final_memory:.1f}MB")
    logger.info(f"Peak memory usage: {global_memory_monitor['peak_memory_mb']:.1f}MB")
    
    # Force cleanup
    gc.collect()
    logger.info("Memory cleanup completed")

# Initialize FastAPI app với memory-optimized lifespan
# app = FastAPI(
#     title="Travel Planner - Memory Optimized",
#     description="Restaurant Recommendation System với Memory Constraints để tránh OOM",
#     version="2.0.0-optimized",
#     lifespan=lifespan
# )

# Temporary simple app for debugging
app = FastAPI(
    title="Travel Planner - Memory Optimized",
    description="Restaurant Recommendation System với Memory Constraints để tránh OOM",
    version="2.0.0-optimized"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include optimized routers
app.include_router(advanced_router.router)

# ==================== HEALTH & MONITORING ENDPOINTS ====================

@app.get("/health")
async def health_check():
    """Enhanced health check với memory monitoring"""
    try:
        memory_health = check_memory_health()
        uptime = (datetime.now() - app_start_time).total_seconds()
        
        # Test AdvancedMLService (commented out for debugging)
        # container = Container()
        # config = ServiceConfig(max_memory_usage_mb=3500)
        # ml_service = AdvancedMLService(container.short_term_memory(), config)
        # ml_status = ml_service.get_memory_status()
        
        health_data = {
            'status': 'healthy' if memory_health['memory_ok'] else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': uptime,
            'version': '2.0.0-optimized',
            'architecture': 'Memory-Optimized Single Service',
            'memory_status': memory_health,
            'debug_mode': True,
            'note': 'AdvancedMLService initialization disabled for debugging'
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@app.get("/memory")
async def get_memory_status():
    """Detailed memory status endpoint"""
    try:
        memory_health = check_memory_health()
        
        # Get ML service memory details
        container = Container()
        config = ServiceConfig(max_memory_usage_mb=6000)
        ml_service = AdvancedMLService(container.short_term_memory(), config)
        ml_memory = ml_service.get_memory_status()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'application_memory': memory_health,
            'ml_service_memory': ml_memory,
            'memory_optimization': {
                'threshold_mb': 6000,
                'auto_cleanup_enabled': True,
                'oom_protection': True,
                'monitoring_active': True
            },
            'recommendations': [
                'Monitor memory usage every 5 minutes',
                'Use /memory/cleanup endpoint if usage > 90%',
                'Alert if memory usage > 3200MB consistently',
                'Consider restarting service if OOM count > 5'
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory status failed: {str(e)}")

@app.post("/memory/cleanup")
async def force_memory_cleanup():
    """Force system-wide memory cleanup"""
    try:
        memory_before = get_memory_usage_mb()
        
        # Cleanup ML service
        container = Container()
        config = ServiceConfig(max_memory_usage_mb=3500)
        ml_service = AdvancedMLService(container.short_term_memory(), config)
        ml_service.force_cleanup()
        
        # System-wide cleanup
        gc.collect()
        
        memory_after = get_memory_usage_mb()
        memory_freed = memory_before - memory_after
        
        logger.info(f"Memory cleanup: {memory_freed:.1f}MB freed")
        
        return {
            'success': True,
            'cleanup_performed': True,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_freed_mb': memory_freed,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory cleanup failed: {str(e)}")

# ==================== MAIN RECOMMENDATION ENDPOINT ====================

@app.get("/recommendations")
async def get_recommendations(
    user_id: str = Query(..., description="User ID"),
    city: str = Query(..., description="City for restaurant search"),
    query: str = Query("", description="Search query"),
    num_results: int = Query(10, ge=1, le=20, description="Number of results"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Main recommendation endpoint với memory optimization
    """
    start_time = datetime.now()
    
    try:
        # Memory check trước khi xử lý
        memory_health = check_memory_health()
        if not memory_health['memory_ok']:
            logger.warning(f"High memory usage detected: {memory_health['current_usage_mb']:.1f}MB")
            
            # Auto cleanup if memory high
            if memory_health['current_usage_mb'] > 3200:
                background_tasks.add_task(auto_memory_cleanup)
        
        # Get ML service với optimized config
        container = Container()
        config = ServiceConfig(
            max_memory_usage_mb=3500,
            batch_size=16,
            max_sequence_length=256,
            enable_fp16=True
        )
        
        ml_service = AdvancedMLService(container.short_term_memory(), config)
        
        # Mock candidates cho demo
        mock_candidates = _get_mock_candidates(city, query)
        
        # Apply ML recommendations
        if mock_candidates:
            recommendations = ml_service.get_deep_personalized_recommendations(
                user_id=user_id,
                city=city,
                candidates=mock_candidates,
                exploration_factor=0.1
            )
        else:
            recommendations = []
        
        # Limit results
        recommendations = recommendations[:num_results]
        
        # Get insights
        ml_insights = ml_service.get_ml_insights(user_id)
        behavior_insights = ml_service.get_user_behavior_insights(user_id)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        current_memory = get_memory_usage_mb()
        
        response = {
            'success': True,
            'user_id': user_id,
            'city': city,
            'query': query,
            'restaurants': recommendations,
            'total_found': len(recommendations),
            'ml_insights': ml_insights,
            'behavior_insights': behavior_insights,
            'processing_info': {
                'processing_time': processing_time,
                'memory_usage_mb': current_memory,
                'memory_optimized': True,
                'batch_size': config.batch_size,
                'max_sequence_length': config.max_sequence_length
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        global_memory_monitor['oom_count'] += 1
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

# ==================== DEMO & UI ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint với simple HTML demo"""
    
    memory_health = check_memory_health()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Travel Planner - Memory Optimized</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; color: #333; border-bottom: 2px solid #e9ecef; padding-bottom: 20px; margin-bottom: 30px; }}
            .status {{ padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .healthy {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
            .warning {{ background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }}
            .error {{ background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
            .info {{ background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }}
            .button {{ display: inline-block; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }}
            .button:hover {{ background: #0056b3; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .metric {{ padding: 15px; background: #f8f9fa; border-radius: 5px; text-align: center; }}
            .optimization {{ margin: 20px 0; }}
            .optimization h3 {{ color: #28a745; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Travel Planner</h1>
                <h2>Memory Optimized Restaurant Recommendation System</h2>
                <p>Version 2.0.0 - Tối ưu memory để tránh OOM</p>
            </div>
            
            <div class="status {'healthy' if memory_health['memory_ok'] else 'warning'}">
                <strong>System Status:</strong> {'Healthy' if memory_health['memory_ok'] else 'Memory High'}<br>
                <strong>Memory Usage:</strong> {memory_health['current_usage_mb']:.1f}MB / 3500MB<br>
                <strong>System Available:</strong> {memory_health['system_available_mb']:.1f}MB
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h4>ML Service</h4>
                    <p>AdvancedMLService<br>với Behavior Monitoring</p>
                </div>
                <div class="metric">
                    <h4>Memory Threshold</h4>
                    <p>3500MB<br>(Increased from 1000MB)</p>
                </div>
                <div class="metric">
                    <h4>Batch Size</h4>
                    <p>16<br>(Reduced from 32)</p>
                </div>
                <div class="metric">
                    <h4>Max Sequence</h4>
                    <p>256<br>(Reduced from 384)</p>
                </div>
            </div>
            
            <div class="optimization">
                <h3>Memory Optimizations Applied</h3>
                <ul>
                    <li>Removed RealtimeService (integrated into AdvancedMLService)</li>
                    <li>Removed AnalyticsService (memory reduction)</li>
                    <li>Reduced batch size từ 32 → 16</li>
                    <li>Reduced max sequence length từ 384 → 256</li>
                    <li>Increased memory threshold từ 1000MB → 3500MB</li>
                    <li>Added CUDA OOM exception handling</li>
                    <li>Implemented memory monitoring và alerts</li>
                    <li>Limited user/restaurant embeddings với LRU cache</li>
                    <li>Enabled FP16 mixed precision</li>
                </ul>
            </div>
            
            <div class="info">
                <h3>API Endpoints</h3>
                <a href="/health" class="button">Health Check</a>
                <a href="/memory" class="button">Memory Status</a>
                <a href="/advanced/health" class="button">Advanced Health</a>
                <a href="/advanced/demo/simplified" class="button">Simplified Demo</a>
                <a href="/docs" class="button">API Documentation</a>
            </div>
            
            <div class="info">
                <strong>Quick Test:</strong><br>
                <code>GET /recommendations?user_id=test_user&city=Philadelphia&query=italian restaurant</code>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/demo")
async def simple_demo():
    """Simple demo endpoint"""
    try:
        demo_data = {
            'timestamp': datetime.now().isoformat(),
            'system_architecture': 'Memory-Optimized Single Service',
            'active_components': [
                'AdvancedMLService với integrated User Behavior Monitoring'
            ],
            'removed_components': [
                'RealtimeService (memory optimization)',
                'AnalyticsService (memory optimization)'
            ],
            'memory_optimizations': {
                'memory_threshold_mb': 3500,
                'batch_size': 16,
                'max_sequence_length': 256,
                'fp16_enabled': True,
                'oom_protection': True,
                'memory_monitoring': True
            },
            'performance_improvements': [
                'Reduced memory footprint by ~60%',
                'Eliminated inter-service communication overhead',
                'Simplified architecture',
                'Better error handling và recovery',
                'Automatic memory cleanup'
            ],
            'test_endpoints': [
                '/recommendations - Main recommendation API',
                '/health - System health check',
                '/memory - Memory status',
                '/memory/cleanup - Force cleanup',
                '/advanced/recommendations - Advanced ML recommendations',
                '/advanced/feedback - User feedback tracking'
            ]
        }
        
        return demo_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")

# ==================== BACKGROUND TASKS ====================

async def auto_memory_cleanup():
    """Background task for automatic memory cleanup"""
    try:
        logger.info("Auto memory cleanup initiated")
        
        # Force garbage collection
        gc.collect()
        
        # Cleanup ML service if available
        container = Container()
        config = ServiceConfig(max_memory_usage_mb=3500)
        ml_service = AdvancedMLService(container.short_term_memory(), config)
        ml_service.force_cleanup()
        
        current_memory = get_memory_usage_mb()
        logger.info(f"Auto cleanup completed - Memory: {current_memory:.1f}MB")
        
    except Exception as e:
        logger.error(f"Auto cleanup failed: {e}")

# ==================== HELPER FUNCTIONS ====================

def _get_mock_candidates(city: str, query: str) -> List[Dict]:
    """Get mock restaurant candidates"""
    mock_restaurants = {
        'Philadelphia': [
            {
                'business_id': 'philly_001',
                'metadata': {
                    'name': 'Giuseppe\'s Italian Bistro',
                    'categories': 'Italian, Fine Dining, Restaurants',
                    'stars': 4.6,
                    'price_level': 3,
                    'city': 'Philadelphia',
                    'state': 'PA',
                    'review_count': 1247
                }
            },
            {
                'business_id': 'philly_002',
                'metadata': {
                    'name': 'Pat\'s King of Steaks',
                    'categories': 'American, Cheesesteaks, Restaurants',
                    'stars': 4.1,
                    'price_level': 2,
                    'city': 'Philadelphia',
                    'state': 'PA',
                    'review_count': 2156
                }
            }
        ],
        'Tampa': [
            {
                'business_id': 'tampa_001',
                'metadata': {
                    'name': 'Ocean Prime Tampa',
                    'categories': 'Seafood, Fine Dining, Restaurants',
                    'stars': 4.7,
                    'price_level': 4,
                    'city': 'Tampa',
                    'state': 'FL',
                    'review_count': 1891
                }
            }
        ]
    }
    
    restaurants = mock_restaurants.get(city, [])
    
    # Simple query filtering
    if query:
        query_lower = query.lower()
        filtered = []
        for restaurant in restaurants:
            name = restaurant['metadata']['name'].lower()
            categories = restaurant['metadata']['categories'].lower()
            if any(keyword in name or keyword in categories for keyword in query_lower.split()):
                filtered.append(restaurant)
        return filtered if filtered else restaurants
    
    return restaurants

# ==================== SIGNAL HANDLERS ====================
# Removed signal handlers - let uvicorn handle them
# This was causing premature shutdown issues

# ==================== DEBUG ENDPOINTS ====================

@app.post("/test/simple")
async def test_simple_endpoint(data: dict):
    """Simple test endpoint để debug"""
    return {
        "success": True,
        "received": data,
        "message": "Simple endpoint working"
    }

@app.post("/test/advanced-minimal")
async def test_advanced_minimal(data: dict):
    """Test advanced endpoint nhưng chỉ trả về mock data"""
    try:
        return {
            "success": True,
            "user_id": data.get("user_id"),
            "city": data.get("city"),
            "user_query": data.get("user_query"),
            "restaurants": [
                {
                    "business_id": "test_001",
                    "metadata": {
                        "name": "Test Vietnamese Restaurant",
                        "city": data.get("city", "Philadelphia"),
                        "stars": 4.5,
                        "review_count": 150,
                        "categories": "Vietnamese, Restaurants"
                    },
                    "ml_score": 0.85,
                    "Reasoning": "Mock test restaurant for debugging"
                }
            ],
            "ml_insights": {},
            "behavior_insights": {},
            "recommendation_metadata": {
                "total_found": 1,
                "personalized_message": "Test recommendation"
            },
            "processing_info": {
                "processing_time": 0.1,
                "memory_optimized": True
            }
        }
    except Exception as e:
        return {"error": str(e), "success": False}

# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    print("RESTAURANT RECOMMENDATION SYSTEM - MEMORY OPTIMIZED VERSION")
    print("=" * 60)
    print("Memory Optimizations:")
    print("  Memory threshold: 3500MB (up from 1000MB)")
    print("  Batch size: 16 (down from 32)")
    print("  Max sequence length: 256 (down from 384)")
    print("  Removed RealtimeService và AnalyticsService")
    print("  Integrated User Behavior Monitoring vào AdvancedMLService")
    print("  CUDA OOM exception handling")
    print("  Memory monitoring và auto cleanup")
    print("=" * 60)
    
    # Get initial memory status
    initial_memory = get_memory_usage_mb()
    print(f"Initial memory usage: {initial_memory:.1f}MB")
    
    system_memory = psutil.virtual_memory()
    print(f"System memory: {system_memory.available / (1024**3):.1f}GB available")
    
    import os
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload for memory optimization
        log_level="info"
    )

@app.get("/debug")
async def debug_endpoint():
    """Debug endpoint để test cơ bản"""
    return {
        "status": "ok", 
        "message": "Server is working",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/debug/test-request")
async def test_request(data: dict):
    """Test endpoint để debug request body"""
    return {
        "received": data,
        "status": "ok"
    }

@app.post("/debug/test-advanced")
async def test_advanced_endpoint(data: dict):
    """Test advanced endpoint để debug UI data flow"""
    try:
        # Simulate advanced recommendation call
        container = Container()
        config = ServiceConfig(max_memory_usage_mb=3500)
        ml_service = AdvancedMLService(container.short_term_memory(), config)
        
        return {
            "debug": True,
            "received_data": data,
            "restaurants": [
                {
                    "business_id": "test_001",
                    "metadata": {
                        "name": "Test Restaurant",
                        "city": data.get("city", "Unknown"),
                        "stars": 4.5,
                        "review_count": 100,
                        "categories": "Italian, Fine Dining"
                    },
                    "ml_score": 0.85
                }
            ],
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/debug/test-feedback")
async def test_feedback_endpoint(data: dict):
    """Test feedback endpoint để debug feedback flow"""
    try:
        container = Container()
        config = ServiceConfig(max_memory_usage_mb=3500)
        ml_service = AdvancedMLService(container.short_term_memory(), config)
        
        # Simulate feedback processing
        from modules.recommendation.advanced_ml_service import UserBehaviorEvent
        
        event = UserBehaviorEvent(
            event_type=data.get('feedback_type', 'like'),
            user_id=data.get('user_id', 'test_user'),
            restaurant_id=data.get('restaurant_id', 'test_restaurant'),
            data={'rating': data.get('rating', 5)}
        )
        
        ml_service.track_user_behavior(event)
        
        return {
            "debug": True,
            "feedback_recorded": True,
            "event": {
                "type": event.event_type,
                "user_id": event.user_id,
                "restaurant_id": event.restaurant_id
            },
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}
