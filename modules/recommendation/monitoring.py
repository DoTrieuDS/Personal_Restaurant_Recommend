"""
Production Monitoring Service
Thu thập metrics, logging và performance monitoring cho recommendation system
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import psutil
import threading
from modules.memory.short_term import SessionStore

@dataclass
class PerformanceMetric:
    """Single performance metric record"""
    metric_name: str
    metric_value: float
    metric_unit: str
    timestamp: datetime
    component: str
    user_id: Optional[str] = None
    additional_data: Optional[Dict] = None

@dataclass
class RecommendationEvent:
    """Recommendation request event"""
    event_id: str
    user_id: Optional[str]
    query: str
    destination: str
    num_candidates: int
    num_results: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class SystemHealthMetrics:
    """System health snapshot"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_requests: int
    cache_hit_rate: float
    avg_response_time: float
    error_rate: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MetricsCollector:
    """
    Metrics collection service cho recommendation system
    """
    
    def __init__(self, memory: SessionStore):
        self.memory = memory
        self.logger = self._setup_monitoring_logger()
        
        # In-memory metrics storage (sẽ flush định kỳ)
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.events_buffer: deque = deque(maxlen=1000)
        self.health_history: deque = deque(maxlen=100)
        
        # Real-time counters
        self.request_counter = 0
        self.error_counter = 0
        self.response_times = deque(maxlen=1000)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Component performance tracking
        self.component_timings = defaultdict(list)
        
        # Thread lock for safe concurrent access
        self.lock = threading.Lock()
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _setup_monitoring_logger(self) -> logging.Logger:
        """Setup dedicated logger cho monitoring"""
        logger = logging.getLogger('recommendation.monitoring')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler cho metrics
        file_handler = logging.FileHandler('recommendation_metrics.log')
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def record_metric(self, 
                     metric_name: str, 
                     metric_value: float, 
                     metric_unit: str,
                     component: str,
                     user_id: Optional[str] = None,
                     additional_data: Optional[Dict] = None):
        """
        Ghi nhận một metric
        """
        metric = PerformanceMetric(
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit=metric_unit,
            component=component,
            user_id=user_id,
            additional_data=additional_data,
            timestamp=datetime.now()
        )
        
        with self.lock:
            self.metrics_buffer.append(metric)
            
            # Track component timings nếu là timing metric
            if metric_unit == 'seconds':
                self.component_timings[component].append(metric_value)
                # Keep only recent 100 measurements
                if len(self.component_timings[component]) > 100:
                    self.component_timings[component] = self.component_timings[component][-100:]
        
        self.logger.debug(f"📊 Metric recorded: {metric_name}={metric_value}{metric_unit} for {component}")
    
    def record_recommendation_event(self, 
                                  event_id: str,
                                  user_id: Optional[str],
                                  query: str,
                                  destination: str,
                                  num_candidates: int,
                                  num_results: int,
                                  response_time: float,
                                  success: bool,
                                  error_message: Optional[str] = None):
        """
        Ghi nhận recommendation event
        """
        event = RecommendationEvent(
            event_id=event_id,
            user_id=user_id,
            query=query,
            destination=destination,
            num_candidates=num_candidates,
            num_results=num_results,
            response_time=response_time,
            success=success,
            error_message=error_message
        )
        
        with self.lock:
            self.events_buffer.append(event)
            self.request_counter += 1
            self.response_times.append(response_time)
            
            if not success:
                self.error_counter += 1
        
        # Log event
        if success:
            self.logger.info(f"✅ Recommendation event: {event_id} - {response_time:.3f}s")
        else:
            self.logger.error(f"❌ Recommendation failed: {event_id} - {error_message}")
    
    def record_cache_hit(self, cache_type: str = "default"):
        """Ghi nhận cache hit"""
        with self.lock:
            self.cache_hits += 1
        
        self.record_metric("cache_hit", 1, "count", f"cache_{cache_type}")
    
    def record_cache_miss(self, cache_type: str = "default"):
        """Ghi nhận cache miss"""
        with self.lock:
            self.cache_misses += 1
        
        self.record_metric("cache_miss", 1, "count", f"cache_{cache_type}")
    
    @contextmanager
    def time_component(self, component_name: str, user_id: Optional[str] = None):
        """
        Context manager để đo timing của component
        
        Usage:
            with metrics.time_component("bge_retrieval"):
                # do retrieval work
                pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_metric(
                metric_name="execution_time",
                metric_value=duration,
                metric_unit="seconds",
                component=component_name,
                user_id=user_id
            )
    
    def get_current_health_metrics(self) -> SystemHealthMetrics:
        """
        Lấy current system health metrics
        """
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Active requests (approximation)
        active_requests = len(self.response_times)
        
        # Cache hit rate
        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / max(total_cache_ops, 1)) * 100
        
        # Average response time
        avg_response_time = sum(self.response_times) / max(len(self.response_times), 1)
        
        # Error rate
        error_rate = (self.error_counter / max(self.request_counter, 1)) * 100
        
        health = SystemHealthMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            active_requests=active_requests,
            cache_hit_rate=cache_hit_rate,
            avg_response_time=avg_response_time,
            error_rate=error_rate
        )
        
        with self.lock:
            self.health_history.append(health)
        
        return health
    
    def get_component_performance_summary(self) -> Dict[str, Dict]:
        """
        Lấy performance summary cho tất cả components
        """
        summary = {}
        
        with self.lock:
            for component, timings in self.component_timings.items():
                if timings:
                    summary[component] = {
                        'avg_time': sum(timings) / len(timings),
                        'min_time': min(timings),
                        'max_time': max(timings),
                        'total_calls': len(timings),
                        'recent_calls': len([t for t in timings if t > 0])  # All positive timings
                    }
        
        return summary
    
    def get_metrics_summary(self, last_minutes: int = 60) -> Dict:
        """
        Lấy metrics summary cho last N minutes
        """
        cutoff_time = datetime.now() - timedelta(minutes=last_minutes)
        
        recent_metrics = []
        recent_events = []
        
        with self.lock:
            # Filter recent metrics
            for metric in self.metrics_buffer:
                if metric.timestamp >= cutoff_time:
                    recent_metrics.append(metric)
            
            # Filter recent events
            for event in self.events_buffer:
                if event.timestamp >= cutoff_time:
                    recent_events.append(event)
        
        # Aggregate metrics
        summary = {
            'time_window': f"Last {last_minutes} minutes",
            'total_requests': len(recent_events),
            'successful_requests': len([e for e in recent_events if e.success]),
            'failed_requests': len([e for e in recent_events if not e.success]),
            'avg_response_time': 0,
            'total_metrics_recorded': len(recent_metrics),
            'component_performance': self.get_component_performance_summary(),
            'current_health': asdict(self.get_current_health_metrics())
        }
        
        if recent_events:
            summary['avg_response_time'] = sum(e.response_time for e in recent_events) / len(recent_events)
        
        return summary
    
    def export_metrics_to_json(self, filepath: str):
        """
        Export current metrics to JSON file
        """
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'metrics': [asdict(m) for m in list(self.metrics_buffer)],
            'events': [asdict(e) for e in list(self.events_buffer)],
            'health_history': [asdict(h) for h in list(self.health_history)],
            'summary': self.get_metrics_summary()
        }
        
        # Convert datetime objects to strings for JSON serialization
        def datetime_converter(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=datetime_converter)
        
        self.logger.info(f"📁 Metrics exported to: {filepath}")
    
    def _start_background_monitoring(self):
        """
        Start background thread cho periodic monitoring
        """
        def monitor_loop():
            while True:
                try:
                    # Record system health every 30 seconds
                    health = self.get_current_health_metrics()
                    
                    # Log warnings nếu có issues
                    if health.cpu_usage > 80:
                        self.logger.warning(f"⚠️ High CPU usage: {health.cpu_usage:.1f}%")
                    
                    if health.memory_usage > 80:
                        self.logger.warning(f"⚠️ High memory usage: {health.memory_usage:.1f}%")
                    
                    if health.error_rate > 5:
                        self.logger.warning(f"⚠️ High error rate: {health.error_rate:.1f}%")
                    
                    if health.avg_response_time > 5:
                        self.logger.warning(f"⚠️ Slow response time: {health.avg_response_time:.2f}s")
                    
                    time.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"❌ Background monitoring error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        self.logger.info("Background monitoring started")

class RecommendationLogger:
    """
    Specialized logger cho recommendation events với structured logging
    """
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = self._setup_structured_logger(log_level)
    
    def _setup_structured_logger(self, log_level: str) -> logging.Logger:
        """Setup structured logger with JSON format"""
        logger = logging.getLogger('recommendation.structured')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # File handler with JSON format
        file_handler = logging.FileHandler('recommendation_events.jsonl')
        file_handler.setLevel(logging.INFO)
        
        # JSON formatter
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'level': record.levelname,
                    'component': record.name,
                    'message': record.getMessage(),
                }
                
                # Add extra fields nếu có
                if hasattr(record, 'user_id'):
                    log_entry['user_id'] = record.user_id
                if hasattr(record, 'event_type'):
                    log_entry['event_type'] = record.event_type
                if hasattr(record, 'duration'):
                    log_entry['duration'] = record.duration
                if hasattr(record, 'error_details'):
                    log_entry['error_details'] = record.error_details
                
                return json.dumps(log_entry)
        
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
        
        return logger
    
    def log_recommendation_request(self, 
                                 user_id: Optional[str],
                                 query: str,
                                 destination: str,
                                 num_results: int):
        """Log recommendation request"""
        self.logger.info(
            "Recommendation request received",
            extra={
                'user_id': user_id,
                'event_type': 'recommendation_request',
                'query': query,
                'destination': destination,
                'num_results': num_results
            }
        )
    
    def log_recommendation_response(self,
                                  user_id: Optional[str],
                                  num_results: int,
                                  duration: float,
                                  success: bool):
        """Log recommendation response"""
        self.logger.info(
            "Recommendation response sent",
            extra={
                'user_id': user_id,
                'event_type': 'recommendation_response',
                'num_results': num_results,
                'duration': duration,
                'success': success
            }
        )
    
    def log_component_performance(self,
                                component: str,
                                operation: str,
                                duration: float,
                                success: bool,
                                additional_data: Optional[Dict] = None):
        """Log component performance"""
        extra_data = {
            'event_type': 'component_performance',
            'component': component,
            'operation': operation,
            'duration': duration,
            'success': success
        }
        
        if additional_data:
            extra_data.update(additional_data)
        
        self.logger.info(
            f"Component {component} {operation} completed",
            extra=extra_data
        )
    
    def log_error(self,
                 component: str,
                 error_type: str,
                 error_message: str,
                 user_id: Optional[str] = None,
                 additional_context: Optional[Dict] = None):
        """Log error with context"""
        error_details = {
            'error_type': error_type,
            'error_message': error_message,
            'component': component
        }
        
        if additional_context:
            error_details.update(additional_context)
        
        self.logger.error(
            f"Error in {component}: {error_message}",
            extra={
                'user_id': user_id,
                'event_type': 'error',
                'error_details': error_details
            }
        )

def demo_monitoring_system():
    """Demo monitoring system"""
    print("DEMO MONITORING SYSTEM")
    print("=" * 50)
    
    # Setup monitoring
    from modules.memory.short_term import SessionStore
    from shared.settings import Settings
    
    settings = Settings()
    memory = SessionStore(settings)
    metrics = MetricsCollector(memory)
    logger = RecommendationLogger()
    
    # Simulate some activity
    print(f"\n1. Simulating recommendation activity...")
    
    # Record some metrics
    with metrics.time_component("bge_retrieval"):
        time.sleep(0.1)  # Simulate BGE work
    
    with metrics.time_component("llama_reranking"):
        time.sleep(0.2)  # Simulate Llama work
    
    # Record cache activity
    metrics.record_cache_hit("faiss")
    metrics.record_cache_miss("faiss")
    metrics.record_cache_hit("user_profile")
    
    # Record recommendation events
    import uuid
    
    for i in range(5):
        event_id = str(uuid.uuid4())[:8]
        metrics.record_recommendation_event(
            event_id=event_id,
            user_id=f"user_{i}",
            query=f"restaurants in city {i}",
            destination=f"City {i}",
            num_candidates=50,
            num_results=10,
            response_time=0.5 + i * 0.1,
            success=i < 4  # Last one fails
        )
        
        # Log structured events
        logger.log_recommendation_request(
            user_id=f"user_{i}",
            query=f"restaurants in city {i}",
            destination=f"City {i}",
            num_results=10
        )
        
        logger.log_recommendation_response(
            user_id=f"user_{i}",
            num_results=10 if i < 4 else 0,
            duration=0.5 + i * 0.1,
            success=i < 4
        )
    
    # Get metrics summary
    print(f"\n2. Current metrics summary...")
    summary = metrics.get_metrics_summary(last_minutes=5)
    
    print(f"   Total requests: {summary['total_requests']}")
    print(f"   Successful: {summary['successful_requests']}")
    print(f"   Failed: {summary['failed_requests']}")
    print(f"   Avg response time: {summary['avg_response_time']:.3f}s")
    
    # Component performance
    print(f"\n3. Component performance...")
    for component, perf in summary['component_performance'].items():
        print(f"   {component}: {perf['avg_time']:.3f}s avg ({perf['total_calls']} calls)")
    
    # System health
    print(f"\n4. System health...")
    health = summary['current_health']
    print(f"   CPU: {health['cpu_usage']:.1f}%")
    print(f"   Memory: {health['memory_usage']:.1f}%")
    print(f"   Cache hit rate: {health['cache_hit_rate']:.1f}%")
    print(f"   Error rate: {health['error_rate']:.1f}%")
    
    # Export metrics
    print(f"\n5. Exporting metrics...")
    export_path = "recommendation_metrics_export.json"
    metrics.export_metrics_to_json(export_path)
    print(f"   Exported to: {export_path}")
    
    print(f"\nMonitoring system demo completed!")
    print(f"Check log files:")
    print(f"   - recommendation_metrics.log")
    print(f"   - recommendation_events.jsonl")
    print(f"   - {export_path}")

if __name__ == "__main__":
    demo_monitoring_system() 