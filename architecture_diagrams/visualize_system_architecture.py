#!/usr/bin/env python3
"""
Restaurant Recommendation System - Architecture Visualization
Vẽ Process Flow và UML Diagrams cho hệ thống
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow, ConnectionPatch
import numpy as np
import seaborn as sns
from typing import List, Tuple, Dict
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SystemArchitectureVisualizer:
    """
    Enhanced System Architecture Visualizer với improved precision và professional styling
    """
    
    def __init__(self):
        # Create output directory
        self.output_dir = 'architecture_diagrams'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Enhanced styling configuration
        self.colors = {
            'primary': '#2E86AB',      # Professional blue
            'secondary': '#A23B72',    # Accent purple
            'success': '#46B59A',      # Green for processes
            'warning': '#F18F01',      # Orange for warnings
            'info': '#C73E1D',         # Red for important items
            'light': '#E8F4FD',        # Light blue background
            'dark': '#1B2951',         # Dark blue text
            'gray': '#6C757D',         # Gray for secondary text
            'white': '#FFFFFF',
            'accent1': '#FFE082',      # Light yellow
            'accent2': '#B39DDB',      # Light purple
            'accent3': '#81C784'       # Light green
        }
        
        # Precise positioning grid
        self.grid_size = 0.1
        self.margin = 0.05
        
        # Enhanced text styling
        self.title_font = {'family': 'Arial', 'size': 16, 'weight': 'bold'}
        self.subtitle_font = {'family': 'Arial', 'size': 12, 'weight': 'bold'}
        self.body_font = {'family': 'Arial', 'size': 10}
        self.small_font = {'family': 'Arial', 'size': 8}
        
    def create_process_flow_diagram(self):
        """
        Enhanced Process Flow Diagram với layout tối ưu và không trùng lặp
        """
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))  # Tăng size
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Professional title với font lớn hơn
        ax.text(9, 11.5, 'Restaurant Recommendation System - Process Flow', 
                ha='center', va='center', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Redesigned flow với positioning rõ ràng
        # ROW 1: Input và Initial Processing (y=9.5)
        step1_pos = (2, 9.5)    # User Query
        step2_pos = (6, 9.5)    # Router
        step3_pos = (10, 9.5)   # ML Service
        step4_pos = (14, 9.5)   # Behavior Monitor
        
        # ROW 2: ML Components (y=7.5)
        bge_pos = (3, 7.5)      # BGE-M3 Encoding
        faiss_pos = (7, 7.5)    # FAISS Search
        profile_pos = (11, 7.5) # User Profile
        collab_pos = (15, 7.5)  # Collaborative Filtering
        
        # ROW 3: Processing (y=5.5)
        content_pos = (4, 5.5)   # Content-based (CHỈ 1 lần)
        behavior_pos = (8, 5.5)  # Behavior-based
        ensemble_pos = (12, 5.5) # Ensemble Scoring
        
        # ROW 4: Final Steps (y=3.5)
        rerank_pos = (6, 3.5)    # T5 Reranking
        response_pos = (12, 3.5) # Final Response
        
        # ROW 5: Feedback (y=1.5)
        feedback_pos = (9, 1.5)  # User Feedback
        
        # Enhanced box drawing với font size lớn hơn
        box_width, box_height = 2.5, 1.0  # Tăng size box
        
        # Define all boxes với content ngắn gọn
        boxes = [
            # Row 1
            (step1_pos, 'User Query\n"seafood restaurant"', self.colors['light'], self.colors['primary']),
            (step2_pos, 'FastAPI Router\n/advanced/recommendations', self.colors['info'], self.colors['white']),
            (step3_pos, 'AdvancedMLService\n+ Monitoring', self.colors['primary'], self.colors['white']),
            (step4_pos, 'User Behavior\nMonitoring', self.colors['secondary'], self.colors['white']),
            
            # Row 2
            (bge_pos, 'BGE-M3 LoRA\nSemantic Encoding', self.colors['success'], self.colors['white']),
            (faiss_pos, 'FAISS Vector DB\nSimilarity Search', self.colors['success'], self.colors['white']),
            (profile_pos, 'UserProfileService\nPersonalization', self.colors['warning'], self.colors['white']),
            (collab_pos, 'Collaborative\nFiltering', self.colors['success'], self.colors['white']),
            
            # Row 3
            (content_pos, 'Content-based\nFiltering', self.colors['success'], self.colors['white']),
            (behavior_pos, 'Behavior-based\nScoring', self.colors['warning'], self.colors['white']),
            (ensemble_pos, 'Ensemble\nScoring', self.colors['accent1'], self.colors['dark']),
            
            # Row 4
            (rerank_pos, 'T5 Reranking\n+ Reasoning', self.colors['secondary'], self.colors['white']),
            (response_pos, 'Personalized\nResponse', self.colors['light'], self.colors['primary']),
            
            # Row 5
            (feedback_pos, 'User Feedback\n& Learning', self.colors['success'], self.colors['white'])
        ]
        
        # Draw all boxes
        for pos, text, fill_color, border_color in boxes:
            self._draw_enhanced_box_v2(ax, pos[0], pos[1], box_width, box_height, 
                                     text, fill_color, border_color)
        
        # Clear connection flow (không trùng lặp)
        connections = [
            # Main horizontal flow (Row 1)
            (step1_pos, step2_pos),
            (step2_pos, step3_pos),
            (step3_pos, step4_pos),
            
            # Processing flows (Row 1 -> Row 2)
            (step1_pos, bge_pos),
            (step2_pos, faiss_pos),
            (step3_pos, profile_pos),
            (step4_pos, collab_pos),
            
            # ML Processing (Row 2 -> Row 3)
            (bge_pos, content_pos),
            (faiss_pos, behavior_pos),
            (profile_pos, ensemble_pos),
            (collab_pos, ensemble_pos),
            
            # Final processing (Row 3 -> Row 4)
            (content_pos, rerank_pos),
            (behavior_pos, rerank_pos),
            (ensemble_pos, response_pos),
            
            # Feedback loop (Row 4 -> Row 5)
            (response_pos, feedback_pos),
            
            # Learning loop (Row 5 -> Row 1)
            (feedback_pos, step4_pos)
        ]
        
        # Draw connections với style tốt hơn
        for start_pos, end_pos in connections:
            self._draw_precise_arrow_v2(ax, start_pos, end_pos, box_width, box_height)
        
        # Add flow labels để rõ ràng hơn
        ax.text(1, 10.5, '1. Input', fontsize=14, fontweight='bold', color=self.colors['primary'])
        ax.text(1, 8.5, '2. ML Processing', fontsize=14, fontweight='bold', color=self.colors['success'])
        ax.text(1, 6.5, '3. Scoring', fontsize=14, fontweight='bold', color=self.colors['warning'])
        ax.text(1, 4.5, '4. Final Output', fontsize=14, fontweight='bold', color=self.colors['secondary'])
        ax.text(1, 2.5, '5. Learning', fontsize=14, fontweight='bold', color=self.colors['info'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'restaurant_recommendation_process_flow.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def create_system_architecture_diagram(self):
        """
        Enhanced System Architecture với font size lớn và layout cân đối
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))  # Tăng height
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Professional title với font lớn
        ax.text(8, 11.5, 'Restaurant Recommendation System - Architecture', 
                ha='center', va='center', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Layer definitions với spacing tốt hơn
        layers = [
            {'name': 'PRESENTATION LAYER', 'y': 9.5, 'height': 1.2, 'color': '#E3F2FD'},
            {'name': 'APPLICATION LAYER', 'y': 7.8, 'height': 1.2, 'color': '#F3E5F5'},
            {'name': 'BUSINESS LAYER', 'y': 6.1, 'height': 1.2, 'color': '#E8F5E8'},
            {'name': 'INFRASTRUCTURE LAYER', 'y': 4.4, 'height': 1.2, 'color': '#FFF3E0'},
            {'name': 'DATA LAYER', 'y': 2.7, 'height': 1.2, 'color': '#FFEBEE'}
        ]
        
        # Draw layers với improved styling
        for layer in layers:
            self._draw_layer_background_v2(ax, 1, layer['y'], 14, layer['height'], 
                                      layer['color'], layer['name'])
        
        # Components với grid-based positioning và font lớn hơn
        component_width, component_height = 3.8, 0.8  # Tăng size
        
        # Presentation Layer (y=9.8)
        self._draw_component_box_v2(ax, 3, 9.8, component_width, component_height, 
                                   'Web User Interface\n(restaurant_ui.html)')
        self._draw_component_box_v2(ax, 8, 9.8, component_width, component_height, 
                                   'FastAPI REST API\n(advanced_router.py)')
        self._draw_component_box_v2(ax, 13, 9.8, component_width, component_height, 
                                   'API Documentation\n(docs, OpenAPI)')
        
        # Application Layer (y=8.1)
        self._draw_component_box_v2(ax, 3, 8.1, component_width, component_height, 
                                   'Advanced Router\n(Memory Optimized)')
        self._draw_component_box_v2(ax, 8, 8.1, component_width, component_height, 
                                   'Request/Response\nModels (Pydantic)')
        self._draw_component_box_v2(ax, 13, 8.1, component_width, component_height, 
                                   'Background Tasks\n& Monitoring')
        
        # Business Layer (y=6.4)
        self._draw_component_box_v2(ax, 3, 6.4, component_width, component_height, 
                                   'AdvancedMLService\n+ Behavior Monitor')
        self._draw_component_box_v2(ax, 8, 6.4, component_width, component_height, 
                                   'UserProfileService\n+ Personalization')
        self._draw_component_box_v2(ax, 13, 6.4, component_width, component_height, 
                                   'FeedbackLearning\nService')
        
        # Infrastructure Layer (y=4.7)
        self._draw_component_box_v2(ax, 3, 4.7, component_width, component_height, 
                                   'BGE-M3 Local LoRA\nEmbedding Service')
        self._draw_component_box_v2(ax, 8, 4.7, component_width, component_height, 
                                   'T5 Reranking\n& Generation')
        self._draw_component_box_v2(ax, 13, 4.7, component_width, component_height, 
                                   'FAISS Vector\nDatabase')
        
        # Data Layer (y=3.0)
        self._draw_component_box_v2(ax, 4, 3.0, component_width, component_height, 
                                   'Restaurant\nMetadata')
        self._draw_component_box_v2(ax, 8, 3.0, component_width, component_height, 
                                   'User Profiles\n& Feedback')
        self._draw_component_box_v2(ax, 12, 3.0, component_width, component_height, 
                                   'Vector\nEmbeddings')
        
        # Enhanced connections với improved flow
        connections = [
            # Vertical connections (top to bottom)
            ((8, 9.4), (8, 8.5)),     # API to Router
            ((8, 7.7), (8, 6.8)),     # Router to ML
            ((8, 6.0), (8, 5.1)),     # ML to T5
            ((8, 4.3), (8, 3.4)),     # T5 to Data
            
            # Cross-layer connections
            ((3, 6.0), (3, 5.1)),     # ML to BGE-M3
            ((13, 6.0), (13, 5.1)),   # Feedback to FAISS
            ((4, 4.3), (4, 3.4)),     # BGE-M3 to Metadata
            ((12, 4.3), (12, 3.4)),   # FAISS to Embeddings
        ]
        
        for start_pos, end_pos in connections:
            self._draw_enhanced_connection_v2(ax, start_pos, end_pos)
        
        # Add memory optimization note
        ax.text(8, 1.5, 'Memory Optimized Architecture: 3.5GB RAM threshold, Batch size 16, FP16 enabled', 
                ha='center', va='center', fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFFDE7', 
                         edgecolor=self.colors['warning'], linewidth=2))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'restaurant_recommendation_architecture.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def create_ml_pipeline_diagram(self):
        """
        Enhanced ML Pipeline với layout tối ưu và font size lớn
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Professional title
        ax.text(8, 9.5, 'Machine Learning Pipeline - Restaurant Recommendation', 
                ha='center', va='center', fontsize=18, fontweight='bold', color=self.colors['dark'])
        
        # Redesigned pipeline với clear flow
        # Stage 1: Input Processing (y=8)
        stage1_y = 8
        query_pos = (2, stage1_y)
        encoding_pos = (5, stage1_y)
        search_pos = (8, stage1_y)
        candidates_pos = (11, stage1_y)
        profile_pos = (14, stage1_y)
        
        # Stage 2: ML Components (y=6)
        stage2_y = 6
        collab_pos = (3, stage2_y)
        content_pos = (6.5, stage2_y)
        behavior_pos = (10, stage2_y)
        context_pos = (13.5, stage2_y)
        
        # Stage 3: Integration (y=4)
        stage3_y = 4
        ensemble_pos = (5, stage3_y)
        exploration_pos = (8, stage3_y)
        rerank_pos = (11, stage3_y)
        
        # Stage 4: Output (y=2)
        stage4_y = 2
        final_pos = (8, stage4_y)
        
        # Enhanced box styling
        box_width, box_height = 2.2, 0.8
        
        # Stage 1 boxes - Input Processing
        boxes_stage1 = [
            (query_pos, 'User Query\n"seafood near me"', self.colors['light']),
            (encoding_pos, 'BGE-M3\nEncoding', self.colors['success']),
            (search_pos, 'FAISS\nVector Search', self.colors['success']),
            (candidates_pos, 'Initial\nCandidates', self.colors['info']),
            (profile_pos, 'User Profile\nContext', self.colors['warning'])
        ]
        
        # Stage 2 boxes - ML Components
        boxes_stage2 = [
            (collab_pos, 'Collaborative\nFiltering', self.colors['primary']),
            (content_pos, 'Content-based\nFiltering', self.colors['primary']),
            (behavior_pos, 'Behavior-based\nScoring', self.colors['primary']),
            (context_pos, 'Context-aware\nScoring', self.colors['primary'])
        ]
        
        # Stage 3 boxes - Integration
        boxes_stage3 = [
            (ensemble_pos, 'Ensemble\nScoring', self.colors['secondary']),
            (exploration_pos, 'Exploration\nStrategy', self.colors['accent1']),
            (rerank_pos, 'T5 Reranking\n& Reasoning', self.colors['secondary'])
        ]
        
        # Stage 4 boxes - Output
        boxes_stage4 = [
            (final_pos, 'Final Ranked\nRecommendations', self.colors['accent2'])
        ]
        
        # Draw all boxes
        all_boxes = boxes_stage1 + boxes_stage2 + boxes_stage3 + boxes_stage4
        
        for pos, text, color in all_boxes:
            self._draw_ml_component_box(ax, pos[0], pos[1], box_width, box_height, text, color)
        
        # Draw stage separators
        stage_separators = [
            (7.3, 'Input & Retrieval'),
            (5.3, 'ML Scoring Components'),
            (3.3, 'Integration & Reranking'),
            (1.3, 'Final Output')
        ]
        
        for y, label in stage_separators:
            ax.axhline(y=y, color=self.colors['light'], linestyle='--', alpha=0.5)
            ax.text(0.5, y + 0.3, label, fontsize=12, fontweight='bold', 
                   color=self.colors['dark'], rotation=90, va='bottom')
        
        # Enhanced connections with clear flow
        connections = [
            # Stage 1 flow
            (query_pos, encoding_pos),
            (encoding_pos, search_pos),
            (search_pos, candidates_pos),
            (candidates_pos, profile_pos),
            
            # Stage 1 to Stage 2
            (candidates_pos, collab_pos),
            (candidates_pos, content_pos),
            (candidates_pos, behavior_pos),
            (profile_pos, context_pos),
            
            # Stage 2 to Stage 3
            (collab_pos, ensemble_pos),
            (content_pos, ensemble_pos),
            (behavior_pos, exploration_pos),
            (context_pos, exploration_pos),
            
            # Stage 3 flow
            (ensemble_pos, rerank_pos),
            (exploration_pos, rerank_pos),
            
            # Stage 3 to Stage 4
            (rerank_pos, final_pos)
        ]
        
        for start_pos, end_pos in connections:
            self._draw_ml_arrow(ax, start_pos, end_pos, box_width, box_height)
        
        # Add ML component weights display
        weights_text = """Component Weights:
        • Collaborative Filtering: 30%
        • Content-based: 30% 
        • Behavior-based: 25%
        • Context-aware: 15%"""
        
        ax.text(1, 0.5, weights_text, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light'], 
                        edgecolor=self.colors['primary'], alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'restaurant_ml_pipeline.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def _draw_ml_component_box(self, ax, x, y, width, height, text, color):
        """Draw ML component box với enhanced styling"""
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor=self.colors['dark'],
            linewidth=2,
            alpha=0.85
        )
        ax.add_patch(box)
        
        # Text với contrast color
        text_color = self.colors['white'] if color in [self.colors['primary'], self.colors['secondary']] else self.colors['dark']
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=10, fontweight='bold', color=text_color)

    def _draw_ml_arrow(self, ax, start_pos, end_pos, box_width, box_height):
        """Draw ML pipeline arrow với improved styling"""
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Calculate connection points
        if start_x < end_x:
            start_x += box_width/2 + 0.1
            end_x -= box_width/2 + 0.1
        elif start_x > end_x:
            start_x -= box_width/2 + 0.1
            end_x += box_width/2 + 0.1
        
        if start_y > end_y:
            start_y -= box_height/2 + 0.1
            end_y += box_height/2 + 0.1
        elif start_y < end_y:
            start_y += box_height/2 + 0.1
            end_y -= box_height/2 + 0.1
        
        # Draw arrow
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=1.8, color=self.colors['primary'],
                                 alpha=0.7, connectionstyle="arc3,rad=0.05"))

    def create_class_diagram(self):
        """
        Enhanced UML Class Diagram với layout tối ưu và font size lớn
        """
        fig, ax = plt.subplots(1, 1, figsize=(18, 14))  # Tăng size lớn hơn
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        # Professional title
        ax.text(9, 13.5, 'UML Class Diagram - Restaurant Recommendation System', 
                ha='center', va='center', fontsize=20, fontweight='bold', color=self.colors['dark'])
        
        # Define class positions với spacing tốt hơn
        class_width, class_height = 4.0, 2.5  # Tăng size boxes
        
        # Core Service Classes (Row 1 - y=11)
        advanced_ml_pos = (3, 11)
        user_profile_pos = (9, 11)
        restaurant_search_pos = (15, 11)
        
        # Supporting Classes (Row 2 - y=8)
        user_profile_class_pos = (3, 8)
        restaurant_prefs_pos = (7, 8)
        feedback_action_pos = (11, 8)
        advanced_router_pos = (15, 8)
        
        # Data & Config Classes (Row 3 - y=5)
        recommendation_request_pos = (3, 5)
        recommendation_response_pos = (7, 5)
        memory_monitor_pos = (11, 5)
        service_config_pos = (15, 5)
        
        # Infrastructure Classes (Row 4 - y=2)
        lora_service_pos = (6, 2)
        session_store_pos = (12, 2)
        
        # Enhanced class definitions với content chi tiết hơn
        classes = [
            # Core Services
            (advanced_ml_pos, 'AdvancedMLService', [
                '+ get_recommendations()',
                '+ track_behavior()',
                '+ get_ml_insights()',
                '+ memory_cleanup()',
                '- collaborative_filtering()',
                '- content_based_scoring()',
                '- behavior_analysis()'
            ]),
            
            (user_profile_pos, 'UserProfileService', [
                '+ get_profile(user_id)',
                '+ update_preferences()',
                '+ record_feedback()',
                '+ get_personalization_boost()',
                '- calculate_completeness()',
                '- analyze_behavior()'
            ]),
            
            (restaurant_search_pos, 'RestaurantSearchPipeline', [
                '+ search_restaurants()',
                '+ apply_filters()',
                '+ personalize_results()',
                '+ generate_recommendations()',
                '- semantic_search()',
                '- apply_city_filtering()'
            ]),
            
            # Supporting Classes
            (user_profile_class_pos, 'UserProfile', [
                '+ user_id: str',
                '+ demographics: UserDemographics',
                '+ preferences: RestaurantPreferences',
                '+ total_searches: int',
                '+ profile_completeness: float',
                '+ last_updated: datetime'
            ]),
            
            (restaurant_prefs_pos, 'RestaurantPreferences', [
                '+ cuisine_types: List[CuisineType]',
                '+ price_levels: List[PriceLevel]',
                '+ dining_styles: List[DiningStyle]',
                '+ group_size_preference: int',
                '+ dietary_restrictions: List[str]'
            ]),
            
            (feedback_action_pos, 'FeedbackAction', [
                '+ user_id: str',
                '+ business_id: str',
                '+ feedback_type: FeedbackType',
                '+ rating: Optional[float]',
                '+ timestamp: datetime',
                '+ context: Dict'
            ]),
            
            (advanced_router_pos, 'AdvancedRouter', [
                '+ recommendations()',
                '+ submit_feedback()',
                '+ get_user_insights()',
                '+ health_check()',
                '+ memory_status()',
                '- validate_request()'
            ]),
            
            # Data & Config Classes
            (recommendation_request_pos, 'RecommendationRequest', [
                '+ user_id: str',
                '+ city: str',
                '+ user_query: str',
                '+ filters: Dict',
                '+ exploration_factor: float',
                '+ num_results: int'
            ]),
            
            (recommendation_response_pos, 'RecommendationResponse', [
                '+ success: bool',
                '+ restaurants: List[Dict]',
                '+ ml_insights: Dict',
                '+ behavior_insights: Dict',
                '+ processing_info: Dict'
            ]),
            
            (memory_monitor_pos, 'MemoryMonitor', [
                '+ max_usage_mb: int',
                '+ get_usage_mb(): float',
                '+ check_memory_available(): bool',
                '+ get_status(): Dict',
                '- monitor_usage()'
            ]),
            
            (service_config_pos, 'ServiceConfig', [
                '+ max_memory_usage_mb: int',
                '+ batch_size: int',
                '+ max_sequence_length: int',
                '+ enable_fp16: bool',
                '+ memory_monitoring: bool'
            ]),
            
            # Infrastructure Classes
            (lora_service_pos, 'OptimizedLocalLoRAService', [
                '+ encode_texts(): np.ndarray',
                '+ search_similar(): Dict',
                '+ get_performance_info(): Dict',
                '- load_model()',
                '- load_vector_database()'
            ]),
            
            (session_store_pos, 'SessionStore', [
                '+ get(key): Any',
                '+ set(key, value, ttl): None',
                '+ delete(key): bool',
                '+ clear(): None',
                '- cleanup_expired()'
            ])
        ]
        
        # Draw all class boxes với enhanced styling
        for pos, class_name, methods in classes:
            self._draw_enhanced_class_box(ax, pos[0], pos[1], class_width, class_height, 
                                        class_name, methods)
        
        # Enhanced relationships với proper styling
        relationships = [
            # Core service dependencies
            (advanced_ml_pos, user_profile_pos, 'uses', 'composition'),
            (user_profile_pos, user_profile_class_pos, 'manages', 'aggregation'),
            (user_profile_class_pos, restaurant_prefs_pos, 'contains', 'composition'),
            
            # Router relationships
            (advanced_router_pos, advanced_ml_pos, 'delegates to', 'association'),
            (advanced_router_pos, recommendation_request_pos, 'receives', 'association'),
            (advanced_router_pos, recommendation_response_pos, 'returns', 'association'),
            
            # Data flow relationships
            (advanced_ml_pos, feedback_action_pos, 'processes', 'association'),
            (advanced_ml_pos, memory_monitor_pos, 'monitors with', 'composition'),
            (advanced_ml_pos, service_config_pos, 'configured by', 'dependency'),
            
            # Infrastructure relationships
            (restaurant_search_pos, lora_service_pos, 'uses', 'association'),
            (user_profile_pos, session_store_pos, 'persists to', 'association'),
        ]
        
        # Draw relationships với enhanced styling
        for start_pos, end_pos, label, rel_type in relationships:
            self._draw_enhanced_class_relationship(ax, start_pos, end_pos, class_width, 
                                                 class_height, label, rel_type)
        
        # Add architecture notes
        notes_text = """Memory-Optimized Architecture:
        • Integrated services to reduce overhead
        • Memory monitoring and auto-cleanup
        • Batch processing optimization
        • LRU caching with size limits"""
        
        ax.text(1, 0.5, notes_text, fontsize=11, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor=self.colors['light'], 
                        edgecolor=self.colors['primary'], alpha=0.9, linewidth=2))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'restaurant_class_diagram.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def _draw_enhanced_class_box(self, ax, x, y, width, height, class_name, methods):
        """Draw enhanced class box với proper UML styling"""
        # Main class box
        class_box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.05",
            facecolor=self.colors['white'],
            edgecolor=self.colors['dark'],
            linewidth=2,
            alpha=0.95
        )
        ax.add_patch(class_box)
        
        # Class name header với larger font
        header_height = 0.4
        ax.text(x, y + height/2 - header_height/2, class_name, 
               ha='center', va='center', fontsize=13, fontweight='bold', 
               color=self.colors['dark'])
        
        # Separator line
        separator_y = y + height/2 - header_height
        ax.plot([x - width/2 + 0.1, x + width/2 - 0.1], 
               [separator_y, separator_y], 
               color=self.colors['dark'], linewidth=1.5)
        
        # Methods list với improved formatting
        methods_start_y = separator_y - 0.15
        method_spacing = 0.22
        
        # Limit methods để fit trong box
        visible_methods = methods[:min(len(methods), 8)]  # Show max 8 methods
        
        for i, method in enumerate(visible_methods):
            method_y = methods_start_y - i * method_spacing
            if method_y > y - height/2 + 0.1:  # Ensure within box bounds
                ax.text(x - width/2 + 0.15, method_y, method, 
                       ha='left', va='center', fontsize=9, 
                       color=self.colors['dark'])
        
        # Add "..." if there are more methods
        if len(methods) > 8:
            ax.text(x - width/2 + 0.15, method_y - method_spacing, '...', 
                   ha='left', va='center', fontsize=9, 
                   color=self.colors['dark'], style='italic')

    def _draw_enhanced_class_relationship(self, ax, start_pos, end_pos, box_width, 
                                        box_height, label, rel_type):
        """Draw enhanced class relationship với UML styling"""
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Calculate connection points
        if abs(start_x - end_x) > abs(start_y - end_y):  # Horizontal connection
            if start_x < end_x:  # Left to right
                conn_start = (start_x + box_width/2, start_y)
                conn_end = (end_x - box_width/2, end_y)
            else:  # Right to left
                conn_start = (start_x - box_width/2, start_y)
                conn_end = (end_x + box_width/2, end_y)
        else:  # Vertical connection
            if start_y < end_y:  # Bottom to top
                conn_start = (start_x, start_y + box_height/2)
                conn_end = (end_x, end_y - box_height/2)
            else:  # Top to bottom
                conn_start = (start_x, start_y - box_height/2)
                conn_end = (end_x, end_y + box_height/2)
        
        # Choose arrow style based on relationship type
        if rel_type == 'composition':
            arrowstyle = '-|>'
            color = self.colors['primary']
        elif rel_type == 'aggregation':
            arrowstyle = '->'
            color = self.colors['secondary']
        elif rel_type == 'dependency':
            arrowstyle = '->'
            color = self.colors['warning']
            linestyle = '--'
        else:  # association
            arrowstyle = '->'
            color = self.colors['dark']
            linestyle = '-'
        
        # Draw relationship arrow
        ax.annotate('', xy=conn_end, xytext=conn_start,
                   arrowprops=dict(arrowstyle=arrowstyle, lw=1.5, color=color,
                                 alpha=0.8, linestyle=linestyle if 'linestyle' in locals() else '-'))
        
        # Add relationship label
        mid_x = (conn_start[0] + conn_end[0]) / 2
        mid_y = (conn_start[1] + conn_end[1]) / 2
        
        ax.text(mid_x, mid_y + 0.15, label, ha='center', va='center', 
               fontsize=8, color=color, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', 
                        edgecolor='none', alpha=0.8))

    def create_sequence_diagram(self):
        """
        Enhanced UML Sequence Diagram với precise timing và professional styling
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Professional title
        ax.text(8, 9.5, 'UML Sequence Diagram - Restaurant Recommendation Flow', 
                ha='center', va='center', fontdict=self.title_font, color=self.colors['dark'])
        
        # Precise actor positioning
        actors = [
            ('User', 1),
            ('Web UI', 3),
            ('Advanced Router', 5.5),
            ('AdvancedMLService', 8),
            ('UserProfileService', 10.5),
            ('BGE-M3 Service', 13),
            ('FAISS DB', 15)
        ]
        
        # Draw actors and lifelines
        actor_y = 8.8
        lifeline_start = 8.5
        lifeline_end = 1.5
        
        for name, x in actors:
            self._draw_actor_box(ax, x, actor_y, name)
            self._draw_lifeline(ax, x, lifeline_start, lifeline_end)
        
        # Precise message sequence
        messages = [
            (1, 3, 8.2, 'Search "seafood restaurant"'),
            (3, 5.5, 7.9, 'POST /advanced/recommendations'),
            (5.5, 8, 7.6, 'get_deep_personalized_recommendations()'),
            (8, 10.5, 7.3, 'get_or_create_profile(user_id)'),
            (10.5, 8, 7.0, 'return UserProfile'),
            (8, 13, 6.7, 'encode_texts(query)'),
            (13, 15, 6.4, 'search_similar(embedding, k=50)'),
            (15, 13, 6.1, 'return candidates'),
            (13, 8, 5.8, 'return encoded_results'),
            (8, 8, 5.5, 'apply_ml_scoring()'),  # Self call
            (8, 10.5, 5.2, 'get_personalized_boost()'),
            (10.5, 8, 4.9, 'return boost_scores'),
            (8, 5.5, 4.6, 'return recommendations'),
            (5.5, 3, 4.3, 'return JSON response'),
            (3, 1, 4.0, 'display results'),
            (1, 3, 3.7, 'click "like" button'),
            (3, 5.5, 3.4, 'POST /advanced/feedback'),
            (5.5, 8, 3.1, 'track_user_behavior()'),
            (8, 10.5, 2.8, 'record_feedback()'),
            (10.5, 8, 2.5, 'return success'),
            (8, 5.5, 2.2, 'return feedback_response'),
            (5.5, 3, 1.9, 'return success')
        ]
        
        # Draw messages
        for from_x, to_x, y, message in messages:
            self._draw_sequence_message(ax, from_x, to_x, y, message)
        
        # Draw activation boxes
        activations = [
            (5.5, 7.9, 1.9),  # Advanced Router
            (8, 7.6, 2.2),    # ML Service
            (10.5, 7.3, 2.5), # Profile Service
            (13, 6.7, 5.8),   # BGE-M3 Service
        ]
        
        for x, start_y, end_y in activations:
            self._draw_activation_box(ax, x, start_y, end_y)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'restaurant_sequence_diagram.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    # Enhanced helper methods
    def _draw_enhanced_box(self, ax, x, y, width, height, text, fill_color, border_color):
        """Draw enhanced box với professional styling"""
        # Create rounded rectangle
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.02",
            facecolor=fill_color,
            edgecolor=border_color,
            linewidth=1.5,
            alpha=0.9
        )
        ax.add_patch(box)
        
        # Add text với proper formatting
        ax.text(x, y, text, ha='center', va='center', 
                fontdict=self.body_font, color=self.colors['dark'], wrap=True)
    
    def _draw_precise_arrow(self, ax, start_pos, end_pos, box_width, box_height):
        """Draw precise arrow between boxes"""
        # Calculate exact connection points
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Adjust for box boundaries
        if start_x < end_x:  # Arrow going right
            start_x += box_width/2
            end_x -= box_width/2
        elif start_x > end_x:  # Arrow going left
            start_x -= box_width/2
            end_x += box_width/2
        
        if start_y > end_y:  # Arrow going down
            start_y -= box_height/2
            end_y += box_height/2
        elif start_y < end_y:  # Arrow going up
            start_y += box_height/2
            end_y -= box_height/2
        
        # Draw arrow
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=self.colors['dark']))
    
    def _draw_layer_background(self, ax, x, y, width, height, color, name):
        """Draw layer background với consistent styling"""
        # Background rectangle
        rect = patches.Rectangle((x, y), width, height, linewidth=1, 
                               edgecolor=self.colors['gray'], facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        
        # Layer name
        ax.text(x + width/2, y + height - 0.15, name, ha='center', va='center',
               fontdict=self.subtitle_font, color=self.colors['dark'])
    
    def _draw_component_box(self, ax, x, y, width, height, text):
        """Draw component box với professional styling"""
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.02",
            facecolor=self.colors['white'],
            edgecolor=self.colors['dark'],
            linewidth=1,
            alpha=0.9
        )
        ax.add_patch(box)
        
        ax.text(x, y, text, ha='center', va='center', 
                fontdict=self.body_font, color=self.colors['dark'])

    def _draw_enhanced_connection(self, ax, start_pos, end_pos, start_width, end_width, box_height):
        """Draw enhanced connection với proper boundary calculation"""
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Calculate exact connection points
        if start_x < end_x:
            start_x += start_width/2
            end_x -= end_width/2
        elif start_x > end_x:
            start_x -= start_width/2
            end_x += end_width/2
        
        if start_y > end_y:
            start_y -= box_height/2
            end_y += box_height/2
        elif start_y < end_y:
            start_y += box_height/2
            end_y -= box_height/2
        
        # Draw arrow với enhanced styling
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color=self.colors['primary']))

    def _draw_actor_box(self, ax, x, y, name):
        """Draw enhanced actor box"""
        box = FancyBboxPatch(
            (x - 0.6, y - 0.15), 1.2, 0.3,
            boxstyle="round,pad=0.02",
            facecolor=self.colors['success'],
            edgecolor=self.colors['dark'],
            linewidth=1
        )
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', 
                fontdict=self.small_font, color=self.colors['white'], weight='bold')

    def _draw_lifeline(self, ax, x, start_y, end_y):
        """Draw enhanced lifeline"""
        ax.plot([x, x], [start_y, end_y], color=self.colors['gray'], 
               linestyle='--', linewidth=1.5)

    def _draw_sequence_message(self, ax, from_x, to_x, y, message):
        """Draw enhanced sequence message"""
        if from_x == to_x:  # Self call
            ax.add_patch(Rectangle((from_x, y - 0.05), 0.4, 0.1, 
                                 facecolor=self.colors['warning'], alpha=0.3))
            ax.text(from_x + 0.6, y, message, ha='left', va='center', 
                   fontdict=self.small_font, color=self.colors['dark'])
        else:
            # Regular message
            ax.annotate('', xy=(to_x, y), xytext=(from_x, y),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color=self.colors['primary']))
            ax.text((from_x + to_x)/2, y + 0.1, message, ha='center', va='bottom',
                   fontdict=self.small_font, color=self.colors['dark'])

    def _draw_activation_box(self, ax, x, start_y, end_y):
        """Draw enhanced activation box"""
        box = Rectangle((x - 0.05, end_y), 0.1, start_y - end_y,
                       facecolor=self.colors['warning'], 
                       edgecolor=self.colors['dark'], 
                       linewidth=1, alpha=0.7)
        ax.add_patch(box)

    def _draw_enhanced_box_v2(self, ax, x, y, width, height, text, fill_color, border_color):
        """Draw enhanced box với font size lớn hơn và spacing tốt hơn"""
        # Create rounded rectangle
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.05",
            facecolor=fill_color,
            edgecolor=border_color,
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(box)
        
        # Add text với font size lớn hơn
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=11, fontweight='bold', color=self.colors['dark'], 
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8, edgecolor='none'))
    
    def _draw_precise_arrow_v2(self, ax, start_pos, end_pos, box_width, box_height):
        """Draw precise arrow với style tốt hơn"""
        # Calculate exact connection points
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Adjust for box boundaries với margin lớn hơn
        margin = 0.3
        
        if start_x < end_x:  # Arrow going right
            start_x += box_width/2 + margin
            end_x -= box_width/2 + margin
        elif start_x > end_x:  # Arrow going left
            start_x -= box_width/2 + margin
            end_x += box_width/2 + margin
        
        if start_y > end_y:  # Arrow going down
            start_y -= box_height/2 + margin
            end_y += box_height/2 + margin
        elif start_y < end_y:  # Arrow going up
            start_y += box_height/2 + margin
            end_y -= box_height/2 + margin
        
        # Draw arrow với style đẹp hơn
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color=self.colors['dark'],
                                 connectionstyle="arc3,rad=0.1"))

    def _draw_layer_background_v2(self, ax, x, y, width, height, color, name):
        """Draw layer background với improved styling"""
        # Background rectangle
        rect = patches.Rectangle((x, y), width, height, linewidth=2, 
                               edgecolor=self.colors['dark'], facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        
        # Layer name với font lớn hơn
        ax.text(x + width/2, y + height - 0.2, name, ha='center', va='center',
               fontsize=14, fontweight='bold', color=self.colors['dark'])

    def _draw_component_box_v2(self, ax, x, y, width, height, text):
        """Draw component box với font size lớn hơn và spacing tốt hơn"""
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.05",
            facecolor=self.colors['white'],
            edgecolor=self.colors['dark'],
            linewidth=1.5,
            alpha=0.95
        )
        ax.add_patch(box)
        
        # Text với font size lớn hơn
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=11, fontweight='bold', color=self.colors['dark'])

    def _draw_enhanced_connection_v2(self, ax, start_pos, end_pos):
        """Draw enhanced connection với improved style"""
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Draw arrow với enhanced styling
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=self.colors['primary'],
                                 alpha=0.8))

def main():
    """Main function để tạo tất cả diagrams với enhanced precision"""
    print("🎨 Generating Restaurant Recommendation System Diagrams...")
    visualizer = SystemArchitectureVisualizer()
    
    print("📊 Creating Process Flow Diagram...")
    visualizer.create_process_flow_diagram()
    
    print("🏗️ Creating System Architecture Diagram...")
    visualizer.create_system_architecture_diagram()
    
    print("🧠 Creating ML Pipeline Diagram...")
    visualizer.create_ml_pipeline_diagram()
    
    print("📋 Creating UML Class Diagram...")
    visualizer.create_class_diagram()
    
    print("🔄 Creating UML Sequence Diagram...")
    visualizer.create_sequence_diagram()
    
    print("✅ All diagrams created successfully!")
    print("📁 Files saved:")
    print("   - restaurant_recommendation_process_flow.png")
    print("   - restaurant_recommendation_architecture.png")
    print("   - restaurant_ml_pipeline.png")
    print("   - restaurant_class_diagram.png")
    print("   - restaurant_sequence_diagram.png")

if __name__ == "__main__":
    main() 