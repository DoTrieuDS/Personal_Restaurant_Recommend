#!/usr/bin/env python3
"""
Additional UML Diagrams for Restaurant Recommendation System
Component Diagram & Deployment Diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon
import numpy as np
import os

class AdditionalDiagramsVisualizer:
    """
    Enhanced Additional Diagrams Visualizer với improved precision và professional styling
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
            'component': '#E3F2FD',    # Light blue for components
            'interface': '#F3E5F5',    # Light purple for interfaces
            'external': '#FFF3E0'      # Light orange for external systems
        }
        
        # Enhanced text styling
        self.title_font = {'family': 'Arial', 'size': 16, 'weight': 'bold'}
        self.subtitle_font = {'family': 'Arial', 'size': 12, 'weight': 'bold'}
        self.body_font = {'family': 'Arial', 'size': 10}
        self.small_font = {'family': 'Arial', 'size': 8}
    
    def create_component_diagram(self):
        """
        Enhanced Component Diagram với precise positioning và clear dependencies
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Professional title
        ax.text(8, 11.5, 'UML Component Diagram - Restaurant Recommendation System', 
                ha='center', va='center', fontdict=self.title_font, color=self.colors['dark'])
        
        # Grid-based component positioning
        components = {
            # Top tier - User Interface
            'web_ui': (2, 10, 3, 1.2, 'Web UI Component'),
            'api_gateway': (8, 10, 3, 1.2, 'API Gateway Component'),
            'ml_component': (14, 10, 3, 1.2, 'Advanced ML Component'),
            
            # Middle tier - Business Logic
            'user_mgmt': (2, 7.5, 3, 1.2, 'User Management Component'),
            'search_rec': (8, 7.5, 3, 1.2, 'Search & Recommendation Component'), 
            'ml_infra': (14, 7.5, 3, 1.2, 'ML Infrastructure Component'),
            
            # Bottom tier - Infrastructure
            'data_access': (2, 5, 3, 1.2, 'Data Access Component'),
            'memory_mgmt': (8, 5, 3, 1.2, 'Memory Management Component'),
            'config': (14, 5, 3, 1.2, 'Configuration Component')
        }
        
        # Component content definitions
        component_content = {
            'web_ui': ['<< interfaces >>', 'restaurant_ui.html', 'JavaScript Client'],
            'api_gateway': ['<< gateway >>', 'FastAPI Router', 'Request Validation', 'Response Formatting'],
            'ml_component': ['<< service >>', 'AdvancedMLService', 'User Behavior Monitoring', 'Memory Optimization'],
            
            'user_mgmt': ['<< service >>', 'UserProfileService', 'Feedback Learning', 'Personalization Engine'],
            'search_rec': ['<< service >>', 'RestaurantSearchPipeline', 'Semantic Search', 'Result Enhancement'],
            'ml_infra': ['<< infrastructure >>', 'BGE-M3 Local LoRA', 'T5 Reranking', 'Model Management'],
            
            'data_access': ['<< repository >>', 'FAISS Vector DB', 'Metadata Management', 'Cache Layer'],
            'memory_mgmt': ['<< service >>', 'Memory Monitor', 'Cleanup Service', 'Resource Optimization'],
            'config': ['<< configuration >>', 'Service Config', 'Environment Settings', 'Feature Flags']
        }
        
        # Draw components
        for key, (x, y, width, height, name) in components.items():
            content = component_content.get(key, [])
            self._draw_component(ax, x, y, width, height, name, content)
        
        # Interfaces (circles below components)
        interfaces = [
            (3.5, 3, 'HTTP Interface'),
            (8, 3, 'REST API'),
            (13, 3, 'ML Processing'),
            (3.5, 2, 'Data Access'),
            (13, 2, 'Memory Management'),
            (8, 2, 'Model Loading')
        ]
        
        for x, y, name in interfaces:
            self._draw_interface(ax, x, y, 1.2, 0.6, name, self.colors['interface'])
        
        # External systems (dashed boxes)
        external_systems = [
            (1, 0.5, 2, 0.8, 'Web Browser'),
            (5, 0.5, 2, 0.8, 'Mobile App'),
            (9, 0.5, 2, 0.8, 'Restaurant Database'),
            (13, 0.5, 2, 0.8, 'User Feedback System'),
            (16, 0.5, 2, 0.8, 'Analytics Platform')
        ]
        
        for x, y, width, height, name in external_systems:
            self._draw_external_system(ax, x, y, width, height, name, self.colors['external'])
        
        # Dependencies (dashed lines)
        dependencies = [
            ((3.5, 10), (8, 10)),       # Web UI -> API Gateway
            ((8, 9.4), (14, 9.4)),      # API Gateway -> ML Component
            ((3.5, 8.7), (8, 8.7)),     # User Mgmt -> Search & Rec
            ((11, 8.7), (14, 8.7)),     # Search & Rec -> ML Infra
            ((8, 6.3), (8, 6.2)),       # Search & Rec -> Memory Mgmt
            ((14, 6.3), (14, 6.2)),     # ML Infra -> Config
        ]
        
        for start, end in dependencies:
            self._draw_dependency(ax, start, end)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'restaurant_component_diagram.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def create_deployment_diagram(self):
        """
        Enhanced Deployment Diagram với 3-tier architecture và precise specifications
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Professional title
        ax.text(8, 9.5, 'UML Deployment Diagram - Restaurant Recommendation System', 
                ha='center', va='center', fontdict=self.title_font, color=self.colors['dark'])
        
        # Tier headers
        tier_headers = [
            (2, 8.8, 'CLIENT TIER'),
            (8, 8.8, 'APPLICATION TIER'),
            (14, 8.8, 'DATA TIER')
        ]
        
        for x, y, text in tier_headers:
            ax.text(x, y, text, ha='center', va='center', 
                   fontdict=self.subtitle_font, color=self.colors['dark'])
        
        # Nodes (deployment targets)
        nodes = {
            # Client Tier
            'web_browser': (1, 7.5, 2.5, 1.5, 'Web Browser'),
            'mobile': (4, 7.5, 2.5, 1.5, 'Mobile Device'),
            
            # Application Tier  
            'web_server': (7, 7.5, 2.5, 1.5, 'Web Server'),
            'ml_node': (10, 7.5, 2.5, 1.5, 'ML Processing Node'),
            
            # Data Tier
            'vector_db': (13, 7.5, 2.5, 1.5, 'Vector Database'),
            'cache': (13, 5.5, 2.5, 1.5, 'Memory Cache'),
            'filesystem': (13, 3.5, 2.5, 1.5, 'File System')
        }
        
        # Node specifications
        node_specs = {
            'web_browser': ['<< device >>', 'Chrome/Firefox/Safari', 'JavaScript Runtime', 'Local Storage'],
            'mobile': ['<< device >>', 'iOS/Android', 'Mobile Browser', 'App Runtime'],
            'web_server': ['<< execution environment >>', 'Ubuntu 20.04 LTS', 'Python 3.9+', 'FastAPI Server'],
            'ml_node': ['<< execution environment >>', 'CUDA 11.8+', 'PyTorch 2.0', 'GPU Memory: 3.5GB'],
            'vector_db': ['<< database server >>', 'FAISS Index', '17,873 Vectors', '1024 Dimensions'],
            'cache': ['<< cache server >>', 'Session Store', 'User Profiles', 'ML Embeddings'],
            'filesystem': ['<< storage >>', 'Model Checkpoints', 'Configuration Files', 'Logs']
        }
        
        # Draw nodes
        for key, (x, y, width, height, name) in nodes.items():
            specs = node_specs.get(key, [])
            self._draw_node(ax, x, y, width, height, name, specs)
        
        # Artifacts (deployed components)
        artifacts = [
            # Web Server artifacts
            (7.2, 6.2, 0.6, 0.3, 'restaurant_ui.server'),
            (7.2, 5.8, 0.6, 0.3, 'advanced_router.py'),
            
            # ML Node artifacts  
            (10.2, 6.2, 0.6, 0.3, 'AdvancedMLService'),
            (10.2, 5.8, 0.6, 0.3, 'BGE-M3 LoRA'),
            (10.2, 5.4, 0.6, 0.3, 'T5 Reranking'),
            
            # Data artifacts
            (13.2, 6.2, 0.6, 0.3, 'faiss_index.bin'),
            (13.2, 4.5, 0.6, 0.3, 'user_profiles.cache'),
            (13.2, 4.1, 0.6, 0.3, 'ml_embeddings.cache'),
            (13.2, 2.8, 0.6, 0.3, 'BGE-M3_embeddings'),
            (13.2, 2.4, 0.6, 0.3, 'config.files')
        ]
        
        for x, y, width, height, name in artifacts:
            self._draw_artifact(ax, x, y, width, height, name)
        
        # Network connections
        connections = [
            ((3.5, 7.5), (7, 7.5), 'HTTPS'),
            ((4.5, 7.5), (7, 7.5), 'HTTPS'),
            ((9.5, 7.5), (10, 7.5), 'Internal API'),
            ((12.5, 7.5), (13, 7.5), 'TCP'),
            ((10, 6.5), (13, 5.5), 'Memory Access'),
            ((10, 6), (13, 3.5), 'File I/O')
        ]
        
        for start, end, label in connections:
            self._draw_connection(ax, start, end, label)
        
        # Deployment specifications box
        spec_text = """DEPLOYMENT SPECIFICATIONS:
• Memory Optimization: 3.5GB RAM threshold
• GPU Requirements: CUDA-compatible, 4GB+ VRAM recommended
• Storage: 500MB for models + vector database
• Network: 1Gbps for optimal performance
• OS: Ubuntu 20.04 LTS, Python 3.9+
• Dependencies: PyTorch 2.0, FAISS, Transformers"""
        
        ax.text(8, 1.5, spec_text, ha='center', va='center', 
                fontdict=self.small_font, bbox=dict(boxstyle="round,pad=0.3",
                facecolor='#FFFDE7', edgecolor=self.colors['warning'], linewidth=1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'restaurant_deployment_diagram.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    def create_data_flow_diagram(self):
        """
        Enhanced Data Flow Diagram với layout tối ưu và font size lớn
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))  # Tăng size
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Professional title
        ax.text(8, 11.5, 'Data Flow Diagram - Restaurant Recommendation System', 
                ha='center', va='center', fontsize=18, fontweight='bold', color=self.colors['dark'])
        
        # External entities (rectangles)
        user_pos = (1.5, 9)
        admin_pos = (14.5, 9)
        
        # Processes (circles) với improved positioning
        process_positions = {
            'receive_query': (4, 9),
            'process_recommendation': (8, 9),
            'generate_response': (12, 9),
            'process_feedback': (2, 6),
            'update_profile': (6, 6),
            'ml_training': (10, 6)
        }
        
        # Data stores (open rectangles)
        datastore_positions = {
            'user_profiles': (3, 3),
            'restaurant_db': (8, 3),
            'ml_models': (13, 3),
            'feedback_data': (5.5, 1.5),
            'cache_sessions': (10.5, 1.5)
        }
        
        # Draw external entities với enhanced styling
        self._draw_external_entity_v2(ax, user_pos[0], user_pos[1], 'User')
        self._draw_external_entity_v2(ax, admin_pos[0], admin_pos[1], 'Admin')
        
        # Draw processes (circles) với enhanced styling
        process_radius = 0.8
        process_labels = {
            'receive_query': '1\nReceive\nUser Query',
            'process_recommendation': '2\nProcess\nRecommendation',
            'generate_response': '3\nGenerate\nPersonalized Results',
            'process_feedback': '4\nProcess\nUser Feedback',
            'update_profile': '5\nUpdate\nUser Profile',
            'ml_training': '6\nML Model\nTraining'
        }
        
        for proc_id, pos in process_positions.items():
            label = process_labels[proc_id]
            self._draw_process_circle_v2(ax, pos[0], pos[1], process_radius, label)
        
        # Draw data stores với enhanced styling
        datastore_width, datastore_height = 2.5, 0.6
        datastore_labels = {
            'user_profiles': 'D1\nUser\nProfiles',
            'restaurant_db': 'D2\nRestaurant\nDatabase',
            'ml_models': 'D3\nML Models\n& Embeddings',
            'feedback_data': 'D4\nFeedback\nData',
            'cache_sessions': 'D5\nCache &\nSessions'
        }
        
        for ds_id, pos in datastore_positions.items():
            label = datastore_labels[ds_id]
            self._draw_datastore_v2(ax, pos[0], pos[1], datastore_width, datastore_height, label)
        
        # Enhanced data flows với clear labels
        data_flows = [
            # User interactions
            (user_pos, process_positions['receive_query'], 'search query'),
            (process_positions['receive_query'], process_positions['process_recommendation'], 'processed query'),
            (process_positions['process_recommendation'], process_positions['generate_response'], 'candidates'),
            (process_positions['generate_response'], user_pos, 'recommendations'),
            
            # Feedback flow
            (user_pos, process_positions['process_feedback'], 'user feedback'),
            (process_positions['process_feedback'], process_positions['update_profile'], 'feedback signals'),
            
            # Data store interactions
            (process_positions['receive_query'], datastore_positions['user_profiles'], 'user context'),
            (datastore_positions['user_profiles'], process_positions['process_recommendation'], 'user preferences'),
            (process_positions['process_recommendation'], datastore_positions['restaurant_db'], 'search criteria'),
            (datastore_positions['restaurant_db'], process_positions['process_recommendation'], 'restaurant data'),
            (process_positions['process_recommendation'], datastore_positions['ml_models'], 'query embedding'),
            (datastore_positions['ml_models'], process_positions['generate_response'], 'similarity scores'),
            
            # Profile and learning flows
            (process_positions['update_profile'], datastore_positions['user_profiles'], 'updated profile'),
            (process_positions['process_feedback'], datastore_positions['feedback_data'], 'feedback events'),
            (datastore_positions['feedback_data'], process_positions['ml_training'], 'training data'),
            (process_positions['ml_training'], datastore_positions['ml_models'], 'trained models'),
            
            # Cache flows
            (process_positions['generate_response'], datastore_positions['cache_sessions'], 'session data'),
            (datastore_positions['cache_sessions'], process_positions['process_recommendation'], 'cached results'),
            
            # Admin flows
            (admin_pos, process_positions['ml_training'], 'system config'),
            (datastore_positions['ml_models'], admin_pos, 'model metrics')
        ]
        
        # Draw data flows với enhanced styling
        for start_pos, end_pos, label in data_flows:
            self._draw_data_flow_v2(ax, start_pos, end_pos, label)
        
        # Add system boundary
        self._draw_system_boundary_v2(ax, 0.5, 0.5, 15, 10.5, 
                                    'Restaurant Recommendation System Boundary')
        
        # Add legend
        legend_text = """Legend:
        ○ Processes (numbered 1-6)
        ⬜ External Entities  
        ▬ Data Stores (D1-D5)
        → Data Flows"""
        
        ax.text(0.5, 11, legend_text, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['light'], 
                        edgecolor=self.colors['primary'], alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'restaurant_data_flow_diagram.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

    # Enhanced helper methods
    def _draw_component(self, ax, x, y, width, height, name, content):
        """Draw enhanced component box"""
        # Main component box
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.02",
            facecolor=self.colors['component'],
            edgecolor=self.colors['dark'],
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # Component symbol (two small rectangles in top-left)
        symbol_x = x - width/2 + 0.1
        symbol_y = y + height/2 - 0.2
        
        ax.add_patch(Rectangle((symbol_x, symbol_y), 0.15, 0.08, 
                              facecolor=self.colors['dark']))
        ax.add_patch(Rectangle((symbol_x + 0.2, symbol_y), 0.15, 0.08, 
                              facecolor=self.colors['dark']))
        
        # Component name
        ax.text(x, y + height/2 - 0.15, name, ha='center', va='center',
                fontdict=self.subtitle_font, color=self.colors['dark'])
        
        # Separator line
        ax.plot([x - width/2 + 0.1, x + width/2 - 0.1], 
               [y + height/2 - 0.35, y + height/2 - 0.35], 
               color=self.colors['dark'], linewidth=1)
        
        # Content lines
        if content:
            line_height = 0.15
            start_y = y + height/2 - 0.5
            for i, line in enumerate(content):
                ax.text(x, start_y - i * line_height, line, ha='center', va='center',
                       fontdict=self.small_font, color=self.colors['dark'])

    def _draw_interface(self, ax, x, y, width, height, name, color):
        """Draw interface as circle"""
        circle = Circle((x, y), width/2, facecolor=color, 
                       edgecolor=self.colors['dark'], linewidth=1.5, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', 
               fontdict=self.small_font, color=self.colors['dark'], weight='bold')

    def _draw_external_system(self, ax, x, y, width, height, name, color):
        """Draw external system với dashed border"""
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor=self.colors['dark'],
            linewidth=1.5,
            linestyle='--',
            alpha=0.6
        )
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, '<< external >>\n' + name, 
               ha='center', va='center', fontdict=self.small_font, color=self.colors['dark'])

    def _draw_dependency(self, ax, start, end):
        """Draw dependency với dashed line"""
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, 
                                 color=self.colors['primary'], linestyle='--'))

    def _draw_node(self, ax, x, y, width, height, name, specs):
        """Draw deployment node với 3D effect"""
        # Main box
        main_box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02",
            facecolor=self.colors['light'],
            edgecolor=self.colors['dark'],
            linewidth=2
        )
        ax.add_patch(main_box)
        
        # 3D effect (offset shadow)
        shadow_offset = 0.05
        shadow_box = FancyBboxPatch(
            (x + shadow_offset, y - shadow_offset), width, height,
            boxstyle="round,pad=0.02",
            facecolor=self.colors['gray'],
            alpha=0.3,
            zorder=-1
        )
        ax.add_patch(shadow_box)
        
        # Node name
        ax.text(x + width/2, y + height - 0.15, name, ha='center', va='center',
                fontdict=self.subtitle_font, color=self.colors['dark'])
        
        # Specifications
        if specs:
            line_height = 0.15
            start_y = y + height - 0.4
            for i, spec in enumerate(specs):
                ax.text(x + width/2, start_y - i * line_height, spec, 
                       ha='center', va='center', fontdict=self.small_font, 
                       color=self.colors['dark'])

    def _draw_artifact(self, ax, x, y, width, height, name):
        """Draw deployment artifact"""
        # Create artifact shape (rectangle with folded corner)
        main_rect = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.01",
            facecolor=self.colors['white'],
            edgecolor=self.colors['dark'],
            linewidth=1
        )
        ax.add_patch(main_rect)
        
        # Folded corner triangle
        corner_size = 0.08
        triangle = Polygon([(x + width - corner_size, y + height),
                           (x + width, y + height - corner_size),
                           (x + width, y + height)],
                          facecolor=self.colors['gray'], alpha=0.5)
        ax.add_patch(triangle)
        
        # Artifact name
        ax.text(x + width/2, y + height/2, name, ha='center', va='center',
                fontdict={'family': 'Arial', 'size': 6}, color=self.colors['dark'])

    def _draw_connection(self, ax, start, end, label):
        """Draw network connection với label"""
        # Connection line
        ax.plot([start[0], end[0]], [start[1], end[1]], 
               color=self.colors['info'], linewidth=2, alpha=0.7)
        
        # Connection points (circles)
        ax.add_patch(Circle(start, 0.05, facecolor=self.colors['info']))
        ax.add_patch(Circle(end, 0.05, facecolor=self.colors['info']))
        
        # Label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.15, label, ha='center', va='center',
                fontdict=self.small_font, color=self.colors['info'], weight='bold',
                bbox=dict(boxstyle="round,pad=0.1", facecolor=self.colors['white'], 
                         edgecolor=self.colors['info'], alpha=0.8))

    def _draw_external_entity_v2(self, ax, x, y, label):
        """Draw external entity với enhanced styling"""
        width, height = 1.5, 0.8
        
        entity_box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="square,pad=0.05",
            facecolor=self.colors['accent1'],
            edgecolor=self.colors['dark'],
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(entity_box)
        
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=12, fontweight='bold', color=self.colors['dark'])

    def _draw_process_circle_v2(self, ax, x, y, radius, label):
        """Draw process circle với enhanced styling"""
        circle = Circle((x, y), radius, 
                       facecolor=self.colors['success'], 
                       edgecolor=self.colors['dark'],
                       linewidth=2, alpha=0.85)
        ax.add_patch(circle)
        
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=9, fontweight='bold', color=self.colors['white'])

    def _draw_datastore_v2(self, ax, x, y, width, height, label):
        """Draw data store với enhanced styling"""
        # Main rectangle với open left side
        rect = Rectangle((x - width/2, y - height/2), width, height,
                        facecolor=self.colors['warning'], 
                        edgecolor=self.colors['dark'],
                        linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        
        # Open left side
        ax.plot([x - width/2, x - width/2], 
               [y - height/2, y + height/2], 
               color='white', linewidth=3)
        
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=9, fontweight='bold', color=self.colors['dark'])

    def _draw_data_flow_v2(self, ax, start_pos, end_pos, label):
        """Draw data flow arrow với enhanced styling và labels"""
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Calculate arrow offset để avoid overlap với shapes
        dx = end_x - start_x
        dy = end_y - start_y
        length = (dx**2 + dy**2)**0.5
        
        if length > 0:
            # Normalize direction
            dx_norm = dx / length
            dy_norm = dy / length
            
            # Offset start and end points
            offset = 0.5
            start_x_offset = start_x + dx_norm * offset
            start_y_offset = start_y + dy_norm * offset
            end_x_offset = end_x - dx_norm * offset
            end_y_offset = end_y - dy_norm * offset
            
            # Draw arrow
            ax.annotate('', xy=(end_x_offset, end_y_offset), 
                       xytext=(start_x_offset, start_y_offset),
                       arrowprops=dict(arrowstyle='->', lw=1.5, 
                                     color=self.colors['primary'], alpha=0.8))
            
            # Add label ở midpoint
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            # Offset label slightly để avoid overlap với arrow
            label_offset = 0.2
            if abs(dx) > abs(dy):  # More horizontal
                label_y = mid_y + label_offset if dy >= 0 else mid_y - label_offset
                label_x = mid_x
            else:  # More vertical
                label_x = mid_x + label_offset if dx >= 0 else mid_x - label_offset
                label_y = mid_y
            
            ax.text(label_x, label_y, label, ha='center', va='center', 
                   fontsize=8, color=self.colors['dark'], fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='white', 
                            edgecolor='none', alpha=0.8))

    def _draw_system_boundary_v2(self, ax, x, y, width, height, label):
        """Draw system boundary với enhanced styling"""
        boundary_rect = Rectangle((x, y), width, height,
                                fill=False, edgecolor=self.colors['secondary'],
                                linewidth=3, linestyle='--', alpha=0.7)
        ax.add_patch(boundary_rect)
        
        ax.text(x + width/2, y + height + 0.2, label, ha='center', va='bottom',
               fontsize=12, fontweight='bold', color=self.colors['secondary'])

def main():
    """Main function để tạo additional diagrams"""
    print("🎨 Generating Additional UML Diagrams...")
    
    visualizer = AdditionalDiagramsVisualizer()
    
    # Create additional diagrams
    print("🧩 Creating Component Diagram...")
    visualizer.create_component_diagram()
    
    print("🚀 Creating Deployment Diagram...")
    visualizer.create_deployment_diagram()
    
    print("📊 Creating Data Flow Diagram...")
    visualizer.create_data_flow_diagram()
    
    print("✅ Additional diagrams created successfully!")
    print("📁 Files saved:")
    print("   - restaurant_component_diagram.png")
    print("   - restaurant_deployment_diagram.png")
    print("   - restaurant_data_flow_diagram.png")

if __name__ == "__main__":
    main() 