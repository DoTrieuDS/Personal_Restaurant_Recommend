# Recommendation System

Hệ thống AI-powered personalize restaurant recommendation với BGE-M3 embeddings và FAISS vector search.

## Quick Start

### Khởi động Server
```bash
# Chế độ nhanh (không preload models)
python start_simple.py

# Chế độ đầy đủ (preload models)  
python start_server.py
```

### Test Giao diện
- **test_working.html** - Test search functionality
- **test_api.html** - Test enhanced recommendation

## Project Structure

```
Restaurant_recommendation_system/
├── main.py                 # FastAPI main application
├── start_simple.py         # Quick start server
├── start_server.py         # Full server with preload
├── requirements.txt        # Python dependencies
├── test_working.html       # Web test interface
├── recommendation_output_spec.py  # Output schemas
│
├── modules/
│   ├── recommendation/     # Core recommendation module
│   │   ├── router.py         # FastAPI endpoints
│   │   ├── service.py        # Business logic
│   │   ├── recommendation_pipeline.py  # Main pipeline
│   │   ├── lora_bge_service.py  # BGE-M3 embedding
│   │   ├── faiss_vectorDB.py  # Vector database
│   │   ├── enhanced_recommendation.py  # Result processing
│   │   ├── planner_integration.py  # Planner format
│   │   ├── faiss_db/      # FAISS database (70MB)
│   │   ├── BGE-M3_embedding/  # Fine-tuned model
│   │   └── data/          # POI embeddings & metadata
│   ├── memory/            # Session management
│   └── domain/            # Schemas & models
│
└── shared/                # Shared config & dependencies
    ├── kernel.py            # Dependency injection
    └── settings.py          # App settings
```

## Core Components

### Recommendation Pipeline
1. **BGE-M3 Embedding** - Text to vector encoding
2. **FAISS Search** - Fast similarity search (17,873 POIs)
3. **Mock Reranking** - Rule-based scoring (placeholder for Llama-3)
4. **Result Enhancement** - Categorization & planning helpers

### API Endpoints
- `GET /` - Server status & info
- `GET /recommendation/health` - Health check  
- `POST /recommendation/search` - Direct search
- `POST /recommendation/enhanced-recommend` - Enhanced recommendations

## Database
- **FAISS Index**: 17,873 POI embeddings (70MB)
- **Metadata**: POI info, ratings, categories
- **BGE-M3 Model**: Fine-tuned embeddings (1024-dim)

## Features
- Semantic POI search with BGE-M3
- Fast FAISS vector similarity  
- T5 reranking
- T5 personal message generation
- Travel-optimized categorization
- Web test interfaces
- Lightweight/Full modes


## Testing
```bash
# API health check
curl http://localhost:8001/recommendation/health

# Search test
curl -X POST http://localhost:8001/recommendation/search \
  -H "Content-Type: application/json" \
  -d '{"query": "restaurant", "top_k": 5}'
```

## Performance
- **Search Time**: ~0.2-0.6s for retrieval
- **Database Size**: ~150MB total
- **Memory Usage**: ~200MB runtime
- **POI Coverage**: 17,873 POIs across multiple cities

---
**Status**: Production-ready recommendation system với mock reranking
**Next**: Integrate real T5 cho advanced reranking & personal message generation
