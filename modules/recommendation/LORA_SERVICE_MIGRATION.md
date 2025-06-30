# LoRA Service Migration Documentation

## Tổng quan
Đã chuyển đổi từ `lora_bge_service.py` (không tồn tại) sang `optimized_local_lora_service.py`

## Các thay đổi đã thực hiện

### 1. **Xóa file duplicate**
- Đã xóa: `/optimized_local_lora_service.py` (file ở root, 454 dòng)
- Giữ lại: `/modules/recommendation/optimized_local_lora_service.py` (666 dòng, có nhiều tính năng hơn)

### 2. **Cập nhật imports**
```python
# Từ (file không tồn tại):
from .lora_bge_service import LoRABGEM3Service

# Sang:
from .optimized_local_lora_service import OptimizedLocalLoRAService
```

### 3. **Cập nhật method calls**
```python
# Từ:
search_results = self.embedding_service.search_similar_pois(query, k)

# Sang:
search_results = self.embedding_service.search_similar(query, k, return_embeddings=False)
```

### 4. **Cập nhật FAISS database path**
```python
# Từ:
faiss_db_path = "modules/recommendation/faiss_db_restaurants"

# Sang:
faiss_db_path = "modules/recommendation/data/indices/restaurants"
```

## Ưu điểm của OptimizedLocalLoRAService

### Memory Optimization
- **Memory threshold**: 6000MB (tăng từ 1000MB)
- **Batch size**: 16 (giảm từ 32)
- **Max sequence length**: 256 (giảm từ 384)
- **Memory monitoring** với `MemoryMonitor` class
- **CUDA OOM handling** tự động

### Tính năng bổ sung
- `LimitedDict` với LRU eviction cho embeddings cache
- Force cleanup methods
- Sequential loading mode
- FP16 mixed precision support
- Detailed performance monitoring

## Lưu ý quan trọng

### Memory Requirements
Service này yêu cầu ít nhất **3.5GB RAM** để hoạt động ổn định. Nếu gặp lỗi "out of memory":

1. **Giảm batch size**: Sửa `batch_size=8` trong config
2. **Giảm max sequence length**: Sửa `max_sequence_length=128`
3. **Tắt các ứng dụng khác** để giải phóng RAM
4. **Sử dụng GPU** nếu có (cần ít nhất 4GB VRAM)

### File Structure
```
modules/recommendation/
├── optimized_local_lora_service.py  # Main service (666 lines)
├── restaurant_search_pipeline.py    # Updated to use OptimizedLocalLoRAService
├── data/
│   └── indices/
│       └── restaurants/            # FAISS database location
│           ├── faiss_index.bin
│           └── mappings.pkl
└── BGE-M3_embedding/              # LoRA adapter files
    ├── adapter_config.json
    ├── adapter_model.bin
    └── tokenizer.json
```

## Các file bị ảnh hưởng
1. `restaurant_search_pipeline.py` - Đã cập nhật imports và method calls
2. `optimized_local_lora_service.py` - Đã cập nhật default FAISS path

## Testing
```python
# Khởi tạo với config tối ưu memory
from modules.recommendation.optimized_local_lora_service import OptimizedLocalLoRAService, OptimizedConfig

config = OptimizedConfig(
    max_memory_threshold_mb=6000,
    batch_size=16,  # Giảm nếu OOM
    max_sequence_length=256  # Giảm nếu OOM
)

service = OptimizedLocalLoRAService(config=config)
```

## Troubleshooting

### "out of memory" error
- Giảm batch_size và max_sequence_length
- Kiểm tra memory usage với Task Manager
- Đảm bảo có ít nhất 4GB RAM free

### "FAISS database not found" error
- Kiểm tra path: `modules/recommendation/data/indices/restaurants/`
- Đảm bảo files `faiss_index.bin` và `mappings.pkl` tồn tại

### Import errors
- Sử dụng absolute imports khi chạy file trực tiếp
- Chạy như module: `python -m modules.recommendation.restaurant_search_pipeline` 