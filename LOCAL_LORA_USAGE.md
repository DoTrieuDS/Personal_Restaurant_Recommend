# 🍽️ Local LoRA Restaurant Recommendation System

Hệ thống đề xuất nhà hàng sử dụng **Local LoRA BGE-M3** hoạt động **hoàn toàn offline**.

## ✅ Đã Hoàn Thành

- ✅ **Local LoRA BGE-M3** checkpoint tích hợp đầy đủ
- ✅ **FAISS database** với 7,661 nhà hàng  
- ✅ **Không cần internet** - hoạt động offline hoàn toàn
- ✅ **Web UI** sẵn sàng sử dụng
- ✅ **Advanced ML features** đã được tích hợp

## 🚀 Khởi Động Nhanh

### Option 1: Sử dụng Batch File (Dễ nhất)
```bash
# Double-click file này:
start_local_lora_server.bat
```

### Option 2: Command Line
```bash
# Set environment và start server:
set SKIP_PRELOAD=1
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🧪 Test Hệ Thống

### 1. Test API
```bash
python test_local_lora_system.py
```

### 2. Test Web UI
- Mở `restaurant_ui.html` trong browser
- API endpoint: `http://localhost:8000`

## 📊 Hệ Thống Thông Tin

### Local LoRA Model
- **Model**: BGE-M3 với LoRA adapter (22.4MB)
- **Device**: CPU (tối ưu cho máy thường)
- **Load time**: ~5 giây
- **Memory usage**: ~2GB RAM

### FAISS Database
- **Vectors**: 7,661 restaurant vectors
- **Cities**: 52 thành phố Mỹ/Canada
- **Search time**: < 100ms

### Web UI Features
- ✅ City selection (Philadelphia, Tampa, Nashville, etc.)
- ✅ Semantic search với Local LoRA
- ✅ Advanced ML recommendations
- ✅ User personalization
- ✅ Real-time feedback system

## 🎯 Test Queries

### Thử các query này:
```
City: Philadelphia
Query: "Italian restaurant"

City: Tampa  
Query: "seafood restaurant"

City: Nashville
Query: "barbecue restaurant"
```

## 🔧 Troubleshooting

### Server không khởi động
```bash
# Check Python modules:
pip install fastapi uvicorn sentence-transformers faiss-cpu torch peft

# Check port conflict:
netstat -an | findstr :8000
```

### OOM (Out of Memory)
```bash
# Sử dụng CPU-only mode:
set SKIP_PRELOAD=1

# Restart với ít RAM hơn:
# Close other applications
```

### Web UI không connect
```bash
# Verify server running:
curl http://localhost:8000/

# Check browser console for errors
# Ensure CORS headers working
```

## 📈 Performance Stats

| Metric | Value |
|--------|--------|
| Model Load Time | ~5s |
| Search Time | <100ms |
| Memory Usage | ~2GB |
| Vector Count | 7,661 |
| Offline Operation | ✅ Yes |

## 🎉 Success Confirmation

Khi hệ thống hoạt động tốt, bạn sẽ thấy:

```
✅ Optimized Pipeline với Local LoRA đã khởi tạo thành công sau 5s
✅ FAISS database loaded: 7661 vectors  
✅ Preload hoàn tất! Local LoRA model và FAISS database đã sẵn sàng
```

**🌐 Mở `restaurant_ui.html` để sử dụng Web UI!** 