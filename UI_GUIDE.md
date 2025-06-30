# Restaurant Recommendation System - User Interface Guide

Hướng dẫn sử dụng đầy đủ cho hệ thống giao diện người dùng của Restaurant Recommendation System.

## Tổng Quan

Dự án này bao gồm 2 giao diện chính:

1. **User Interface** (`restaurant_ui.html`) - Dành cho người dùng cuối tìm kiếm nhà hàng
2. **Admin Dashboard** (`admin_dashboard.html`) - Dành cho quản trị viên theo dõi hệ thống

## Khởi Động Hệ Thống

### 1. Khởi động Backend API

```bash
# Trong thư mục gốc của dự án
python main.py
```

Server sẽ chạy tại: `http://localhost:8000`

### 2. Mở User Interface

Mở file `restaurant_ui.html` trong trình duyệt web hoặc sử dụng Live Server.

### 3. Mở Admin Dashboard (tùy chọn)

Mở file `admin_dashboard.html` trong tab/cửa sổ khác.

---

## User Interface - Hướng Dẫn Sử Dụng

### **Tính Năng Chính**

#### **Dashboard Thống Kê** (cũ)
- **6,523 Restaurants Available**: Tổng số nhà hàng trong database
- **50+ Cities Covered**: Số thành phố được hỗ trợ
- **78% Recommendation Accuracy**: Độ chính xác của hệ thống ML
- **4.2 User Satisfaction**: Điểm hài lòng trung bình

#### **Search Panel**

**1. City Selection**
```
Philadelphia, PA
Tampa, FL  
Indianapolis, IN
New Orleans, LA
Nashville, TN
Las Vegas, NV
Charlotte, NC
Portland, OR
Scottsdale, AZ
Cleveland, OH
```

**2. Query Input**
- Nhập mô tả món ăn/nhà hàng bạn muốn
- Ví dụ: "Italian restaurant", "sushi", "fine dining"

**3. User ID (Personalization)**
- Nhập User ID để bật tính năng cá nhân hóa
- Để trống nếu muốn tìm kiếm chung

**4. Number of Results**
- Chọn số lượng kết quả: 5, 10, 15, hoặc 20

#### **Advanced Options**

**Use Personalization**
- Bật/tắt tính năng cá nhân hóa dựa trên lịch sử người dùng

**Advanced ML Features**  
- Sử dụng Advanced ML Service (Phase 5-7)
- Bao gồm: Collaborative Filtering, Deep Learning, Context-Aware

**Show ML Components**
- Hiển thị breakdown của các ML components
- Useful cho debugging và phân tích

**Exploration Factor (0.0 - 1.0)**
- `0.0`: An toàn, đề xuất theo sở thích đã biết
- `1.0`: Thám hiểm, đề xuất những điều mới lạ
- Mặc định: `0.1`

### **Kết Quả Tìm Kiếm**

#### **Restaurant Card Information**

**Basic Info:**
- **Name & Rank**: Tên nhà hàng và thứ hạng
- **City & Address**: Vị trí địa lý
- **Rating & Reviews**: Số sao và số lượt đánh giá
- **Cuisine Types**: Loại ẩm thực
- **Price Level**: Mức giá

**Scoring System:**
- **Final Score**: Điểm tổng hợp cuối cùng (0.0-1.0)
- **Similarity**: Độ tương đồng với query
- **Personal**: Điểm cá nhân hóa (nếu có User ID)
- **Confidence**: Độ tin cậy của đề xuất

#### **ML Components Breakdown** (nếu bật)

- **Collaborative**: Điểm từ Collaborative Filtering
- **Content**: Điểm từ Content-based Filtering  
- **Deep Learning**: Điểm từ Neural Networks
- **Context**: Điểm từ Context-aware Algorithms

#### **Why Recommended**
Tags giải thích tại sao nhà hàng được đề xuất:
- "Highly rated restaurant"
- "Excellent match for your search"
- "Matches your preferences"
- "Popular with many reviews"

#### **User Match Factors**
Tags thể hiện các yếu tố phù hợp với người dùng:
- Dựa trên lịch sử tương tác
- Sở thích ẩm thực
- Patterns hành vi

#### **Feedback System**
- **Like Button**: Ghi nhận phản hồi tích cực
- **Dislike Button**: Ghi nhận phản hồi tiêu cực
- **Real-time Learning**: Cải thiện đề xuất tương lai

### **API Endpoints Usage**

#### **Standard Search**
```javascript
POST /restaurant/search
{
    "city": "Philadelphia",
    "user_query": "Italian restaurant", 
    "user_id": "user_001",
    "num_results": 10,
    "use_personalization": true
}
```

#### **Advanced ML Search** 
```javascript
POST /advanced/recommendations
{
    "city": "Philadelphia",
    "user_query": "Italian restaurant",
    "user_id": "user_001", 
    "num_results": 10,
    "use_personalization": true,
    "exploration_factor": 0.1
}
```

#### **Feedback Submission**
```javascript
POST /restaurant/feedback
{
    "user_id": "user_001",
    "business_id": "restaurant_id_123",
    "feedback_type": "like",
    "rating": 5
}
```

---

## Admin Dashboard - Hướng Dẫn Sử Dụng

### **Navigation Sections**

#### **Overview**
- **Real-time Statistics**: Requests, Users, Response Time, Uptime
- **Request Trends Chart**: 24H/7D/30D views
- **User Satisfaction Distribution**: Pie chart breakdown

#### **Analytics** 
- **ML Performance Metrics**: Accuracy, CTR, Satisfaction
- **ML Components Radar Chart**: Performance comparison
- **Top Performing Cities**: Request volume & satisfaction

#### **Performance**
- **System Health Indicators**: API, Database, ML Models, Cache
- **Response Time Trends**: Performance over time
- **Resource Usage Tables**: CPU, Memory, Disk, Network

#### **Users**
- **User Statistics**: Total, New, Returning users
- **User Segments Chart**: Food Explorers, Local Loyalists, etc.
- **Recent Activity Table**: Real-time user interactions

#### **Restaurants**
- **Database Statistics**: 6,523 restaurants, 52 cities
- **Top Recommended Table**: Most popular restaurants
- **Performance Metrics**: CTR by restaurant

#### **System Health**
- **Live System Monitoring**: All services status
- **Health Score Charts**: Component-wise performance
- **Operational Alerts**: Real-time system issues

### **Key Performance Indicators (KPIs)**

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Recommendation Accuracy | 78% | 80% | Good |
| Click-through Rate | 34% | 30% | Excellent |
| Avg Response Time | 450ms | 500ms | Good |
| P95 Response Time | 780ms | 1000ms | Good |
| Cache Hit Rate | 62% | 85% | Needs Improvement |
| System Uptime | 99.9% | 99.5% | Excellent |

---

## Technical Implementation

### **Frontend Technologies**
- **HTML5**: Semantic markup
- **CSS3**: Modern styling với CSS Grid/Flexbox
- **JavaScript ES6+**: Async/await, Fetch API
- **Chart.js**: Data visualization
- **Font Awesome**: Icons
- **Google Fonts**: Typography (Inter)

### **Responsive Design**
- **Desktop First**: Optimized cho desktop experience
- **Mobile Responsive**: Media queries cho mobile/tablet
- **Cross-browser Compatible**: Chrome, Firefox, Safari, Edge

### **Performance Features**
- **Lazy Loading**: Charts chỉ load khi cần
- **Debounced Refresh**: Tránh spam API calls
- **Error Handling**: Graceful error messages
- **Loading States**: Visual feedback cho user

---

## UI/UX Design Principles

### **Color Scheme**
- **Primary**: `#667eea` (Blue gradient start)
- **Secondary**: `#764ba2` (Purple gradient end)
- **Success**: `#28a745` (Green)
- **Warning**: `#ffc107` (Yellow)
- **Danger**: `#dc3545` (Red)
- **Background**: `#f8f9fa` (Light gray)

### **Typography**
- **Font Family**: Inter (Google Fonts)
- **Weights**: 300, 400, 500, 600, 700
- **Hierarchy**: Clear heading/body text distinction

### **Interactive Elements**
- **Hover Effects**: Subtle animations
- **Focus States**: Accessibility compliant
- **Visual Feedback**: Button states & loading indicators
- **Smooth Transitions**: 0.3s ease transitions

---

## Troubleshooting

### **Common Issues**

#### **1. "API error: 500" khi search**
```
Solution: 
- Kiểm tra backend server đang chạy
- Restart main.py 
- Kiểm tra log trong terminal
```

#### **2. "No restaurants found"**
```
Solution:
- Thử với các cities được support
- Kiểm tra spelling của query
- Thử với query đơn giản hơn ("restaurant")
```

#### **3. Charts không hiển thị**
```
Solution:
- Kiểm tra connection internet (Chart.js CDN)
- Refresh page
- Check console errors (F12)
```

#### **4. Personalization không hoạt động**
```
Solution:
- Nhập User ID hợp lệ
- Tích "Use Personalization" checkbox
- Thử với user đã có interaction history
```

### **Debug Mode**

Mở Developer Tools (F12) để xem:
- **Console**: JavaScript errors & API responses
- **Network**: API request/response details
- **Application**: Local storage & session data

---

## Mobile Usage

### **Responsive Breakpoints**
- **Desktop**: > 768px (Full layout)
- **Tablet**: 768px - 480px (Adapted layout)  
- **Mobile**: < 480px (Stacked layout)

### **Mobile-Specific Features**
- **Touch-friendly**: Buttons và interactive elements
- **Swipe Support**: Horizontal scrolling cho tables
- **Hamburger Menu**: Collapsible navigation (Admin Dashboard)

---

## Security Considerations

### **Data Privacy**
- **No sensitive data storage**: Chỉ User ID trong memory
- **HTTPS recommended**: Production deployment
- **CORS enabled**: Cross-origin requests support

### **API Security**
- **Rate limiting**: Backend protection
- **Input validation**: XSS prevention
- **Error handling**: No sensitive info exposure

---

## Production Deployment

### **Performance Optimization**
```html
<!-- Minify CSS/JS files -->
<!-- Enable gzip compression -->
<!-- Use CDN for external resources -->
<!-- Implement service worker for caching -->
```

### **Monitoring**
- **Real User Monitoring (RUM)**: Track actual user performance
- **Error Tracking**: Sentry integration
- **Analytics**: Google Analytics for usage patterns

### **SEO (nếu cần)**
```html
<!-- Meta tags -->
<meta name="description" content="AI-powered restaurant recommendations">
<meta name="keywords" content="restaurant,recommendation,AI,food">
<meta property="og:title" content="Restaurant Recommendation System">
```

---

## Support & Contribution

### **Bug Reports**
Báo cáo bugs qua GitHub Issues với:
- Browser & version
- Steps to reproduce  
- Expected vs actual behavior
- Screenshots (nếu có)

### **Feature Requests**
- User story format
- Business justification
- Technical feasibility assessment

### **Code Contribution**
- Follow existing code style
- Add comments for complex logic
- Test trên multiple browsers
- Update documentation

---

**🎉 Chúc bạn có trải nghiệm tuyệt vời với Restaurant Recommendation System!** 