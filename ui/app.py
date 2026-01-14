# ui/app.py
import streamlit as st
import requests
from PIL import Image
import io
import base64

# --- CẤU HÌNH ---
API_URL = "http://127.0.0.1:8000/predict"  # Địa chỉ API FastAPI đang chạy
st.set_page_config(page_title="Knee OA Detection", page_icon="🦵", layout="wide")

# --- CSS TÙY CHỈNH CHO ĐẸP ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🦵 Knee Osteoarthritis Detection AI")
st.markdown("Hệ thống chẩn đoán mức độ thoái hóa khớp gối tự động sử dụng **EfficientNet-B0** & **Grad-CAM**.")
st.write("---")

# --- SIDEBAR (Upload ảnh) ---
with st.sidebar:
    st.header("📤 Input X-ray")
    uploaded_file = st.file_uploader("Chọn ảnh X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Hiển thị ảnh nhỏ ở sidebar
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã chọn", use_container_width=True)

# --- MAIN SECTION ---
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Nút bấm Dự đoán
    if st.sidebar.button("🚀 CHẨN ĐOÁN NGAY"):
        with st.spinner("Đang phân tích... Vui lòng chờ..."):
            try:
                # 1. Gửi ảnh sang API FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)
                
                # 2. Xử lý kết quả trả về
                if response.status_code == 200:
                    result = response.json()
                    pred_class = result['prediction']
                    confidence = result['confidence']
                    heatmap_b64 = result['heatmap_base64']
                    
                    # Giải mã ảnh Heatmap từ Base64
                    heatmap_bytes = base64.b64decode(heatmap_b64)
                    heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                    
                    # --- HIỂN THỊ KẾT QUẢ ---
                    with col1:
                        st.subheader("🖼️ Ảnh gốc")
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        st.subheader("🔥 Vùng tổn thương (AI Focus)")
                        st.image(heatmap_img, use_container_width=True)
                    
                    # Hiển thị thông số kết quả
                    st.write("---")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="color: #d9534f;">KẾT QUẢ: Mức độ {pred_class}</h2>
                        <h3>Độ tin cậy: {confidence}</h3>
                        <p>AI tập trung vào các vùng đỏ trên Heatmap để đưa ra dự đoán này.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error(f"Lỗi API: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Không thể kết nối đến API! Hãy chắc chắn bạn đã chạy 'python -m uvicorn api.main:app' chưa?")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")

else:
    # Màn hình chờ khi chưa upload
    with col1:
        st.info("👈 Vui lòng tải ảnh X-ray lên từ thanh bên trái.")
    with col2:
        st.empty()