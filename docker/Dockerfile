# 1. Chọn nền Python nhẹ (Slim version)
FROM python:3.9-slim

# 2. Tối ưu: Không tạo file .pyc rác và hiện log ngay lập tức
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Cài thư viện hệ thống (Bắt buộc để xử lý ảnh)
# ... (các đoạn trên giữ nguyên)

# 3. Cài thư viện hệ thống
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ... (các đoạn dưới giữ nguyên)

# 4. Tạo thư mục làm việc
WORKDIR /app

COPY requirements.txt .

# --- THÊM ĐOẠN NÀY VÀO ---
# Cài PyTorch bản CPU (Nhẹ, ổn định, không lỗi download)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
# -------------------------

# Cài các thư viện còn lại trong file requirements
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy toàn bộ code vào trong
COPY . .

# 8. Mở sẵn cổng (Chỉ mang tính chất tài liệu)
EXPOSE 8000
EXPOSE 8501

# Lệnh mặc định (Sẽ bị Docker Compose ghi đè, để bash cho an toàn)
CMD ["bash"]