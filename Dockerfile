# Sử dụng Python 3.10 thay vì 3.12
FROM python:3.10

# Cập nhật hệ thống và cài đặt thư viện cần thiết
RUN apt-get update && apt-get install -y \
    python3-distutils \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    gcc \
    g++ \
    make

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy toàn bộ mã nguồn vào container
COPY . .

# Cài đặt các package trong requirements.txt mà không dùng venv
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Chạy ứng dụng với Gunicorn
CMD gunicorn items.wsgi --bind 0.0.0.0:$PORT
