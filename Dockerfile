# Sử dụng Python 3.10 thay vì 3.12
FROM python:3.10

# Cập nhật hệ thống và cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    python3-distutils \
    python3-venv \
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

# Tạo virtual environment
RUN python -m venv venv
RUN . venv/bin/activate && pip install --upgrade pip setuptools wheel

# Cài đặt các package trong requirements.txt
RUN . venv/bin/activate && pip install -r requirements.txt

# Chạy ứng dụng với Daphne
CMD . venv/bin/activate && daphne -b 0.0.0.0 -p $PORT items.asgi:application