# Sử dụng Python 3.10
FROM python:3.10

# Đặt thư mục làm việc
WORKDIR /app

# Copy requirements và cài đặt các dependencies
COPY requirements.txt /app/
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code vào container
COPY . /app/

# Expose port 8000
EXPOSE 8000

# Chạy migrate và khởi động server bằng Gunicorn
CMD ["sh", "-c", ". venv/bin/activate && python manage.py migrate && gunicorn items.wsgi:application --bind 0.0.0.0:${PORT:-8000}"]
