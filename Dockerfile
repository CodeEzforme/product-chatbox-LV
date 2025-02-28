FROM python:3.10

# Đặt thư mục làm việc
WORKDIR /app

# Copy và cài đặt các dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào container
COPY . /app/

# Expose port
EXPOSE 8000

# Chạy lệnh migrate và khởi động Django Server
CMD ["sh", "-c", "python manage.py migrate && gunicorn items.wsgi:application --bind 0.0.0.0:8000"]
