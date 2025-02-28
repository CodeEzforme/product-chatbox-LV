# Sử dụng Python 3.10 làm base image
FROM python:3.10

# Đặt thư mục làm việc trong container
WORKDIR /app

# Copy toàn bộ mã nguồn vào container
COPY . /app/

# Cài đặt dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Chạy lệnh migrate để cập nhật database
RUN python manage.py migrate

# Expose cổng 8000
EXPOSE 8000

# Chạy Gunicorn để serve Django app
CMD ["gunicorn", "items.wsgi:application", "--bind", "0.0.0.0:8000"]
