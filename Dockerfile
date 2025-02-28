# Sử dụng Python 3.10
FROM python:3.10

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file requirements.txt và cài đặt dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . /app/

# Thu gọn static files
RUN python manage.py collectstatic --noinput

# Chạy migrate, tạo superuser rồi chạy Gunicorn
CMD python manage.py migrate && gunicorn items.wsgi:application --bind 0.0.0.0:$PORT