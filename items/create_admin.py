import os
import django

# Cấu hình Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "items.settings")
django.setup()

from django.contrib.auth.models import User

# Tạo Superuser nếu chưa tồn tại
admin_username = os.getenv("ADMIN_USERNAME", "admin")
admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
admin_password = os.getenv("ADMIN_PASSWORD", "admin123")

if not User.objects.filter(username=admin_username).exists():
    User.objects.create_superuser(admin_username, admin_email, admin_password)
    print(f"✅ Superuser '{admin_username}' đã được tạo!")
else:
    print(f"⚠️ Superuser '{admin_username}' đã tồn tại.")
