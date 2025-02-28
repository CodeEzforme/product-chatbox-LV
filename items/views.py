from django.http import JsonResponse
from .models import Item
from .serializers import ItemSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# giới hạn request 30
from django_ratelimit.decorators import ratelimit
from django.core.cache import cache #bộ nhớ ip request

# box-chat
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.http import require_http_methods
# import json

from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import json

# Cấu hình số lần vi phạm và thời gian block
BLOCK_DURATION = 10  # 1 giờ (tính bằng giây)
VIOLATION_LIMIT = 1000   # Số lần vượt quá giới hạn trước khi block

# kết hợp mô hình PhoBERT + VIT
import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel, AutoTokenizer, AutoModel
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# ✅ Cấu hình mô hình
# ----------------------------
# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
model_phobert = AutoModel.from_pretrained("vinai/phobert-base").to(device)

# Load ViT
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model_vit = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)

# Định nghĩa mô hình VQA
class VQAModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VQAModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, combined_features):
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load dữ liệu huấn luyện để lấy LabelEncoder
df_train = pd.read_csv(r"C:\HOCTAP\Back-end\mohinh\items\data3\train_kaggle.csv")
le = LabelEncoder()
df_train["encoded_answer"] = le.fit_transform(df_train["answer"])
num_classes = len(df_train["encoded_answer"].unique())

# Khởi tạo và load mô hình VQA
model_vqa = VQAModel(input_dim=768+768, hidden_dim=512, output_dim=num_classes).to(device)
checkpoint_path = r"C:\HOCTAP\Back-end\mohinh\items\data3\model_16_1\vqa_epoch_15.pth"  # Đường dẫn tới file model
model_vqa.load_state_dict(torch.load(checkpoint_path, map_location=device))
model_vqa.eval()

# ----------------------------
# ✅ Cấu hình giới hạn request
# ----------------------------
VIOLATION_LIMIT = 30 # Số lần vi phạm trước khi block

# ----------------------------
# ✅ Hàm lấy địa chỉ IP từ request
# ----------------------------
def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

# Hàm tiền xử lý câu hỏi (tách từ với PhoBERT)
def word_segment(text):
    return " ".join(tokenizer.tokenize(text))
# ----------------------------
# ✅ Hàm dự đoán từ mô hình VQA
# ----------------------------
def predict_answer(image_path, question):
    # Tiền xử lý ảnh
    image = Image.open(image_path).convert("RGB")
    image_inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    image_features = model_vit(**image_inputs).last_hidden_state[:, 0, :]

    # Tiền xử lý câu hỏi
    question_processed = word_segment(question)
    question_inputs = tokenizer(question_processed, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    question_features = model_phobert(**question_inputs).last_hidden_state[:, 0, :]

    # Kết hợp đặc trưng ảnh và câu hỏi
    combined_features = torch.cat((image_features, question_features), dim=1)

    # Dự đoán
    with torch.no_grad():
        output = model_vqa(combined_features)
        _, predicted_idx = torch.max(output, 1)

    # Chuyển chỉ số thành câu trả lời
    predicted_answer = le.inverse_transform(predicted_idx.cpu().numpy())
    return predicted_answer[0]

# ----------------------------
# ✅ API Chat + Dự đoán từ mô hình
# ----------------------------
@csrf_exempt
@ratelimit(key='ip', rate='30/m', block=False)  # Giới hạn 30 requests/phút
def chat_api(request):
    if request.method == 'POST':
        # 📡 Lấy địa chỉ IP người dùng
        ip_address = get_client_ip(request)
        print(f"📡 Địa chỉ IP người dùng: {ip_address}")

        # Kiểm tra block IP
        block_key = f"blocked_{ip_address}"
        violation_key = f"violations_{ip_address}"

        if cache.get(block_key):
            return JsonResponse({'error': '🚫 IP của bạn đã bị chặn. Hãy thử lại sau.'}, status=403)

        # Kiểm tra rate limit
        if getattr(request, 'limited', False):
            violations = cache.get(violation_key, 0) + 1
            cache.set(violation_key, violations, timeout=BLOCK_DURATION)
            if violations >= VIOLATION_LIMIT:
                cache.set(block_key, True, timeout=BLOCK_DURATION)
                return JsonResponse({'error': '🚫 Bạn đã bị chặn sau nhiều lần vi phạm.'}, status=403)
            return JsonResponse({'error': f'⚠️ Quá nhiều yêu cầu! Vi phạm {violations}/{VIOLATION_LIMIT}.'}, status=429)

        # ✅ Lấy câu hỏi và hình ảnh từ request
        user_message = request.POST.get('message', '')
        uploaded_image = request.FILES.get('image', None)

        if not user_message or not uploaded_image:
            return JsonResponse({'error': '⚠️ Vui lòng gửi cả câu hỏi và hình ảnh.'}, status=400)

        # Lưu hình ảnh tạm thời
        image_path = default_storage.save(uploaded_image.name, uploaded_image)
        image_full_path = default_storage.path(image_path)

        try:
            # 🎯 Gọi hàm dự đoán từ mô hình
            predicted_answer = predict_answer(image_full_path, user_message)

            return JsonResponse({
                'question': user_message,
                'answer': predicted_answer
            }, status=200)

        except Exception as e:
            return JsonResponse({'error': f'❌ Lỗi trong quá trình dự đoán: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Method Not Allowed'}, status=405)
# hết kết hợp mô hình PhoBERT + VIT


# @csrf_exempt
# @ratelimit(key='ip', rate='30/m', block=False)  # Theo dõi nhưng không block ngay lập tức
# def chat_api(request):
#     if request.method == 'POST':
#         # 🔥 Lấy và in ra địa chỉ IP
#         ip_address = get_client_ip(request)
#         print(f"📡 Địa chỉ IP người dùng (ratelimit key='ip'): {ip_address}")

#         # Key lưu trong cache
#         block_key = f"blocked_{ip_address}"
#         violation_key = f"violations_{ip_address}"

#         # ✅ Kiểm tra nếu IP đã bị block
#         if cache.get(block_key):
#             return JsonResponse({
#                 'error': '🚫 IP của bạn đã bị chặn do vi phạm nhiều lần. Hãy thử lại sau.'
#             }, status=403)

#         # ✅ Kiểm tra nếu vượt quá rate limit
#         if getattr(request, 'limited', False):
#             # Tăng số lần vi phạm
#             violations = cache.get(violation_key, 0) + 1
#             cache.set(violation_key, violations, timeout=BLOCK_DURATION)

#             # Nếu vi phạm >= giới hạn, block IP
#             if violations >= VIOLATION_LIMIT:
#                 cache.set(block_key, True, timeout=BLOCK_DURATION)
#                 return JsonResponse({
#                     'error': '🚫 Bạn đã bị chặn sau nhiều lần vi phạm.'
#                 }, status=403)

#             # Trả về cảnh báo nếu chưa bị block nhưng đã vi phạm
#             return JsonResponse({
#                 'error': f'⚠️ Bạn đã gửi quá nhiều yêu cầu! Đây là vi phạm {violations}/{VIOLATION_LIMIT}.'
#             }, status=429)

#         # ✅ Xử lý tin nhắn và hình ảnh nếu chưa bị block
#         user_message = request.POST.get('message', '')
#         uploaded_image = request.FILES.get('image', None)

#         response_text = f"Bạn vừa hỏi: '{user_message}'"

#         # Nếu có ảnh, lưu và trả link
#         if uploaded_image:
#             image_path = default_storage.save(uploaded_image.name, uploaded_image)
#             image_url = f"/media/{image_path}"
#             response_text += f"\n📷 Đã nhận ảnh: {image_url}"

#         return JsonResponse({'reply': response_text}, status=200)

#     # Nếu không phải POST method
#     return JsonResponse({'error': 'Method Not Allowed'}, status=405)
# # end box-chat

# # ✅ Hàm lấy IP từ request
# def get_client_ip(request):
#     x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
#     if x_forwarded_for:
#         # Nếu đi qua proxy hoặc load balancer
#         ip = x_forwarded_for.split(',')[0]
#     else:
#         # Nếu kết nối trực tiếp
#         ip = request.META.get('REMOTE_ADDR')
#     return ip

@api_view(['GET', 'POST'])
def item_list(request, format=None):
    
    #get all items
    if request.method == 'GET':
        #serialize them
        #return json
        items = Item.objects.all()
        serializer = ItemSerializer(items, many=True)
        # return JsonResponse({'items': serializer.data})
        return Response(serializer.data)

    if request.method == 'POST':
        serializer = ItemSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
         
@api_view(['GET', 'PUT', 'DELETE'])
def item_detail(request, id, format=None):
    
    try:
        item = Item.objects.get(pk=id)
    except Item.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer =  ItemSerializer(item)
        return Response(serializer.data)
    
    elif request.method == 'PUT':
        serializer =  ItemSerializer(item, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        item.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)