from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from torchvision import transforms
from model import SimpleCNN  # Đảm bảo file model.py ở cùng thư mục và chứa class SimpleCNN
import numpy as np
from flask_cors import CORS  # Import thư viện Flask-CORS

app = Flask(__name__)
CORS(app)  # Enable CORS cho toàn bộ ứng dụng Flask

# *** CẤU HÌNH DEVICE (GPU nếu có, CPU nếu không) ***
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# *** TRANSFORMATIONS CHO ẢNH (GIỐNG VALIDATION/TEST) ***
image_size = 128  # *** ĐIỀU CHỈNH KÍCH THƯỚC ẢNH NẾU MODEL CỦA BẠN YÊU CẦU KHÁC ***
test_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # *** KIỂM TRA VÀ ĐIỀU CHỈNH THÔNG SỐ CHUẨN HÓA NẾU CẦN THIẾT ***
])

# *** LOAD MODEL PYTORCH (MỘT LẦN KHI SERVER KHỞI ĐỘNG) ***
model_path = 'cat_dog_model.pth'  # *** ĐẢM BẢO FILE model.pth Ở CÙNG THƯ MỤC HOẶC ĐƯỜNG DẪN ĐÚNG ***
model = SimpleCNN(num_classes=2).to(device)  # num_classes=2 cho chó/mèo
try:
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Load weights
    print(f"Model loaded successfully from: {model_path}") # In thông báo khi load model thành công
except FileNotFoundError:
    print(f"Error: Model file not found at: {model_path}. Please check the path.")
    exit() # Dừng chương trình nếu không tìm thấy model
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval()  # Chuyển sang chế độ evaluation

# *** HÀM PHÂN LOẠI ẢNH CỦA BẠN (ĐÃ CHỈNH SỬA CHO PYTORCH) ***
def predict_image(image_data):
    """
    Hàm này nhận dữ liệu ảnh (dạng bytes) và trả về kết quả dự đoán (ví dụ: "mèo" hoặc "chó")
    SỬ DỤNG MODEL PYTORCH ĐÃ LOAD Ở NGOÀI HÀM.
    """
    try:
        # 1. Đọc ảnh từ dữ liệu bytes
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # 2. Tiền xử lý ảnh (SỬ DỤNG TRANSFORMATIONS ĐÃ ĐỊNH NGHĨA)
        image_tensor = test_transforms(image).unsqueeze(0).to(device)  # Áp dụng transformations và chuyển lên device

        # 3. Dự đoán bằng model đã load (TRONG CHẾ ĐỘ EVALUATION)
        with torch.no_grad():  # Tắt tính toán gradient khi dự đoán
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)  # Chuyển output thành probabilities
            _, predicted_class = torch.max(output, 1)  # Lấy lớp có xác suất cao nhất

        # 4. Xử lý kết quả dự đoán
        predicted_class = predicted_class.item()  # Lấy giá trị số của lớp dự đoán (0 hoặc 1)
        probabilities_np = probabilities.cpu().numpy()[0]  # Chuyển probabilities về numpy array và về CPU

        labels = ["mèo", "chó"]  # *** ĐẢM BẢO THỨ TỰ NHÃN ĐÚNG VỚI THỨ TỰ LỚP MODEL HUẤN LUYỆN (0: mèo, 1: chó HOẶC NGƯỢC LẠI) ***
        predicted_label = labels[predicted_class]  # Lấy nhãn tương ứng từ index lớp
        confidence = float(probabilities_np[predicted_class])  # Lấy độ tin cậy của lớp dự đoán (chuyển sang float để JSON serialize được)

        return {"label": predicted_label, "confidence": confidence}  # Trả về kết quả dạng JSON

    except Exception as e:
        return {"error": str(e)}  # Xử lý lỗi nếu có


# *** API ENDPOINT ĐỂ NHẬN ẢNH VÀ TRẢ VỀ KẾT QUẢ DỰ ĐOÁN ***
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Không có file ảnh được tải lên"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "Không có file ảnh được chọn"}), 400

    try:
        image_data = image_file.read()  # Đọc dữ liệu ảnh dạng bytes
        prediction_result = predict_image(image_data)  # Gọi hàm phân loại ảnh của bạn
        return jsonify(prediction_result)  # Trả về kết quả dạng JSON

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Lỗi server


if __name__ == '__main__':
    app.run(debug=True)  # Chạy server Flask (debug=True để dễ dàng phát triển)