from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
import torch
from torchvision import transforms
from backend.model import SimpleCNN  # Import từ backend cho đúng cấu trúc
import numpy as np
from flask_cors import CORS
import os  # Thêm import os để lấy biến môi trường
import logging # Thêm logging để theo dõi lỗi tốt hơn

app = Flask(__name__)
CORS(app)

# Cấu hình logging (ghi log ra console trên Render)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 128
test_transforms = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

model_path = "backend/cat_dog_model.pth"  # Đường dẫn đến model
model = SimpleCNN(num_classes=2).to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Model loaded successfully from: {model_path}") # Sử dụng logger.info
except FileNotFoundError:
    error_message = f"Error: Model file not found at: {model_path}. Please check the path."
    logger.error(error_message) # Sử dụng logger.error
    exit()
except Exception as e:
    error_message = f"Error loading model: {e}"
    logger.error(error_message) # Sử dụng logger.error
    exit()

model.eval()

labels = ["mèo", "chó"] # Labels, có thể cấu hình sau nếu cần


def predict_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_tensor = test_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted_class = torch.max(output, 1)

        predicted_class = predicted_class.item()
        probabilities_np = probabilities.cpu().numpy()[0]


        predicted_label = labels[predicted_class]
        confidence = float(probabilities_np[predicted_class])

        return {"label": predicted_label, "confidence": confidence}

    except Exception as e:
        error_message = f"Error during prediction: {e}"
        logger.error(error_message) # Sử dụng logger.error
        return {"error": str(e)}


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Không có file ảnh được tải lên"}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "Không có file ảnh được chọn"}), 400

    try:
        image_data = image_file.read()
        prediction_result = predict_image(image_data)
        return jsonify(prediction_result)

    except Exception as e:
        error_message = f"Error processing image: {e}"
        logger.error(error_message) # Sử dụng logger.error
        return jsonify({"error": str(e)}), 500


# Phục vụ frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")


@app.route("/")
def serve_index():
    return send_from_directory(frontend_dir, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(frontend_dir, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Lấy port từ biến môi trường
    app.run(host="0.0.0.0", port=port) # Loại bỏ debug=True cho production