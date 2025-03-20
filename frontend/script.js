document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const predictButton = document.getElementById('predictButton');
    const previewImageDiv = document.getElementById('previewImage');
    const resultSection = document.getElementById('resultSection');
    const predictionResultDiv = document.getElementById('predictionResult');
    const confidenceLevelDiv = document.getElementById('confidenceLevel');
    const errorSection = document.getElementById('errorSection');
    const errorMessageDiv = document.getElementById('errorMessage');

    let selectedImageFile = null;

    imageUpload.addEventListener('change', (event) => {
        selectedImageFile = event.target.files[0];
        if (selectedImageFile) {
            predictButton.disabled = false;
            errorSection.classList.add('hidden'); // Ẩn khu vực lỗi nếu có lỗi trước đó
            errorMessageDiv.textContent = '';

            // Hiển thị ảnh preview (tùy chọn)
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImageDiv.innerHTML = `<img src="${e.target.result}" alt="Ảnh xem trước" style="max-width: 100%;">`;
            };
            reader.readAsDataURL(selectedImageFile);
        } else {
            predictButton.disabled = true;
            previewImageDiv.innerHTML = ''; // Xóa ảnh preview
        }
    });

    predictButton.addEventListener('click', async () => {
        if (!selectedImageFile) {
            return; // Không có ảnh nào được chọn
        }

        const formData = new FormData();
        formData.append('image', selectedImageFile);

        resultSection.classList.add('hidden'); // Ẩn khu vực kết quả cũ
        errorSection.classList.add('hidden'); // Ẩn khu vực lỗi cũ
        errorMessageDiv.textContent = '';
        predictionResultDiv.textContent = 'Đang dự đoán...'; // Hiển thị thông báo đang tải
        confidenceLevelDiv.textContent = '';


        try {
            const response = await fetch('/predict', { // Đảm bảo đường dẫn đúng với endpoint Flask
                method: 'POST',
                body: formData
            });
        
            console.log("Response object:", response); // IN RA RESPONSE OBJECT VÀO CONSOLE (THÊM DÒNG NÀY)
        
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        
            const data = await response.json();
            console.log("Data (parsed JSON):", data); // IN RA DATA JSON VÀO CONSOLE (THÊM DÒNG NÀY)
        
            if (data.error) {
                errorSection.classList.remove('hidden');
                errorMessageDiv.textContent = `Lỗi dự đoán: ${data.error}`;
                predictionResultDiv.textContent = 'Lỗi';
                confidenceLevelDiv.textContent = '';
            } else {
                resultSection.classList.remove('hidden');
                predictionResultDiv.textContent = `Nhãn dự đoán: ${data.label}`;
                confidenceLevelDiv.textContent = `Độ tin cậy: ${(data.confidence * 100).toFixed(2)}%`;
            }
        
        } catch (error) {
            errorSection.classList.remove('hidden');
            errorMessageDiv.textContent = `Lỗi mạng hoặc server: ${error.message}`;
            predictionResultDiv.textContent = 'Lỗi';
            confidenceLevelDiv.textContent = '';
        }
    });
});