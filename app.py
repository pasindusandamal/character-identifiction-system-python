from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

app = Flask(__name__)

model = load_model("sign.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

uploaded_files = []

@app.route('/predict', methods=['POST'])

def predict():
    global uploaded_files
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        predictions = []
        for file in files:
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            uploaded_files.append(file_path)
            data = preprocess_image(file_path)
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index][2:].strip()
            predictions.append(class_name)
        return jsonify({
            "status": "uploaded",
            "message": "Images uploaded successfully."
        })

@app.route('/final_result')
def final_result():
    global uploaded_files
    predictions = []
    for file_path in uploaded_files:
        data = preprocess_image(file_path)
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index][2:].strip()
        predictions.append(class_name)
    uploaded_files = []  # Clear uploaded files for next submission
    return render_template('result.html', predictions=predictions)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
