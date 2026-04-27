from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join('model', 'saved model', 'model.h5')
CLASS_NAMES = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Tumor',
    3: 'Pituitary Tumor',
}

_model = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((128, 128))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        model = get_model()
        image = preprocess_image(filepath)
        probabilities = model.predict(image, verbose=0)[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])

        prediction = CLASS_NAMES.get(class_index, 'Unknown')
        message = f'Prediction: {prediction} ({confidence:.2%} confidence)'
    except Exception as exc:
        message = f'Prediction failed: {exc}'

    return render_template('result.html', prediction=message, image_name=file.filename)


if __name__ == '__main__':
    app.run(debug=True)
