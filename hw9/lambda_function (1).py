import numpy as np
import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image

# Load model once (outside handler for Lambda optimization)
MODEL_PATH = "hair_classifier_empty.onnx"
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def download_image(url):
    """Download image from URL"""
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size=(200, 200)):
    """Prepare image for model input"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_image(img):
    """Convert image to model input format with ImageNet normalization"""
    x = np.array(img, dtype='float32')
    
    # ImageNet normalization (from homework 8)
    x = x / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        x[:, :, i] = (x[:, :, i] - mean[i]) / std[i]
    
    # Add batch dimension and transpose to NCHW format
    x = np.expand_dims(x, axis=0)
    x = np.transpose(x, (0, 3, 1, 2))
    
    return x


def predict(url):
    """Download image and run prediction"""
    img = download_image(url)
    img = prepare_image(img)
    input_data = preprocess_image(img)
    
    result = session.run([output_name], {input_name: input_data})
    prediction = float(result[0][0][0])
    
    return prediction


def lambda_handler(event, context):
    """AWS Lambda handler"""
    url = event.get('url')
    prediction = predict(url)
    
    return {
        'prediction': prediction
    }


# For local testing
if __name__ == "__main__":
    test_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    result = predict(test_url)
    print(f"Prediction: {result:.6f}")
