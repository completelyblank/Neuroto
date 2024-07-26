from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load the TensorFlow model
model = tf.keras.models.load_model('models/PhotoClassifier.h5')

class_names = ['landscape', 'minimalist', 'monochrome', 'nightime', 'portrait', 'street_photography', 'vintage']

# Load the YOLOv8 model
yolo_model = YOLO('yolov8n.pt')  # Use the appropriate YOLOv8 model file

# Mapping YOLO class IDs to human-readable labels (YOLOv8 uses COCO dataset labels)
object_labels = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
    28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
    53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'sofa', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'TV monitor', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}

def preprocess_image(image):
    img = np.array(image.convert('RGB'))
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    resize = tf.image.resize(img_tensor, (256, 256))
    resize_np = resize.numpy().astype(np.uint8)
    resize_np = np.expand_dims(resize_np, axis=0) / 255.0
    return resize_np

def detect_objects(image):
    results = yolo_model(image)
    detected_objects = []
    
    for result in results:
        for bbox in result.boxes:
            confidence = bbox.conf.item()
            class_id = int(bbox.cls.item())
            if confidence > 0.5:  # Adjust the threshold for object detection confidence
                detected_objects.append(object_labels[class_id])
    
    return detected_objects


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    username = request.form['username']
    image = Image.open(io.BytesIO(file.read()))
    img = preprocess_image(image)

    # Predict the class of the image
    yhat = model.predict(img)
    class_index = np.argmax(yhat)
    predicted_class = class_names[class_index]

    # Perform object detection
    detected_objects = detect_objects(np.array(image))
    objects_str = ', '.join(detected_objects) if detected_objects else 'None detected'

    # Create the caption
    caption = f"{predicted_class}: {objects_str}"

    return jsonify({'caption': caption, 'class': predicted_class, 'objects': detected_objects})

if __name__ == '__main__':
    app.run(debug=True)
