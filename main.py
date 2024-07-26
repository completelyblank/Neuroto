# Libraries
import tensorflow as tf
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.models import load_model


# Set up GPU configurations if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define the directory and valid image extensions
data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']  # accepted image types

# Validate images by checking if they are in the directory and have the right type
for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    if os.path.isdir(class_path):
        for image in os.listdir(class_path):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = Image.open(image_path)
                if img.format.lower() not in image_exts:
                    print(f"Image is not in ext list {image_path}")
                    os.remove(image_path)
            except Exception as e:
                print(f"Issue with Image {image_path}")

# Build data pipeline
data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(256, 256),
    batch_size=32
)

# Normalize the data
data = data.map(lambda x, y: (x / 255.0, y))

# Split the data 70:20:10
len_data = len(list(data))
train_size = int(len_data * 0.7)
val_size = int(len_data * 0.2)
test_size = int(len_data * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Build the model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(os.listdir(data_dir)), activation='softmax')  # Adjust for the number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
logdir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# Initialize metrics
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

# Loop through batches in the test data
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    yhat_labels = np.argmax(yhat, axis=1)

    pre.update_state(y, yhat_labels)
    re.update_state(y, yhat_labels)
    acc.update_state(y, yhat_labels)

# Print the results
print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

# Evaluate the model
model.evaluate(test)

# Read the image using OpenCV
img = cv2.imread('data/portrait/1jWx9suY2k3Ifq4B8A_vz9g.jpeg')

# Display the original image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.show()

# Convert the image to a tensor
img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

# Resize the image using TensorFlow
resize = tf.image.resize(img_tensor, (256, 256))

# Convert the resized tensor back to a NumPy array
resize_np = resize.numpy().astype(np.uint8)

# Display the resized image
plt.imshow(cv2.cvtColor(resize_np, cv2.COLOR_BGR2RGB))
plt.title("Resized Image")
plt.show()

# Expand dimensions to match the model's input shape
resize_np = np.expand_dims(resize_np, axis=0) / 255.0  # Normalize and add batch dimension

# Predict using the model
yhat = model.predict(resize_np)

# Get the predicted class index
class_index = np.argmax(yhat)

# Map the class index to class names
class_names = ['landscape', 'minimalist', 'monochrome', 'nightime', 'portrait', 'street_photography', 'vintage']
predicted_class = class_names[class_index]

print(f"Predicted Class Index: {class_index}")
print(f"Predicted Class: {predicted_class}")

# Verify if the class names are correctly mapped
print("Class Names:", class_names)

#Save The Model
model.save(os.path.join('models', "PhotoClassifier.h5"))
new_model=load_model(os.path.join('models', 'PhotoClassifier.h5'))
yhatnewt= new_model.predict(np.expand_dims(resize/255, 0))
