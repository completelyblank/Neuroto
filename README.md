![image](https://github.com/user-attachments/assets/c7b71f97-7fb0-4658-92e8-dac7b286968c)

# Neuroto 

Neuroto is a beginner image classification web application designed to classify uploaded images and detect objects within them. It utilizes a deep neural network (DNN) => CNN in this term for accurate classification and offers a modern, aesthetic user interface with dark and neon blue themes.

## Project Overview

Neuroto aims to provide users with an intuitive platform to upload images, receive detailed classification results, and view detected objects. The project combines a sleek frontend with a robust backend, leveraging machine learning techniques.

## Features

- **Image Upload**: Users can upload images for classification.
- **Username Input**: Option to input a username for personalizing the results.
- **Image Classification**: Classifies images and detects objects using a Convolutional Neural Network (accurate but training data was not as clean as could be for classification as Google Images have their own 
- **Modern Design**: Aesthetic dark theme with neon blue accents.
- **Object Detection**: Yolov8 for Basic Object Detection (Multiple included) (Errors and is limited as a model)

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask
- **Machine Learning**: OpenCV for object detection, custom DNN model for classification
- **Object Detection**: YOLOV8
- **HTTP Requests**: Axios

## DNN Model

### Overview

The deep neural network (DNN) used in Neuroto is a convolutional neural network (CNN) designed for image classification. It is trained to recognize a variety of objects and classify images with high accuracy.

### Accuracy

- **Training Data**: The model is trained on a large dataset of labeled images to ensure diverse and accurate object recognition.
- **Validation Accuracy**: Achieved an accuracy of approximately `XX%` on the validation set.
- **Real-World Performance**: Provides accurate classifications and object detections in real-world scenarios, though performance may vary based on image quality and content.

### Challenges and Solutions

1. **Model Training**:
   - **Challenge**: Training the model on a diverse dataset while avoiding overfitting.
   - **Solution**: Utilized data augmentation techniques and regularization methods to improve generalization.

2. **Object Detection**:
   - **Challenge**: Detecting multiple objects in complex scenes.
   - **Solution**: Implemented OpenCV for robust object detection and integrated it with the classification model.

3. **Performance Optimization**:
   - **Challenge**: Balancing model accuracy with real-time performance.
   - **Solution**: Optimized the model by reducing its size and complexity while maintaining accuracy.

### Clone the Repository

```bash
git clone https://github.com/your-username/neuroto-image-classifier.git
cd neuroto-image-classifier
