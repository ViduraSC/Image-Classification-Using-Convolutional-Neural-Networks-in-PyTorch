# CIFAR-10 Image Classification Web Application

This project is a web-based application that classifies images using a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. The app allows users to upload an image, which is then classified into one of 10 categories (plane, car, bird, cat, deer, dog, frog, horse, ship, or truck) using a trained deep learning model built with PyTorch.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Details](#model-details)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Dependencies](#dependencies)
- [Credits](#credits)

## Overview

This project is built with Flask as the backend web framework and uses PyTorch for model training and inference. The model is a Convolutional Neural Network (CNN) that has been trained on the CIFAR-10 dataset to classify images into 10 common object categories. Users can upload an image through the app, and the model will return the predicted class label.

## Features

- Image classification using a deep learning model trained on CIFAR-10.
- Simple and intuitive web interface for image upload and classification.
- Displays the prediction along with the uploaded image for easy comparison.
- Supports live predictions on user-uploaded images.

## Model Details

The model used in this project is a CNN with the following architecture:
- **Conv2d** layers followed by **MaxPooling** for feature extraction.
- **Fully connected layers** for classification.
- Trained on the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes.

The CIFAR-10 classes used in the model are: **Plane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck**.

## Installation

To set up this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd project_folder
   ```

2. **Install Dependencies**:
   Make sure you have Python 3.6+ installed. Then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Download the Pretrained Model**:
   Place the `trained_net.pth` file (pretrained model weights) in the root directory of the project.

4. **Set Up Folder Structure**:
   Create the `static/uploads` folder to store uploaded images.
   ```bash
   mkdir -p static/uploads
   ```

## Usage

1. **Run the Application**:
   Start the Flask server by running:
   ```bash
   python app.py
   ```
   
2. **Open the Web Application**:
   Open a web browser and navigate to `http://127.0.0.1:5000`.

3. **Upload an Image**:
   - Click the "Choose File" button to upload an image.
   - Click the "Upload and Classify" button to send the image to the model for prediction.
   
4. **View Results**:
   The app will display the predicted class along with the uploaded image.

## Folder Structure

```plaintext
project_folder/
├── app.py                    # Main Flask application
├── model.py                  # Model architecture (NeuralNet class)
├── trained_net.pth           # Trained PyTorch model weights
├── requirements.txt          # Project dependencies
├── static/
│   ├── css/
│   │   └── styles.css        # CSS styling
│   ├── js/
│   │   └── script.js         # JavaScript for loading message
│   └── uploads/              # Folder to store uploaded images
└── templates/
    └── index.html            # HTML template for the web page
```

## Dependencies

- Python 3.6+
- Flask
- PyTorch
- Torchvision
- PIL (Pillow)

To install dependencies:
```bash
pip install -r requirements.txt
```

## Credits

This project was created using:
- **PyTorch** for building and training the model.
- **Flask** for developing the web application interface.

--- 

