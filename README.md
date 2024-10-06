
# Payment Prediction ML

This project is a Flask-based web application designed to train a machine learning model using TensorFlow, make predictions on payment data, and display the results using a web interface.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Complexities](#complexities)
- [Solutions](#solutions)
- [Challenges](#challenges)
- [License](#license)

## Installation

1. Clone the repository:

```sh
git clone https://github.com/faisal-fida/payment_prediction_ml.git
cd payment_prediction_ml
```

2. Install the dependencies:

```sh
pip install -r requirements.txt
```

3. Run the application:

```sh
python app.py
```

## Usage

The application provides the following functionalities:

1. **Train the Model**: Run `prediction.py` to train the model on MNIST data and save the trained model.
2. **Web Interface**: Use Flask to provide a web interface for viewing predictions.

## Features

- **Machine Learning Model**: The project uses TensorFlow to create, train, and evaluate a neural network model for payment prediction.
- **Data Handling**: The project involves normalizing and splitting data into training and testing sets.
- **Web Integration**: Flask is used to create a web interface for displaying predictions and allowing users to upload images.

- **Efficient Model Training**: Implemented a sequential neural network model using TensorFlow, including layers for data flattening and dense layers with ReLU and softmax activations.
- **Data Normalization**: Utilized TensorFlow utilities to normalize the dataset, ensuring the model trains effectively.
- **User-Friendly Interface**: Developed a responsive web interface with Tailwind CSS for users to upload images and view predictions.

- **Model Accuracy**: Ensuring the model achieves a high accuracy required fine-tuning the architecture and parameters.
- **Data Normalization**: Properly normalizing the data was crucial for the model's performance.
- **File Handling**: Managing file uploads and predictions through a web interface required careful handling of HTTP requests and response rendering.
