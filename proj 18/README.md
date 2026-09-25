🌱 AI-Based Plant Disease Detection
📌 Project Overview

AI-Based Plant Disease Detection is a machine-learning project that identifies plant diseases from images of plant leaves.

The system uses a Convolutional Neural Network (CNN) to analyze leaf images and classify them into different disease categories. The trained model can be used to predict the disease of a new leaf image.

🎯 Objectives

Detect plant diseases automatically from leaf images.

Reduce the need for manual disease identification.

Train a CNN-based image-classification model.

Evaluate the model using a separate test dataset.

Display the predicted disease and prediction confidence.

Provide a reusable framework that can be extended to additional crops and diseases.

🏗️ System Architecture
                Plant Leaf Image
                       │
                       ▼
              Image Preprocessing
               Resize + Normalize
                       │
                       ▼
                 CNN Model
                       │
                       ▼
              Feature Extraction
                       │
                       ▼
               Disease Classifier
                       │
                       ▼
              Predicted Disease
                       │
                       ▼
             Prediction Confidence

📁 Project Structure
AI-Plant-Disease-Detection/
│
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── preprocess.py
│
├── testbench/
│   └── test_model.py
│
├── simulation/
│   ├── evaluate.py
│   └── results/
│       ├── accuracy.png
│       ├── loss.png
│       ├── confusion_matrix.png
│       └── sample_predictions.png
│
└── models/
    └── plant_disease_model.keras

📊 Dataset

The project expects the dataset to be organized into folders according to disease classes.

Example:

dataset/
├── train/
│   ├── Healthy/
│   ├── Early_Blight/
│   └── Late_Blight/
│
├── validation/
│   ├── Healthy/
│   ├── Early_Blight/
│   └── Late_Blight/
│
└── test/
    ├── Healthy/
    ├── Early_Blight/
    └── Late_Blight/


Each folder contains images belonging to that class.

For a larger project, the same structure can be extended to multiple crops and diseases.

🧠 Model

The baseline model is a Convolutional Neural Network consisting of:

Image resizing

Data augmentation

Convolution layers

Max-pooling layers

Dropout

Fully connected layer

Softmax output layer

The model predicts one class from the classes present in the training dataset.

⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/AI-Plant-Disease-Detection.git
cd AI-Plant-Disease-Detection


Install the required packages:

pip install -r requirements.txt

🚀 Training

Run:

python src/train.py


The trained model will be saved in:

models/plant_disease_model.keras


Training graphs will be saved in:

simulation/results/

🔍 Prediction

To classify an individual leaf image:

python src/predict.py path/to/leaf_image.jpg


Example output:

Predicted Disease : Early_Blight
Confidence        : 94.27%

🧪 Testbench

The testbench verifies that the trained model can load correctly and produce valid predictions.

Run:

pytest testbench/test_model.py -v


The testbench checks:

Model loading

Input image processing

Output shape

Prediction probability

Confidence value

Predicted class

📈 Simulation and Evaluation

The evaluation script generates:

Training accuracy graph

Validation accuracy graph

Training loss graph

Validation loss graph

Confusion matrix

Sample predictions

Run:

python simulation/evaluate.py


Results are stored in:

simulation/results/

📌 Performance Metrics

The following metrics can be used to evaluate the system:

Metric	Description
Accuracy	Percentage of correctly classified images
Precision	Correct positive predictions among positive predictions
Recall	Correctly detected samples among actual samples
F1-Score	Combination of precision and recall
Confusion Matrix	Shows class-by-class predictions

Record the actual values obtained from your experiment here instead of using assumed values.

Example:

Test Accuracy : XX.XX%
Precision      : XX.XX%
Recall         : XX.XX%
F1-Score       : XX.XX%

🔮 Future Improvements

Add more plant species.

Add more disease classes.

Use transfer-learning models such as MobileNet or ResNet.

Deploy the model as a web application.

Deploy the model on an edge device.

Connect the system to a camera for real-time detection.

Add disease-treatment recommendations from a verified agricultural source.

👨‍💻 Technologies Used

Python

TensorFlow / Keras

NumPy

OpenCV

Matplotlib

Scikit-learn

Pytest

📜 License

This project is intended for educational and research purposes.