# fracture_detection_computer_vision


An AI-driven end-to-end pipeline for detecting bone fractures in radiographic (X-ray) images using advanced image processing techniques and deep learning, specifically a DenseNet-121 CNN model. This system aims to support radiologists by improving diagnostic speed and accuracy.


📌 Table of Contents

🎯 Project Objective

📚 Dataset

🧠 Model Architecture

🖼️ Preprocessing Techniques

📊 Performance & Evaluation

🛠️ Installation & Usage

🚀 Future Work

📎 Contributors


🎯 Project Objective

Goal: Develop a deep learning-based model to detect bone fractures in X-ray images.

Motivation: Manual analysis of X-rays is time-consuming and subject to human error. Automating fracture detection helps radiologists make quicker and more accurate decisions.


📚 Dataset: MURA (Musculoskeletal Radiographs)

Provided by Stanford ML Group

Total Images: 40,561

Studies: 14,863

Regions Covered: Elbow, Finger, Forearm, Hand, Humerus, Shoulder, Wrist

Labels: Each study is classified by radiologists as either Normal or Abnormal.


📦 Data Pipeline

Load and label images from MURA dataset

Generate metadata (image path, label, body part)

Prepare structured dataset for preprocessing and model training


🧠 Model Architecture: DenseNet-121

Base Model: Pretrained DenseNet-121

Why DenseNet? Efficient parameter usage and strong performance in medical image analysis due to dense connectivity.

Input: Preprocessed 224x224 images

Custom Layers:

Global Average Pooling

Fully Connected Dense Layers

Dropout for regularization

Output: Sigmoid activation for binary classification (Fracture / No Fracture)


🖼️ Preprocessing Techniques

To enhance the quality of radiographic images and improve model accuracy:

Gamma Correction – Adjust image brightness

Gaussian Blur – Reduce image noise

CLAHE – Improve local contrast

Unsharp Masking – Sharpen fine details

Edge Enhancement – Emphasize fracture edges


🧪 Example:

Preprocessing enhances subtle fracture patterns that might be missed in raw images.


📊 Performance & Evaluation

Training Evaluation: Model was trained with efficient image preprocessing.

Explainability: Used Grad-CAM to generate heatmaps highlighting fracture-relevant regions in the image, improving model interpretability for clinicians.
