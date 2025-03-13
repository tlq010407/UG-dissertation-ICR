# Intelligent Character Recognition (ICR) System

## Overview
This project focuses on enhancing an **Intelligent Character Recognition (ICR)** system through **machine learning techniques** to improve the recognition of handwritten and printed characters. It employs **deep learning models**, including **Convolutional Neural Networks (CNNs)**, **Recurrent Neural Networks (RNNs)** such as **Long Short-Term Memory (LSTM)**, and **Gated Recurrent Units (GRU)** to develop a robust, multi-language character recognition system.

## Features
- **Multi-Language Character Recognition**
  - Supports **Chinese, English, and Korean** characters.
  - Trained on **large-scale datasets** for improved accuracy.
- **Machine Learning & Deep Learning Techniques**
  - Uses **CNN, LSTM, GRU**, and **ensemble methods**.
  - Implements **hyperparameter tuning** and **regularization**.
- **Data Preprocessing & Augmentation**
  - **Noise reduction** with Median Filtering.
  - **Contrast enhancement** for better character distinction.
- **Web-Based Implementation**
  - **Streamlit-powered UI** for character image uploads.
  - **Real-time character recognition processing.**

## Technology Stack
- **Programming Languages**: Python
- **Machine Learning Frameworks**: TensorFlow, Keras
- **Backend**: Flask, Firebase
- **Frontend**: Streamlit
- **Database**: Firebase Realtime Database
- **Visualization**: Matplotlib, Seaborn

## Dataset Preparation
### 1. Data Collection
- **Chinese**: CASIA-HWDB Dataset (7,000+ characters)
- **English**: Handwritten OCR Dataset
- **Korean**: Custom dataset with frequent characters

### 2. Preprocessing & Augmentation
- **Image resizing (64x64, 128x128)**
- **Grayscale conversion & normalization**
- **Noise reduction using Median Filters**
- **Data augmentation (rotation, zoom, shift)**

## Model Development
### 1. CNN Model (Baseline)
- Extracts features using convolutional layers.
- Uses **Max Pooling** and **Dropout** for dimensionality reduction.

### 2. CNN + RNN Hybrid Model
- Combines **CNN for feature extraction** and **GRU/LSTM for sequence learning**.

### 3. Hyperparameter Tuning & Regularization
- **Optimized CNN-LSTM model** achieves **84.02% accuracy** on English characters.
- **CNN with L2 Regularization** significantly improves generalization.

## Web Application (Streamlit UI)
- **Upload Character Image** (PNG, JPG, JPEG supported)
- **Process Image in Real-time** with trained ML models
- **Display Recognized Characters** and probabilities
- **Supports Multi-language Recognition**

## How to Run
### 1. Clone the repository
```sh
 git clone https://github.com/your-repo/ICR-System.git
```

### 2. Install Dependencies
```sh
 pip install -r requirements.txt
```

### 3. Run the Web App
```sh
 streamlit run app.py
```

## Results & Performance
| Model                         | Language | Accuracy |
|--------------------------------|----------|----------|
| CNN Basic Model               | Chinese  | 63.93%   |
| CNN + Median Filter           | Chinese  | 68.06%   |
| CNN + More Layers             | Chinese  | 82.70%   |
| CNN + GRU                     | English  | 69.65%   |
| CNN + GRU + ImageDataGenerator| English  | 71.41%   |
| CNN + L2 Regularization       | English  | 80.21%   |
| CNN + LSTM                    | English  | 82.99%   |
| CNN + Hyperparameter Tuning   | English  | 84.02%   |

## Future Enhancements
- **Expand to Japanese and other languages.**
- **Improve model efficiency with lightweight architectures.**
- **Implement real-time recognition via mobile app integration.**

