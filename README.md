# Parkinsons-Finger-Tapping-Project
This project aims to develop a computer-based assessment tool that analyzes finger tapping movements in Parkinson’s patients. Videos and signal data are used to measure how smoothly and quickly a patient taps their fingers, providing consistent insights into their motor control. This repository contains the relevant files used for this project.

## 📌 Project Overview

Parkinson’s Disease affects motor function, often impairing fine motor skills such as finger tapping.
This project leverages signal processing and machine learning techniques to:
- Process raw finger tapping amplitude & time data
- Extract meaningful statistical and signal-based features
- Classify subjects as:
  - 1 → Parkinson’s Disease
  - 0 → Healthy Control
- Evaluate model performance using robust validation techniques
- Export trained models for deployment (e.g., Streamlit app)

## 📊 Dataset Description

Each patient folder contains:
  - ```Amplitude.txt``` → Finger tapping amplitude values
  - ```Time.txt``` → Corresponding timestamps

Additionally:
  - ```updrs_scores.csv``` → Clinical motor severity scores (UPDRS)

### Target Variable
The label is derived from the patient ID:
  - ```PD*``` → Parkinson’s Disease (1)
  - Otherwise → Healthy Control (0)
    
## ⚙️ Pipeline Overview

### 1️⃣ Data Retrieval
- Loads data from Google Drive
- Reads amplitude and time signals
- Loads UPDRS scores

### 2️⃣ Data Preprocessing
- Pads signals to a fixed length (1800 samples)
- Converts raw lists to NumPy arrays
- Handles missing or inconsistent files

### 3️⃣ Feature Engineering
Signal-based feature extraction includes:
- Statistical features:
  - Mean
  - Standard deviation
  - Variance
  - Min / Max
- Signal smoothing using Savitzky–Golay filter
- Peak detection using ```scipy.signal.find_peaks```
- Frequency-based and amplitude-based derived metrics

### 4️⃣ Model Training & Evaluation
Models evaluated:
- Random Forest
- SVM (RBF Kernel)
  
Validation strategy:
- Stratified K-Fold Cross-Validation
- Cross-validated predictions
- Metrics:
  - Recall
  - ROC-AUC
  - Confusion Matrix
  - Classification Report
 
## 🚀 Deployment
Models are saved for integration into:
- Streamlit Web App
- Clinical decision support tools
- API-based inference system
  
Example inference flow:
```
model = joblib.load("left_hand_model.pkl")
prediction = model.predict(features)
```
## 🧠 Why This Matters
Early detection of Parkinson’s Disease can:
- Improve treatment planning
- Support clinical decision-making
- Enable remote motor function assessment

This pipeline demonstrates how signal processing + ML can support neurological diagnosis.

## 🔮 Future Improvements
- Deep Learning (LSTM / CNN on raw signals)
- Frequency-domain feature extraction (FFT)
- Feature selection & dimensionality reduction (PCA)
- Hyperparameter tuning (GridSearchCV)
- Class imbalance handling (SMOTE)
- External validation dataset

## 📜 License
This project is for academic and research purposes.
