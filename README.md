# SkinAI — Flask Skin Cancer Detection App

A Flask-based web application that detects skin cancer (Malignant vs Benign) from uploaded lesion images using a trained VGG16 deep learning model.

## Project Structure

```
ai/
│
├── .gitattributes                ← Git LFS configurations
├── README.md
│
├── model/
│   └── vgg16_malignant_vs_benign.h5  ← Trained VGG16 binary classification model
│
└── app/
    ├── app.py                    ← Flask application
    ├── requirements.txt          ← Python dependencies
    ├── skinai.db                 ← SQLite database (automatically seeded on startup)
    ├── static/
    │   ├── style.css             ← Styling for the dashboard and pages
    │   └── uploads/              ← Directory for uploaded patient images
    └── templates/
        ├── base.html             ← Layout template wrapper
        ├── login.html            ← Authentication page
        ├── dashboard.html        ← Analytics dashboard
        ├── predict.html          ← Diagnostic image upload form
        ├── result.html           ← Patient diagnosis details
        └── patients.html         ← Searchable history of patient cases
```

## Setup & Run

### 1. Prerequisites
Ensure you have Python 3.9+ installed. You also need [Git LFS](https://git-lfs.github.com) installed to pull the large `.h5` model file if cloning the repository:
```bash
git lfs install
git clone https://github.com/louayamri999/ai.git
cd ai
```

### 2. Install Dependencies
Install the required packages listed in `requirements.txt`:
```bash
pip install -r app/requirements.txt
```

### 3. Run the Web Application
```bash
cd app
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

## Default Credentials

| Username | Password  |
|----------|-----------|
| `admin`  | `admin123`|

> **Security Note:** You can change this in production by editing the seeded user's password hash within the `init_db()` function in `app.py`.

## Model and Classification Details

- **Model Type:** VGG16 transfer learning model.
- **Classification Class:** Binary (Malignant vs. Benign).
- **Decision Boundary Threshold:** `0.2976` (optimized from the Precision-Recall curve).
- **Behavior:** 
  - Output probability `> 0.2976` is predicted as **Malignant** (High Risk).
  - Output probability `≤ 0.2976` is predicted as **Benign** (Low Risk).

## Features

- **🔐 Session-Based Authentication:** Clean and secure login flow for clinical operators.
- **📊 Analytics Dashboard:** Interactive counters summarizing total patients, malignant cases, benign cases, and high-risk percentages.
- **🔬 Prediction Engine:** Seamless image preprocessing (224x224 scaling, normalisation) feeding into the VGG16 classifier.
- **🗂️ Patient Case Management:** Search, filter by risk level, sort, and manage patient diagnostic records.
- **🗑️ Automated Cleanup:** Deleting a patient's record automatically removes their uploaded image from local storage.
