# SkinAI — Flask Skin Cancer Detection App

## Project Structure

```
your-project/
│
├── model/
│   └── vgg16_skin_cancer.h5      ← your trained model (place it here)
│
└── app/
    ├── app.py                    ← Flask application
    ├── requirements.txt
    ├── static/
    │   └── style.css
    └── templates/
        ├── base.html
        ├── login.html
        ├── dashboard.html
        ├── predict.html
        ├── result.html
        └── patients.html
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r app/requirements.txt

# 2. Place your model
# Copy vgg16_skin_cancer.h5 into the model/ folder

# 3. Run the app
cd app
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Default Login

| Username | Password  |
|----------|-----------|
| admin    | admin123  |

> Change this in production by updating the seeded user's password hash in `init_db()`.

## Model Output Classes

The app expects the model to output probabilities for 7 classes (HAM10000 dataset):

1. Actinic keratosis
2. Basal cell carcinoma
3. Benign keratosis
4. Dermatofibroma
5. Melanoma
6. Melanocytic nevi
7. Vascular lesion

If your model uses different classes or a different order, edit the `CLASS_NAMES` list in `app.py`.

## Features

- 🔐 Login / session auth
- 📊 Dashboard with stats (total, malignant, benign)
- 🔬 Image upload + VGG16 prediction
- 🗂️ Patient records with search & sort
- 🗑️ Delete records (removes image file too)
- ⚠️  Risk classification (Melanoma, BCC, AK = high-risk)
