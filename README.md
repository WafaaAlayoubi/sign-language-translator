# ✋ Sign Language Number Translator (0-9)

This project is a machine learning-based Sign Language Translator that recognizes hand signs representing numbers from 0 to 9 using a webcam and predicts them in real time.

![Demo](demo.gif)

## 🎯 Features
- Real-time prediction with webcam
- Recognizes numbers 0 to 9 using hand signs
- Uses MediaPipe for hand detection
- XGBoost model for classification

## 🧠 Tech Used
- MediaPipe (hand tracking)
- XGBoost (machine learning classifier)
- OpenCV (webcam input and display)
- scikit-learn (model training and evaluation)
- NumPy, Pandas (data handling)
- Joblib (model saving/loading)

## 📷 Dataset Creation

For this project, I used a dataset of hand sign images. Unfortunately, I can't attach the dataset because it's too large. However, here's how you can generate a similar dataset:

1. **Download your own image dataset** of hand signs (from 0 to 9).
2. **Run the script** `createDSAll.py`. This script will:
   - Take 10 folders, named from `0` to `9`, each containing images of the corresponding hand sign.
   - Convert the images into a CSV file by extracting MediaPipe hand landmarks' coordinates.

3. **Result**: The script will generate a CSV file similar to mine, which you can use for training.

You can use the XGBoost model I’ve implemented or any other model you prefer for training.

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/WafaaAlayoubi/sign-language-translator.git
cd sign-language-translator
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Train the model (or use my model)**
```bash
python XGBoost.py
```
4. **Run the app**
```bash
python app.py
```
