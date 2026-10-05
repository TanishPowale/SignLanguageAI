# 🤟 SignLanguageAI

### Real-Time Sign Language to Text & Speech

SignLanguageAI is a real-time sign language recognition system that uses a webcam to recognize ASL alphabet gestures and convert them into text and speech.

## Features

* Real-time ASL alphabet recognition
* Hand landmark detection using MediaPipe
* Random Forest based sign classification
* Prediction stabilization
* Word suggestions
* Sentence building
* Text-to-speech

## Technologies

Python • OpenCV • MediaPipe • Scikit-learn • Random Forest • NumPy • Pandas • Joblib • Wordfreq • pyttsx3

## Dataset

The project uses the **ASL Alphabet Dataset** for training.

MediaPipe extracts **21 hand landmarks**, with X, Y and Z coordinates for each landmark.

**21 × 3 = 63 features**

Approximately **63,676 landmark samples** were prepared for training and testing.

The original dataset is not included in the repository because of its large size.

## Machine Learning Model

The system uses a **Random Forest Classifier** trained on the normalized hand landmark features.

**Input:** 63 normalized landmark features
**Output:** ASL alphabet letter

The final model achieved approximately **99.86% test accuracy**.

The application uses:

```text
sign_model_robust.pkl
```

The `.pkl` file is not included in GitHub because of its large size. To run the application, place the model inside the project's `models` folder.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/TanishPowale/SignLanguageAI.git
cd SignLanguageAI
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the model

Place:

```text
sign_model_robust.pkl
```

inside the `models` folder.

### 5. Run

```bash
python app/live_sentence.py
```

A working webcam is required.

## Controls

`B` → Backspace
`C` → Clear
`S` → Speak
`Q` → Quit

## Future Scope

* Indian Sign Language (ISL)
* Complete word and sentence recognition
* Two-hand gesture recognition
* Facial expression recognition
* Mobile and web deployment

## Author

**Tanish Powale**
git status