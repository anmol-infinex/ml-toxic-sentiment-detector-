# 🧠 ML Toxic Sentiment Detector

## 📌 Overview
This project is a machine learning-based system that classifies user input as:

- ✅ **Good**
- ❌ **Bad**

It is designed to detect harmful, abusive, or negative language in real-time using a hybrid approach combining machine learning and rule-based logic.

---

## ⚙️ Features

- 🔍 Detects toxic or negative sentences
- ⚡ Real-time prediction (fast & lightweight)
- 🧠 Hybrid system:
  - TF-IDF (Text Vectorization)
  - Logistic Regression (ML Model)
  - Rule-based enhancements
- 🧩 Handles complex NLP cases:
  - Negation → "not bad"
  - Contrast → "good but slow"
  - Sarcasm → "great job ruining everything"
- 📊 Confidence score for predictions
- 🛡️ Detects system-related threats (cybersecurity use-case)

---

## 🏗️ Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

## 📁 Project Structure

ML/
├── train.py # Train ML model
├── detector.py # Core classification logic
├── predict.py # User input interface (CLI)
├── preprocess.py # Text preprocessing
├── vocabulary.py # Rule-based signals
├── config.py # Configurations
├── train.csv # Dataset
├── models/
│ └── sentiment_model.joblib
└── README.md

## 🚀 How It Works

1. User enters a sentence
2. Text is preprocessed
3. TF-IDF converts text into numerical features
4. Logistic Regression predicts probabilities
5. Rule-based system refines the prediction
6. Final output includes:
   - Label (**good / bad**)
   - Confidence score
   - Detected bad words or phrases

---

## ▶️ How to Run

### 1. Clone the repository

bash

git clone https://github.com/your-username/ml-toxic-sentiment-detector.git

cd ml-toxic-sentiment-detector

pip install -r requirements.txt

python train.py

python predict.py

---------------------------------------

💡 Example
Enter a sentence: i will destroy your system

Prediction: bad
Confidence: 0.94
Bad word/phrase found: destroy

--------------------------------------

🎯 Applications


💬 Chat moderation systems
🌐 Social media content filtering
📝 Customer feedback analysis
🔐 Cybersecurity threat detection
🤖 AI assistants content safety

------------------------------------

📊 Model Performance
Training Accuracy: ~99%
Real-world accuracy: ~80–90%
Handles edge cases better than basic ML models

--------------------------------------

🧠 Key Learnings
Built an end-to-end ML pipeline
Combined ML + rule-based logic
Improved model using real-world test cases
Handled edge cases like sarcasm, negation, and contrast
Designed system for practical applications

## 👨‍💻 Author

**Anmol Rathod**  
BSc IT (Cyber Security + AI/ML)

🔗 [LinkedIn Profile](https://www.linkedin.com/in/anmol-rathod-aabb13360)  
📫 Open to internships and collaboration opportunities
