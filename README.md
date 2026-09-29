# 🧠 ML Toxic Sentiment Detector

> A lightweight NLP security project that classifies text as **Good** or **Bad** using TF-IDF, Logistic Regression, and rule-based signals.

## 📌 Overview

This project is an end-to-end machine learning system for detecting harmful, abusive, or negative language in text. It combines a traditional ML pipeline with rule-based enhancements to make predictions more practical for real-world inputs.

### What it demonstrates
- 🧠 Natural Language Processing with **TF-IDF**
- 📈 Text classification with **Logistic Regression**
- 🛡️ Rule-based detection for security-related phrases
- ⚡ Fast, lightweight prediction suitable for real-time use
- 📊 Confidence scores and detected signals
- 🧩 Handling of negation, contrast, and selected edge cases

## ⚙️ How It Works

**Input → Preprocessing → TF-IDF → Logistic Regression → Rule-Based Refinement → Final Prediction**

The final result includes a **good/bad label**, confidence score, and relevant detected words or phrases.

## 🏗️ Tech Stack

Python · Scikit-learn · Pandas · NumPy · Joblib

## 🚀 Applications

Content moderation · Social media filtering · Customer feedback analysis · AI safety · Cybersecurity text screening

## ▶️ Quick Start

```bash
git clone https://github.com/anmol-infinex/ml-toxic-sentiment-detector-.git
cd ml-toxic-sentiment-detector-
pip install -r requirements.txt
python train.py
python predict.py
```

## 📊 Model Performance

The training pipeline reports model accuracy and classification metrics after training. Current project documentation reports approximately **99% training accuracy** and **80–90% real-world accuracy**, with edge-case handling as an additional focus.

## 👨‍💻 Author

**Anmol Rathod** · BSc IT (Cyber Security + AI/ML)

🔗 [LinkedIn](https://www.linkedin.com/in/anmol-rathod-aabb13360)  
📫 Open to internships and collaboration opportunities