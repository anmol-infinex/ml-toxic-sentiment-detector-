# ML Toxic Sentiment Detector

A lightweight NLP project that classifies text as **good** or **bad** using TF-IDF, Logistic Regression, and rule-based signals.

## Overview

The project combines machine learning with deterministic text rules to explore practical content-safety classification.

### Pipeline

`Input text → preprocessing → TF-IDF → Logistic Regression → rule-based refinement → prediction`

## Features

- Toxic / negative text detection
- Confidence score
- Rule-based signal detection
- Simple CLI inference
- Lightweight scikit-learn pipeline
- Experimentation with negation, contrast, and adversarial wording

## Tech Stack

Python · scikit-learn · pandas · NumPy · Joblib

## Project Structure

```text
ML/
├── train.py
├── detector.py
├── predict.py
├── preprocess.py
├── vocabulary.py
├── config.py
├── train.csv
├── models/
│   └── sentiment_model.joblib
└── README.md
```

## Run Locally

```bash
git clone https://github.com/anmol-infinex/ml-toxic-sentiment-detector-.git
cd ml-toxic-sentiment-detector-
pip install -r requirements.txt
python train.py
python predict.py
```

## Example

```text
Input: i will destroy your system

Prediction: bad
Confidence: 0.94
Detected signal: destroy
```

## Applications

Content moderation · NLP experimentation · AI safety research · cybersecurity-oriented text analysis

## Notes

Reported training accuracy in the original experiment was approximately 99%. Real-world performance depends on the dataset, preprocessing, and evaluation methodology; benchmark numbers should not be treated as production accuracy without a reproducible test set.

## Author

**Anmol Rathod**  
BSc IT (Hons) · AI/ML + Cybersecurity

[LinkedIn](https://www.linkedin.com/in/anmol-rathod-aabb13360) · [GitHub](https://github.com/anmol-infinex)
