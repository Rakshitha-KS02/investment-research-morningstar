# investment-research-morningstar

# GECS Industry & Business Activity Classification using Machine Learning

> End-to-end Machine Learning system for automatically classifying companies into Morningstar's Global Equity Classification System (GECS) using unstructured business descriptions.

## Overview

This project was developed as part of the MS Business Analytics Capstone at DePaul University in collaboration with Morningstar.

The objective was to automate the classification of companies into the **Morningstar Global Equity Classification System (GECS)** using machine learning and natural language processing, reducing the need for manual industry mapping.

The solution predicts:

- **Task 1:** Industry Classification (145 Industry Classes)
- **Task 2:** Hierarchical Business Activity Classification
  - Sector
  - Industry Group
  - Industry
  - Business Activity

The project combines text analytics, feature engineering, supervised learning, and interactive prediction into a single classification framework.

---

## Business Problem

Organizations receive thousands of company profiles containing unstructured business descriptions.

Manually assigning GECS classifications is:

- Time consuming
- Expensive
- Inconsistent
- Difficult to scale

This project automates the classification process while maintaining strong prediction accuracy.

---

## Dataset

The model was trained using Morningstar company disclosures containing:

- Company profiles
- Segment names
- Segment descriptions
- Revenue information
- Existing GECS labels
- Business activity hierarchy

Additional lookup tables were used to enrich missing descriptions using official Morningstar definitions.

---

## Project Architecture

```
Company Description
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Linear Support Vector Machine
        │
        ▼
Industry Prediction
        │
        ▼
Hierarchical GECS Mapping
        │
        ▼
Interactive Recommendation Engine
```

---

## Machine Learning Pipeline

### Text Processing

- Missing value imputation
- Text normalization
- TF-IDF Vectorization
- Business-specific stop word removal
- 1-gram to 3-gram feature generation

### Numerical Features

- Revenue
- Revenue Share
- Log-transformed Company Revenue
- Largest Business Segment Indicator

### Models

- Linear Support Vector Classifier (LinearSVC)
- Balanced class weighting
- Train/Test Split
- Stratified Sampling

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- LinearSVC
- Label Encoding
- Pipeline API
- Column Transformer

---

## Feature Engineering

The final model combines both textual and numerical information.

### Text Features

- Segment Name
- Long Profile
- Segment Description

### Numerical Features

- Revenue
- Revenue Share
- Total Company Revenue
- Largest Segment Flag

This hybrid approach improved classification performance compared to using text alone.

---

## Model Performance

### Task 1 — Industry Classification

- **145 Industry Classes**
- **Macro F1 Score: 0.79**
- Balanced Linear SVM
- TF-IDF + Numerical Features

### Task 2 — Business Activity Classification

Predicts the complete GECS hierarchy:

- Sector
- Industry Group
- Industry
- Business Activity

The final system produces an integrated recommendation with confidence scores and official business activity definitions.

---

## Interactive Prediction Engine

Users can enter:

- Company description
- Segment information
- Revenue

The model predicts:

- Top industry predictions
- Confidence scores
- GECS Industry Code
- Sector
- Industry Group
- Industry
- Business Activity
- Official business activity definition

---

## Repository Structure

```
├── task1-exp6.py
├── task2-exp6.py
├── datasets/
├── outputs/
├── screenshots/
└── README.md
```

---

## Key Learnings

- NLP feature engineering
- TF-IDF vectorization
- Multi-class classification
- Business taxonomy prediction
- Hybrid machine learning models
- Model evaluation using Macro F1
- Building reusable Scikit-learn pipelines

---

## Future Improvements

- Fine-tune transformer-based models (BERT/FinBERT)
- Add probability calibration
- Deploy as a Streamlit web application
- Build REST API using FastAPI
- Containerize with Docker
- Integrate cloud deployment on AWS

---

## Authors

**Rakshitha Kavitha Suresh**

MS Business Analytics  
DePaul University

---

## Acknowledgements

This project was completed as part of the MS Business Analytics Capstone at DePaul University in collaboration with Morningstar. The work focuses on applying Natural Language Processing and Machine Learning techniques to automate GECS industry classification from company disclosures.
