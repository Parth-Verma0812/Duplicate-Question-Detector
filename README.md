# Duplicate-Question-Detector

# Duplicate Question Detector

This project predicts whether two questions are duplicates using Natural Language Processing and Machine Learning.

## Live Demo
https://duplicate-question-detector-ml.streamlit.app

---

## Project Overview

Duplicate questions are common in Q&A platforms like Quora or StackOverflow.  
This project uses Custom NLP features + vectorization to identify semantic similarity between question pairs.

The model combines:

- Text preprocessing & cleaning
- Token-based similarity features
- Length-based features
- Fuzzy string matching
- Count Vectorization
- Machine Learning classification

---

## Features Used

###  Basic Features
- Question length
- Word count
- Common words
- Unique word ratio

###  Token Features
- Common non-stopwords
- Stopword overlap
- First/last word match

###  Length Features
- Absolute length difference
- Average token length
- Longest common substring ratio

###  Fuzzy Matching
- Fuzzy ratio
- Partial ratio
- Token sort ratio
- Token set ratio

---

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLTK
- FuzzyWuzzy
- NumPy & Pandas
- BeautifulSoup

---

## Dataset

- **Dataset:** Quora Question Pairs
- Source: Kaggle
- Task: Binary classification (Duplicate vs Non-duplicate)
