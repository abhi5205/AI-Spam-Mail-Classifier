# Implementation Plan - Spam Email Classifier

This plan outlines the design, architecture, and step-by-step development of a beginner-friendly, modular, and visually striking **Spam Email/SMS Classifier** using Python, Streamlit, and Scikit-Learn.

---

## 1. Project Directory Structure

A professional, GitHub-ready folder structure will be used:

```text
Spam-Mail-Classifier/
│
├── data/
│   └── SMSSpamCollection.tsv          # Downloaded raw dataset
│
├── models/
│   ├── model.pkl                      # Saved Multinomial Naive Bayes model
│   └── vectorizer.pkl                 # Saved TF-IDF vectorizer
│
├── src/
│   ├── __init__.py
│   └── preprocessing.py               # Modular NLP preprocessing utilities
│
├── train_model.py                     # Script to download data, train model, and print metrics
├── app.py                             # Beautiful, modern Streamlit UI
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Version control exclusions
└── README.md                          # Detailed project documentation & instructions
```

---

## 2. Technical Roadmap

### Phase 1: Environment & Preprocessing (`src/preprocessing.py`)
- Setup `requirements.txt` with standard library versions (`pandas`, `numpy`, `scikit-learn`, `nltk`, `streamlit`).
- Implement the NLP pipeline:
  - Download necessary NLTK corpora (`stopwords`, `punkt`) automatically.
  - Lowercase the input text.
  - Remove punctuation using regular expressions or `string.punctuation`.
  - Tokenize into individual words.
  - Filter out English stopwords.

### Phase 2: Model Training Pipeline (`train_model.py`)
- Programmatically download the dataset from a fast, reliable source (e.g., PyCon NLP dataset mirror or UCI).
- Load the dataset using `pandas` and name the columns `label` and `message`.
- Preprocess the text data using the modular preprocess function from `src/preprocessing.py`.
- Split the dataset into 80% training and 20% testing sets using `train_test_split`.
- Vectorize text using `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) fit on training data.
- Train a `MultinomialNB` (Multinomial Naive Bayes) classifier.
- Evaluate the model:
  - Print accuracy score.
  - Calculate and display confusion matrix.
  - Generate classification report (Precision, Recall, F1-Score).
- Save both the trained classifier and the TF-IDF vectorizer to the `models/` directory using `pickle`.

### Phase 3: Premium Streamlit Web Application (`app.py`)
To make this app stand out and feel highly premium:
- **Visual Design**:
  - A clean, modern dark/light-compatible dashboard using Streamlit's thematic variables.
  - Use custom CSS injected with `st.markdown(..., unsafe_allow_html=True)` to create sleek gradients, glassmorphism containers, and beautiful response cards.
  - Use clear visual feedback: a glowing green container for **"Ham" (Safe)** and a sharp red/orange container for **"Spam" (Caution)**.
- **User Experience (UX)**:
  - Interactive prediction with a large text area for copy-pasting emails/SMS.
  - Real-time length/character counter.
  - A metrics/insights card displaying model details (accuracy, precision) so beginners can understand how the AI works.
  - Beautiful charts/gauges for prediction confidence.
  - Quick-sample buttons for users to immediately test pre-selected spam/ham messages with one click.

### Phase 4: Project Files & Documentation (`README.md`, `.gitignore`)
- Setup a comprehensive `.gitignore` to exclude datasets, cached files, virtual environments, and `.pkl` models from cluttered commits.
- Write a professional `README.md` with:
  - Clear architectural breakdown.
  - Installation instructions.
  - Performance metrics.
  - Future improvement ideas.
