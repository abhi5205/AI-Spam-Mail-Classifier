import os
import urllib.request
import zipfile
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.preprocessing import clean_text

PRIMARY_DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-nlp/master/data/sms.tsv"
FALLBACK_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_PATH = os.path.join(DATA_DIR, "SMSSpamCollection.tsv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def download_dataset():
    """
    Downloads the SMS Spam Collection dataset programmatically.
    Uses a highly reliable primary URL and falls back to UCI if needed.
    """
    if os.path.exists(DATASET_PATH):
        print(f"[*] Dataset already exists locally at: {DATASET_PATH}")
        return

    print("[*] Dataset not found locally. Initiating download...")
    try:
        print(f"[*] Fetching dataset from primary mirror: {PRIMARY_DATA_URL}")
        urllib.request.urlretrieve(PRIMARY_DATA_URL, DATASET_PATH)
        print("[+] Download complete!")
    except Exception as e:
        print(f"[-] Primary download failed: {e}")
        print(f"[*] Attempting fallback download from UCI Repository: {FALLBACK_ZIP_URL}")
        
        zip_path = os.path.join(DATA_DIR, "smsspamcollection.zip")
        try:
            urllib.request.urlretrieve(FALLBACK_ZIP_URL, zip_path)
            print("[*] Extracting ZIP archive...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract('SMSSpamCollection', DATA_DIR)
                
            raw_path = os.path.join(DATA_DIR, "SMSSpamCollection")
            if os.path.exists(raw_path):
                os.rename(raw_path, DATASET_PATH)
                
            os.remove(zip_path)
            print("[+] Fallback download and extraction complete!")
        except Exception as fallback_err:
            raise RuntimeError(
                f"CRITICAL: Failed to download dataset. Check internet connection. Error: {fallback_err}"
            )


def main():
    download_dataset()

    print("\n[*] Loading dataset...")
    df = pd.read_csv(DATASET_PATH, sep='\t', names=['label', 'message'])
    
    print(f"[+] Loaded {len(df)} SMS messages.")
    print("[*] Label distribution:")
    print(df['label'].value_counts())

    print("\n[*] Preprocessing text messages (lowercase, punctuation & stopword removal)...")
    df['cleaned_message'] = df['message'].apply(clean_text)
    
    df = df[df['cleaned_message'] != ""]
    print(f"[+] Preprocessing complete! Remaining messages: {len(df)}")

    print("\n[*] Splitting dataset into training and testing sets (80-20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_message'], 
        df['label'], 
        test_size=0.20, 
        random_state=42,
        stratify=df['label'] 
    )
    print(f"[+] Training set size: {len(X_train)}")
    print(f"[+] Testing set size: {len(X_test)}")

    print("\n[*] Extracting features using TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000) 
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"[+] Vectorization complete. Feature matrix shape: {X_train_tfidf.shape}")
    
    print("\n[*] Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    print("[+] Model training complete!")

    print("\n[*] Evaluating model on test dataset...")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("-" * 50)
    print(f"[ACCURACY] Model Accuracy: {accuracy * 100:.2f}%")
    print("-" * 50)
    print("[CONFUSION MATRIX] Confusion Matrix:")
    print(conf_matrix)
    print("-" * 50)
    print("[METRICS] Classification Report:")
    print(class_report)
    print("-" * 50)

    model_path = os.path.join(MODELS_DIR, "model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")

    print(f"\n[*] Saving trained components to '{MODELS_DIR}'...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[+] Saved Naive Bayes classifier: {model_path}")

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"[+] Saved TF-IDF Vectorizer: {vectorizer_path}")
    print("\n[SUCCESS] Setup and training completed successfully! You are now ready to run the Streamlit app.")


if __name__ == "__main__":
    main()
