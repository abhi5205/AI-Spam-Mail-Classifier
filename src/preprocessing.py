import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK datasets are downloaded locally
for corpus in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{corpus}" if corpus == "stopwords" else f"tokenizers/{corpus}")
    except LookupError:
        nltk.download(corpus, quiet=True)


def clean_text(text: str) -> str:
    """
    Normalizes and cleans input text for NLP classification.
    
    Processing Steps:
    1. Lowecases input text.
    2. Strips standard punctuation marks.
    3. Tokenizes strings into individual word units.
    4. Removes common English stopwords and non-alphanumeric tokens.
    
    Args:
        text (str): Raw message content.
        
    Returns:
        str: Space-separated cleaned word tokens.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    
    return " ".join(cleaned_tokens)
