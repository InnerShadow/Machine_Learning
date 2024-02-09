import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import math

def calculate_tf(word_tokens):
    word_count = Counter(word_tokens)
    total_words = len(word_tokens)
    tf = {word: count / total_words for word, count in word_count.items()}
    return tf


def calculate_idf(documents, word):
    document_count = sum(1 for doc in documents if word in doc)
    if document_count == 0:
        return 0
    idf = math.log2(len(documents) / document_count)
    return idf


def calculate_tfidf(tf, idf):
    return tf * idf


def __main__():
    nltk.download('punkt')
    nltk.download('wordnet')

    files = ['p1.txt', 'p2.txt', 'p3.txt']
    texts = []

    lemmatizer = WordNetLemmatizer()

    for filename in files:
        with open(filename, 'r') as f:
            text = f.read()

        tokens = word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        lemmatized_text = ' '.join(lemmatized_tokens)
        lemmatized_text = lemmatized_text.replace('\n', ' ').replace('.', '').replace(',', '').replace('’', '').replace('-', '').replace('?', '').replace('!', '').replace(';', '').replace(':', '').replace('\'', '').lower()

        texts.append(lemmatized_text)

    for i, doc in enumerate(texts):
        word_tokens = doc.split()
        tf_scores = calculate_tf(word_tokens)
        unique_words = set(word_tokens)

        tfidf_scores = {}
        for word in unique_words:
            tf = tf_scores[word]
            idf = calculate_idf(texts, word)
            tfidf = calculate_tfidf(tf, idf)
            tfidf_scores[word] = tfidf

        print(f"\nTF-IDF Scores for File {files[i]}:")
        for word, tfidf in tfidf_scores.items():
            if tfidf > 0.01 : print(f"{word}: {tfidf}")
            

if __name__ == '__main__':
    __main__()

