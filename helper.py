import re
from bs4 import BeautifulSoup
import pickle
import numpy as np
import string
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


import nltk
nltk.download('stopwords')


with open(r"C:\ds\PROJECT\QUORA_QUESTION_PAIR\cv_model.pkl", "rb") as f:
    cv = pickle.load(f)


with open(r"C:\ds\PROJECT\QUORA_QUESTION_PAIR\tfidf_model.pkl", "rb") as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words('english'))


def preprocess(q):
    q = str(q).lower().strip()

    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q

import re
import string
import difflib
from nltk.corpus import stopwords
from nltk.util import ngrams

stop_words = set(stopwords.words('english'))
def extract_features(text1, text2):

    # --------------------
    # SAFETY
    # --------------------
    if not isinstance(text1, str):
        text1 = ""
    if not isinstance(text2, str):
        text2 = ""

    # --------------------
    # PREPROCESS
    # --------------------
    t1 = preprocess(text1)
    t2 = preprocess(text2)

    w1 = t1.split()
    w2 = t2.split()

    s1 = set(w1)
    s2 = set(w2)

    features = {}

    # --------------------
    # TF-IDF COSINE
    # --------------------
    try:
        v1 = tfidf.transform([t1])
        v2 = tfidf.transform([t2])
        features["tfidf_cosine"] = cosine_similarity(v1, v2)[0][0]
    except Exception:
        features["tfidf_cosine"] = 0.0

    # --------------------
    # LENGTH FEATURES
    # --------------------
    features["q1_len"] = len(t1)
    features["q2_len"] = len(t2)
    features["len_diff"] = abs(len(t1) - len(t2))

    features["q1_word"] = len(w1)
    features["q2_word"] = len(w2)
    features["word_count_ratio"] = len(w1) / (len(w2) + 1)

    # --------------------
    # WORD OVERLAP
    # --------------------
    common_words = s1 & s2
    union_words = s1 | s2

    features["common_word"] = len(common_words)
    features["total_word"] = len(w1) + len(w2)
    features["word_share"] = len(common_words) / (features["total_word"] + 1)
    features["jaccard_sim"] = len(common_words) / (len(union_words) + 1)
    features["overlap_coeff"] = len(common_words) / (min(len(s1), len(s2)) + 1)
    features["unique_word_diff"] = abs(len(s1) - len(s2))

    # --------------------
    # STOPWORDS
    # --------------------
    sw1 = set(w for w in w1 if w in stop_words)
    sw2 = set(w for w in w2 if w in stop_words)

    features["stopword_diff"] = abs(len(sw1) - len(sw2))
    features["common_stopword_ratio"] = len(sw1 & sw2) / (len(sw1 | sw2) + 1)

    # --------------------
    # CONTENT WORDS
    # --------------------
    cw1 = set(w for w in w1 if w not in stop_words)
    cw2 = set(w for w in w2 if w not in stop_words)

    features["content_overlap"] = len(cw1 & cw2)

    # --------------------
    # POSITIONAL
    # --------------------
    features["first_word_match"] = int(w1[0] == w2[0]) if w1 and w2 else 0
    features["first_word_same"] = int(w1[0] == w2[0]) if w1 and w2 else 0
    features["last_word_match"] = int(w1[-1] == w2[-1]) if w1 and w2 else 0

    # --------------------
    # N-GRAMS
    # --------------------
    features["bigram_overlap"] = len(set(ngrams(w1, 2)) & set(ngrams(w2, 2)))
    features["trigram_overlap"] = len(set(ngrams(w1, 3)) & set(ngrams(w2, 3)))

    # --------------------
    # CHARACTER FEATURES
    # --------------------
    features["char_overlap"] = len(set(t1) & set(t2)) / (len(set(t1) | set(t2)) + 1)

    features["avg_word_len_diff"] = abs(
        (sum(len(w) for w in w1) / (len(w1) + 1)) -
        (sum(len(w) for w in w2) / (len(w2) + 1))
    )

    features["punct_diff"] = abs(
        sum(c in string.punctuation for c in text1) -
        sum(c in string.punctuation for c in text2)
    )

    # --------------------
    # STRING SIMILARITY
    # --------------------
    features["sequence_ratio"] = difflib.SequenceMatcher(None, t1, t2).ratio()

    i = 0
    while i < min(len(t1), len(t2)) and t1[i] == t2[i]:
        i += 1
    features["prefix_match_len"] = i

    i = 0
    while i < min(len(t1), len(t2)) and t1[-1 - i] == t2[-1 - i]:
        i += 1
    features["suffix_match_len"] = i

    # --------------------
    # DIGITS + EXACT
    # --------------------
    features["digit_overlap"] = len(set(filter(str.isdigit, t1)) & set(filter(str.isdigit, t2)))
    features["exact_match"] = int(t1 == t2)

    # --------------------
    # HANDCRAFTED FEATURES (ORDER FIX)
    # --------------------
    feature_df = pd.DataFrame([features])
    feature_df = feature_df.reindex(columns=TRAIN_FEATURE_ORDER, fill_value=0.0)

    # --------------------
    # BAG OF WORDS (CV)
    # --------------------
    vectors = cv.transform([t1, t2])
    q1_vec = vectors[0].toarray()
    q2_vec = vectors[1].toarray()

    combined = np.hstack([q1_vec, q2_vec])

    feature_names = cv.get_feature_names_out()
    bow_cols = [f"q1_{w}" for w in feature_names] + [f"q2_{w}" for w in feature_names]

    bow_df = pd.DataFrame(combined, columns=bow_cols)

    # --------------------
    # FINAL DATAFRAME
    # --------------------
    final_df = pd.concat([feature_df, bow_df], axis=1)

    return final_df

TRAIN_FEATURE_ORDER = [
    "q1_len",
    "q2_len",
    "q1_word",
    "q2_word",
    "common_word",
    "total_word",
    "word_share",
    "jaccard_sim",
    "len_diff",
    "common_stopword_ratio",
    "first_word_same",
    "unique_word_diff",
    "overlap_coeff",
    "exact_match",
    "char_overlap",
    "avg_word_len_diff",
    "word_count_ratio",
    "punct_diff",
    "first_word_match",
    "last_word_match",
    "bigram_overlap",
    "trigram_overlap",
    "stopword_diff",
    "content_overlap",
    "sequence_ratio",
    "prefix_match_len",
    "suffix_match_len",
    "digit_overlap",
    "tfidf_cosine"
]

