# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import pandas as pd
import re, nltk, spacy, string
import en_core_web_sm
import pickle as pk
from nltk.corpus.reader import reviews
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# importing pickel files
cnt_vect = pk.load(open('cnt_vect.pkl','rb'))            # Count Vectorizer
tfidf_transformer_var = pk.load(open('tfidf_transformer_var.pkl','rb')) # TFIDF Transformer
final_logistic_model = pk.load(open('final_model_logistic_best.pkl','rb'))                          # Classification Model
user_user_ratings = pk.load(open('user_user_final_rating.pkl','rb'))   # User-User Recommendation System 

nlp_parser = spacy.load('en_core_web_sm',disable=['ner','parser'])

dfA1 = pd.read_csv('sample30.csv',sep=",")


# removing special chars from our text
def remove_special_charac(text_data, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text_data)
    return text_data
# making all letters lower case
def update_to_lowercase(words):
    low_words = []
    for word in words:
        new_word = word.lower()
        low_words.append(new_word)
    return low_words
# removing punctuation from text
def remove_punctuation_and_splcharc(words):
    updated_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_word = remove_special_charac(new_word, True)
            updated_words.append(new_word)
    return updated_words
# removing stopwords from our text
stopword_list= stopwords.words('english')

def remove_stop_words(words):
    updated_words = []
    for word in words:
        if word not in stopword_list:
            updated_words.append(word)
    return updated_words
#stemming the text
def handle_stem_words(words):
    stemmer = LancasterStemmer()
    stem_updated = []
    for word in words:
        stem = stemmer.stem(word)
        stem_updated.append(stem)
    return stem_updated
#lemmatizing our text
def handle_lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemma_updated = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemma_updated.append(lemma)
    return lemma_updated
#normalizing our text
def normalize_words(words):
    words = update_to_lowercase(words)
    words = remove_punctuation_and_splcharc(words)
    words = remove_stop_words(words)
    return words

def lemmatize(words):
    lemmas = handle_lemmatize_verbs(words)
    return lemmas


def model_result(text):
    word_vect = cnt_vect.transform(text)
    tfidf_vect = tfidf_transformer_var.transform(word_vect)
    output_final = model.predict(tfidf_vect)
    return output_final

def normalize_and_lemmaize_words(input_text):
    input_text = remove_special_charac(input_text)
    words = nltk.word_tokenize(input_text)
    words = normalize_words(words)
    lemmas = lemmatize_words(words)
    return ' '.join(lemmas)

def recommend_products(user_name):
    user_user_ratings_mat = pk.load(open('/content/user_user_final_rating.pkl','rb'))
    prod_list0 = pd.DataFrame(user_user_ratings_mat.loc[user_name].sort_values(ascending=False)[0:20])
    prod_frame0 = prod_df[prod_df.name.isin(prod_list0.index.tolist())]
    output_df0 = prod_frame0[['name','reviews_text']]
    output_df0['lemmatized_text'] = output_df0['reviews_text'].map(lambda text: normalize_and_lemmaize_words(text))
    output_df0['predicted_sentiment'] = model_result(output_df0['lemmatized_text'])
    return output_df0
    
def top5_products(dfB1):
    all_prod=dfB1.groupby(['name']).agg('count')
    rec_prod = dfB1.groupby(['name','predicted_sentiment']).agg('count')
    rec_prod=rec_prod.reset_index()
    merged0 = pd.merge(rec_prod,all_product['reviews_text'],on='name')
    merged0['%percentage'] = (merged0['reviews_text_x']/merged0['reviews_text_y'])*100
    merged0=merged0.sort_values(ascending=False,by='%percentage')
    final_prod0 = pd.DataFrame(merged0['name'][merged0['predicted_sentiment'] ==  1][:5])
    return final_prod0