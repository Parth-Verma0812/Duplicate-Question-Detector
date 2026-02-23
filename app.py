import streamlit as st
import numpy as np
import pickle
import pandas as pd
import distance
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import re
from bs4 import BeautifulSoup

nltk.download('punkt_tab')
nltk.download('stopwords')

model=pickle.load(open('approach_3_model.pkl','rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))

def preprocess(q):

  q=str(q).lower().strip()

  # Replacing certain special characters with their string equivalents
  q=q.replace('%', ' percent')
  q=q.replace('$', ' dollar ')
  q=q.replace('₹', ' rupee ')
  q=q.replace('€', ' euro ')
  q=q.replace('@', ' at ')
  q=q.replace('&','and')

  # The pattern '[math]' appeared in the text several time so we need to replace it
  q=q.replace('[math]','')

  # replacing some numbers with their string equivae=lents
  q=q.replace(',000,000,000 ','b ')
  q=q.replace(',000,000 ','m ')
  q=q.replace(',000 ','k ')
  q=re.sub(r'([0-9]+)000000000',r'\1b',q)
  q=re.sub(r'([0-9]+)000000',r'\1m',q)
  q=re.sub(r'([0-9]+)000',r'\1k',q)


  # Decontracting words

  # Source - https://stackoverflow.com/a/19794953
  # Posted by arturomp, modified by community. See post 'Timeline' for change history
  # Retrieved 2026-02-23, License - CC BY-SA 3.0

  contractions = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
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
  "he'd": "he had",
  "he'd've": "he would have",
  "he'll": "he shall",
  "he'll've": "he shall have",
  "he's": "he has",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how has",
  "I'd": "I had",
  "I'd've": "I would have",
  "I'll": "I shall",
  "I'll've": "I shall have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it shall",
  "it'll've": "it shall have",
  "it's": "it has",
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
  "she'd": "she had",
  "she'd've": "she would have",
  "she'll": "she shall",
  "she'll've": "she shall have",
  "she's": "she has",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so as",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that has",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there has",
  "they'd": "they had",
  "they'd've": "they would have",
  "they'll": "they shall",
  "they'll've": "they shall have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what shall",
  "what'll've": "what shall have",
  "what're": "what are",
  "what's": "what has",
  "what've": "what have",
  "when's": "when has",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where has",
  "where've": "where have",
  "who'll": "who shall",
  "who'll've": "who shall have",
  "who's": "who has",
  "who've": "who have",
  "why's": "why has",
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
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you shall",
  "you'll've": "you shall have",
  "you're": "you are",
  "you've": "you have"
  }

  decontracted=[]

  for word in q.split():
    if word in contractions:
      word=contractions[word]
    decontracted.append(word)

  q=" ".join(decontracted)

  q=q.replace("'ve",' have')
  q=q.replace("n't",' not')
  q=q.replace("'re",' are')
  q=q.replace("'ll",' will')

  ## Removing html tags using beautiful soup
  q=BeautifulSoup(q).get_text()

  ## Removing Punctuation

  pattern=re.compile(r'\W')
  q=re.sub(pattern,' ',q).strip()

  return q


def common_words(row):
  w1=set(map(lambda word : word.lower().strip(),row['question1'].split(" ")))
  w2=set(map(lambda word : word.lower().strip(),row['question2'].split(" ")))
  return len(w1 & w2)

def find_total_unique_words(row):
  w1=set(map(lambda word : word.lower().strip(),row['question1'].strip()))
  w2=set(map(lambda word : word.lower().strip(),row['question2'].strip()))
  return(len(w1)+len(w2))

def add_token_features(row):
  q1=row['question1']
  q2=row['question2']

  safe_div=0.001

  stop_words=stopwords.words('english')
  token_features=[0.0]*8

  ## Converting the sentences into the tokens
  q1_tokens=word_tokenize(q1)
  q2_tokens=word_tokenize(q2)

  if len(q1_tokens)==0 or len(q2_tokens)==0:
    return token_features

  ## Extracting the non stopwords from the questions
  q1_words=set([word for word in q1_tokens if word not in stop_words])
  q2_words=set([word for word in q2_tokens if word not in stop_words])

  ## Extracting the stopwords from the each questions
  q1_stopwords=set([word for word in q1_tokens if word in stop_words])
  q2_stopwords=set([word for word in q2_tokens if word in stop_words])


  ## Extracting the common non_stopwords from the quespairs
  common_word_count=len(set(q1_words).intersection(set(q2_words)))

  ## Extracting the count of common stopwords
  common_stopwords_count=len(set(q1_stopwords).intersection(set(q2_stopwords)))

  ## Extracting the count of common tokens
  common_token_counts=len(set(q1_tokens).intersection(set(q2_tokens)))

  token_features[0]=common_word_count/(min(len(q1_words),len(q2_words))+safe_div)
  token_features[1]=common_word_count/(max(len(q1_words),len(q2_words))+safe_div)
  token_features[2]=common_stopwords_count/(min(len(q1_stopwords),len(q2_stopwords))+safe_div)
  token_features[3]=common_stopwords_count/(max(len(q1_stopwords),len(q2_stopwords))+safe_div)
  token_features[4]=common_token_counts/(min(len(q1_tokens),len(q2_tokens))+safe_div)
  token_features[5]=common_token_counts/(max(len(q1_tokens),len(q2_tokens))+safe_div)

  ## Check if the last word is same or not
  token_features[6] =int(q1_tokens[-1]==q2_tokens[-1])

  ## Check if the first word is same or not
  token_features[7]=int(q1_tokens[0]==q2_tokens[0])

  return token_features



def fetch_length_features(row):
  q1=row['question1']
  q2=row['question2']

  len_features=[0.0]*3

  q1_tokens=word_tokenize(q1)
  q2_tokens=word_tokenize(q2)

  if len(q1_tokens)==0 or len(q2_tokens)==0:
    return len_features

  ## Absolute length difference
  len_features[0]=abs(len(q1_tokens)-len(q2_tokens))

  ## Average token length of both the questions
  len_features[1]=(len(q1_tokens)+len(q2_tokens))/2


  ## Finding the maximum substring ratio

  strs=list(distance.lcsubstrings(q1,q2))
  len_features[2]=len(strs[0])/min(len(q1_tokens),len(q2_tokens)+1)

  return len_features



def fetch_fuzzy_features(row):
  q1=row['question1']
  q2=row['question2']

  fuzz_features=[0.0]*4

  # fuzzy ratio
  fuzz_features[0]=fuzz.QRatio(q1,q2)

  # Partial fuzzy ratio
  fuzz_features[1]=fuzz.partial_ratio(q1,q2)

  # token_sort_ratio
  fuzz_features[2]=fuzz.token_sort_ratio(q1,q2)

  ## token_set_ratio
  fuzz_features[3]=fuzz.token_set_ratio(q1,q2)

  return fuzz_features


def pipeline(q1,q2):

  row = pd.Series({
        'question1': q1,
        'question2': q2
    })

  input_query=[]

  q1=preprocess(q1)
  q2=preprocess(q2)

  input_query.append(len(q1))
  input_query.append(len(q2))

  input_query.append(len(q1.split(" ")))
  input_query.append(len(q2.split(" ")))

  input_query.append(common_words(row))
  input_query.append(find_total_unique_words(row))
  input_query.append(common_words(row)/find_total_unique_words(row))


  ## token_features

  token_features=add_token_features(row)
  input_query.extend(token_features)

  ## length based features

  len_features=fetch_length_features(row)
  input_query.extend(len_features)

  ## fuzzy wuzzy features

  fuzzy_features=fetch_fuzzy_features(row)
  input_query.extend(fuzzy_features)

  ## applying the vectorization

  q1_vectorized=cv.transform([q1]).toarray()
  q2_vectorized=cv.transform([q2]).toarray()

  input=np.hstack((np.array(input_query).reshape(1,22),q1_vectorized,q2_vectorized))

  prediction=model.predict(input)[0]

  if prediction==1:
    return 'Duplicate Questions'
  else:
    return 'Non-duplicate Questions'
  

st.title("Duplicate Question Detector")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Check Duplicate"):

    if q1 and q2:
        result = pipeline(q1,q2)
        st.success(result)
    else:
        st.warning("Please enter both questions.")