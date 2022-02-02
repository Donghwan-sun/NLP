import numpy as np
import pandas as pd
import re
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
#import urllib.request
np.random.seed(seed=0)

pd.get_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# 데이터 로드
data = pd.read_csv('./data/archive/Reviews.csv')
data = data[['Text', 'Summary']]

# 데이터 정제
# 데이터 개수 비교
print("TEXT 개수: {} Summary 개수: {}".format(data['Text'].count(), data['Summary'].count()))
print("TEXT 중복 배제 개수: ", data['Text'].nunique())
print("Summary 중복 배제 개수: ", data['Summary'].nunique())
print("결측값 확인\n", data.isnull().sum())

#결측치 제거
data.dropna(axis=0, inplace=True)

#동일한 의미를 가진 줄인 단어를 정규화
with open('contraction.json', 'r', encoding='utf8') as f:
    contractions = json.load(f)

# NLTK 불용어 개수: 179개
stop_words = set(stopwords.words('english'))
print('불용어 개수 :',len(stop_words))
print(stop_words)

def preprocess_sentence(sentence, remove_stopwords = True):
    sentence = sentence.lower() # 텍스트 소문자화
    sentence = BeautifulSoup(sentence, "lxml").text # <br />, <a href = ...> 등의 html 태그 제거
    sentence = re.sub(r'\([^)]*\)', '', sentence) # 괄호로 닫힌 문자열  제거 Ex) my husband (and myself) for => my husband for
    sentence = re.sub('"','', sentence) # 쌍따옴표 " 제거
    sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")]) # 약어 정규화
    sentence = re.sub(r"'s\b","",sentence) # 소유격 제거. Ex) roland's -> roland
    sentence = re.sub("[^a-zA-Z]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    sentence = re.sub('[m]{2,}', 'mm', sentence) # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah

    # 불용어 제거 (Text)
    if remove_stopwords:
        tokens = ' '.join(word for word in sentence.split() if not word in stop_words if len(word) > 1)
    # 불용어 미제거 (Summary)
    else:
        tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
    return tokens

temp_text = 'Everything I bought was great, infact I ordered twice and the third ordered was<br />for my mother and father.'
temp_summary = 'Great way to start (or finish) the day!!!'
print(preprocess_sentence(temp_text))
print(preprocess_sentence(temp_summary, 0))