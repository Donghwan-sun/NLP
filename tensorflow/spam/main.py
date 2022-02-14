import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATAPATH = "./data/spam.csv"
data = pd.read_csv(DATAPATH, encoding="latin1")
print(data['v1'].isnull().sum())
print(data['v2'].isnull().sum())
columns = list(data.columns)
print(columns)
for i in range(2, 5):
    del data[columns[i]]

print(data.columns)
data['v1'] = data['v1'].replace(['ham', 'spam'], [0, 1])
print(data.info())
#유니크한 값찾기
print(data['v2'].nunique())
print(len(data))
data.drop_duplicates(subset=["v2"], inplace=True)
print(len(data))
data['v1'].value_counts().plot(kind='bar')
plt.show()

print('정상 메일과 스팸 메일 개수')
print(data.groupby('v1').size().reset_index(name='count'))

print(f'정상 메일의 비율 = {round(data["v1"].value_counts()[0]/len(data) * 100,3)}%')
print(f'스팸 메일의 비율 = {round(data["v1"].value_counts()[1]/len(data) * 100,3)}%')
X_data = data['v2']
Y_data = data['v1']

#이킷 런의 train_test_split에 stratify의 인자로서 레이블 데이터를 기재하면 훈련 데이터와 테스트 데이터를 분리할 때 레이블의 분포가 고르게 분포하도록 합니다
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=0, stratify=Y_data)

print('--------훈련 데이터의 비율-----------')
print(f'정상 메일 = {round(Y_train.value_counts()[0]/len(Y_train) * 100,3)}%')
print(f'스팸 메일 = {round(Y_train.value_counts()[1]/len(Y_train) * 100,3)}%')

print('--------테스트 데이터의 비율-----------')
print(f'정상 메일 = {round(Y_test.value_counts()[0]/len(Y_test) * 100,3)}%')
print(f'스팸 메일 = {round(Y_test.value_counts()[1]/len(Y_test) * 100,3)}%')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
tokenizer_json = tokenizer.to_json()

with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

X_train_encoded = tokenizer.texts_to_sequences(X_train)
print(X_train_encoded[:5])
word_to_index = tokenizer.word_index
print(word_to_index)
