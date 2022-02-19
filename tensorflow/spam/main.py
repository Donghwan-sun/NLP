import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json

from sklearn.model_selection import train_test_split
import tensorflow as tf
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

threshold = 2
total_cnt = len(word_to_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합(vocabulary)에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
# 지나치게 낮은 단어들은 자연어 처리에서 제외하고 싶다면 케라스 토크나이저 선언 시에 단어 집합의 크기를 제한
tokenizer = Tokenizer(num_words = total_cnt - rare_cnt + 1)

vocab_size = len(word_to_index) + 1

print('단어 집합의 크기: {}'.format((vocab_size)))

print('메일의 최대 길이 : %d' % max(len(sample) for sample in X_train_encoded))

print('메일의 평균 길이 : %f' % (sum(map(len, X_train_encoded))/len(X_train_encoded)))

plt.hist([len(sample) for sample in X_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

max_len = 189
X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len)
print("훈련 데이터의 크기(shape):", X_train_padded.shape)

from model import RNN
batch_size = 16
embedding_dim = 32
hidden_units = 32

# 손실함수 정의
@tf.function
def sparse_cross_entropy_loss(labels, logits):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        logtis = model(x)
        loss = sparse_cross_entropy_loss(y, logtis)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

train_data = tf.data.Dataset.from_tensor_slices((X_train_padded, Y_train)).shuffle(4135).batch(batch_size=batch_size, drop_remainder=True)
#train_data = iter(train_data)

import os
import time
rnn = RNN(vocab_size, embedding_dim, hidden_units,batch_size=16)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
epochs = len(X_train_padded)/ batch_size

for epoch in range(int(epochs)):
    start = time.time()
    hidden = rnn.reset_states()
    for (idx, (input, target)) in enumerate(train_data):
        loss = train_step(rnn, input, target)
        if idx % 1 == 0 :
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch + 1, idx, loss))
    if (epoch + 1) % 5 == 0:
      rnn.save_weights(checkpoint_prefix.format(epoch=epoch))
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

rnn.save_weights(checkpoint_prefix.format(epoch=epoch))
print("트레이닝이 끝났습니다!")

test_rnn = RNN(vocab_size, embedding_dim, hidden_units, batch_size)
test_rnn.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
test_rnn.build(input_shape=(1, 189))
test_rnn.summary()