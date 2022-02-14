import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy import data, datasets
import random
from time import time

SEED = 5
random.seed(SEED)

BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10
# cuda 사용 가능여부
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("활용 디바이스:", DEVICE)

TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)
# 데이터셋의 3가지 종류(train, vaildation, test 중 train과 test를 스플릿
train_set, test_set = datasets.IMDB.splits(TEXT, LABEL)


# min_freq는 학습 데이터에서 최소 5번 이상 등장한 단어만을 단어 집합에 추가하겠다는 의미
TEXT.build_vocab(train_set, min_freq=5) # 단어 집합 생성
LABEL.build_vocab(train_set)

vocab_size = len(TEXT.vocab)
n_classes = 2

print(f'단어집 크기 {vocab_size}')
print(f'클래스의 개수 {n_classes}')

# stoi로 단어와 각 단어의 정수 인덱스가 저장되어져 있는 딕셔너리 객체에 접근할 수 있음
# <unk>: Unknown으로 알수없는 단어라는 의미 <pad> padding 각 문장끼리 서로 길이 다를수 있어 <pad>를 활용하여 채움


## 데이터 로드 만들기
# train_set에서 trian set 과 vaildation set을 나눔
trainset, valset = train_set.split(split_ratio=0.8)


train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (trainset, valset, test_set), batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)

print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_iter)))
print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_iter)))
print('검증 데이터의 미니 배치의 개수 : {}'.format(len(val_iter)))

batch = next(iter(train_iter)) # 첫번째 배치
print(batch.text.shape)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (trainset, valset, test_set), batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)

class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0)) # 첫번째 히든 스테이트를 0벡터로 초기화
        x, _ = self.gru(x, h_0)  # GRU의 리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        h_t = x[:,-1,:] # (배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.

        self.dropout(h_t)

        logit = self.out(h_t)  # (배치 크기, 은닉 상태의 크기) -> (배치 크기, 출력층의 크기)

        return logit

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(model, optimizer, train_iter):
    model.train() # 모델을 학습 모드로 변환 / 평가 모드는 model.eval() 로 할 수 있다.
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)  # 각 x 와 y을 앞서 설정한 DEVICE(GPU 혹은 CPU) 에 보내는 것
        y.data.sub_(1)                    # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()             # 반복 때마다 기울기를 새로 계산하므로, 이 함수로 초기화

        logit = model(x)                  # x를 모델에 넣어서 가설(hypothesis)를 획득
        loss = F.cross_entropy(logit, y)  # 가설과 groud truth를 비교하여 loss 계산
        loss.backward()                   # loss 를 역전파 알고리즘으로 계산
        optimizer.step()                  # 계산한 기울기를 앞서 정의한 알고리즘에 맞추어 가중치를 수정

def evaluate(model, val_iter):
    """evaluate model"""
    model.eval() # 모델을 학습 모드로 변환 / 평가 모드는 model.eval() 로 할 수 있다.
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))