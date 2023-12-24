
import json
import torch
import torch.nn as nn
from konlpy.tag import Okt
from torch.nn.utils.rnn import pad_sequence
import torch

okt = Okt()


# 저장된 어휘(vocab) 로드
with open('vocab.json', 'r') as f:
    vocab = json.load(f)


class FashionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(FashionClassifier, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0)  # 패딩 인덱스 추가
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text):
        embedded = self.embedding(text).mean(1)
        return self.fc(embedded)


# 모델 정의와 인스턴스 생성은 동일하게 진행
model = FashionClassifier(len(vocab)+1, 100, 6)

# 저장된 모델 가중치 로드
model.load_state_dict(torch.load('item_classifier_model_weights.pth'))


# 우선 평가모드로 바꿔야 한다.
model.eval()


# 사용할 데이터를 학습했을때와 동일하게 토큰화하고 숫자 인덱스로 변환해야 한다.
# (벡터화 과정은 이미 모델안에 기술되어 있다. 따라서 같은 과정을 거치게 될것이다.)
# .get() 함수는 파이썬 딕셔너리 메소드이다. vocab에 딕셔너리의 키로 단어가 저장되고 해당 단어에 해당하는 인덱스 정수가 값으로 저장되어있다.
# 따라서 vocab에 있는 인덱스대로 인덱싱을 하되, 없다면 0으로 준다는 소리다. 그리고 0은 OOV특수 토큰으로 바뀌고,
# 엔베딩과정에서 다시 벡터화로 실수들로 바뀔 것이다.
def text_pipeline(text):
    return [vocab.get(word, 0) for word in okt.morphs(text)]
