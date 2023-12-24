
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
fashion_item_classifier_model = FashionClassifier(len(vocab)+1, 150, 6)

# 저장된 모델 가중치 로드
fashion_item_classifier_model.load_state_dict(
    torch.load('item_classifier_model_weights.pth'))


# 사용할 데이터를 학습했을때와 동일하게 토큰화하고 숫자 인덱스로 변환해야 한다.
# (벡터화 과정은 이미 모델안에 기술되어 있다. 따라서 같은 과정을 거치게 될것이다.)
# .get() 함수는 파이썬 딕셔너리 메소드이다. vocab에 딕셔너리의 키로 단어가 저장되고 해당 단어에 해당하는 인덱스 정수가 값으로 저장되어있다.
# 따라서 vocab에 있는 인덱스대로 인덱싱을 하되, 없다면 0으로 준다는 소리다. 그리고 0은 OOV특수 토큰으로 바뀌고,
# 엔베딩과정에서 다시 벡터화로 실수들로 바뀔 것이다.
def text_pipeline(text):
    return [vocab.get(token, 0) for token in okt.morphs(text)]


def predict(fashion_keywords, fashion_item_classifier_model, text_pipeline):
    fashion_item_classifier_model.eval()  # 모델을 평가 모드로 전환
    output_dict = {}

    with torch.no_grad():  # 그라디언트 계산 비활성화
        for keyword in fashion_keywords:
            # 텍스트를 토큰화하고 숫자 인덱스로 변환
            processed_text = text_pipeline(keyword)

            # LongTensor로 변환하고 배치 차원 추가
            # 대부분의 파이토치 모델은 텐서 형태의 입력을 기대한다. 따라서 반드시 텐서형태로 입력 데이터를 바꿔줘야 한다.
            # unsqueeze() 함수는 텐서의 차원을 하나 더 하는 함수이고 인자로 위치를 받는다. 0은 가장 바깥쪽을 의미한다.
            # 위 함수를 사용하는 이유는, 원래 모델은 데이터가 여러개가 있다고 가정한다. 그런데 지금 현재 샘플 데이터는 단일 텍스트이다.
            # 현재 processed_text 는 [323,423,2, 0 ..] 이런 형태일것이다. 여기에 차원을 더해서 [[323,423,2, 0 ..]]형태로 만들어 주는 것이다.
            processed_text = torch.LongTensor(processed_text).unsqueeze(0)

            # 모델을 사용하여 예측 수행
            output = fashion_item_classifier_model(processed_text)

            # 딕셔너리에 키값으로 아이템 이름을, 그 값으로 예측 인덱스를 대입
            output_dict[keyword] = output.argmax(1).item()

    return output_dict


# 예측 실행 예시
# fashion_keywords = ['롱패딩', '청바지', '나시', '목도리', '패딩 목도리', '사탕', '핸드폰', '검정 스웨터', '아이스크림', '면 티',
#                     '아디다스 신발', '운동화', '크록스 신발', '쪼리', '노래', '모자', '넥타이', '시계', '손목 시계', '셔츠', '겨울 바지']
# predicted_class = predict(
#     fashion_keywords, fashion_item_classifier_model, text_pipeline)
# print("예측된 클래스:", predicted_class)


def classifier_category(predicted_class):
    # predicted_class 에서 분류된 것들을 인덱스 별로 분리해야 한다.
    top_keywords = []
    bottom_keywords = []
    outer_keywords = []
    shoes_keywords = []
    accessary_keywords = []
    not_fashion_items = []

    # 0: 상의, 1: 하의, 2: 아우터, 3: 신발, 4: 악세사리, 5: 비패션아이템
    for keyword, index in predicted_class.items():
        if index == 0:
            top_keywords.append(keyword)
        elif index == 1:
            bottom_keywords.append(keyword)
        elif index == 2:
            outer_keywords.append(keyword)
        elif index == 3:
            shoes_keywords.append(keyword)
        elif index == 4:
            accessary_keywords.append(keyword)
        else:
            not_fashion_items.append(keyword)

    return top_keywords, bottom_keywords, outer_keywords, shoes_keywords, accessary_keywords, not_fashion_items


# print(classifier_category(predicted_class))
