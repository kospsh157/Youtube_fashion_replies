
import json
import torch
import torch.nn as nn
from konlpy.tag import Okt
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader, TensorDataset

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
# 전체 단어수에 +1를 하는 이유는 패딩토큰을 추가하기 위한 것이다. 거의 대부분 패팅 토큰은 항상 추가해야 하기때문에
    # 자주 보게 되는 현상일 것이다.
fashion_item_classifier_model = FashionClassifier(len(vocab)+1, 300, 6)

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
            # 모델에 넣어주기 까지, 과정을 정리하자면,
            # 1. text_pipeline를 통해서 학습할때 사용한 똑같은 토큰나이저로 텍스트를 토큰화하고,
            # 2. 역시 같은 어휘 사전을 이용해서 사전에 정의되어 있는대로 토큰을 사용하고, 없으면 0으로 준다.
            # ( 여기서 0은 패딩 토큰이다. )
            # 3. 그리고 나서 모델은 항상 텐서 형태를 바라기 때문에 텐서로 바꿔 주고 비로소 입력해준다.
            output = fashion_item_classifier_model(processed_text)

            # 딕셔너리에 키값으로 아이템 이름을, 그 값으로 예측 인덱스를 대입
            output_dict[keyword] = output.argmax(1).item()

    return output_dict


# 예측 실행 예시
# fashion_keywords = ['플란넬셔츠', '셔츠', '티셔츠', '후드집업', '스트라이프셔츠', '스웨터', '레이스탑', '저지', '가디건', '데님셔츠', '니트', '블라우스', '긴팔티', '터틀넥', '후드티', '반팔티', '크롭탑', '탱크탑', '캐미솔', '폴로셔츠', '데님', '컬러팬츠', '카프리', '스커트', '쇼츠', '사각팬츠', '큐롯', '바지', '숏팬츠', '캐주얼팬츠', '플레어팬츠', '레깅스', '슬랙스', '청바지', '와이드팬츠', '린넨팬츠', '스키니', '조거팬츠', '하이웨이스트바지', '트레이닝복', '모토자켓', '패딩', '가죽자켓', '베스트', '애나멜코트', '코트', '바시티자켓',
#                     '카디건', '덴임자켓', '후드집업', '블레이저', '트렌치코트', '더플코트', '봄버자켓', '바람막이', '퍼자켓', '자켓', '필드재킷', '무스탕', '청자켓', '부츠', '블로퍼', '메리제인', '뮬', '컴뱃부츠', '앵클부츠', '스니커즈', '샌들', '플랫', '운동화', '슬리퍼', '플랫폼', '구두', '힐', '로퍼', '옥스퍼드', '글래디에이터', '워커', '에스파듀', '코르크슈즈', '브로치', '백팩', '헤어밴드', '페도라', '발찌', '토트백', '크로스바디백', '스카프', '팔찌', '선글라스', '키링', '반지', '장갑', '목걸이', '클러치', '귀걸이', '시계', '벨트', '지갑', '모자']
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


# print('')
# print('')

# print(
#     '예측된 클래스가지고 다시 카테고리별로 정리해서 나열')
# print(classifier_category(predicted_class))

# 모델 평가
# test = fashion_keywords


# def create_dataset(keywords_list):
#     items = pad_sequence([torch.tensor(x)
#                          for x in keywords_list], batch_first=True)
#     labels = torch.tensor(df['label'].values)
#     return TensorDataset(items, labels)


# test_dataset = create_dataset(test)
# test_loader = DataLoader(test_dataset, batch_size=2)


# fashion_item_classifier_model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for items, labels in test_loader:
#         outputs = fashion_item_classifier_model(items)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print(f'Accuracy: {100 * correct / total}%')


'''



1. 문제점 고친거 보여주고
2. 벡엔드 대충 만들었다고 보여주고
3. 프로젝트는 마무리 (현재 상황 보여주고 마감)



'''
