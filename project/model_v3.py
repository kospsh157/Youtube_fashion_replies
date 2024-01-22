import json
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from konlpy.tag import Okt

# 랜덤 시드 설정
torch.manual_seed(0)

# 유니크한 아이템 생성
items = [
    '티셔츠', '스웨터', '블라우스', '면 티', '셔츠', '후드 티', '카디건', '니트', '윈드 스토퍼', '조끼',
    '드레스 셔츠', '목 폴라', '터틀 넥', '기모 티', '옥스포드 셔츠', '쉬프트 드레스', '집업 후드', '나시', '맨투맨', '남방',
    '하프 폴라', '후드', '짧은 티', '긴 팔', '상의', '오버사이즈 셔츠', '바디콘 드레스', '언더 셔츠', '반 팔', '린넨 셔츠',
    '기모 후드', '풀오버', '집업', '폴로 셔츠', '셔츠', '드레스', '하프 집업', '와이셔츠', '탑', '민소매',
    '바디돌 드레스', '볼레로', '티', '기본 티', '런닝구', '후드 티', '집업후드', '박스 티셔츠', '크롭 탑', '스웻터',
    '피케 셔츠', '후디', '수트', '긴팔', '폴라', '저지', '데님셔츠', '탱크 탑', '반팔', '린넨셔츠',
    '탱크탑', '크롭탑', '폴로셔츠', '베스트', '탱크탑', '베스트', '저지', '터틀넥', '옥스퍼드셔츠', '니트',
    '레이스탑', '저지',

    '청바지', '슬랙스', '페그드 팬츠', '치노 팬츠', '퀼로트', '스커트', '팬츠', '바지', '하의', '스타킹',
    '트레이닝복 바지', '배기 팬츠', '코듀로이 바지', '스키니 팬츠', '치노 바지', '코듀로이 큐롯 팬츠', '숏 팬츠', '핫 팬츠', '츄리닝 바지', '데님 바지',
    '카고', '스커트', '치마', '테니스 치마', '가죽 바지', '조거', '시티 쇼츠', '와이드', '드레스 팬츠', '레더 팬츠',
    '청바지', '와이드 팬츠', '숏 팬츠', '부츠컷', '니커스', '스커트', '팬츠', '바지', '커프드 팬츠', '스타킹',
    '가우초 팬츠', '골덴 바지', '코듀로이 바지', '스트레이트 팬츠', '벨 보텀 팬츠', '카고 팬츠', '숏팬츠', '레깅스', '카프리 팬츠', '데님바지',
    '카고', '스커트', '치마', '테니스치마', '가죽바지', '조거', '스키니', '니커즈 팬츠', '오버롤', '버뮤다 팬츠',
    '레깅스', '래깅스',

    '자켓 코트', '트렌치 코트', '패딩 재킷', '데님 재킷', '가죽 자켓', '하프 코트', '노카라 코트', '바람막이', '레인 코트', '야상',
    '아우터', '덕다운', '블루종', '청자켓', '미니멀 자켓', '캐시미어 코트', '조끼', '벨벳 재킷', '경량 패딩', '퀄팅 자켓',
    '패딩', '롱 트렌치코트', '트윌 자켓', '항공 점퍼', '레더 재킷', '트러커 재킷', '스키 재킷', '맥시 코트', '롱 코트', '롱 패딩',
    '코트', '트렌치코트', '패딩재킷', '데님재킷', '라이더 자켓', '블레이저', '파카', '바람 막이', '가디건', '니트 조끼',
    '아우터', '덕 다운', '블루종', '청 자켓', '오버셔츠', '캐시미어코트', '무스탕', '벨벳재킷', '재킷', '퀼트 재킷',
    '청 자켓', '롱트렌치코트', '데님 재킷', '항공점퍼', '레더재킷', '트러커재킷', '스키재킷', '점퍼', '롱코트', '뽀글이 점퍼',
    '롱패딩',

    '런닝화', '하이힐', '킬 힐', '샌들', '가죽 로퍼', '스니커즈', '플랫 슈즈', '워커', '슬립온', '신발',
    '컨버스', '나이키 신발', '아디다스 신발', '플레인 토', '군화', '부츠', '컴뱃 부츠', '슬링백 슈즈', '스트랩 샌들', '플랫폼 슈즈',
    '글래디에이터 샌들', '보트 슈즈', '태슬', '메리제인', '메리제인 슈즈', '뉴발란스 신발', '장화', '크록스', '슬리퍼', '펌프스',
    '로퍼', '하이 힐', '롱부츠', '어글리 슈즈', '케쥬얼 구두', '스니커즈', '몽크 스트랩', '웨지힐', '삭스 앵클부츠', '쪼리',
    '컨버스', '나이키신발', '아디다스신발', '퓨마신발', '플레인 토', '가죽 구두', '컴뱃부츠', '뮬 슈즈', '앵클부츠', '부츠 슈즈',
    '나막신', '에스파드리유', '더비', '옥스퍼드슈즈', '윙팁', '모카신', '운동화', '스트레이트 팁', '백밴드 슈즈', '구두',
    '운동화',

    '목걸이', ' 귀걸이', '팔찌', ' 시계', '반지', '헤어밴드', '선글라스', '스카프', '캡', '벨트',
    '악세사리', '악세', '브로치', '커프스', '토트백', '클러치', '백팩', '핸드백', '목도리', '헤어클립',
    '마스크', '양말', '콘택트 렌즈', ' 렌즈', '안경', '가방', '헤드폰', '헤어 핀', '컨텍트 렌즈', '넥타이',
    '보석', '네일', '귀 걸이', '시계', '열쇠고리', '헤어 밴드', '키링', '빽', '모자', '가죽 벨트',
    '타투', '악새', '백', '패딩 목도리', '가방백', '헤나', '백팩', '빽팩', '털 장갑', '헤어 클립',
    ' 마스크', ' 양말', '콘택트렌즈', '렌즈', '네일 아트', '핀', '벙어리 장갑', '헤어핀', '컨텍트렌즈', '넥 타이',
    '모자', '목거리', '목걸이',
]


# 0: 상의, 1: 하의, 2: 아우터, 3: 신발, 4: 악세사리, 5: 비패션아이템
labels = [0]*72 + [1]*62 + [2]*61 + [3]*61 + [4]*63
# 319개

print(len(items))
print(len(labels))
df = pd.DataFrame({'item': items, 'label': labels})

# 레이블 인코딩
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# 한국어 토큰화 및 인덱싱
okt = Okt()
counter = Counter()
for item in df['item']:
    counter.update(okt.morphs(item))
vocab = {word: i+1 for i, word in enumerate(counter)}


def text_pipeline(text):
    return [vocab.get(word, 0) for word in okt.morphs(text)]


df['item'] = df['item'].apply(text_pipeline)

# 데이터 분할
train, test = train_test_split(df, test_size=0.2, random_state=0)

# 텐서 데이터셋 생성


def create_dataset(df):
    items = pad_sequence([torch.tensor(x)
                         for x in df['item']], batch_first=True)
    labels = torch.tensor(df['label'].values)
    return TensorDataset(items, labels)


train_dataset = create_dataset(train)
test_dataset = create_dataset(test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# 모델 정의


class FashionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(FashionClassifier, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0)  # 패딩 인덱스 추가
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text):
        embedded = self.embedding(text).mean(1)
        return self.fc(embedded)


model = FashionClassifier(len(vocab)+1, 300, 6)  # 임베딩 차원 300, 분류 6개
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
for epoch in range(40):
    model.train()
    total_loss = 0
    for items, labels in train_loader:
        optimizer.zero_grad()
        pred = model(items)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')


# 모델의 state_dict 저장하기
torch.save(model.state_dict(), 'item_classifier_model_weights.pth')

# 어휘(vocab) 저장하기
with open('vocab.json', 'w') as f:
    json.dump(vocab, f)


# # 다른곳에서 로드 할 때

# # 모델 정의와 인스턴스 생성은 동일하게 진행
# model = FashionClassifier(len(vocab)+1, 100, 6)

# # 저장된 모델 가중치 로드
# model.load_state_dict(torch.load('item_classifier_model_weights.pth'))

# # 저장된 어휘(vocab) 로드
# with open('vocab.json', 'r') as f:
#     vocab = json.load(f)

# # Okt 토크나이저 초기화 (별도의 상태 저장 필요 없음)
# okt = Okt()


# 모델 평가
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for items, labels in test_loader:
        outputs = model(items)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {100 * correct / total}%')
