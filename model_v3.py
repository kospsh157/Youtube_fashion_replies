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
    '티셔츠', '스웨터', '블라우스', '폴로셔츠', '셔츠', '후드티', '카디건', '니트', '탑', '조끼',
    '드레스 셔츠', '목 폴라', '티', '티셔츠', '옥스포드 셔츠', '후드 티', '집업 후드', '폴로 셔츠', '크롭탑', '니트',
    '하프 폴라', '후드', '짧은 티', '긴 팔', '상의', '저지', '데님 셔츠', '탱크탑', '반 팔', '린넨 셔츠',
    '티셔츠', '스웨터', '블라우스', '폴로 셔츠', '셔츠', '후드 티', '카디건', '니트', '탑', '조끼',
    '드레스셔츠', '목폴라', '티', '티셔츠', '옥스포드 셔츠', '후드 티', '집업후드', '폴로셔츠', '크롭 탑', '니트',
    '하프폴라', '후드', '짧은티', '긴팔', '상의', '저지', '데님셔츠', '탱크 탑', '반팔', '린넨셔츠',

    '청바지', '슬랙스', '숏팬츠', '치노 팬츠', '레깅스', '스커트', '팬츠', '바지', '하의', '스타킹',
    '트레이닝복 바지', '골덴 바지', '코듀로이 바지', '카고 팬츠', '치노 바지', '치노', '숏 팬츠', '레깅스', '츄리닝 바지', '데님 바지',
    '카고', '스커트', '치마', '테니스 치마', '가죽 바지', '조거', '스키니', '와이드', '드레스 팬츠', '레더 팬츠',
    '청바지', '슬랙스', '숏 팬츠', '치노팬츠', '레깅스', '스커트', '팬츠', '바지', '하의', '스타킹',
    '트레이닝복바지', '골덴바지', '코듀로이바지', '카고팬츠', '치노바지', '치노', '숏팬츠', '레깅스', '츄리닝바지', '데님바지',
    '카고', '스커트', '치마', '테니스치마', '가죽바지', '조거', '스키니', '와이드', '드레스팬츠', '레더팬츠',

    '코트', '트렌치 코트', '패딩 재킷', '데님 재킷', '가죽 자켓', '블레이저', '파카', '바람막이', '레인 코트', '야상',
    '아우터', '덕다운', '블루종', '청자켓', '청재킷', '캐시미어 코트', '블레이저', '벨벳 재킷', '재킷', '퀼트 재킷',
    '패딩', '롱 트렌치코트', '데님재킷', '항공 점퍼', '레더 재킷', '트러커 재킷', '스키 재킷', '맥시 코트', '롱 코트', '롱 패딩',
    '코트', '트렌치코트', '패딩재킷', '데님재킷', '가죽자켓', '블레이저', '파카', '바람 막이', '레인코트', '야상',
    '아우터', '덕 다운', '블루종', '청 자켓', '청 재킷', '캐시미어코트', '블레이저', '벨벳재킷', '재킷', '퀼트 재킷',
    '패딩', '롱트렌치코트', '데님 재킷', '항공점퍼', '레더재킷', '트러커재킷', '스키재킷', '맥시코트', '롱코트', '롱패딩',

    '런닝화', '하이힐', '부츠', '샌들', '로퍼', '스니커즈', '플랫 슈즈', '워커', '슬립온', '신발',
    '컨버스', '나이키 신발', '아디다스 신발', '퓨마 신발', '군화', '부츠', '컴뱃 부츠', '펌프스', '스트랩 샌들', '플랫폼 슈즈',
    '글래디에이터 샌들', '에스파드리유', '모카신', '옥스퍼드 슈즈', '메리제인 슈즈', '뉴발 신발', '장화', '크록스', '슬리퍼', '구두',
    '런닝화', '하이 힐', '부츠', '샌들', '로퍼', '스니커즈', '플랫슈즈', '워커', '슬립 온', '신발',
    '컨버스', '나이키신발', '아디다스신발', '퓨마신발', '군화', '부츠', '컴뱃부츠', '펌프스', '스트랩샌들', '플랫폼슈즈',
    '글래디에이터샌들', '에스파드리유', '모카신', '옥스퍼드슈즈', '메리제인슈즈', '뉴발신발', '장화', '크록스', '슬리퍼', '구두',

    '목걸이', '귀걸이', '팔찌', '시계', '반지', '헤어밴드', '선글라스', '스카프', '모자', '벨트',
    '악세사리', '악세', '브로치', '커프스', '토트백', '클러치', '백팩', '핸드백', '목도리', '헤어클립',
    '마스크', '양말', '콘택트 렌즈', '렌즈', '안경', '가방', '헤드폰', '헤어 핀', '컨텍트 렌즈', '넥타이',
    '목걸이', '귀걸이', '팔찌', '시계', '반지', '헤어 밴드', '선글라스', '스카프', '모자', '벨트',
    '악세사리', '악세', '브로치', '커프스', '토트백', '클러치', '백팩', '핸드백', '목도리', '헤어 클립',
    '마스크', '양말', '콘택트렌즈', '렌즈', '안경', '가방', '헤드폰', '헤어핀', '컨텍트렌즈', '넥 타이',

    '책', '볼펜', '노트북', '컵', '휴대폰', '키보드', '마우스', '모니터', '가방', '의자',
    '물건', '안테나', '자동차', '노래', '음식', '쓰레기', '테이블', '스피커', '우산', '램프',
    '카메라', '사람', '플랜터', '빌딩', '땅', '문신', '패션', '친구', '유튜브', '유튜버',
    '졸라', '존나', '구독자', '방장', '팀원', '왜', '인사', '안녕하세요', '댓글', '모델',
    '칼라', '쿠션', '야외', '빌딩', '인터넷', '조금', '패션', '모델', '예쁘', '이쁘',
    '추천', '구독', '구독자', '실내', '팀원', '와', '돈', '안녕하세요', '댓글', '개',
]
labels = [0]*60 + [1]*60 + [2]*60 + [3]*60 + [4]*60 + [5] * \
    60  # 0: 상의, 1: 하의, 2: 아우터, 3: 신발, 4: 악세사리, 5: 비패션아이템

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

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

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


model = FashionClassifier(len(vocab)+1, 100, 6)  # 임베딩 차원 10, 분류 6개
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
for epoch in range(20):
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


# # 모델 평가
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for items, labels in test_loader:
#         outputs = model(items)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print(f'Accuracy: {100 * correct / total}%')
