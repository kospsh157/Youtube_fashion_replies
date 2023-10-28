import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification

# 사전 학습된 KoELECTRA 모델과 토크나이저 로드
MODEL_NAME = "monologg/koelectra-small-v2-discriminator"
tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v2-finetuned-nsmc").eval()

def predict_sentiment(sentence):
    # 토크나이징
    tokens = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        # 모델을 통해 감정 분석 예측 수행
        output = model(tokens)[0]
    prediction = torch.argmax(output, dim=1).item()
    
    # 0은 부정, 1은 긍정으로 라벨링 되어있음
    return "긍정" if prediction == 1 else "부정"

# 예제 문장으로 감정 분석
sentence = "유튜브 댓글 내용을 여기에 입력하세요."
result = predict_sentiment(sentence)
print(f"'{sentence}'는 {result}입니다.")
