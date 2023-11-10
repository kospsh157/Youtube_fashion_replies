import openai


# 한글로 약 1000개의 글자당, 0.12불 요금


# OpenAI API 키 설정
# 여기서 'your-api-key'를 실제 API 키로 대체하세요.

# fasion_keywords = ['jacket', 'denim', 'vintage', 'checkered',
#                    't-shirt', 'cap', 'jumper', 'slim', 'sleeve', 'white']
# # GPT 모델을 사용하여 텍스트 생성 요청
# response = openai.Completion.create(
#     engine="text-davinci-003",  # 모델을 지정 (예: text-davinci-003)
#     prompt=f"A stylish model is showcasing a combination of fashion items. The outfit includes {fasion_keywords}, creating a modern and sophisticated look. The ensemble is completed with accessories that enhance the overall style. Non-fashion items, if any, are not part of the attire.",
#     max_tokens=2000  # 생성할 토큰의 최대 개수
# )

# # 생성된 텍스트 출력
# print(response.choices[0].text.strip())


# OpenAI API 키 설정
# 여기서 'your-api-key'를 실제 API 키로 대체하세요.
openai.api_key = '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'


response = openai.Completion.create(
    model="text-davinci-003",  # GPT-4 모델을 사용
    prompt="이것은 테스트 메시지입니다.",  # 여기에 원하는 프롬프트를 넣으세요
    max_tokens=50  # 생성할 토큰의 최대 수
)

print(response.choices[0].text.strip())
