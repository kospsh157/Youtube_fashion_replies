# REST API 호출, 이미지 파일 처리에 필요한 라이브러리
import requests
import json
import urllib
from PIL import Image

# [내 애플리케이션] > [앱 키] 에서 확인한 REST API 키 값 입력
REST_API_KEY = 'ada696dfddc16d1227e321d7574f06c1'

# 이미지 생성하기 요청


def t2i(fashion_items, negative_prompt):
    # combined_srt = ", ".join(fashion_items)
    prompt = "Generate a realistic image of a well-groomed male model on a simple urban street, showcasing " + \
        ", ".join(fashion_items) + ". The items should be sharply in focus, capturing their texture and fit in a real-world setting. The model's appearance is natural and photorealistic, with a poised demeanor that complements the understated street scene. The lighting and composition should resemble a high-quality photograph, avoiding any cartoonish or cybernetic aesthetic, to truly emphasize the fashion items in a contemporary, believable manner."
    # prompt += combined_srt

    r = requests.post(
        'https://api.kakaobrain.com/v2/inference/karlo/t2i',
        json={
            'prompt': prompt,
            'negative_prompt': negative_prompt
        },
        headers={
            'Authorization': f'KakaoAK {REST_API_KEY}',
            'Content-Type': 'application/json'
        }
    )
    # 응답 JSON 형식으로 변환
    response = json.loads(r.content)
    return response


# 프롬프트에 사용할 제시어


negative_prompt = "sleeping cat, dog, ugly face, cropped"

# # 이미지 생성하기 REST API 호출
# response = t2i(
#     ["blue shirt", "beige pants", "black shoes", "layered white-T"], negative_prompt)

# # 응답의 첫 번째 이미지 생성 결과 출력하기
# result = Image.open(urllib.request.urlopen(
#     response.get("images")[0].get("image")))
# result.show()
