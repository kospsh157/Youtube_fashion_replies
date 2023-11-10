# REST API 호출, 이미지 파일 처리에 필요한 라이브러리
import requests
import json
import urllib
from PIL import Image

# [내 애플리케이션] > [앱 키] 에서 확인한 REST API 키 값 입력
REST_API_KEY = 'ada696dfddc16d1227e321d7574f06c1'

# 이미지 생성하기 요청

# 칼로 불용어 설정
negative_prompt = "news paper, sleeping cat, dog, ugly face, cropped, foods, alphabet, latters, Home Appliances, Electronics, Furniture, Beverages, Tools, \
    Toys, Books, Music items, sports equipment, pet supplies, gardening supplies, artwork, office supplies, building materials, outdoor activity gear"


def t2i():
    # combined_srt = ", ".join(fashion_items)
    # prompt = "Generate a realistic image of a well-groomed male model on a simple urban street, showcasing " + \
    #     ", ".join(fashion_items) + ". The items should be sharply in focus, capturing their texture and fit in a real-world setting. The model's appearance is natural and photorealistic, with a poised demeanor that complements the understated street scene. The lighting and composition should resemble a high-quality photograph, avoiding any cartoonish or cybernetic aesthetic, to truly emphasize the fashion items in a contemporary, believable manner."
    # # prompt += combined_srt
    # prompt = "On a realistic cool background, a model in a cool pose is wearing " + \
    #     ", ".join(fashion_items)

    prompt = "A stylish model is wearing a vintage-style denim jacket. The jacket is decorated with a check pattern and is layered with a slim fit white t-shirt underneath. The model is wearing a comfortable jumper and a casual cap on her head. Overall, this combination expresses a classic yet modern sensibility, and the long sleeves of the jacket and the slim silhouette of the jumper stand out in particular."

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


# 이미지 생성하기 REST API 호출
response = t2i()

# 응답의 첫 번째 이미지 생성 결과 출력하기
result = Image.open(urllib.request.urlopen(
    response.get("images")[0].get("image")))
result.show()
