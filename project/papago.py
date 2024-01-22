import requests

# 파파고 키
client_id = "8zyFrQmcAHuJWA46WWw5"
client_secret = "XjJxSZhHds"


def translate_with_papago(text, source='ko', target='en'):

    url = "https://openapi.naver.com/v1/papago/n2mt"

    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }

    data = {
        "source": source,
        "target": target,
        "text": text
    }

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        return response.json()['message']['result']['translatedText']
    else:
        return "Error:", response.status_code, response.text
