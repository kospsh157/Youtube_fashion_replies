
# OpenAI API 키 설정
import openai
import time
openai.api_key = "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"
response = openai.Image.create(
    prompt="a white siamese cat",
    n=1,
    size="1024x1024"
)
image_url = response['data'][0]['url']
