import requests

YOUR_API_KEY = 'AIzaSyAMZtdzCRfSJaDwjSHjHpJdHB2x4en0BiM'  # YouTube Data API 키
SEARCH_TERM = '가을 패션'  # 검색하려는 키워드, 예: 'OpenAI'

# YouTube Search API URL
SEARCH_URL = f'https://www.googleapis.com/youtube/v3/search?part=id&maxResults=10&q={SEARCH_TERM}&key={YOUR_API_KEY}'

response = requests.get(SEARCH_URL)
data = response.json()

video_ids = [item['id']['videoId']
             for item in data['items'] if item['id']['kind'] == "youtube#video"]

print(video_ids)
