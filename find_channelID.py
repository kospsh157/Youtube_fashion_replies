import requests

API_KEY = "AIzaSyAMZtdzCRfSJaDwjSHjHpJdHB2x4en0BiM"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"

params = {
    'key': API_KEY,
    'part': 'id',
    'forUsername': 'ytnnews24'
}

response = requests.get(CHANNELS_URL, params=params)

if response.status_code == 200:
    result = response.json()
    if result['items']:
        channel_id = result['items'][0]['id']
        print("Channel ID:", channel_id)
    else:
        print("No channel found for the username sbsnews8.")
else:
    print("API 요청 오류:", response.status_code)
