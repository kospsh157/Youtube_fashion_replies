import requests


def search_youtube_channels(api_key, query, params):
    base_url = "https://www.googleapis.com/youtube/v3/search"

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def get_channel_details(api_key, channel_id):
    base_url = "https://www.googleapis.com/youtube/v3/channels"
    params = {
        'part': 'statistics',
        'id': channel_id,
        'key': api_key
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def main(api_key):
    params = {
        'part': 'snippet',
        'q': '패션',
        'type': 'channel',
        'key': api_key,
        'maxResults': 50,
        'pageToken': None,
        'regionCode': 'KR'
    }
    results = []
    while True:

        channels = search_youtube_channels(api_key, "패션", params)
        if channels['items']:
            results.extend(channels['items'])
        else:
            print('items가 존재하지 않음')
            break

        if channels.get('nextPageToken'):
            params['pageToken'] = channels.get('nextPageToken')
        else:
            break

        # if len(results) >= 200:
        #     print('총 결과 길이: ', len(results))

        #     print(results)
        #     break

    if not channels:
        print("채널 검색 중 오류 발생")
        return

    filtered_channels = []

    for item in results:
        details = get_channel_details(api_key, item['snippet']['channelId'])
        if not details:
            continue
        subscriber_count = int(
            details['items'][0]['statistics']['subscriberCount'])
        if subscriber_count > 100000:  # 10만 구독자 이상인 채널만 필터링
            filtered_channels.append({
                "name": item['snippet']['title'],
                "id": item['snippet']['channelId'],
                "subscribers": subscriber_count
            })

    for channel in filtered_channels:
        print(
            f"Channel Name: {channel['name']}, Channel ID: {channel['id']}, Subscribers: {channel['subscribers']}")


if __name__ == "__main__":
    # 여기에 YouTube Data API v3의 API 키를 입력하세요.
    API_KEY = 'AIzaSyAMZtdzCRfSJaDwjSHjHpJdHB2x4en0BiM'
    main(API_KEY)
