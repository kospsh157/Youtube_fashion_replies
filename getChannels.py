import requests
from DB_injector import inputChannels


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


def main(api_key, query):
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

    cnt = 1
    while True:

        channels = search_youtube_channels(api_key, query, params)
        if channels['items']:
            print(
                f'현재 {cnt}회 while문 반복에서, 총 {len(channels["items"])}개의 채널을 찾았습니다!')
            results.extend(channels['items'])
        else:
            print('items가 존재하지 않음')
            break

        if channels.get('nextPageToken'):
            params['pageToken'] = channels.get('nextPageToken')
        else:
            break

        # results가 10000개가 넘으면 반복문 탈출
        if len(results) > 1000:
            print(f'한도를 넘었습니다. 종료합니다. 현재 총 {len(results)}채널 입니다.')
            break

        cnt += 1

    if not channels:
        print("채널 검색 중 오류 발생")
        return

    filtered_channels = []

    sub_cnt = 0
    for item in results:

        details = get_channel_details(api_key, item['snippet']['channelId'])
        if not details:
            continue
        subscriber_count = int(
            details['items'][0]['statistics']['subscriberCount'])
        if subscriber_count > 100000:  # 10만 구독자 이상인 채널만 필터링
            print(f'십만구독자가 넘는 유튜버를 찾았습니다. 현재 총 {sub_cnt}명입니다. ')
            filtered_channels.append({
                "name": item['snippet']['title'],
                "id": item['snippet']['channelId'],
                "subscribers": subscriber_count
            })
            sub_cnt += 1

    for channel in filtered_channels:
        print(
            f"Channel Name: {channel['name']}, Channel ID: {channel['id']}, Subscribers: {channel['subscribers']}")

    # 왜그런지는 알수없지만 중복되어 수집함
    # 중복되는 값을 지우고 DB에 전달, 채널ID가 기본키이기 때문에 중복은 반드시 제거해야함
    seen = set()
    list = []
    for channel in filtered_channels:
        if channel['id'] not in seen:
            seen.add(channel['id'])
            list.append(channel)

    # DB에 삽입
    inputChannels(list, query)


if __name__ == "__main__":
    API_KEY = 'AIzaSyAMZtdzCRfSJaDwjSHjHpJdHB2x4en0BiM'
    API_KEY2 = 'AIzaSyA4ltLYUhWYUEa3rbevQNCAELquiG-fWPg'
    main(API_KEY, '패션')
