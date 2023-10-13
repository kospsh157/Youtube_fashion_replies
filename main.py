import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta

'''
구글api는 한번 쿼리를 날리면 500개의 검색결과를 찾는다. 이 수는 500개로 정해져있다. 
날짜 파라미터를 사용해서 여러번 호출하면 데이터를 모을 수 있다. 
예를 들어 21년 1월 부터 2월까지 한달치로 정하면 그 한달치에 해당하는 제목들 500개를 받아온다. 

그리고 다시 2월 부터 3월까지 범위를 지정하고 달마다 500개씩 데이터를 쌓아가는 것이다. 

# 순서 [sbs뉴스, 한국경제tv, kbs뉴스, ytn뉴스]
channel_ids = ["UCkinYTS9IHqOEwR1Sze2JTw", "UCF8AeLlUbEpKju6v1H6p8Eg", "UCcQTRi69dsVYHN3exePtZ1A", "UChlgI3UHCOnwUGzWzbJ3H5w"] 

'''

# .json()는 서버로부터 받은 응답 객체를 파이썬의 오브젝트(리스트나 딕셔너리)로 반환한다.


# nextPageToken 속성은 구글 api가 보내는 응답에 있는 속성으로, 이 속성값 안에 있는 데이터로 다시 호출하면 바로
# 이전 데이터에 이어서 다시 연속적으로 응답을 해준다
# 만약 nextPageToken이게 없다면 해당 응답이 마지막 페이지라는 것이다. 따라서 이걸 이용해서 계속해서 모든 데이터를
# 받을 때 까지 호출 할수 있다.

API_KEY = 'AIzaSyAMZtdzCRfSJaDwjSHjHpJdHB2x4en0BiM'
# CHANNEL_ID = 'UCF8AeLlUbEpKju6v1H6p8Eg'  # 원하는 채널의 ID
CHANNEL_IDS = ["UCkinYTS9IHqOEwR1Sze2JTw",
               "UCF8AeLlUbEpKju6v1H6p8Eg"]  # 여러 채널 ID를 리스트로 추가


SEARCH_KEYWORD = '한국'
MAX_RESULTS = 50  # 한 번의 요청으로 가져올 수 있는 최대 결과 수


# API URL
SEARCH_ENDPOINT = 'https://www.googleapis.com/youtube/v3/search'
# 키워드와 원하는 채널명들을 넣어주면 그에 해당하는 비디오들을 찾아서 id들을 반환

VIDEOS_ENDPOINT = 'https://www.googleapis.com/youtube/v3/videos'
# 위에서 받은 비디오 고유 id들을 이용해서 이걸 가지고 비디오들의 각각의 통계 정보들을 반환.


total_request_cnt = 0

start_date = "2021-01-01T00:00:00Z"
end_date = "2021-01-31T23:59:59Z"
start_date_obj = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
all_video_data = []

for channel_id in CHANNEL_IDS:
    print('channel_id:', channel_id)
    next_page_token = None
    while True:
        print(f'총 요청 횟수: {total_request_cnt}')
        total_request_cnt += 1

        next_date_obj = start_date_obj + relativedelta(weeks=1)

        search_params = {
            'key': API_KEY,
            'q': SEARCH_KEYWORD,
            'type': 'video',
            'order': 'date',
            'channelId': channel_id,
            'maxResults': MAX_RESULTS,
            'publishedAfter': start_date_obj.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'publishedBefore': next_date_obj.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'part': 'id',
            'pageToken': next_page_token,
        }

        search_response = requests.get(SEARCH_ENDPOINT, params=search_params)
        search_data = search_response.json()

        total_results = search_data["pageInfo"]["totalResults"]
        print(f'한번 videoID호출에서 받아온 ID들: {total_results} ')

        video_ids = [item['id']['videoId'] for item in search_data['items']]

        # Fetch video details using video IDs
        videos_params = {
            'key': API_KEY,
            'id': ','.join(video_ids),
            'part': 'snippet,statistics',
        }
        videos_response = requests.get(VIDEOS_ENDPOINT, params=videos_params)
        all_video_data.extend(videos_response.json()['items'])

        # Get the next page token
        next_page_token = search_data.get('nextPageToken')

        if not next_page_token:
            print('@@@@@@@@@@@@일주일치가 끝났으니 다음 일주일으로 넘어가자@@@@@@@@@@@@@@@.')
            print(all_video_data)

            start_date_obj = next_date_obj

            end_date_obj = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")
            if start_date_obj >= end_date_obj:
                break


# publishedAt 문자열을 datetime 객체로 변환하는 함수
def convert_to_datetime(published_str):
    return datetime.strptime(published_str, '%Y-%m-%dT%H:%M:%SZ')


# publishedAt 기준으로 오름차순 정렬
sorted_video_data = sorted(
    all_video_data, key=lambda x: convert_to_datetime(x['snippet']['publishedAt']))


for item in sorted_video_data:
    title = item['snippet']['title']
    published_at = item['snippet']['publishedAt']
    views = item['statistics'].get('viewCount', 0)
    likes = item['statistics'].get('likeCount', 0)
    dislikes = item['statistics'].get('dislikeCount', 0)
    comments = item['statistics'].get('commentCount', 0)

    print(f"Title: {title}")
    print(f"Published At: {published_at}")
    print(f"Views: {views}")
    print(f"Likes: {likes}")
    print(f"Dislikes: {dislikes}")
    print(f"Comments: {comments}")
    print("="*50)


print('총 비디오 데이터 개수: ', len(sorted_video_data))
