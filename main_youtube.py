import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from getReply import getReply
from DB_injector import insert_video_data
from DB_retriever import get_channel_ids
from DB_connector import get_db_connection
from DB_SERVER_connector import DB_server_connentor

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

# 메인계정
# API_KEY = 'AIzaSyAMZtdzCRfSJaDwjSHjHpJdHB2x4en0BiM'

# 비빅바
API_KEY = 'AIzaSyA4ltLYUhWYUEa3rbevQNCAELquiG-fWPg'

# 하마
# API_KEY = 'AIzaSyA-SXNjsNcNijuLnete6DQLk4X_F7URIis'

# 바밤바
# API_KEY = 'AIzaSyBQU5HopruqfPu9kqc1XOtEs69O4IEfW7k'
# CHANNEL_ID = 'UCF8AeLlUbEpKju6v1H6p8Eg'  # 원하는 채널의 ID
# CHANNEL_IDS = ["UCkinYTS9IHqOEwR1Sze2JTw",
#                "UCF8AeLlUbEpKju6v1H6p8Eg"]  # 여러 채널 ID를 리스트로 추가


# postgresql 접속해서 id 받아와야함
channel_ids = get_channel_ids()
# 각 항목이 튜플로 감싸져있어서 풀고 깨끗하게 채널만 담김 리스트형태로 만들어야함.
CHANNELS = [channel[0] for channel in channel_ids]
SEARCH_KEYWORD = '패션'
MAX_RESULTS = 50  # 한 번의 요청으로 가져올 수 있는 최대 결과 수가 50개임
# API URL
SEARCH_ENDPOINT = 'https://www.googleapis.com/youtube/v3/search'
# 키워드와 원하는 채널명들을 넣어주면 그에 해당하는 비디오들을 찾아서 id들을 반환

VIDEOS_ENDPOINT = 'https://www.googleapis.com/youtube/v3/videos'
# 위에서 받은 비디오 고유 id들을 이용해서 이걸 가지고 비디오들의 각각의 통계 정보들을 반환.

total_request_cnt = 0
all_video_data = []
for channel_id in CHANNELS:
    print('channel_id:', channel_id)
    next_page_token = None
    start_date = "2023-10-01T00:00:00Z"
    end_date = "2023-10-31T23:59:59Z"
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
    while True:
        print(f'총 요청 횟수: {total_request_cnt}')
        total_request_cnt += 1
        next_date_obj = start_date_obj + relativedelta(days=31)
        print(f'현재 자료 수집중인 기간: {start_date_obj} 부터 {next_date_obj} 여기까지')
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

        # search_data['items']가 0이면 관련 비디오 영상이 없다는 뜻이다.
        video_ids = [item['id']['videoId'] for item in search_data['items']]

        if len(search_data['items']) == 0:
            print('지금 검색으로는 영상이 없음')
        else:
            print('@@@@@현재 채널의 동영상 개수@@@@@@: ', len(search_data['items']))

        # Fetch video details using video IDs
        videos_params = {
            'key': API_KEY,
            'id': ','.join(video_ids),
            'part': 'snippet,statistics',
        }
        videos_response = requests.get(VIDEOS_ENDPOINT, params=videos_params)
        all_video_data.extend(videos_response.json()['items'])

        # 해당 비디오에 대한 댓글들도 가져오기
        # 동시에 all_video_data에서 해당 아이디 찾아서 리플 추가하기
        for id in video_ids:
            replies = getReply(id, API_KEY)
            for item in all_video_data:
                if item['id'] == id:
                    item['replies'] = replies

        # Get the next page token
        next_page_token = search_data.get('nextPageToken')

        # 여기서 위에 next_page_token 값이 없다면, 일주일치는 다 받은 상태이고 다음주로 넘어간다.
        if not next_page_token:
            start_date_obj = next_date_obj

            # 정해진 시간이 넘으면 종료한다.
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")
            if start_date_obj >= end_date_obj:
                break
    print('현재까지 모은 @@총 동영상@@ 개수: ', len(all_video_data))

# publishedAt 문자열을 datetime 객체로 변환하는 함수


def convert_to_datetime(published_str):
    return datetime.strptime(published_str, '%Y-%m-%dT%H:%M:%SZ')


# publishedAt 기준으로 오름차순 정렬
sorted_video_data = sorted(
    all_video_data, key=lambda x: convert_to_datetime(x['snippet']['publishedAt']))


for item in sorted_video_data:

    video_id = item['id']
    title = item['snippet']['title']
    published_at = item['snippet']['publishedAt']
    views = item['statistics'].get('viewCount', 0)
    likes = item['statistics'].get('likeCount', 0)
    dislikes = item['statistics'].get('dislikeCount', 0)
    comments = item['statistics'].get('commentCount', 0)
    replies = item['replies']

    print(f"Video_id: {video_id}")
    print(f"Title: {title}")
    print(f"Published At: {published_at}")
    print(f"Views: {views}")
    print(f"Likes: {likes}")
    print(f"Dislikes: {dislikes}")
    print(f"Comments: {comments}")
    print(f"Replies: {replies}")
    print("="*50)


print('총 비디오 데이터 개수: ', len(sorted_video_data))


# 비디오 데이타 삽입
# db에 한다고 쳐도 일단 로컬에도 똑같이 저장해야 하기 때문에 같이 불러야 한다.
try:
    insert_video_data(sorted_video_data, SEARCH_KEYWORD, get_db_connection)
except Exception as e:
    print('로컬 데이터 삽입중에 에러 발생: ', e)




# try:
#     # 로컬 DB서버에 데이타 삽입
#     insert_video_data(video_dats_list, query, DB_server_connentor)
# except Exception as e:
#     print('DB서버에 데이터 삽입중에 에러 발생: ', e)



# 주기적으로 함수를 실행하고 그럼 api서버가 따로 있어야 하고 거기서 댓글을 주기적으로 모으고 다시 db서버로 전달한다.
# 웹서버는 독립적으로 api서버와 연결되어 있지 않은 상태로, db와 접속해서 알아서 데이터를 받아온다.
#

'''
2일에 한번 데이터 수집.
일단 채널은 수동으로 추가.
댓글은 2일에 한번씩 수집






'''
