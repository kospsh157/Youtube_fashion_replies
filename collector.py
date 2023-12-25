
from dotenv import load_dotenv
import os
from DB_SERVER_get_channels import get_channels
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from getReply import getReply


'''
만약 유튜브에서 api한계치에 도달했을때 응답객체
{
  "error": {
    "code": 403,
    "message": "The request cannot be completed because you have exceeded your quota.",
    "errors": [
      {
        "message": "The request cannot be completed because you have exceeded your quota.",
        "domain": "youtube.quota",
        "reason": "quotaExceeded"
      }
    ]
  }
}

한계치에 도달했을때, 스위칭? 
스위칭은 못하고... 의미도 없고... 결국 수동을 해야 한다느 건데... 그럼 이렇게 자동화 코드로 만들 필요도 없다...

우선 주기를 이틀로 잡는다. 
이틀마다 다음 함수를 실행해서 db에 패션 관련 동영상 데이터를 저장한다.



'''


# 에러 핸들러 함수
def handle_api_error(error_response, api_number):
    error_code = error_response.get('error', {}).get('code')
    error_message = error_response.get('error', {}).get('message')
    specific_errors = error_response.get('error', {}).get('errors', [])

    if error_code == 403:
        print("접근 거부됨: ", error_message)
        for error in specific_errors:
            print("세부 사항: ", error.get('message'))
            # 특정 도메인이나 이유에 대한 추가적인 핸들링
            if error.get('domain') == 'youtube.quota':
                print("유튜브 할당량 초과")

                change_api_key(api_number)

    elif error_code == 404:
        print("Nof Found에러", error_message)
    # 그 밖에 에러 코드 마다 핸들러 달아줄려면 여기에다가 작성..
    else:
        print("예상치 못한 에러 입니다.")

# 한도 도달했을때, api키 돌려쓰기 함수


def change_api_key(api_number):
    if api_number != 4:
        return api_number+1
    else:
        return 1


class GetYoutubeComments:
    # API key 받아오기
    load_dotenv('api.env')
    API_KEY1 = os.getenv('youtube_API_KEY1')
    API_KEY2 = os.getenv('youtube_API_KEY2')
    API_KEY3 = os.getenv('youtube_API_KEY3')
    API_KEY4 = os.getenv('youtube_API_KEY4')
    # 한번의 요청에 받아오는 댓글의 수(기본적으로 50개가 이미 최고임. 유튜브에서는 한번에 요청에 이 이상 주지 않음.)
    MAX_RESULT = 50
    # API URL
    SEARCH_ENDPOINT = 'https://www.googleapis.com/youtube/v3/search'

    # 키워드와 원하는 채널명들을 넣어주면 그에 해당하는 비디오들을 찾아서 id들을 반환
    VIDEOS_ENDPOINT = 'https://www.googleapis.com/youtube/v3/videos'

    # 위에서 받은 비디오 고유 id들을 이용해서 이걸 가지고 비디오들의 각각의 통계 정보들을 반환.

    # 일단 end_date 는 하드코딩으로 2일치만 하자 그냥 2일에 한번씩 실행해서 2일동안 데이타만 수집하는 것.
    def __init__(self, keyword, api_number, start_date):
        self.channels_list = get_channels()
        self.keyword = keyword
        self.api_number = api_number
        self.start_date = start_date + 'T00:00:00Z'
        self.end_date = (datetime.strptime(self.start_date, "%Y-%m-%dT%H:%M:%SZ") +
                         relativedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.all_video_data = []
        self.max_result = GetYoutubeComments.MAX_RESULT

        if self.api_number == 1:
            self.api_key = GetYoutubeComments.API_KEY1
        elif self.api_number == 2:
            self.api_key = GetYoutubeComments.API_KEY2
        elif self.api_number == 3:
            self.api_key = GetYoutubeComments.API_KEY3
        elif self.api_number == 4:
            self.api_key = GetYoutubeComments.API_KEY4

    # publishedAt 문자열을 datetime 객체로 변환하는 함수

    def convert_to_datetime(published_str):
        return datetime.strptime(published_str, '%Y-%m-%dT%H:%M:%SZ')

    # 날짜가 들어오면 해당 날짜의 월에 해당하는 제일 마지막 날을 구하는 함수
    # 아직은 사용하지 않는다.. 일단 일주일 간격으로 무조건 받아오겠끔 하드 코딩 했다.
    # def get_end_of_month(date):
    #     next_month = date.replace(day=1) + relativedelta(months=1)
    #     end_of_month = next_month - timedelta(days=1)
    #     return end_of_month

    def __call__(self):
        total_request_cnt = 0
        for channel_id in self.channels_list:
            print('channel_id:', channel_id)
            next_page_token = None
            start_date = self.start_date
            end_date = self.end_date

            start_date_obj = datetime.strptime(
                start_date, "%Y-%m-%dT%H:%M:%SZ")

            while True:
                print(f'총 요청 횟수: {total_request_cnt}')
                total_request_cnt += 1
                next_date_obj = start_date_obj + relativedelta(days=2)
                print(
                    f'현재 자료 수집중인 기간: {start_date_obj} 부터 {next_date_obj} 여기까지')
                search_params = {
                    'key': self.api_key,
                    'q': self.keyword,
                    'type': 'video',
                    'order': 'date',
                    'channelId': channel_id,
                    'maxResults': GetYoutubeComments.MAX_RESULT,
                    'publishedAfter': start_date_obj.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    'publishedBefore': next_date_obj.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    'part': 'id',
                    'pageToken': next_page_token,
                }

                search_response = requests.get(
                    GetYoutubeComments.SEARCH_ENDPOINT, params=search_params)
                search_data = search_response.json()

                # 에러 발생시
                if search_response.status_code != 200:
                    error_response = search_response.json()
                    handle_api_error(error_response, self.api_number)

                # search_data['items']가 0이면 관련 비디오 영상이 없다는 뜻이다.
                video_ids = [item['id']['videoId']
                             for item in search_data['items']]

                if len(search_data['items']) == 0:
                    print('지금 검색으로는 영상이 없음')
                else:
                    print('@@@@@현재 채널의 동영상 개수@@@@@@: ',
                          len(search_data['items']))

                # Fetch video details using video IDs
                videos_params = {
                    'key': self.api_key,
                    'id': ','.join(video_ids),
                    'part': 'snippet,statistics',
                }
                videos_response = requests.get(
                    GetYoutubeComments.VIDEOS_ENDPOINT, params=videos_params)
                self.all_video_data.extend(videos_response.json()['items'])

                # 해당 비디오에 대한 댓글들도 가져오기
                # 동시에 all_video_data에서 해당 아이디 찾아서 리플 추가하기
                for id in video_ids:
                    replies = getReply(id, self.api_key)
                    for item in self.all_video_data:
                        if item['id'] == id:
                            item['replies'] = replies

                # Get the next page token
                next_page_token = search_data.get('nextPageToken')

                # 여기서 위에 next_page_token 값이 없다면, 일주일치는 다 받은 상태이고 다음주로 넘어간다.
                if not next_page_token:
                    start_date_obj = next_date_obj

                    # 정해진 시간이 넘으면 종료한다.
                    end_date_obj = datetime.strptime(
                        end_date, "%Y-%m-%dT%H:%M:%SZ")
                    if start_date_obj >= end_date_obj:
                        self.real_end_date = start_date_obj.strftime(
                            "%Y-%m-%dT%H:%M:%SZ")
                        print(
                            '지정된 end_date 날짜가 넘어서 수집을 종료합니다. read_end_date: ', self.real_end_date)
                        break
            print('이제 for문 한 번 돌았음. 현재까지 모은 총 동영상 개수: ',
                  len(self.all_video_data))
        # publishedAt 기준으로 오름차순 정렬
        # 이 GetYoutubeComments클래스는 결국 GetYoutubeComments('패션', 1) 이런식으로 호출할 수 있고, 리턴값은 날짜별로 정렬된
        # 댓글들을 포함한 비디오의 정보들이 담긴 리스트형태이다.
        return sorted(self.all_video_data, key=lambda x: self.convert_to_datetime(x['snippet']['publishedAt']))
