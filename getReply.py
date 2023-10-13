import requests


# 유튜브 댓글은 영상 하나마다 불러올수있음.
# 영상 여러개를 한번에 다 못 받음.

# 함수로 만들어서 메인에서 가져다 쓰자.

'''
    처음 요청시 pageToken 사용: 처음에는 pageToken을 지정하지 않고 API 요청을 보냅니다.
    응답에서 nextPageToken 사용: API의 응답에는 nextPageToken이 포함될 수 있습니다. 
    이 토큰은 다음 페이지의 데이터를 요청하기 위한 것입니다.

'''


# 댓글은 편의를 위해 100개만 받아온다. nextPage까지 검사 안함.
def getReply(video_id, api_key):
    URL = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": api_key,
        "maxResults": 100,
    }

    response = requests.get(URL, params)
    data = response.json()

    # 댓글은 가상 상위의 댓글만 가져온다 대댓글은 가져오지 않는다.
    topReplys = []
    for item in data['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        topReplys.append(comment)

    return topReplys
