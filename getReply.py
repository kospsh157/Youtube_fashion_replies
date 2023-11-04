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

    data = None

    try:
        response = requests.get(URL, params)
        data = response.json()
    except requests.RequestException as e:
        print(f"API 요청 중 오류 발생: {e}")

    # 댓글은 가장 상위의 댓글만 가져온다 대댓글은 가져오지 않는다.
    topReplys = []

    try:
        if 'items' in data:
            for item in data['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                topReplys.append(comment)
        else:
            print("items 키가 응답에 없습니다. 다음 원인을 확인하세요:")
            if 'error' in data:
                if "disabled comments" in data['error']['message']:
                    print('해당 비디오는 댓글이 중지된 비디오임')
                    return "댓글이 중지된 비디오"
                else:
                    print(data['error']['message'])
            else:
                print(data)

    except Exception as e:
        print('에러 발생')
        print(e)

    print('해당 영상의 댓글 총 개수:', len(topReplys))
    return topReplys


# API_KEY = 'AIzaSyAMZtdzCRfSJaDwjSHjHpJdHB2x4en0BiM'
# print(getReply('e4yWqRNVZCo', API_KEY))
