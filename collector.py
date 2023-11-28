
from dotenv import load_dotenv
import os
from DB_SERVER_get_channels import get_channels


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

'''


class GetYoutubeComments:
    def __init__(self):
        load_dotenv('api.env')
        self.api_key1 = os.getenv('youtube_API_KEY1')
        self.api_key2 = os.getenv('youtube_API_KEY2')
        self.api_key3 = os.getenv('youtube_API_KEY3')
        self.api_key4 = os.getenv('youtube_API_KEY4')
        self.channels_list = get_channels()
    
    def __call__(self):
        # 여기서 이제 main_youtube.py 코드 수정해서 가져와야함.
        

    
    
