import schedule
import time
import datetime
from collector import GetYoutubeComments
from datetime import datetime, timedelta
import pytz


def job():
    # 짝수날에만 실행하여 결국 2일마다 한번씩 실행되도록 한다.
    korea_timezone = pytz.timezone('Asia/Seoul')
    current_time_in_korea = datetime.now(korea_timezone)
    current_date_in_korea = current_time_in_korea.date()
    if current_date_in_korea.day % 2 == 0:
        # 여기에 유튜브 댓글을 수집하는 함수를 호출
        two_days_ago = current_time_in_korea - timedelta(days=2)
        # 수집 시작 날짜는 현재 날짜에서 2일 전이어야함.
        formatted_date = two_days_ago.strftime('%Y-%m-%d')

        # ('검색 키워드', 'api 넘버', '시작 날짜')
        GetYoutubeComments('패션', 1, formatted_date)


# 매일 정오에 'job' 함수 실행
# 스케줄러는 매일 정오에 실행되지만 실제로 짝수날에만 데이터 수집을 한다.
schedule.every().day.at("12:00").do(job)

# 무한 반복되어야 한다. run_pendind()이 항상 돌아가고 있어야 스케쥴러가 셋팅된대로 작동하는 것이다.
# sleep 주기는 100초로 하였고, 따라서 오차가 최대 100초 차이날 수 있다.
while True:
    schedule.run_pending()
    time.sleep(100)


'''
    1. 스케줄러가 정해진 시간에 job을 실행한다.
    2. GETYOUTUBECOMMNETS 클래스가 작동해서 유튜브 댓글을 2일 단위로 모은다.  
    3. 스케쥴러에서 시작 날짜를 계속 현재 날짜에서 2일 전 시간으로 전해줘야 한다.
    4. 

    




    변환의 필요성:
호환성: Llama 모델이 처음 제공될 때 transformers 라이브러리와 호환되지 않는 형식으로 되어 있을 수 있습니다. 
따라서, 이 모델을 transformers 라이브러리와 호환되는 형식으로 변환해야만, transformers 라이브러리의 기능과 도구들을 이용하여 
모델을 쉽게 로드하고 활용할 수 있습니다.

편의성 및 기능성: transformers 라이브러리는 다양한 사전 훈련된 모델을 쉽게 로드하고 사용할 수 있는 편리한 인터페이스를 제공합니다. 
이 라이브러리를 사용하면 모델을 더 쉽게 사용, 조정, 평가할 수 있으며, NLP 관련 다양한 작업을 효율적으로 수행할 수 있습니다.

변환 과정:
이 변환 과정은 Llama 모델의 원본 파일(가중치 및 구성)을 transformers 라이브러리에서 요구하는 특정 형식(예: 모델 가중치 파일, 구성 파일 등)으로 변환합니다. 
이를 통해 모델이 transformers 라이브러리와 호환되도록 합니다.
결론:
따라서, 이러한 변환 작업은 Llama 모델을 Hugging Face의 transformers 라이브러리를 통해 더 쉽고 효과적으로 활용하기 위한 필수적인 단계입니다.





'''
