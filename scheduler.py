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
        formatted_date = two_days_ago.strftime('%Y-%m-%d')
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


'''
