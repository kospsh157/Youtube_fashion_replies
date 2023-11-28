import schedule
import time


def job():
    # 여기에 유튜브 댓글을 수집하는 함수를 호출

    pass


# 매일 정오에 'job' 함수 실행
schedule.every().day.at("12:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(100)
