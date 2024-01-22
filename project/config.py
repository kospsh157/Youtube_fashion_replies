from dotenv import load_dotenv
import os

load_dotenv()

DB_PASS = os.getenv('DB_PASS')

GOOGLE_API_1 = os.getenv('GOOGLE_API_1')
GOOGLE_API_2 = os.getenv('GOOGLE_API_2')
GOOGLE_API_3 = os.getenv('GOOGLE_API_3')
GOOGLE_API_4 = os.getenv('GOOGLE_API_4')

OPEN_API = os.getenv('OPEN_API')

kakao_kalro = os.getenv('kakao_kalro')

papago_id = os.getenv('papago_id')
papago_secret = os.getenv('papago_secret')
