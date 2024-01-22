from dotenv import load_dotenv
import os

load_dotenv()

DB_PASS = os.getenv('DB_PASS')

GOOGLE_API_1 = os.getenv('GOOGLE_API_1')
GOOGLE_API_2 = os.getenv('GOOGLE_API_2')
GOOGLE_API_3 = os.getenv('GOOGLE_API_3')
GOOGLE_API_4 = os.getenv('GOOGLE_API_4')

OPEN_API = os.getenv('OPEN_API')

KAKAO_KALRO = os.getenv('KAKAO_KALRO')

PAPAGO_ID = os.getenv('PAPAGO_ID')
PAPAGO_CLIENT = os.getenv('PAPAGO_CLIENT')
