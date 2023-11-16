import os
import subprocess
import kaggle

# os.environ['KAGGLE_USERNAME'] = 'kos157'
# os.environ['KAGGLE_KEY'] = '08379c489e459e165c37944a6d940156'

# Kaggle 데이터셋 다운로드
subprocess.run(['kaggle', 'datasets', 'download', '-d', 'danielshanbalico/dog-emotion'])

# 다운로드한 zip 파일 압축 해제
subprocess.run(['unzip', '*.zip'])