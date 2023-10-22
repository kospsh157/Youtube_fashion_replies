import os
os.environ['KAGGLE_USERNAME'] = 'kos157'
os.environ['KAGGLE_KEY'] = '08379c489e459e165c37944a6d940156'


cmd1 = "kaggle datasets download -d danielshanbalico/dog-emotion"
# 윈도우에서는  
# 7-ZIP을 설치하고,
# & 'C:\Program Files\7-Zip\7z.exe' e *.zip 
cmd2 = "unzip '*.zip'"

os.system(cmd1)
os.system(cmd2)

