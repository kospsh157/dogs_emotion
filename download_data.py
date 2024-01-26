import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 불러오기
load_dotenv()
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

cmd1 = "kaggle datasets download -d danielshanbalico/dog-emotion"
# 윈도우에서는
# 7-ZIP을 설치하고,
# & 'C:\Program Files\7-Zip\7z.exe' e *.zip
cmd2 = "unzip '*.zip'"

os.system(cmd1)
os.system(cmd2)
