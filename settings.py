import os
from configparser import ConfigParser

configur = ConfigParser()
configur.read('config.ini')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'modelfiles')
DATA_PATH = os.path.join(BASE_DIR, 'data')



