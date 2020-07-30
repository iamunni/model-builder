import os
from configparser import ConfigParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


configur = ConfigParser()
configur.read(BASE_DIR+'/config.ini')


MODEL_PATH = os.path.join(BASE_DIR, 'modelfiles')
DATA_PATH = os.path.join(BASE_DIR, 'data')



