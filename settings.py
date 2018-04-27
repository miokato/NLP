import os
import datetime

# PROJECT = os.getenv('PROJECT', None)
PROJECT = 'nagoya'
TIMESTAMP = datetime.datetime.now().strftime('%y%m%d-%H%M')

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT, 'data', PROJECT)
MODEL_DIR = os.path.join(ROOT, 'models')

KRS_MODEL_FILE = '{name}-{time}.h5'.format(name=PROJECT, time=TIMESTAMP)
TXT_FILE = '{}.txt'.format(PROJECT)

RAW_DATA = os.path.join(ROOT, DATA_DIR, 'raw.txt')
CLEANED = os.path.join(ROOT, DATA_DIR, 'cleaned.txt')
HIRAGANA = os.path.join(ROOT, DATA_DIR, 'raw_yomi.txt')
CHAR_ID_PKL = os.path.join(ROOT, DATA_DIR, 'char_id.pkl')
ID_CHAR_PKL = os.path.join(ROOT, DATA_DIR, 'id_char.pkl')
INPUTS = os.path.join(ROOT, DATA_DIR, 'inputs.txt')
OUTPUTS = os.path.join(ROOT, DATA_DIR, 'outputs.txt')
XY_VECTORS_NPZ = os.path.join(ROOT, DATA_DIR, 'xy_vectors.npz')

STOP_WORD = '#'
MODEL = os.path.join(ROOT, MODEL_DIR, KRS_MODEL_FILE)