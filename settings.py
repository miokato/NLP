import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = 'data/rap'
RAW_DATA = os.path.join(ROOT, DATA_DIR, 'raw_yomi.txt')
CHAR_ID_PKL = os.path.join(ROOT, DATA_DIR, 'char_id.pkl')
ID_CHAR_PKL = os.path.join(ROOT, DATA_DIR, 'id_char.pkl')
INPUTS = os.path.join(ROOT, DATA_DIR, 'inputs.txt')
OUTPUTS = os.path.join(ROOT, DATA_DIR, 'outputs.txt')
XY_VECTORS_NPZ = os.path.join(ROOT, DATA_DIR, 'xy_vectors.npz')
