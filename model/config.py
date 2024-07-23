BATCH_SIZE = 256
SAVE_FREQ = 5
TEST_FREQ = 5
TOTAL_EPOCH = 300

RESUME = False
SAVE_DIR = '/home/ipcteam/congnt/face/face_recognition/model/weights'

MULTI_GPU = True
DATA = 'Vn'
CASIA_DATA_DIR = '/home/ipcteam/congnt/face/face_recognition/data/data_Casia'
UMD_DATA_DIR = '/home/ipcteam/congnt/face/face_recognition/data/data_umd'
VN_DATA_DIR = '/home/ipcteam/congnt/face/face_recognition/data/data_VN'
LFW_DATA_DIR = '/home/ipcteam/congnt/face/face_recognition/data/data_Lfw'

LOSS = 'CosFace'
OPTIMIZE = 'SGD'
MODEL = 'Shuffle'
THRESHOLD = 0.5

DIVIE_LR = [10,30]