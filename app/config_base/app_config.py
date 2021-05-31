MUXER_OUTPUT_WIDTH = 640
MUXER_OUTPUT_HEIGHT = 480
MUXER_BATCH_TIMEOUT_USEC = 40
TILED_OUTPUT_WIDTH = 640
TILED_OUTPUT_HEIGHT = 480
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 2
OSD_DISPLAY_TEXT = 1
UNTRACKED_OBJECT_ID = 0xffffffffffffffff
MAX_ELEMENTS_IN_DISPLAY_META = 16
DROP_FRAME_INTERVAL = 5
DRAW_TRT_POSE_ARTIFACTS = False
BODY_LABELS = {0: 'nose', 1: 'lEye', 2: 'rEye', 3: 'lEar', 4: 'rEar', 5: 'lShoulder', 6: 'rShoulder',
               7: 'lElbow', 8: 'rElbow', 9: 'lWrist', 10: 'rWrist', 11: 'lHip', 12: 'rHip', 13: 'lKnee', 14: 'rKnee',
               15: 'lAnkle', 16: 'rAnkle', 17: 'neck'}
BODY_IDX = dict([[v, k] for k, v in BODY_LABELS.items()])

POSE_PREDICT_WINDOW = 5
POSE_VEC_DIM = 36
MOTION_DICT = {0: 'nothing', 1: 'fall', 2: 'fight'}
PREDICTION_THRESHOLD = 0
MODEL_PATH = 'models/lstm_model.h5'
DATA_TRANSFORMER_PATH = 'models/min_max_scalar.pickle'
