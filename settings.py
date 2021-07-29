import os

# The Root Directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_DIR = os.path.join(ROOT_DIR, 'data')


MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_H5 = os.path.join(MODEL_DIR, 'CNN_for4lungcondition_20210717.h5')

WAVE_OUTPUT_FILE = os.path.join(DATA_DIR, "recorded.wav")




# Audio configurations
INPUT_DEVICE = 0
MAX_INPUT_CHANNELS = 1  # Max input channels
DEFAULT_SAMPLE_RATE = 44100   # Default sample rate of microphone or recording device
DURATION = 3   # 3 seconds
CHUNK_SIZE = 1024
