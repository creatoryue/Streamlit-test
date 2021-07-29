import sounddevice as sd
import numpy as np
import pyaudio
import soundfile as sf
from scipy.io.wavfile import write
import librosa
# from settings import DATA_DIR, saveWavFile, readWavFile
from settings import MAX_INPUT_CHANNELS, DEFAULT_SAMPLE_RATE, CHUNK_SIZE, INPUT_DEVICE, DURATION

duration = DURATION  # seconds
sd.default.samplerate = DEFAULT_SAMPLE_RATE
sd.default.channels = MAX_INPUT_CHANNELS

class Sound():
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.channels = MAX_INPUT_CHANNELS
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.chunk = CHUNK_SIZE
        self.device = INPUT_DEVICE
        self.frames = []
        self.duration = DURATION
        
        self.myrecording = np.array([])
        self.fs = 44100
        print("hello")
        
    def recording(self,fn):
        #Recording using pyaudio
        self.audio = pyaudio.PyAudio()
        stream = self.audio.open(
                        format=self.format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk,
                        input_device_index=self.device)
        self.frames = []
        for i in range(0, int(self.sample_rate / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            self.frames.append(data)
            
        stream.stop_stream()
        stream.close()
        self.audio.terminate()
        
        
        #Recording using sounddevice
        # self.myrecording = sd.rec(int(duration * fs), dtype='int')
        # sd.wait(duration)
        
        # Save as WAV file # **fs needs int dtype
        # filename = saveWavFile(fn)
        
        # write(fn, fs, self.myrecording)  
        
        return self.myrecording
    
    
    def play(self, fn):
        # ssdata, fs = sf.read('data\\{}.wav'.format(fn), dtype='float32')
        self.read(fn) #read the wav file
        sd.play(self.myrecording, self.fs)
        sd.wait()

    def read(self, fn):
        # filename = readWavFile(fn)
        ssdata, fs = librosa.load(fn)
        # ssdata, fs = librosa.load(DATA_DIR+'\\{}'.format(fn))

        # ssdata, fs = sf.read('data\\{}'.format(fn), dtype='float32')
        self.myrecording = ssdata
        self.fs = fs
        return ssdata
