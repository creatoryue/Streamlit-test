import streamlit as st
import numpy as np
import os
import librosa
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

from settings import DATA_DIR, saveWavFile, readWavFile
from src import loadModel, sound

classes = ['COPD-Mild', 'COPD-Severe', 'Interstitial Lung Disease', 'Normal']


# In[]
'''
# Classification for lung condition.
'''

cnn = loadModel.CNN
s = sound.Sound()

'''##Load model'''
# if st.button("Load CNN model"):
load_state = st.text("loading...")
cnn.model = cnn.loadTrainingModel(self=cnn)
load_state = st.text("Successful...")
    

'''## 2.Record your own voice for 32 sec.'''
# st.header('Record your own voice.')
filename_user = st.text_input('Enter a filename: ')

stt_button = Button(label="Speak", width=100)
stt_button.js_on_event("button_click", CustomJS(code="""
    <input type="file" accept="audio/*" capture>
    
    """))

result = streamlit_bokeh_events(
    stt_button,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0)

#state_recordButton = st.button("Click to Record")
#if state_recordButton:
#    if filename_user == "":
#        st.warning("Choose a filename.")
#    else:
#        #record the sound data and create WAV file
#        fn = saveWavFile(filename_user)
#        recorddata = s.recording(fn)
#        st.text('Record completed!')



'''## 3.Show and Play your own voice.'''
filenames = os.listdir(DATA_DIR)
selected_filename = st.selectbox('Select a file', filenames) # selected_filename contains "XXX.wav" but not whole, correct filepath


# state_checkbox = st.checkbox('Show dataframe')
# if state_checkbox:
#     chart_data = s.read(selected_filename)
#     st.line_chart(chart_data) 
    
state_playButton = st.button("Click to Play")
if state_playButton:
    # st.text(selected_filename)
    # s.read(selected_filename)
    fn = readWavFile(selected_filename)
    s.play(fn)
        
# '''## 5.Show the recording data'''

    
    
'''## 4.Predict'''
state_predictButton = st.button("Predition")
if state_predictButton:
    
    # Read the sound file 

    fn = readWavFile(selected_filename)

    s_pred = sound.Sound()
    s_pred.read(fn)
    
    sample_data = s_pred.myrecording
    
    st.text('Read the sound file {} completed'.format(selected_filename))
    
    data_pred = cnn.samplePred(cnn, sample_data)
    data_pred_class = np.argmax(np.round(data_pred), axis=1)
    
    # s2 is the number of the classes
    s1 = classes[data_pred_class[0]]
    # s1 is the percentage of the predicted class
    s2 = np.round(float(data_pred[0,data_pred_class])*100, 4)
    st.text("Predict class: {} for {}%".format(s1, s2))

    
    
     



# In[]

# if st.button("Play the recording"):
    # sd.playrec(myrecording)
    # plt.plot(myrecording)

#Selec a voice file
# st.header('Classification for lung condition.')
# folder_path='.\data'
# filenames = os.listdir(folder_path)
# selected_filename = st.selectbox('Select a file', filenames)




#        record_state.text(f"Saving sample as {filename}.mp3")
#
#        path_myrecording = f"./samples/{filename}.mp3"
#
#        st.audio(read_audio(path_myrecording))

#        fig = create_spectrogram(path_myrecording)
#        st.pyplot(fig)
        

#import numpy as np
#import pandas as pd
##import time
#import pyaudio
#
#import streamlit as st
#import speech_recognition as sr
#
#def takecomand():
#    r=sr.Recognizer()
#    with sr.Microphone() as source:
#        st.write("answer please....")
#        audio=r.listen(source)
#        try:
#            text=r.recognize_google(audio)
#            st.write("You  said :",text)
#        except:
#            st.write("Please say again ..")
#        return text
#
#if st.button("Click me"):
#    takecomand()


#FORMAT = pyaudio.paInt16 
#CHANNELS = 1
#RATE = 44100
#INPUT_BLOCK_TIME = 0.05
#INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)
#
#
#class LungTester(object):
#    def __init__(self):
#        self.pa = pyaudio.PyAudio()
#        self.stream = self.open_mic_stream()
#        self.list = []
#        self.numpydata = np.array([])
#
#    def stop(self):
#        self.stream.close()
#    
#    def find_input_device(self):
#        device_index = None            
#        for i in range( self.pa.get_device_count() ):     
#            devinfo = self.pa.get_device_info_by_index(i)   
#            print( "Device %d: %s"%(i,devinfo["name"]) )
#
#            for keyword in ["mic","input"]:
#                if keyword in devinfo["name"].lower():
#                    print( "Found an input: device %d - %s"%(i,devinfo["name"]) )
#                    device_index = i
#                    return device_index
#
#        if device_index == None:
#            print( "No preferred input found; using default input device." )
#
#        return device_index
#    
#    def open_mic_stream( self ):
#        device_index = self.find_input_device()
#
#        stream = self.pa.open(   format = FORMAT,
#                                 channels = CHANNELS,
#                                 rate = RATE,
#                                 input = True,
#                                 input_device_index = device_index,
#                                 frames_per_buffer = INPUT_FRAMES_PER_BLOCK)
#        
#        return stream
#    
#    def listen(self):
#        try:
#            block = self.stream.read(INPUT_FRAMES_PER_BLOCK)
#        except IOError as e:
#            # dammit. 
#            self.errorcount += 1
#            print( "(%d) Error recording: %s"%(self.errorcount,e) )
#            self.noisycount = 1
#            return
#        
#        #Do something
#        self.list.append(block)
#        self.numpydata = np.frombuffer(np.array(self.list), dtype=np.int16)
#        
#        return self.numpydata
#'''
## To Test your Lung condition
##Let's START!!
#'''
#
#left_column, right_column = st.beta_columns(2)
#pressed = left_column.button('Press me?')
#if pressed:
#    right_column.write("GOGO")
#    


##left_column, right_column = st.beta_columns(2)
##pressed = left_column.button('Press me?')
##if pressed:
##    right_column.write("GOGO")
##    'Starting a long computation...'
##    # Add a placeholder
##    latest_iteration = st.empty()
##    bar = st.progress(0)
##    
##    for i in range(100):
##      # Update the progress bar with each iteration.
##      latest_iteration.text(f'Iteration {i+1}')
##      bar.progress(i + 1)
##      time.sleep(0.01)
##    
##    '...and now we\'re done!'
#
#
##left_column, right_column, mid_column = st.beta_columns(3)
##pressed = left_column.button('Press me?')
##if pressed:
##    right_column.write("Woohoo!")
##    pressed = mid_column.button('HAHA')
##
##expander = st.beta_expander("FAQ")
##expander.write("Here you could put in some really, really long explanations...")
#
#df = pd.DataFrame({
#  'first column': [1, 2, 3, 4],
#  'second column': [10, 20, 30, 40]
#})
#
#option = st.sidebar.selectbox(
#    'Which number do you like best?',
#     df['first column'])
#
#'You selected:', option
#
#
#
#df = pd.DataFrame({
#  'first column': [1, 2, 3, 4],
#  'second column': [10, 20, 30, 40]
#})
#
#df
#
#option = st.selectbox(
#    'Which number do you like best?',
#     df['second column'])
#
#'You selected: ', option
#
#if st.checkbox('Show dataframe'):
#    chart_data = pd.DataFrame(
#       np.random.randn(20, 3),
#       columns=['a', 'b', 'c'])
#
#    chart_data
#
#map_data = pd.DataFrame(
#    np.random.randn(100, 2) / [3, 5] + [23.5, 121],
#    columns=['lat', 'lon']) #Lat緯度 Lon經度
#
#st.map(map_data)
#
#"""
## Line Chart for randon
#$ Line Chart for randon
#This is a ezample for line chart:
#"""
#chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])
#
#st.line_chart(chart_data)
#
#
#"""
## My first app
## HHHHEEEE
#Here's our first attempt at using data to create a table:
#"""
#
#df = pd.DataFrame({
#  'first column': [1, 2, 3, 4],
#  'second column': [10, 20, 30, 40]
#})
#
#df
