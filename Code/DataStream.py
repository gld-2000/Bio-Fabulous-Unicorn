import UnicornPy
import time
import matplotlib.pyplot as plt
import numpy as np
# Data acquisition imports
from sklearn.cross_decomposition import CCA
#Cursor imports
import pygame


## Relevant variables
WINDOW_SIZE = 2    #Size of scrolling window to be measured (in seconds) (SHOULD BE EVENLY DIVISIBLE BY 0.04 to ensure that we have an even number of samples at 250Hz)
MEASUREMENT_INTERVAL = 0.5   #Time between measurements (in seconds) (SHOULD BE EVENLY DIVISIBLE BY 0.04 to ensure that we have an even number of samples at 250Hz)
NUM_INTERVALS = int(WINDOW_SIZE // MEASUREMENT_INTERVAL)   #Number of intervals required to hold WINDOW_SIZE seconds of data
HEADSET_SERIAL_NUMBER = 'UN-2024.08.43'   #Serial number of the headset to connect to
TEST_SIGNAL = False   #Should be set to false when collecting real data
#Frequencies as directions
FREQ_UP    = 15 #Hz
FREQ_DOWN  = 10 #Hz
FREQ_LEFT  = 7.5 #Hz
FREQ_RIGHT = 6 #Hz


#Data analysis variables
numHarmonics = 2    #The number of harmonics to consider for the CCA analysis
freqsOfInterest = [FREQ_RIGHT, FREQ_LEFT, FREQ_DOWN, FREQ_UP]  #SSVEP frequencies of interest

#Cursor movement variables
CURSOR_SPEED   = 50    # how many pixels the cursor moves each step
# brush settings
BRUSH_COLOR  = (30, 30, 200)  # blue
BRUSH_RADIUS = 8              # size of the brush in pixels
# canvas size 
CANVAS_WIDTH  = 800 #pixels
CANVAS_HEIGHT = 600 #pixels

# cursor starts in the middle of the screen
cursor_x = CANVAS_WIDTH  // 2
cursor_y = CANVAS_HEIGHT // 2


## Safety checks
if (UnicornPy.SamplingRate != 250):
    exit("Unexpected sampling rate. Expected 250Hz, got ", UnicornPy.SamplingRate, ". Terminating program...")



"""
Attempt to connect to the headset 10 times before giving up.
Bluetooth availability must be checked before calling this function.

args: deviceName - the serial number of the device to connect to (we use 'UN-2024.08.43')
"""
def attemptConnection(deviceName):
    loopNum = 1
    headset = None
    while (headset == None and loopNum <= 10):
        availableDevices = UnicornPy.GetAvailableDevices(True)
        print("Recognized Devices: ", availableDevices)
        for device in availableDevices:
            if device == deviceName:
                try:
                    print("Attempting to connect to ", deviceName, "...", loopNum)
                    headset = UnicornPy.Unicorn(device)
                except Exception as e:
                    print(f"Failed to connect to device {device}: {e}")
        loopNum += 1
    if headset != None:
        print("Device connected successfully!")
    return headset

"""
Allocate a buffer to hold the acquired data.
The buffer is sized to hold 1 32-bit data point per sample per channel, for a given duration.

args: numChannels - the number of channels being measured
    numSamplesPerInterval - the number of samples needed for a single time interval

returns: a bytearray of the appropriate size to hold numIntervals of acquired data
"""
def allocateBuffer(numChannels, numSamplesPerInterval):
    bytesPerSample = 32/8   #Each sample is 32-bits long
    bufferSizePerInterval = int(np.ceil(bytesPerSample * numSamplesPerInterval * numChannels))
    return bytearray(bufferSizePerInterval)

"""
Allocate an array of buffers to hold the acquired data.
Each buffer will be sized to contain one interval, and there will be numIntervals buffers.
The buffer is sized to hold 1 32-bit data point per sample per channel, for some duration.

args: numChannels - the number of channels being measured
    numSamplesPerInterval - the number of samples to be stored in each interval
    numIntervals - number of windows required to hold WINDOW_SIZE seconds of data

returns: a list of numIntervals bytearrays, each the appropriate size to hold numSamplesPerInterval of acquired data
"""
def allocateBufferList(numChannels, numSamplesPerInterval, numIntervals):
    bytesPerSample = 32/8   #Each sample is 32-bits long
    bufferSizePerInterval = int(np.ceil(bytesPerSample * numSamplesPerInterval * numChannels))
    return [bytearray(bufferSizePerInterval) for _ in range(numIntervals)]

"""
Make the LEDs on the headset dance (for fun :D)

args: headset - the connected Unicorn object
"""
def danceLEDs(headset):
    digitalOut = 0b00000001
    headset.SetDigitalOutputs(digitalOut)
    time.sleep(0.15)
    while not(digitalOut & 0b10000000):
        digitalOut = (digitalOut << 1)
        headset.SetDigitalOutputs(digitalOut)
        time.sleep(0.15)
    while not(digitalOut & 0b00000001):
        digitalOut = (digitalOut >> 1)
        headset.SetDigitalOutputs(digitalOut)
        time.sleep(0.15)
    for _ in range(4):
        headset.SetDigitalOutputs(0b10011001)
        time.sleep(0.25)
        headset.SetDigitalOutputs(0b01100110)
        time.sleep(0.25)
    headset.SetDigitalOutputs(0b00000000)


##Data Analysis Functions
"""Generates the reference sine waves for the CCA algorithm"""
def generate_reference(freq, fs, window_len, harmonics=2):
    t = np.arange(0, window_len, 1/fs) #time_vector from 0 to window_length with step size = 1 / sampling_rate
    ref = []
    for h in range(1, harmonics+1):
        ref.append(np.sin(2*np.pi*freq*h*t))
        ref.append(np.cos(2*np.pi*freq*h*t))
    return np.array(ref).T  #shape: (samples, 2*harmonics)
    #transpose

"""
Detect frequency using CCA

args: eeg_window - EEG data with shape (numChannels, numSamples)
    ref - Reference waveforms for CCA
    freqs - The frequencies of interest for classification

returns: A list of the scores for the given frequencies
"""
def detect_ssvep(eeg_window, ref, freqs):
    scores = [len(freqs)]
    for i in range(len(freqs)):
        cca = CCA(n_components=1)
        cca.fit(eeg_window.T, ref)      #Must transform since sklearn.cross_decomposition.CCA expects input shape like this:(n_samples, n_features(channels here))
        U, V = cca.transform(eeg_window.T, ref)
        corr = np.corrcoef(U.T, V.T)[0,1]
        scores[i] = corr
    return scores

def classifyFromScores(scores, freqs):
    return freqs[np.argmax(scores)]

##Cursor Movement Functions
"""
input CCA classification
each number is a detected frequency in Hz  in order over time

which direction are we moving in based on the frequency
"""
def get_direction(freq):
    if freq == FREQ_UP:    return 'up'
    if freq == FREQ_DOWN:  return 'down'
    if freq == FREQ_LEFT:  return 'left'
    if freq == FREQ_RIGHT: return 'right'
    return None 

"""
Update the pygame drawing window to display the new direction

args: freq - dominant frequency
    direction - String of either 'up', 'down', 'left', or 'right'
"""
def updateDrawingWindow(freq, direction):
    # check if the user closed the window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    
    # remember where the cursor was before moving
    prev_x = cursor_x
    prev_y = cursor_y

    # move the cursor in that direction
    if direction == 'up':    cursor_y -= CURSOR_SPEED
    if direction == 'down':  cursor_y += CURSOR_SPEED
    if direction == 'left':  cursor_x -= CURSOR_SPEED
    if direction == 'right': cursor_x += CURSOR_SPEED

    # make sure the cursor doesn't go outside the window
    cursor_x = max(0, min(CANVAS_WIDTH,  cursor_x))
    cursor_y = max(0, min(CANVAS_HEIGHT, cursor_y))

    # draw a line from old position to new position to create a smooth brushstroke
    pygame.draw.line(drawing_surface, BRUSH_COLOR, (prev_x, prev_y), (cursor_x, cursor_y), BRUSH_RADIUS * 2)
    # draw a circle at the new position to give the stroke smooth rounded ends
    pygame.draw.circle(drawing_surface, BRUSH_COLOR, (cursor_x, cursor_y), BRUSH_RADIUS)

    # drawnig
    # paste the permanent drawing surface first (contains all brushstrokes so far)
    screen.blit(drawing_surface, (0, 0))

    # draw the cursor as red circle
    pygame.draw.circle(screen, (220, 50, 50), (cursor_x, cursor_y), 10, 2)
    pygame.draw.line(screen, (220, 50, 50), (cursor_x - 15, cursor_y), (cursor_x + 15, cursor_y), 1)
    pygame.draw.line(screen, (220, 50, 50), (cursor_x, cursor_y - 15), (cursor_x, cursor_y + 15), 1)

    # show text on screen: what frequency was detected and which direction
    label = font.render(f"Freq: {freq} Hz  →  Direction: {direction}", True, (50, 50, 50))
    screen.blit(label, (10, 10))

    pygame.display.flip()  


## Main Code

#help(UnicornPy)
print("Api Version: ", UnicornPy.GetApiVersion())

#Check bluetooth adapter
btAdapter = UnicornPy.GetBluetoothAdapterInfo()
print("Bluetooth Adapter: ", btAdapter.Name)
#If it is the correct bluetooth adapter and it has no problems, try to connect to the device
if btAdapter.IsRecommendedDevice and (not btAdapter.HasProblem):
    headset = attemptConnection(HEADSET_SERIAL_NUMBER)
    #If the device did not successfully connect, exit the program
    if headset == None:
        exit("Failed to connect to device. Terminating program...")
#If the bluetooth adapter is not connected, exit the program
else:
    exit("Bluetooth adapter is not suitable for use with the device. Terminating program...")

#danceLEDs(headset)

##Start doing actual useful stuff

#Configure channels (disable those that are not EEG)
channelConfigs = headset.GetConfiguration()
#print("Channels to be recorded:")
for channel in channelConfigs.Channels:
    #Enable all EEG, disable all others
    if channel.Name.startswith("EEG"):
        channel.Enabled = True
    else:
        channel.Enabled = False
    #print(channel.Name, channel.Enabled)
headset.SetConfiguration(channelConfigs)

print("Number of active channels: ", headset.GetNumberOfAcquiredChannels())


#Calculate the number of scans to be acquired per interval, and ensure that is is an integer (rounding up if necessary)
scansPerInterval = int(np.ceil(MEASUREMENT_INTERVAL * UnicornPy.SamplingRate))

#Allocate the memory required to hold the acquired data
dataBuffer = allocateBuffer(headset.GetNumberOfAcquiredChannels(),
                            numSamplesPerInterval = scansPerInterval)
data = np.zeros((headset.GetNumberOfAcquiredChannels()*scansPerInterval*NUM_INTERVALS), dtype=np.uint32)

print("Buffer allocated with size", len(dataBuffer), "bytes to record intervals of", MEASUREMENT_INTERVAL, "seconds.")


#Create the reference waves for CCA
window_duration = (scansPerInterval * NUM_INTERVALS) / UnicornPy.SamplingRate
referenceWaves = []
i = 0
for f in freqsOfInterest:
    referenceWaves[i] = generate_reference(freq = f,
                                           fs = UnicornPy.SamplingRate,
                                           window_len = window_duration,
                                           harmonics = numHarmonics)
    i += 1


#Initialize drawing window
pygame.init()
screen = pygame.display.set_mode((CANVAS_WIDTH, CANVAS_HEIGHT))
pygame.display.set_caption("BCI Cursor")
font = pygame.font.SysFont('Arial', 20)

# this surface remembers every brushstroke permanently and never gets wiped
drawing_surface = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
drawing_surface.fill((255, 255, 255))  # start with a white blank canvas


#Set up a try/finally block to ensure that we stop acquisition and clear the buffer if the program terminates for any reason (including errors)
try:
    #Begin data acquisition on the headset
    headset.StartAcquisition(TEST_SIGNAL)
    
    #Collect the first window of data from the headset (equal to WINDOW_SIZE)
    for i in range(NUM_INTERVALS):
        headset.GetData(scansPerInterval, dataBuffer, len(dataBuffer))
        #Combine the data and convert it to 32-bit
        startInted = i*scansPerInterval*headset.GetNumberOfAcquiredChannels()
        endIndex = (i+1)*scansPerInterval*headset.GetNumberOfAcquiredChannels()
        data[startInted:endIndex] = np.frombuffer(dataBuffer, dtype=np.uint32)

    #Reshape the array to include 8 channels of data
    data = data.reshape((headset.GetNumberOfAcquiredChannels(), -1), order='F')


    #Load new data in continuously
    while True: 
        #Placeholder for the data processing functions
        scores = detect_ssvep(eeg_window = data,
                              ref = referenceWaves,
                              freqs = freqsOfInterest)
        
        #Identify the dominant frequency
        identifiedFreq = classifyFromScores(scores, freqsOfInterest)

        #Update the drawing window based on the classification
        updateDrawingWindow(identifiedFreq, get_direction(identifiedFreq))

        #Collect the new interval of data from the headset
        headset.GetData(scansPerInterval, dataBuffer, len(dataBuffer))
        
        #Delete the old data and add the new data to the end of the array
        #Manipulate raw data into the correct format and add it to the end of the array (same manipulation method as above, just in one line)
        data = np.concatenate((data[:, scansPerInterval:scansPerInterval*NUM_INTERVALS], np.frombuffer(dataBuffer, dtype=np.uint32).reshape((headset.GetNumberOfAcquiredChannels(), -1), order='F')), axis=1)

        #Print the real-time data for debugging purposes
        # plt.cla()
        # plt.plot(data[0,:])
        # plt.pause(0.05)


#Ensure the headset stops acquisition when the program terminates
finally:
    headset.StopAcquisition()
    dataBuffer.clear()

    #Plot the last-seen data for debugging purposes
    # for i in range(headset.GetNumberOfAcquiredChannels()):
    #     plt.plot(data[i,:])
    # plt.show()
