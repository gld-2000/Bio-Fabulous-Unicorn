import UnicornPy
import time
import matplotlib.pyplot as plt
import numpy as np
# Data acquisition imports
from sklearn.cross_decomposition import CCA
#Cursor imports
import pygame
#window & priority imports
import tkinter as tk
import os
import subprocess
import ctypes
import sys

# Get screen resolution
root = tk.Tk()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()

#open psychopy as child process
psychopy_process = subprocess.Popen([
    r"C:\Program Files\PsychoPy\python.exe", #path to pyschopy's python
    r'C:\Users\NTC\PsychoPy Experiments\Flashing Rectangles\stimuli for datastream.py' #path to psychopy experiment
])

# --- Give PsychoPy a moment to start up ---
time.sleep(30)

# Convert PsychoPy norm units to pixels
def norm_to_px(norm_x, norm_y):
    px_x = (norm_x + 1) / 2 * screen_w
    px_y = (1 - norm_y) / 2 * screen_h
    return int(px_x), int(px_y)

# Inner edges of the four rectangles
x_min_norm = -0.85   # right edge of left rectangle
x_max_norm =  0.85   # left edge of right rectangle
y_max_norm =  0.775  # bottom edge of top rectangle
y_min_norm = -0.775  # top edge of bottom rectangle

# Convert to pixel coordinates
px_left,  px_top    = norm_to_px(x_min_norm, y_max_norm)
px_right, px_bottom = norm_to_px(x_max_norm, y_min_norm)

win_x = px_left
win_y = px_top
win_w = px_right  - px_left
win_h = px_bottom - px_top

#win_w and win_h will be fed to the old code below

"""
ONTO OLD DATASTREAM CODE
"""

## Relevant variables
WINDOW_SIZE = 2    #Size of scrolling window to be measured (in seconds) (SHOULD BE EVENLY DIVISIBLE BY 0.04 to ensure that we have an even number of samples at 250Hz)
MEASUREMENT_INTERVAL = 0.5   #Time between measurements (in seconds) (SHOULD BE EVENLY DIVISIBLE BY 0.04 to ensure that we have an even number of samples at 250Hz)
NUM_INTERVALS = int(WINDOW_SIZE // MEASUREMENT_INTERVAL)   #Number of intervals required to hold WINDOW_SIZE seconds of data

#Frequencies as directions
FREQ_UP    = 5.5 #Hz
FREQ_DOWN  = 6 #Hz
FREQ_LEFT  = 6.5 #Hz
FREQ_RIGHT = 7 #Hz

#Neutral / confidence gate parameters
CCA_r_thresh = 0.25        # r is correlation coefficient
CCA_margin_thresh = 0.05   # separation between best (r1) and 2nd-best (r2)
dwell_n = 3            # consecutive updates required to move

#Data analysis variables
numHarmonics = 2    #The number of harmonics to consider for the CCA analysis
freqsOfInterest = [FREQ_RIGHT, FREQ_LEFT, FREQ_DOWN, FREQ_UP]  #SSVEP frequencies of interest

#Cursor movement variables
CURSOR_SPEED   = 50    # how many pixels the cursor moves each step
# brush settings
BRUSH_COLOR  = (30, 30, 200)  # blue
BRUSH_RADIUS = 8              # size of the brush in pixels
# canvas size 
CANVAS_WIDTH  = win_w #pixels
CANVAS_HEIGHT = win_h #pixels

#Other variables
# cursor starts in the middle of the screen
cursor_x = CANVAS_WIDTH  // 2
cursor_y = CANVAS_HEIGHT // 2
# idle threshold trackers
last_winner_idx = None
streak = 0

HEADSET_SERIAL_NUMBER = 'UN-2024.08.43'   #Serial number of the headset to connect to
TEST_SIGNAL = False   #Should be set to false when collecting real data

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
    scores = np.empty(len(freqs), dtype=np.float64)
    for i in range(len(freqs)):
        cca = CCA(n_components=1)
        cca.fit(eeg_window.T, ref[i])      #Must transform since sklearn.cross_decomposition.CCA expects input shape like this:(n_samples, n_features(channels here))
        U, V = cca.transform(eeg_window.T, ref[i])
        corr = np.corrcoef(U.T, V.T)[0,1]
        scores[i] = corr
    return scores

def decision_from_scores(scores, freqs):
    """
    Returns:
      winner_freq (float)
      r1 (float) best score
      r2 (float) second best score
      winner_idx (int)
    """
    idx_sorted = np.argsort(scores)          # ascending
    winner_idx = idx_sorted[-1]
    second_idx = idx_sorted[-2] if len(scores) > 1 else winner_idx
    r1 = float(scores[winner_idx])
    r2 = float(scores[second_idx])
    return freqs[winner_idx], r1, r2, int(winner_idx)

##Cursor Movement Functions
"""
input CCA classification
each number is a detected frequency in Hz  in order over time

which direction are we moving in based on the frequency
"""
FREQ_TO_DIR = {
    float(FREQ_UP): "up",
    float(FREQ_DOWN): "down",
    float(FREQ_LEFT): "left",
    float(FREQ_RIGHT): "right",
}

def get_direction(freq):
    if freq is None:
        return None
    # use rounding to avoid float equality issues
    f = round(float(freq), 2)
    for k, v in FREQ_TO_DIR.items():
        if round(k, 2) == f:
            return v
    return None

"""
Update the pygame drawing window to display the new direction

args: freq - dominant frequency
    direction - String of either 'up', 'down', 'left', or 'right'
    cursor_x - X-position of the cursor
    cursor_y - Y-position of the cursor

returns: cursor_x, cursor_y
"""
def updateDrawingWindow(freq, direction, cursor_x, cursor_y, r1=None, r2=None, streak=0):
    # check if the user closed the window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    
    # remember where the cursor was before moving
    prev_x = cursor_x
    prev_y = cursor_y

   # move only if a direction is selected (otherwise stay still)
    if direction == 'up':    cursor_y -= CURSOR_SPEED
    elif direction == 'down':  cursor_y += CURSOR_SPEED
    elif direction == 'left':  cursor_x -= CURSOR_SPEED
    elif direction == 'right': cursor_x += CURSOR_SPEED
    # else: neutral -> no movement

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
    if direction is None:
        msg = "NEUTRAL"
    else:
        msg = f"{freq} Hz → {direction.upper()}"

    if r1 is not None and r2 is not None:
        msg += f"   (r1={r1:.3f}, r2={r2:.3f}, Δ={r1-r2:.3f}, streak={streak})"

    label = font.render(msg, True, (50, 50, 50))
    screen.blit(label, (10, 10))

    pygame.display.flip()

    return cursor_x, cursor_y


## Main Code

#help(UnicornPy)
#print("Api Version: ", UnicornPy.GetApiVersion())

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
data = np.empty((headset.GetNumberOfAcquiredChannels()*scansPerInterval*NUM_INTERVALS), dtype=np.uint32)

print("Buffer allocated with size", len(dataBuffer), "bytes to record intervals of", MEASUREMENT_INTERVAL, "seconds.")


#Create the reference waves for CCA
window_duration = (scansPerInterval * NUM_INTERVALS) / UnicornPy.SamplingRate
referenceWaves = []
for f in freqsOfInterest:
    referenceWaves.append(generate_reference(freq = f,
                                           fs = UnicornPy.SamplingRate,
                                           window_len = window_duration,
                                           harmonics = numHarmonics))


#Initialize drawing window
pygame.init()
screen = pygame.display.set_mode((CANVAS_WIDTH, CANVAS_HEIGHT))
pygame.display.set_caption("BCI Cursor")
font = pygame.font.SysFont('Arial', 20)

# this surface remembers every brushstroke permanently and never gets wiped
drawing_surface = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
drawing_surface.fill((255, 255, 255))  # start with a white blank canvas

# Set pygame window as topmost
hwnd = pygame.display.get_wm_info()['window']
HWND_TOPMOST = -1
SWP_NOMOVE   = 0x0002
SWP_NOSIZE   = 0x0001
ctypes.windll.user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)

# Attach to foreground thread to bypass Windows' background process restriction
foreground_hwnd = ctypes.windll.user32.GetForegroundWindow()
foreground_tid  = ctypes.windll.user32.GetWindowThreadProcessId(foreground_hwnd, None)
current_tid     = ctypes.windll.kernel32.GetCurrentThreadId()

ctypes.windll.user32.AttachThreadInput(foreground_tid, current_tid, True)
ctypes.windll.user32.BringWindowToTop(hwnd)
ctypes.windll.user32.SetForegroundWindow(hwnd)
ctypes.windll.user32.AttachThreadInput(foreground_tid, current_tid, False)

# Hide taskbar completely
taskbar_hwnd = ctypes.windll.user32.FindWindowW("Shell_TrayWnd", None)
SW_HIDE = 0
SW_SHOW = 5
ctypes.windll.user32.ShowWindow(taskbar_hwnd, SW_HIDE)

#Set up a try/finally block to ensure that we stop acquisition and clear the buffer if the program terminates for any reason (including errors)
try:
    #Begin data acquisition on the headset
    headset.StartAcquisition(TEST_SIGNAL)
    
    #Collect the first window of data from the headset (equal to WINDOW_SIZE)
    for i in range(NUM_INTERVALS):
        headset.GetData(scansPerInterval, dataBuffer, len(dataBuffer))
        #Combine the data and convert it to 32-bit
        startIntex = i*scansPerInterval*headset.GetNumberOfAcquiredChannels()
        endIndex = (i+1)*scansPerInterval*headset.GetNumberOfAcquiredChannels()
        data[startIntex:endIndex] = np.frombuffer(dataBuffer, dtype=np.uint32)

    #Reshape the array to include channels of data
    data = data.reshape((headset.GetNumberOfAcquiredChannels(), -1), order='F')
    

    #Load new data in continuously
    start = 0   #For timing purposes
    start = time.perf_counter_ns()  # start timing once

    while True:
        # 1) CCA scoring on current window
        eeg_float = data.astype(np.float64)
        scores = detect_ssvep(
            eeg_window=eeg_float,
            ref=referenceWaves,
            freqs=freqsOfInterest
        )

        # 2) Winner + confidence metrics
        winner_freq, r1, r2, winner_idx = decision_from_scores(scores, freqsOfInterest)

        confident = (r1 >= CCA_r_thresh) and ((r1 - r2) >= CCA_margin_thresh)

        # 3) Dwell logic
        if confident:
            if last_winner_idx == winner_idx:
                streak += 1
            else:
                last_winner_idx = winner_idx
                streak = 1
        else:
            last_winner_idx = None
            streak = 0

        # 4) Gate: move only if dwell satisfied
        if streak >= dwell_n:
            freq = winner_freq
            direction = get_direction(freq)
        else:
            freq = None
            direction = None

        # 5) Draw/move (neutral => no move)
        cursor_x, cursor_y = updateDrawingWindow(freq, direction, cursor_x, cursor_y, r1, r2, streak)

        # 6) Timing
        end = time.perf_counter_ns()
        print("Time elapsed:", (end - start) * 1e-6, "ms")
        start = time.perf_counter_ns()

        # 7) Acquire next chunk + slide window
        headset.GetData(scansPerInterval, dataBuffer, len(dataBuffer))
        new_block = np.frombuffer(dataBuffer, dtype=np.uint32).reshape(
            (headset.GetNumberOfAcquiredChannels(), -1), order='F'
        )
        data = np.concatenate((data[:, scansPerInterval:], new_block), axis=1)


#Ensure the headset stops acquisition when the program terminates
finally:
    
    headset.StopAcquisition()
    dataBuffer.clear()

    #Plot the last-seen data for debugging purposes
    # for i in range(headset.GetNumberOfAcquiredChannels()):
    #     plt.plot(data[i,:])
    # plt.show()

    
    # Restore taskbar
    ctypes.windll.user32.ShowWindow(taskbar_hwnd, SW_SHOW)

    #terminate psychopy when finished
    psychopy_process.terminate()
    psychopy_process.wait()
    print(f"PsychoPy terminated, return code: {psychopy_process.returncode}")