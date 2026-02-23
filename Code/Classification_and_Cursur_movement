import mne
import numpy as np
from sklearn.cross_decomposition import CCA
from collections import deque
import time

#parameters
fs = 250                #Sampling rate
window_length = 2.0     #seconds
step_size = 0.5         #seconds
channels = ['Oz','PO8','PO7','Pz'] 
freqs = [6, 7.5, 10, 15]  #SSVEP frequencies
harmonics = 2  #number of harmonics to include #maybe we will change this because 6 and 7.5 have the 3rd harmonic in the range of EEG frequencies and that might cause some confusion for the classifier
#bias
smoothing_num = 4  #number of windows to ensure smoothness in the cursor movement 
#Keep the last number (example: 4) of predictions and then send them together to control the cursor movement.
#This is to ensure that the cursor movement is smooth and not jittery due to noise in the EEG signal.
#The majority vote of the last 5 predictions will determine the final direction of the cursor movement.

#Map frequency to direction
freq2dir = {15: "UP", 6: "RIGHT", 10: "DOWN", 7.5: "LEFT"}

def generate_reference(freq, fs, window_len, harmonics=2):
    t = np.arange(0, window_len, 1/fs) #time_vector from 0 to window_length with step size = 1 / sampling_rate
    ref = []
    for h in range(1, harmonics+1):
        ref.append(np.sin(2*np.pi*freq*h*t))
        ref.append(np.cos(2*np.pi*freq*h*t))
    return np.array(ref).T  #shape: (samples, 2*harmonics)
    #transpose 


def detect_ssvep(eeg_window, fs):
    """Detect frequency using CCA"""
    scores = []
    for f in freqs:
        window_duration = eeg_window.shape[1] / fs
        ref = generate_reference(f, fs, window_duration, harmonics)
        cca = CCA(n_components=1)
        cca.fit(eeg_window.T, ref)
        U, V = cca.transform(eeg_window.T, ref)
        corr = np.corrcoef(U.T, V.T)[0,1]
        scores.append(corr)
    return freqs[np.argmax(scores)]
#sklearn.cross_decomposition.CCA expects input shape like this:(n_samples, n_features(channels here))

#(Offline test)
raw = mne.io.read_raw_bdf("F:\\lea\\back to study\\2nd semester\\Biofablab\\Git codes\\Recordings\\Flicker recordings\\Wet_OSCAR-\\UnicornRecorder_17_02_2026_17_04_15.bdf", preload=True)

raw.pick('eeg')

#Rename hardware channels to anatomical names
mapping = {
    'EEG 1': 'Fz',
    'EEG 2': 'C3',
    'EEG 3': 'Cz',
    'EEG 4': 'C4',
    'EEG 5': 'Pz',
    'EEG 6': 'PO7',
    'EEG 7': 'Oz',
    'EEG 8': 'PO8'
}

raw.rename_channels(mapping)

#pick only occipital/parietal channels for SSVEP
raw.pick(['Pz','PO7','Oz','PO8'])

#raw.filter(5, 40)       
#raw.notch_filter(50)   


data = raw.get_data()     #shape: (n_channels, n_samples)
n_samples = data.shape[1]
window_samples = int(window_length * fs)
step_samples = int(step_size * fs)
frequency_list = []

#real-time sliding window simulation 
pred_buffer = deque(maxlen=smoothing_num)

print("Starting sliding window SSVEP detection simulation...")
count = 0
for start in range(0, n_samples - window_samples, step_samples):
    window = data[:, start:start+window_samples]
    freq_detected = detect_ssvep(window, fs)
    frequency_list.append(freq_detected)   #store detected frequencies for analysis
    pred_buffer.append(freq_detected)
    count += 1
    #smoothing
    direction = max(set(pred_buffer), key=pred_buffer.count)
    print(f"Detected freq: {freq_detected:.2f} Hz -> Direction: {freq2dir[direction]}")
    
    #Simulate real-time update every step_size
    time.sleep(step_size)

print(f"Total windows processed: {count}")
print(frequency_list)

import pygame
import numpy as np
import time

# frequencies as directions
FREQ_UP    = 15 #Hz
FREQ_DOWN  = 10 #Hz
FREQ_LEFT  = 7.5 #Hz
FREQ_RIGHT = 6 #Hz

CURSOR_SPEED   = 50    # how many pixels the cursor moves each step

# brush settings
BRUSH_COLOR  = (30, 30, 200)  # blue
BRUSH_RADIUS = 8              # size of the brush in pixels

# canvas size 
CANVAS_WIDTH  = 800 #pixels
CANVAS_HEIGHT = 600 #pixels

#input list from CCA classifier
# each number is a detected frequency in Hz  in order over time

# which direction are we moving in based on the frequency
def get_direction(freq):
    if freq == FREQ_UP:    return 'up'
    if freq == FREQ_DOWN:  return 'down'
    if freq == FREQ_LEFT:  return 'left'
    if freq == FREQ_RIGHT: return 'right'
    return None 

pygame.init()
screen = pygame.display.set_mode((CANVAS_WIDTH, CANVAS_HEIGHT))
pygame.display.set_caption("BCI Cursor")
font = pygame.font.SysFont('Arial', 20)

# this surface remembers every brushstroke permanently and never gets wiped
drawing_surface = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
drawing_surface.fill((255, 255, 255))  # start with a white blank canvas

# cursor starts in the middle of the screen
cursor_x = CANVAS_WIDTH  // 2
cursor_y = CANVAS_HEIGHT // 2

# go through each frequency in the list one by one
for freq in frequency_list:

    # check if the user closed the window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    # figure out which direction this frequency means
    direction = get_direction(freq)

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
    time.sleep(0.3)        # wait 0.3 seconds before processing the next frequency -> we can change this to contorl how fast the cursor moves

# keep the window open after all frequencies are processed
print("Done! Close the window to exit.")
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()