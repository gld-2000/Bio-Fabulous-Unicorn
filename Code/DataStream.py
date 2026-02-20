import UnicornPy
import time
import matplotlib.pyplot as plt
import numpy as np


## Relevant variables
WINDOW_SIZE = 2    #Size of scrolling window to be measured (in seconds) (SHOULD BE EVENLY DIVISIBLE BY 0.04 to ensure that we have an even number of samples at 250Hz)
MEASUREMENT_INTERVAL = 0.5   #Time between measurements (in seconds) (SHOULD BE EVENLY DIVISIBLE BY 0.04 to ensure that we have an even number of samples at 250Hz)
NUM_INTERVALS = int(WINDOW_SIZE // MEASUREMENT_INTERVAL)   #Number of intervals required to hold WINDOW_SIZE seconds of data
HEADSET_SERIAL_NUMBER = 'UN-2024.08.43'   #Serial number of the headset to connect to
TEST_SIGNAL = True   #Should be set to false when collecting real data

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
        #time.sleep(0.2)
        
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
