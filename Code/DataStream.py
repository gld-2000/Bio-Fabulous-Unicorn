from matplotlib.pylab import ceil
import UnicornPy
import time

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
The buffer is sized to hold 1 32-bit data point per sample per channel, for some duration.

args: numChannels - the number of channels being measured
    timeDuration - the duration of measurement to be stored (in seconds)
    samplingRate - the sampling rate of the device (default is 250Hz)

returns: a bytearray of the appropriate size to hold the acquired data
"""
def allocateBuffer(numChannels, timeDuration, samplingRate=250):
    bytesPerSample = 32/8   #Each sample is 32-bits long
    numSamples = ceil(samplingRate * timeDuration)  #Ensures that the number of samples is an integer
    return bytearray(bytesPerSample * numSamples * numChannels)

"""
Make the LEDs on the headset dance (for fun :D)

args: headset - the connected Unicorn object
"""
def danceLEDs(headset):
    digitalOut = 0b00000001
    while True:
        headset.SetDigitalOutputs(digitalOut)
        time.sleep(0.25)
        digitalOut = (digitalOut << 1) % 0xFF


## Relevant variables
MEASUREMENT_DURATION = 2    #Size of scrolling window to be measured (in seconds) (SHOULD BE EVENLY DIVISIBLE BY 0.04 to ensure that we have an even number of samples at 250Hz)
HEADSET_SERIAL_NUMBER = 'UN-2024.08.43'   #Serial number of the headset to connect to

## Safety checks
if (UnicornPy.SamplingRate != 250):
    exit("Unexpected sampling rate. Expected 250Hz, got ", UnicornPy.SamplingRate, ". Terminating program...")


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


##Start doing actual useful stuff

#Configure channels (disable those that are not EEG)
channelConfigs = headset.GetConfiguration()
print("Channels to be recorded:")
for channel in channelConfigs.Channels:
    #Enable all EEG, disable all others
    if channel.Name.startswith("EEG"):
        channel.Enabled = True
    else:
        channel.Enabled = False
    print(channel.Name, channel.Enabled)
headset.SetConfiguration(channelConfigs)

#Allocate the memory required to hold the acquired data
dataBuffer = allocateBuffer(headset.GetNumberOfAcquiredChannels(),
                            timeDuration=2,
                            samplingRate=UnicornPy.SamplingRate)


#testSignal = True
#headset.StartAcquisition(testSignal)
#headset.StopAcquisition()

dataBuffer.clear()
