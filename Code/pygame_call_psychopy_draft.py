"""
Code to make DataStream.py call and execute the PsychoPy Program
DataStream.py should automatically have normal priority
PsychoPy is hardcoded to have high priority for high-fidelity timing
"""
import subprocess
#open psychopy as child process
psychopy_process = subprocess.Popen(['python', 'C:/path/to/psychopy_script.py']) #change file path here

#terminate psychopy when finished
psychopy_process.terminate()