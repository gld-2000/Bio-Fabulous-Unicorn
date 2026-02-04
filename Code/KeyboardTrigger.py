import keyboard
import socket

def on_key_event(event):
    if event.event_type == keyboard.KEY_DOWN:
        sendBytes = ''
        if event.name == '1':
            # Send trigger
            sendBytes = b"1"
        elif event.name == '2':
            # Send trigger
            sendBytes = b"2"
        elif event.name == '3':
            # Send trigger
            sendBytes = b"3"
        elif event.name == 'l':
            # Send trigger
            sendBytes = b"4"

        #add your keys/triggers here

        if len(sendBytes)>0:
            print('Key: ' + event.name + ' Sending: ' + str(sendBytes))
            socket.sendto(sendBytes, endPoint)

# Initialize socket
socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
endPoint = ("127.0.0.1", 1000)

keyboard.on_press(on_key_event)
keyboard.wait('esc')