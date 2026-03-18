"""
Code to set pygame window size and z-score
Size needs to be calculated from dimensions of rectangles, converted from norm to pixels
New z-score overrides PsychoPy fullscreen mode
PsychoPy should also be demoting its own z-score
"""
import tkinter as tk
import pygame
import os

# Get screen resolution
root = tk.Tk()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()

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

print(f"Pygame window: x={win_x}, y={win_y}, w={win_w}, h={win_h}")

# Launch pygame window at that position and size
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{win_x},{win_y}"
pygame.init()
screen = pygame.display.set_mode((win_w, win_h), pygame.NOFRAME)

# Set pygame window as topmost
import ctypes
hwnd = pygame.display.get_wm_info()['window']
HWND_TOPMOST    = -1
SWP_NOMOVE      = 0x0002
SWP_NOSIZE      = 0x0001
ctypes.windll.user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)