import cv2
import numpy as np
import pyautogui as pg
import time
import os

# -----------------------------
# Configuration
# -----------------------------
IMAGES = ["kz.png", "igen.png", "ok.png"]
RIGHT_CLICK_INDEX = 0      # index of the image that needs right-click
CONFIDENCE = 0.9           # similarity threshold
DELAY_AFTER_CLICK = 1.5    # seconds to wait after each click

# Load and preprocess images
templates = []
for img_file in IMAGES:
    if not os.path.exists(img_file):
        raise FileNotFoundError(f"{img_file} doesn't exist.")

    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:  # RGBA -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    templates.append(img_gray)

# -----------------------------
# Main loop
# -----------------------------
while True:
    # Take screenshot and convert to grayscale
    screenshot = pg.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    for idx, template_gray in enumerate(templates):
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= CONFIDENCE:
            # Calculate center of the found template
            button_x = max_loc[0] + template_gray.shape[1] // 2
            button_y = max_loc[1] + template_gray.shape[0] // 2

            click_type = "right" if idx == RIGHT_CLICK_INDEX else "left"
            print(f"Found {IMAGES[idx]} at ({button_x}, {button_y}) — {click_type}-clicking...")

            # Move mouse explicitly before clicking
            pg.moveTo(button_x, button_y, duration=0.05)

            if idx == RIGHT_CLICK_INDEX:
                pg.rightClick(x=button_x, y=button_y)
            else:
                pg.click(x=button_x, y=button_y)

            time.sleep(DELAY_AFTER_CLICK)
            break  # Only click one image per loop
    else:
        # None of the templates found
        print("No buttons found, retrying...")
        time.sleep(0.5)
