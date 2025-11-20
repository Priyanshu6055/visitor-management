# ui_utils.py
import cv2
import numpy as np
from config import DISPLAY_W, DISPLAY_H, STATUS_BAR_H

def draw_status_bar(frame, left_text, right_text):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    bar_h = min(STATUS_BAR_H, h // 6)
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, left_text, (15, int(bar_h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    text_size, _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, right_text, (w - text_size[0] - 15, int(bar_h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)

def upscale_display(frame):
    h, w = frame.shape[:2]
    scale = min(DISPLAY_W / w, DISPLAY_H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    y_off = (DISPLAY_H - new_h) // 2
    x_off = (DISPLAY_W - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas
