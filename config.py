# config.py
import torch

# Detection thresholds (tweakable)
COSINE_SIM_THRESHOLD = 0.65     # similarity cutoff for "same person"
GLOBAL_COOLDOWN_SEC = 6         # seconds before next registration allowed (short)
STABLE_DETECTIONS_N = 3         # number of frames to average before registering (small)
PADDING_PX = 30
MIN_DET_PROB = 0.70             # detection probability threshold
MOTION_RESET_PX = 60
DOWNSAMPLE_EVERY = 1            # process every frame

# UI settings
WINDOW_NAME = "Visitor Management System"
DISPLAY_W, DISPLAY_H = 1280, 800
STATUS_BAR_H = 70

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Storage
SAVE_FOLDER = "visitor_images"
