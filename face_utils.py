# face_utils.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from config import DEVICE

def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)

def get_embedding(model, face_img_bgr):
    try:
        face_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb).resize((160, 160))
        face_array = np.asarray(face_pil).astype(np.float32)
        face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)
        face_tensor = (face_tensor - 127.5) / 128.0
        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(face_tensor)
            emb = l2_normalize(emb)
        return emb.cpu().numpy()[0]
    except Exception as e:
        print(f"❌ Error computing embedding: {e}")
        return None

def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b) / denom)
