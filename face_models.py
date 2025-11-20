# face_models.py
from facenet_pytorch import InceptionResnetV1, MTCNN
from config import DEVICE
import torch

def load_models():
    print("🔄 Loading models...")
    model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    detector = MTCNN(
        keep_all=True,
        device=DEVICE,
        min_face_size=40,
        thresholds=[0.6, 0.6, 0.6],
        factor=0.6
    )
    print("✅ Models loaded successfully!")
    return model, detector
