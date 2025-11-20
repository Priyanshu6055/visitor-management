from facenet_pytorch import MTCNN
import torch, cv2
print("✅ Torch OK, CUDA:", torch.cuda.is_available())
print("✅ OpenCV version:", cv2.__version__)
MTCNN()
print("✅ facenet_pytorch MTCNN initialized successfully!")
