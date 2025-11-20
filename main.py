# main.py
import cv2
import time
import numpy as np
from collections import deque
from config import *
from camera import init_camera
from face_models import load_models
from face_utils import get_embedding
from visitor_manager import VisitorManager
from ui_utils import draw_status_bar, upscale_display

def iou(boxA, boxB):
    # Intersection-over-Union for box stability check
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0])) * max(1, (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0])) * max(1, (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def main():
    model, detector = load_models()
    visitors = VisitorManager()
    cap = init_camera()

    # per-face short buffers
    emb_buffer = deque(maxlen=STABLE_DETECTIONS_N)   # collect embeddings for averaging before registering
    box_buffer = None
    last_box = None
    stable_count = 0
    fullscreen = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

    print("🎥 Visitor Management System started.\n")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("❌ Frame grab failed.")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = detector.detect(frame_rgb)

        drew_any = False

        if boxes is not None and len(boxes) > 0:
            idx = int(np.argmax(probs))
            if probs[idx] >= MIN_DET_PROB:
                x1, y1, x2, y2 = map(int, boxes[idx])
                x1 = max(0, x1 - PADDING_PX)
                y1 = max(0, y1 - PADDING_PX)
                x2 = min(frame_bgr.shape[1], x2 + PADDING_PX)
                y2 = min(frame_bgr.shape[0], y2 + PADDING_PX)
                cur_box = (x1, y1, x2, y2)

                # fast visual bounding
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (90, 200, 255), 2)

                # get embedding for this frame
                face_img = frame_bgr[y1:y2, x1:x2]
                if face_img.size == 0:
                    # skip if crop invalid
                    pass
                else:
                    emb = get_embedding(model, face_img)
                    if emb is not None:
                        # 1) Fast per-frame recognition (immediate)
                        rec_status, vid, sim = visitors.handle_recognition(emb)
                        if rec_status == "recognized":
                            # quick known person detection - show and continue (no register)
                            draw_text = f"Known Visitor #{vid}"
                            color = (0, 0, 255)
                            detail = f"Sim: {sim:.2f}" if sim is not None else None
                            # mark recent to avoid bursts
                            visitors.recent_embeddings.append(emb)
                            # draw
                            cv2.putText(frame_bgr, draw_text, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            if detail:
                                cv2.putText(frame_bgr, detail, (x1, y2 + 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                            drew_any = True
                            # reset buffers for new detection cycle
                            emb_buffer.clear()
                            stable_count = 0
                            last_box = cur_box
                        else:
                            # 2) Not recognized: collect embeddings for short stability window
                            if last_box is not None and iou(cur_box, last_box) > 0.6:
                                stable_count += 1
                            else:
                                stable_count = 1
                                emb_buffer.clear()
                            last_box = cur_box
                            emb_buffer.append(emb)

                            # show detecting
                            cv2.putText(frame_bgr, f"DETECTING... {stable_count}/{STABLE_DETECTIONS_N}",
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            drew_any = True

                            # 3) when buffer fills up (stable enough), try averaged registration
                            if stable_count >= STABLE_DETECTIONS_N and len(emb_buffer) >= 1:
                                avg_emb = np.mean(np.stack(list(emb_buffer)), axis=0)
                                reg_status, reg_id, reg_sim = visitors.attempt_register_with_average(avg_emb, face_img)
                                if reg_status == "registered":
                                    cv2.putText(frame_bgr, f"NEW Visitor #{reg_id}", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    # clear buffer after register
                                    emb_buffer.clear()
                                    stable_count = 0
                                    last_box = None
                                elif reg_status == "duplicate":
                                    cv2.putText(frame_bgr, f"KNOWN Visitor #{reg_id}", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    emb_buffer.clear()
                                    stable_count = 0
                                    last_box = None
                                elif reg_status == "cooldown":
                                    remaining = int(GLOBAL_COOLDOWN_SEC - (time.time() - visitors.last_registration_time))
                                    cv2.putText(frame_bgr, f"WAIT {remaining}s", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                                    # keep buffer but do not register now

        else:
            # nothing detected - reset short-term buffers so new face gets fresh start
            emb_buffer.clear()
            stable_count = 0
            last_box = None

        # Top status bar
        draw_status_bar(frame_bgr, f"Visitors: {visitors.visitor_count}", "Press Q:Quit  R:Reset  F:Fullscreen")
        cv2.imshow(WINDOW_NAME, upscale_display(frame_bgr))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            visitors.registered_embeddings = []
            visitors.visitor_count = 0
            visitors.last_registration_time = 0
            visitors.recent_embeddings.clear()
            emb_buffer.clear()
            stable_count = 0
            last_box = None
            print("\n🔄 Visitor data reset!\n")
        elif key == ord('f'):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if not fullscreen:
                cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

    cap.release()
    cv2.destroyAllWindows()
    print(f"📊 Total visitors: {visitors.visitor_count}")

if __name__ == "__main__":
    main()
