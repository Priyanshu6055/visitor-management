# visitor_manager.py
import os
import cv2
from datetime import datetime
from collections import deque
from face_utils import cosine_similarity
from config import SAVE_FOLDER, COSINE_SIM_THRESHOLD, GLOBAL_COOLDOWN_SEC

class VisitorManager:
    def __init__(self):
        self.registered_embeddings = []     # list of known face embeddings
        self.visitor_count = 0
        self.last_registration_time = 0
        # short-term memory to prevent burst duplicates (very recent embeddings)
        self.recent_embeddings = deque(maxlen=30)

        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)
            print(f"📁 Created folder: {SAVE_FOLDER}")

    def find_best_match(self, emb):
        """Return (is_dup, best_sim, best_index)."""
        best = -1.0
        best_idx = None
        for i, stored in enumerate(self.registered_embeddings):
            sim = cosine_similarity(emb, stored)
            if sim > best:
                best = sim
                best_idx = i
            if sim >= COSINE_SIM_THRESHOLD:
                return True, sim, i
        return False, (best if best >= 0 else None), best_idx

    def register_visitor(self, embedding, face_img):
        """Register new visitor and save face image."""
        self.registered_embeddings.append(embedding)
        self.visitor_count += 1
        self._save_face(face_img, self.visitor_count)
        self.last_registration_time = datetime.now().timestamp()
        # push into recent memory as well
        self.recent_embeddings.append(embedding)
        print(f"✅ New Visitor #{self.visitor_count} registered.")
        return self.visitor_count

    def handle_recognition(self, emb):
        """
        Quick recognition: check current embedding vs registered embeddings.
        Returns (status, visitor_id_or_None, similarity_or_None)
        status in: "recognized", "not_recognized"
        """
        is_dup, sim, idx = self.find_best_match(emb)
        if is_dup:
            # recognized; return visitor id (1-based)
            return "recognized", idx + 1, sim
        # not recognized quickly
        return "not_recognized", None, (sim if sim is not None else None)

    def attempt_register_with_average(self, avg_emb, face_img):
        """
        Final check with averaged embedding and register if safe.
        Returns (status, visitor_id_or_None, similarity_or_None)
        status in: "registered", "duplicate", "cooldown"
        """
        # check duplicate again with averaged embedding
        is_dup, sim, idx = self.find_best_match(avg_emb)
        if is_dup:
            return "duplicate", idx + 1, sim

        # cooldown check
        now_ts = datetime.now().timestamp()
        if now_ts - self.last_registration_time < GLOBAL_COOLDOWN_SEC:
            return "cooldown", None, None

        # register
        vid = self.register_visitor(avg_emb, face_img)
        return "registered", vid, None

    def _save_face(self, face_img, vid):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visitor_{vid:03d}_{ts}.jpg"
        path = os.path.join(SAVE_FOLDER, filename)
        cv2.imwrite(path, face_img)
        print(f"💾 Saved: {filename}")
