import json
import base64
import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
from channels.generic.websocket import AsyncWebsocketConsumer

# ---------------- Load Model ---------------- #
model = joblib.load("gesture_api/gesture_recognition_model.pkl")
label_encoder = joblib.load("gesture_api/label_encoder.pkl")

# ---------------- Mediapipe setup ---------------- #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Temporal smoothing
prediction_history = deque(maxlen=5)


# ---------------- Feature Extraction ---------------- #
def extract_features(landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = landmarks[0]
    relative = landmarks - wrist

    fingertips = [4, 8, 12, 16, 20]
    distances = [np.linalg.norm(landmarks[i] - wrist) for i in fingertips]

    vectors = [landmarks[i] - wrist for i in fingertips]
    angles = []
    for i in range(len(vectors) - 1):
        norm1 = np.linalg.norm(vectors[i])
        norm2 = np.linalg.norm(vectors[i + 1])
        if norm1 == 0 or norm2 == 0:
            continue
        cos_angle = np.dot(vectors[i], vectors[i + 1]) / (norm1 * norm2)
        angles.append(np.arccos(np.clip(cos_angle, -1, 1)))

    features = np.concatenate([relative.flatten(), np.array(distances), np.array(angles)])
    return features


# ---------------- Safe Decode ---------------- #
def decode_frame(frame_data):
    try:
        if not frame_data:
            return None

        # strip prefix if exists (data:image/jpeg;base64,...)
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("⚠️ Frame decode failed (empty frame)")
            return None
        return frame
    except Exception as e:
        print("🔥 Decode error:", e)
        return None


# ---------------- WebSocket Consumer ---------------- #
class GestureConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print("🔌 WebSocket connected")
        await self.accept()
        await self.send(json.dumps({"message": "✅ WebSocket connected"}))

    async def disconnect(self, close_code):
        print("❌ WebSocket disconnected", close_code)

    async def receive(self, text_data=None, bytes_data=None):
        try:
            if not text_data:
                return

            data = json.loads(text_data)
            frame_data = data.get("image") or data.get("frame")
            frame = decode_frame(frame_data)

            if frame is None:
                await self.send(json.dumps({"prediction": None, "confidence": 0.0}))
                return

            # Mediapipe detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            prediction = None
            confidence = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    features = extract_features(hand_landmarks.landmark).reshape(1, -1)

                    proba = model.predict_proba(features)[0]
                    pred_index = np.argmax(proba)
                    confidence = float(proba[pred_index])

                    if confidence > 0.7:
                        pred = label_encoder.inverse_transform([pred_index])[0]
                        prediction_history.append(pred)

                        # Temporal smoothing
                        prediction = max(set(prediction_history), key=prediction_history.count)

            await self.send(json.dumps({
                "prediction": prediction,
                "confidence": confidence
            }))

        except Exception as e:
            error_msg = f"🔥 Error: {str(e)}"
            print(error_msg)
            await self.send(json.dumps({"error": error_msg}))
