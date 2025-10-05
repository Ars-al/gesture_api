from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import os
from django.conf import settings

# Load model & encoder
# model = joblib.load("gesture_recognition_model.pkl")
# label_encoder = joblib.load("label_encoder.pkl")

MODEL_PATH = os.path.join(  settings.BASE_DIR, "gesture_api", "gesture_recognition_model.pkl")
ENCODER_PATH = os.path.join(settings.BASE_DIR, "gesture_api", "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# Mediapipe init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
prediction_history = deque(maxlen=5)


def extract_features(landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = landmarks[0]
    relative = landmarks - wrist
    fingertips = [4, 8, 12, 16, 20]
    distances = [np.linalg.norm(landmarks[i] - wrist) for i in fingertips]
    vectors = [landmarks[i] - wrist for i in fingertips]
    angles = []
    for i in range(len(vectors) - 1):
        cos_angle = np.dot(vectors[i], vectors[i + 1]) / (
            np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[i + 1])
        )
        angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
    features = np.concatenate([relative.flatten(), np.array(distances), np.array(angles)])
    return features


class PredictGesture(APIView):
    
    def get(self, request):
        return Response({"message": "API is working! Use POST for predictions."})
    
    def post(self, request):
        try:
            file = request.FILES["file"]
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    features = extract_features(hand_landmarks.landmark).reshape(1, -1)
                    proba = model.predict_proba(features)[0]
                    pred_index = np.argmax(proba)
                    confidence = proba[pred_index]

                    if confidence > 0.7:
                        prediction = label_encoder.inverse_transform([pred_index])[0]
                        prediction_history.append(prediction)
                        final_pred = max(
                            set(prediction_history), key=prediction_history.count
                        )
                        return Response(
                            {
                                "prediction": final_pred,
                                "confidence": float(confidence),
                            },
                            status=status.HTTP_200_OK,
                        )

            return Response(
                {"prediction": None, "confidence": 0}, status=status.HTTP_200_OK
            )

        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
            
def camera_view(request):
    return render(request, "recognition/camera.html")