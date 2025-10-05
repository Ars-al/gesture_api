from django.urls import path
from .views import PredictGesture, camera_view

urlpatterns = [
    # path("predict/", PredictGesture.as_view(), name="predict-gesture"),
    path("camera/", camera_view, name="camera-view"),
]