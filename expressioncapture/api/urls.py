from django.urls import path
from .views import index, processVideo

urlpatterns = [
    path("", index, name="index"),
    path("process-video",processVideo, name="process-video"),
]
