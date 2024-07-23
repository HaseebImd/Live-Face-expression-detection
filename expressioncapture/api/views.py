import os
import cv2
import torch
import numpy as np
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from torchvision import transforms
from trainedmodel.model import (
    CNN,
)  # Adjust this import statement to match your CNN model's location and name
from django.conf import settings
import matplotlib

matplotlib.use("Agg")  # Use 'Agg' backend for non-GUI environments
import tempfile
import base64
from django.shortcuts import render
import matplotlib.pyplot as plt

# Load the trained model
model_path = os.path.join(settings.BASE_DIR, "trainedmodel/model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define emotion labels
emotion_labels = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

# Define image transformations
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ]
)


def index(request):
    return render(request, "home.html")


@csrf_exempt
def processVideo(request):
    if request.method == "POST" and request.FILES.get("video"):
        video_file = request.FILES["video"]

        # Save the uploaded video to a temporary file
        temp_video_path = tempfile.NamedTemporaryFile(delete=False).name
        with open(temp_video_path, "wb+") as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        # Open the video file
        cap = cv2.VideoCapture(temp_video_path)

        # Initialize variables to track emotions
        emotion_counts = {label: 0 for label in emotion_labels.values()}
        last_frame = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 5 != 0:  # Process every 5th frame
                continue

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Perform face detection
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.3, minNeighbors=5
            )

            face_regions = []
            for x, y, w, h in faces:
                # Extract the face region
                face = gray_frame[y : y + h, x : x + w]

                # Apply transformations
                transformed_face = transform(face).unsqueeze(0).to(device)
                face_regions.append(transformed_face)

            if face_regions:
                face_regions = torch.cat(face_regions, dim=0)

                # Perform emotion prediction in batch
                with torch.no_grad():
                    emotion_predictions = model(face_regions)
                    predicted_emotions = (
                        torch.argmax(emotion_predictions, dim=1).cpu().numpy()
                    )

                for i, (x, y, w, h) in enumerate(faces):
                    predicted_emotion_index = predicted_emotions[i]
                    predicted_emotion = emotion_labels[predicted_emotion_index]

                    # Increment the count for the predicted emotion
                    emotion_counts[predicted_emotion] += 1

                    # Draw rectangle around the face and put emotion label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        predicted_emotion,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )

            last_frame = frame

        cap.release()
        os.remove(temp_video_path)

        if any(emotion_counts.values()):  # Check if there are any non-zero counts
            # Plot pie chart of emotion frequencies
            labels = list(emotion_counts.keys())
            sizes = list(emotion_counts.values())
            plt.figure(figsize=(8, 6))
            plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
            plt.title("Emotion Frequency Analysis")
            plt.axis("equal")

            # Convert the last frame to RGB format
            last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)

            # Display the last frame along with the pie chart
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(last_frame_rgb)
            ax[0].axis("off")
            ax[1].pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
            ax[1].set_title("Emotion Frequency Analysis")
            ax[1].axis("equal")

            # Save the combined image and pie chart to a temporary file
            temp_image_path = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ).name
            plt.savefig(temp_image_path)
            plt.close(fig)

            # Encode the image to base64
            with open(temp_image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()

            os.remove(temp_image_path)

            # Return the image in the response
            return JsonResponse({"image": encoded_image}, status=200)

        else:
            return JsonResponse(
                {"error": "No faces detected or all emotion counts are zero"},
                status=200,
            )

    return JsonResponse({"error": "Invalid request"}, status=400)
