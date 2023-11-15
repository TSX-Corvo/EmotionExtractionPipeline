# load json and create model
from __future__ import division
from typing import List, Tuple
from keras.models import model_from_json
import os
import numpy as np
import cv2
import os
import subprocess

# loading the model
json_file = open("fer.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fer.h5")
print("Loaded model from disk")

# setting image resizing parameters
WIDTH = 48
HEIGHT = 48
x = None
y = None
labels = ["Anger", "Disgust", "Fear", "Enjoyment", "Sadness", "Surprise", "Neutral"]

# loading image


def detect_emotion(img_filename: str) -> List[float]:
    full_size_image = cv2.imread(img_filename)
    # print("Image Loaded")
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face.detectMultiScale(gray, 1.3, 10)

    # detecting faces
    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        cropped_img = np.expand_dims(
            np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
        )
        cv2.normalize(
            cropped_img,
            cropped_img,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_L2,
            dtype=cv2.CV_32F,
        )
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # predicting the emotion
        yhat = loaded_model.predict(cropped_img)

        # scores = np.argsort(np.max(yhat, axis=0))

        # max_index = scores[-1] if scores[-1] != 6 else scores[-2]

        return yhat


def extract_frames(input_folder="videos", output_folder="output", fps="1/3"):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if not filename.endswith(".mp4"):
            continue

        # Generate the output file name by replacing the extension
        output_filename = os.path.splitext(filename)[0]

        output_dir = os.path.join(output_folder, output_filename)
        os.makedirs(output_dir, exist_ok=True)

        # Construct the ffmpeg command
        command = f"ffmpeg -i {os.path.join(input_folder, filename)} -vf fps={fps} {output_dir}/%03d.jpg"

        # Execute the command using subprocess
        subprocess.run(command, shell=True)


def extract_emotions(folder="output"):
    results: List[str] = []
    # Iterate through all subfolders and files in the specified folder
    for _dir in os.listdir(folder):
        scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        for filename in os.listdir(os.path.join(folder, _dir)):
            subscores = detect_emotion(os.path.join(folder, _dir, filename))

            try:
                scores = np.add(scores, subscores)
            except:
                print(f"Couldn't recognize emotion in file {_dir}/{filename}")
                print(f"Scores: {subscores}")

        sorted_scores = np.argsort(np.max(scores, axis=0))

        max_index = sorted_scores[-1] if sorted_scores[-1] != 6 else sorted_scores[-2]

        results.append(f"{_dir}.mp4, {labels[max_index]}\n")

    results = sorted(results)

    with open("data.csv", "w") as output:
        for result in results:
            output.write(result)


if __name__ == "__main__":
    extract_frames(input_folder="videos", output_folder="output", fps="1/2")
    extract_emotions()
