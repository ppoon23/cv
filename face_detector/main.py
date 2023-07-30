import requests
import sys
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from utils import visualize

IMG_LINK = "https://source.unsplash.com/HAe8lLfH02E"
IMAGE_PATH = "img.jpg"
ANNOTATED_IMAGE_PATH = "annotated_img.jpg"
MODEL_LINK = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
MODEL_PATH = "detector.tflite"


def get_image(overwrite=True):
    if (not Path(IMAGE_PATH).exists()) or overwrite:
        try:
            r = requests.get(IMG_LINK, allow_redirects=True)
            with open(IMAGE_PATH, "wb") as f:
                f.write(r.content)
        except Exception:
            sys.exit("Cannot get image")


def get_model(overwrite=True):
    if (not Path(MODEL_PATH).exists()) or overwrite:
        try:
            r = requests.get(MODEL_LINK, allow_redirects=True)
            print(r.ok)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
        except Exception:
            sys.exit("Cannot get model")


def run_detection():
    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.4,
    )

    with vision.FaceDetector.create_from_options(options) as detector:
        image = mp.Image.create_from_file(IMAGE_PATH)
        result = detector.detect(image)

    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(filename=ANNOTATED_IMAGE_PATH, img=rgb_annotated_image)


if __name__ == "__main__":
    get_image()
    get_model()
    run_detection()
