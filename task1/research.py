import cv2
from . import utility


def angle_research(image,
                   angles=[0, 30, 45, 90, 180],
                   scale = 1.0):
    rotation_results = []
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    for angle in angles:
        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            scale
        )

        result = cv2.warpAffine(
            image,
            matrix,
            (width, height)
        )

        rotation_results.append(result)

    utility.show_images(
        rotation_results,
        [f"Rotation = {angle}°" for angle in angles],
        figsize=(18, 5)
    )

def threshold_research(image, thresholds = [50, 100, 127, 180, 220]):
    binary_results = []

    for value in thresholds:
        _, result = cv2.threshold(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            value,
            255,
            cv2.THRESH_BINARY
        )

        binary_results.append(result)

    utility.show_images(
        binary_results,
        [f"Threshold = {value}" for value in thresholds],
        figsize=(18, 5)
    )