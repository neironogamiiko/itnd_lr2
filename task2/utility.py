import cv2
import dlib

def detect_landmarks(image,
                     detector,
                     predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 1)

    if len(faces) == 0:
        return None

    face = max(
        faces,
        key=lambda rect: rect.width() * rect.height()
    )

    landmarks = predictor(gray, face)

    points = [
        (landmarks.part(i).x, landmarks.part(i).y)
        for i in range(68)
    ]

    return points

def draw_landmarks(image, points):
    result = image.copy()

    for point in points:
        cv2.circle(
            result,
            point,
            2,
            (0, 0, 255),
            -1
        )

    return result