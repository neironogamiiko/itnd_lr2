import cv2
import dlib
from pathlib import Path
from . import utility

def landmarks_research(images,
                       predictor_path: Path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))

    for name, image in images.items():
        points = utility.detect_landmarks(
            image,
            detector,
            predictor
        )

        if points is None:
            print(f"{name}: Face not detected.")
            continue

        result = utility.draw_landmarks(image, points)

        print(f"{name}: Detected {len(points)} facial landmarks.")

        cv2.imshow(name, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()