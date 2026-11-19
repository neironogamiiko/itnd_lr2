import cv2
from pathlib import Path
from task1 import utility, research

import os
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

##############
#   TASK 1   #
##############

NAME = 'Horbatiuk Mykyta'
ANGLE = 45
THRESHOLD = 115

image_path = Path("/home/frasero/Downloads/lucy.jpeg")
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")


rotated_image = utility.rotate_image(image, angle=ANGLE)
utility.add_text(rotated_image, text=f"Rotation: {ANGLE} degree\n{NAME}")

binary_image = utility.image2binary(image, threshold=THRESHOLD)
utility.add_text(binary_image, text=f"Binarization threshold: {THRESHOLD}\n{NAME}")

utility.show_images(
    [image, rotated_image, binary_image],
    ["Base image", "Rotated image", "Binary image"]
)

research.angle_research(image, scale=.5)
research.threshold_research(image)

##############
#   TASK 2   #
##############

from task2 import research

image1_path = Path("/home/frasero/Downloads/lfw/Serena_Williams/Serena_Williams_0001.jpg")
image2_path = Path("/home/frasero/Downloads/lfw/Serena_Williams/Serena_Williams_0017.jpg")
image3_path = Path("/home/frasero/Downloads/lfw/Serena_Williams/Serena_Williams_0027.jpg")

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)
image3 = cv2.imread(image3_path)

for image_path, image in [
    (image1_path, image1),
    (image2_path, image2),
    (image3_path, image3)
]:
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

predictor_path = Path(
    "/home/frasero/Downloads/shape_predictor_68_face_landmarks.dat"
)

if not predictor_path.exists():
    raise FileNotFoundError(f"Model not found: {predictor_path}")

research.landmarks_research(
    {
        "Front face": image1,
        "Different lighting": image2,
        "Head rotation": image3
    },
    predictor_path
)