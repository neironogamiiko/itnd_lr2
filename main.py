import cv2
from pathlib import Path
import utility

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