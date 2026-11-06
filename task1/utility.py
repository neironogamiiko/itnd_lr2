import matplotlib.pyplot as plt
import cv2

###########################################################
#  Варіант 2: Поворот, переведення зображення в бінарне.  #
###########################################################

def show_images(images, titles, figsize=(15, 5)):
    plt.figure(figsize=figsize)

    for i, (image, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), i)

        if len(image.shape) == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height)
    )

    return rotated

def add_text(image, text):
    color = 255 if len(image.shape) == 2 else (0, 255, 0)

    for i, line in enumerate(text.split("\n")):
        position = (20, 40 + i * 40)

        cv2.putText(
            image, line, position,
            cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 4
        )

        cv2.putText(
            image, line, position,
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )

def image2binary(image, threshold=127):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return binary