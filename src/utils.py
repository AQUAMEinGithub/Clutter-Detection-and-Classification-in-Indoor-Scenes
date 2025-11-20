import cv2
import matplotlib.pyplot as plt
def load_image(path):
    """
    Load an image from disk using OpenCV.

    Parameters
    ----------
    path : str
        File path to the image.

    Returns
    -------
    img : np.ndarray
        Loaded BGR image, or None if loading fails.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return img

def save_image(path, img):
    """
    Save an image to disk.

    Parameters
    ----------
    path : str
        Output file path.
    img : np.ndarray
        Image to save.
    """
    cv2.imwrite(path, img)


def draw_boxes(img, boxes, color=(0, 255, 0)):
    """
    Draw bounding boxes on an image.

    Parameters
    ----------
    img : np.ndarray
        Image on which to draw.
    boxes : list of tuples
        Bounding boxes in format (x, y, w, h, *optional_class_info).
    color : tuple
        BGR color for the boxes.

    Returns
    -------
    img_drawn : np.ndarray
        Image with drawn boxes.
    """
    img_drawn = img.copy()
    for box in boxes:
        if len(box) >= 4:
            x, y, w, h = box[:4]
            cv2.rectangle(img_drawn, (x, y), (x + w, y + h), color, 2)
    return img_drawn

def show_image(title, img):
    """
    Display an image using matplotlib (handles BGR → RGB conversion).

    Parameters
    ----------
    title : str
        Title for the plot window.
    img : np.ndarray
        Image to display.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()