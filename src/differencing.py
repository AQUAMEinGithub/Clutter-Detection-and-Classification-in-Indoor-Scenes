import cv2
import numpy as np

def compute_difference_mask(aligned_tidy, cluttered):
    """
        Compute the raw pixel-wise difference between the aligned tidy image
        and the cluttered image.

        Parameters
        ----------
        aligned_tidy : np.ndarray
            Tidy image warped into cluttered coordinate frame.
        cluttered : np.ndarray
            Original cluttered image.

        Returns
        -------
        diff_gray : np.ndarray
            Grayscale absolute-difference map highlighting changed regions.
        """
    diff = cv2.absdiff(aligned_tidy, cluttered)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    return diff_gray

def threshold_mask(diff_gray, thresh=20):
    """
    Threshold the grayscale difference map to obtain a binary foreground mask.

    Parameters
    ----------
    diff_gray : np.ndarray
        Raw grayscale difference image.
    thresh : int, optional
        Threshold for binarization.

    Returns
    -------
    mask : np.ndarray
        Binary mask where changed pixels = 255.
    """
    _, mask = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)
    return mask

def clean_mask(mask):
    """
    Refine the binary difference mask using morphological operations.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask from thresholding.

    Returns
    -------
    cleaned : np.ndarray
        Noise-reduced and hole-filled mask suitable for region extraction.

    Notes
    -----
    Uses morphological opening to remove small noise
    and closing to merge fragmented regions.
    """
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

def extract_bounding_boxes(mask):
    """
    Extract bounding boxes around connected components in the mask.

    Parameters
    ----------
    mask : np.ndarray
        Cleaned binary mask representing changed regions.

    Returns
    -------
    boxes : list of tuples
        List of bounding boxes in the format (x, y, w, h) for each region.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        boxes.append((x,y,w,h))
    return boxes