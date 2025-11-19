from .alignment import align_images
from .differencing import (
    compute_difference_mask,
    threshold_mask,
    clean_mask,
    extract_bounding_boxes
)
from .detection import load_yolo, run_yolo
from .utils import load_image, save_image, draw_boxes


def run_pipeline(tidy_path, cluttered_path, save_path=None):
    """
    Steps:
    1. Load images
    2. Align tidy to cluttered
    3. Compute difference mask
    4. Run YOLO on cluttered
    5. Filter YOLO detections by mask
    6. Draw bounding boxes
    7. Save output (optional)
    """
    pass