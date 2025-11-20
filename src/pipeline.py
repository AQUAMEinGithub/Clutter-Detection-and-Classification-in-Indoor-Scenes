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
    """
    Full clutter detection pipeline.

    Steps:
    1. Load images
    2. Align tidy → cluttered
    3. Compute difference mask
    4. Threshold + clean mask
    5. Extract difference bounding boxes
    6. Run YOLO on cluttered
    7. Filter YOLO boxes using difference mask
    8. Draw boxes and optionally save output

    Returns
    -------
    result : dict
        {
            "aligned": aligned image,
            "diff_gray": raw abs-diff map,
            "mask": thresholded mask,
            "cleaned": cleaned mask,
            "diff_boxes": boxes from differencing,
            "yolo_boxes": YOLO detections,
            "filtered_boxes": YOLO detections that overlap changes,
            "output": drawn output image
        }
    """
    # ------------------------------
    # 1. Load images
    # ------------------------------
    tidy = load_image(tidy_path)
    cluttered = load_image(cluttered_path)

    # ------------------------------
    # 2. Align tidy → cluttered
    # ------------------------------
    aligned, H, matches = align_images(tidy, cluttered)
    if aligned is None:
        print("Alignment failed. Exiting pipeline.")
        return {"error": "homography_failed"}

    # ------------------------------
    # 3. Compute difference mask
    # ------------------------------
    diff_gray = compute_difference_mask(aligned, cluttered)
    mask = threshold_mask(diff_gray)
    cleaned = clean_mask(mask)

    # ------------------------------
    # 4. Extract bounding boxes
    # ------------------------------
    diff_boxes = extract_bounding_boxes(cleaned)

    # ------------------------------
    # 5. Run YOLO detection on cluttered image
    # ------------------------------
    yolo_model = load_yolo()
    yolo_boxes = run_yolo(yolo_model, cluttered)

    # ------------------------------
    # 6. Filter YOLO boxes using differencing mask
    # (Keep only YOLO detections overlapping changed regions)
    # ------------------------------
    filtered_boxes = []
    for box in yolo_boxes:
        x, y, w, h, cls_name, conf = box

        # crop mask
        submask = cleaned[y:y + h, x:x + w]
        if submask.size > 0 and np.any(submask > 0):  # overlap
            filtered_boxes.append(box)

    # ------------------------------
    # 7. Draw boxes
    # ------------------------------
    output = cluttered.copy()
    output = draw_boxes(output, filtered_boxes, color=(0, 255, 0))  # green clutter boxes
    output = draw_boxes(output, diff_boxes, color=(255, 0, 0))  # blue diff regions

    # ------------------------------
    # 8. Save image (optional)
    # ------------------------------
    if save_path is not None:
        save_image(save_path, output)

    # ------------------------------
    # Return all data for debugging
    # ------------------------------
    return {
        "aligned": aligned,
        "diff_gray": diff_gray,
        "mask": mask,
        "cleaned": cleaned,
        "diff_boxes": diff_boxes,
        "yolo_boxes": yolo_boxes,
        "filtered_boxes": filtered_boxes,
        "output": output
    }