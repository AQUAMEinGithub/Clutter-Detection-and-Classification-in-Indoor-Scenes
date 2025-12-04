# src/run_demo.py
import os
import numpy as np
import cv2

from src.pipeline import run_pipeline
from src.utils import save_image, draw_boxes


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))

    tidy_dir = os.path.join(project_root, "data", "tidy")
    cluttered_dir = os.path.join(project_root, "data", "cluttered")
    output_dir = os.path.join(project_root, "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        {
            "name": "base",
            "diff_thresh": 20,
            "yolo_conf": 0.25,
            "yolo_iou": 0.45,
            "overlap_ratio_thresh": 0.0,
        },
        {
            "name": "low_diff",
            "diff_thresh": 15,
            "yolo_conf": 0.25,
            "yolo_iou": 0.45,
            "overlap_ratio_thresh": 0.0,
        },
        {
            "name": "high_diff",
            "diff_thresh": 30,
            "yolo_conf": 0.25,
            "yolo_iou": 0.45,
            "overlap_ratio_thresh": 0.0,
        },
        {
            "name": "strict_overlap",
            "diff_thresh": 20,
            "yolo_conf": 0.25,
            "yolo_iou": 0.45,
            "overlap_ratio_thresh": 0.10,
        },
    ]


    # Enumerate all image files in the tidy directory (supports .jpg/.jpeg/.png).
    image_names = sorted(
        f
        for f in os.listdir(tidy_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not image_names:
        print(f"No images found in {tidy_dir}")
        return

    print("Found tidy images:", image_names)

    for img_name in image_names:
        tidy_path = os.path.join(tidy_dir, img_name)
        cluttered_path = os.path.join(cluttered_dir, img_name)

        if not os.path.exists(cluttered_path):
            print(f"Skip {img_name}: no matching file in {cluttered_dir}")
            continue

        base, _ = os.path.splitext(img_name)  # "1.png" -> "1"

        print(f"\n=== Image {img_name} ===")

        outputs = []

        for cfg in configs:
            print(f"  -> config {cfg['name']}")
            result = run_pipeline(
                tidy_path,
                cluttered_path,
                save_path=None,
                diff_thresh=cfg["diff_thresh"],
                yolo_conf=cfg["yolo_conf"],
                yolo_iou=cfg["yolo_iou"],
                overlap_ratio_thresh=cfg["overlap_ratio_thresh"],
            )

            if "error" in result:
                print("     ERROR:", result["error"])
                outputs.append(None)
                continue

            diff_n = len(result["diff_boxes"])
            yolo_n = len(result["yolo_boxes_clutter"])
            filt_n = len(result["filtered_boxes"])
            final_n = len(result["final_boxes"])

            print(
                f"     diff_boxes={diff_n} | "
                f"yolo_clutter={yolo_n} | "
                f"filtered={filt_n} | final={final_n}"
            )

            aligned = result["aligned"].copy()
            tidy_boxes = result["yolo_boxes_tidy"]
            aligned_vis = draw_boxes(aligned, tidy_boxes, color=(255, 0, 255))

            clutter_vis = result["output"].copy()

            h1, w1 = aligned_vis.shape[:2]
            h2, w2 = clutter_vis.shape[:2]
            h = min(h1, h2)
            w = min(w1, w2)
            if (h1, w1) != (h, w):
                aligned_vis = cv2.resize(aligned_vis, (w, h))
            if (h2, w2) != (h, w):
                clutter_vis = cv2.resize(clutter_vis, (w, h))

            tile = np.hstack([aligned_vis, clutter_vis])

            label = f"{cfg['name']}: final={final_n}"
            cv2.putText(
                tile,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            outputs.append(tile)

        valid_imgs = [img for img in outputs if img is not None]
        if len(valid_imgs) == 0:
            print("  -> All configs failed for this image, skip composing grid.")
            continue

        h, w = valid_imgs[0].shape[:2]

        filled = []
        for img in outputs:
            if img is None:
                filled.append(np.zeros((h, w, 3), dtype=np.uint8))
            else:
                if img.shape[0] != h or img.shape[1] != w:
                    img = cv2.resize(img, (w, h))
                filled.append(img)

        if len(filled) != 4:
            print(
                f"  -> Expected 4 configs, got {len(filled)}. "
                "Grid composition assumes 4; skipping."
            )
            continue

        top_row = np.hstack([filled[0], filled[1]])
        bottom_row = np.hstack([filled[2], filled[3]])
        grid = np.vstack([top_row, bottom_row])

        save_path = os.path.join(output_dir, f"{base}_grid.png")
        save_image(save_path, grid)
        print(f"  -> Saved grid to {save_path}")


if __name__ == "__main__":
    main()
