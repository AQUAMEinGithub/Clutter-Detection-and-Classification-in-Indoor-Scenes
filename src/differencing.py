import cv2
import numpy as np

def extract_sift_features(image):
    # SIFT keypoints + descriptors
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)
    return kp, desc


def match_features(desc1, desc2, ratio=0.75):
    # FLANN + Lowe’s ratio test
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    return good


def estimate_homography(kp1, kp2, matches, thresh=4.0):
    # Compute homography with RANSAC
    if len(matches) < 4:
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, thresh)
    return H, mask


def align_images(tidy_image, cluttered_image):
    # Full alignment pipeline
    kp1, desc1 = extract_sift_features(tidy_image)
    kp2, desc2 = extract_sift_features(cluttered_image)

    matches = match_features(desc1, desc2)
    H, mask = estimate_homography(kp1, kp2, matches)

    if H is None:
        print("Homography failed.")
        return None, None, matches

    h, w = cluttered_image.shape[:2]
    aligned = cv2.warpPerspective(tidy_image, H, (w, h))
    return aligned, H, matche