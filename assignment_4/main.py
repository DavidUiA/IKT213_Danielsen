import numpy as np
import cv2

reference_img = cv2.imread('reference_img.png')
align_this_img = cv2.imread('align_this.jpg', 0)


def main():
    cv2.imwrite("harris.png", harris_edge_detection(reference_img))

    aligned, matches_img = feature_image_align(align_this_img, reference_img, 10, 0.7)
    cv2.imwrite("aligned.png", aligned)
    cv2.imwrite("matches.png", matches_img)

    #cv2.imshow("", edges)
    #cv2.waitKey(0)

def harris_edge_detection(reference_image):
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    reference_image[dst > 0.01 * dst.max()] = [0, 0, 255]

    return reference_image

def feature_image_align(image_to_align, reference_image, max_features, good_match_precent):
    # using sift
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image_to_align, None)
    kp2, des2 = sift.detectAndCompute(reference_image, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = max_features

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = image_to_align.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        reference_image = cv2.polylines(reference_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        M = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(image_to_align, kp1, reference_image, kp2, good, None, **draw_params)

    ref_h, ref_w = reference_image.shape[:2]
    aligned = cv2.warpPerspective(image_to_align, M, (ref_w, ref_h))

    return aligned, img3

main()
