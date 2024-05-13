import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, \
    SamPredictor  # Import SAM segmentation algorithm

# Load the template image and test image
template_image = cv2.imread(r"path\to\templete.jpg")
test_image = cv2.imread(r"path\to\test.jpg")

template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

# Initialize SAM segmentation algorithm
DEVICE = torch.device('cuda')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = r"path\to\sam_vit_h_4b8939.pth"  # Path to the SAM model checkpoint
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(model=sam,
                                           points_per_side=32,
                                           pred_iou_thresh=0.9,
                                           stability_score_thresh=0.92,
                                           crop_n_layers=1,
                                           crop_n_points_downscale_factor=2,
                                           min_mask_region_area=400)
predictor = SamPredictor(sam)


def add_SAM_auto_segmentation(image):
    masks = mask_generator.generate(image)
    full_mask = np.zeros_like(masks[0]["segmentation"]).astype(int)
    for i in range(len(masks)):
        x, y = np.where(masks[i]['segmentation'])
        full_mask[x, y] = i + 1

    return np.uint8(full_mask)


# Segment the object of interest in the template image
template_segmented = add_SAM_auto_segmentation(template_image)
print(template_segmented.shape)

test_segmented = add_SAM_auto_segmentation(test_image)
print(test_segmented.shape)

template_segmented = np.array(template_segmented, dtype=np.uint8)
test_segmented = np.array(test_segmented, dtype=np.uint8)

# Compute key points and descriptors for the segmented template image
detector = cv2.ORB_create()

kp_template, des_template = detector.detectAndCompute(template_segmented, None)

# Compute key points and descriptors for the test image
kp_test, des_test = detector.detectAndCompute(test_segmented, None)

# Match key points between segmented template and test images
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(des_template, des_test)

# Draw matched key points between template and test images
matched_img = cv2.drawMatches(template_segmented, kp_template, test_segmented, kp_test, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
matched_img_normal = cv2.drawMatches(template_image, kp_template, test_image, kp_test, matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Estimate affine transformation between segmented template and test images
src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_test[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M = cv2.estimateAffine2D(src_pts, dst_pts)[0]

# Calculate rotation and translation
rotation_rad = np.arctan2(M[1, 0], M[0, 0])
rotation_deg = np.degrees(rotation_rad)
translation = M[:, 2]

print("Estimated Rotation (degrees):", rotation_deg)
print("Estimated Translation:", translation)

# Display images and annotations
fig, axes = plt.subplots(2, 2, figsize=(16, 6))
axes[0][0].imshow(template_image)
axes[0][0].set_title('Template Image')

# Show segmented mask on the template image
axes[0][0].imshow(template_segmented, alpha=0.5)

# Draw bounding box around the object in the test image
h, w, _ = template_image.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)

test_segmented_mat = cv2.UMat(np.uint8(test_segmented * 255))
# Add rotation and translation text on the bounding box

h, w, _ = template_image.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
# test_image_box = cv2.polylines(test_image.copy(), [np.int32(dst)], True, (255, 0, 0), 2)

cv2.putText(test_segmented_mat, f'Rotation: {rotation_deg:.2f} degrees', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 0, 0),
            3)
cv2.putText(test_segmented_mat, f'Translation: {translation}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 0, 0),
            3)

test_segmented_mat = test_segmented_mat.get().astype(np.uint8)  # Converting back to uint8
axes[1][0].imshow(test_segmented_mat, alpha=0.5)
axes[1][0].set_title('Test Image with Bounding Box')

axes[0][1].imshow(matched_img)
axes[0][1].set_title('Matched Keypoints')

axes[1][1].imshow(matched_img_normal)
axes[1][1].set_title('Matched Keypoints on normal Image')
plt.show()
