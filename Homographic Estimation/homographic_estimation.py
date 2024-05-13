import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, \
    SamPredictor  # Import SAM segmentation algorithm

# Load the template image and test image
template_image = cv2.imread(r"/inpu.jpg")
test_image = cv2.imread(r"/test.jpg")

template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

# Convert the images to PyTorch tensors and move them to the appropriate device
template_image_tensor = torch.tensor(template_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to("cuda")
test_image_tensor = torch.tensor(test_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to('cuda')

# Initialize SAM segmentation algorithm
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = r"/sam_vit_h_4b8939.pth"  # Path to the SAM model checkpoint
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

# Function that inputs the output and plots image and mask
def show_anns(anns, axes=None):
    if len(anns) == 0:
        return
    if axes:
        ax = axes
    else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m ** 0.5)))


def add_SAM_auto_segmentation(image):
    masks = mask_generator.generate(image)
    full_mask = np.zeros_like(masks[0]["segmentation"]).astype(int)
    for i in range(len(masks)):
        x, y = np.where(masks[i]['segmentation'])
        full_mask[x, y] = i + 1
    return np.uint8(full_mask)

# Segment the object of interest in the template image
template_segmented = add_SAM_auto_segmentation(template_image)
test_segmented = add_SAM_auto_segmentation(test_image)

template_segmented = np.array(template_segmented, dtype=np.uint8)
test_segmented = np.array(test_segmented, dtype=np.uint8)

# Compute keypoints and descriptors for template and test images
detector = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

kp_template, des_template = detector.detectAndCompute(template_segmented, None)
kp_test, des_test = detector.detectAndCompute(test_segmented, None)

# Match keypoints between template and test images
matches = matcher.match(des_template, des_test)

# Draw matched keypoints between template and test images
matched_img = cv2.drawMatches(template_segmented, kp_template, test_segmented, kp_test, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

matched_img_normal = cv2.drawMatches(template_image, kp_template, test_image, kp_test, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Estimate transformation between template and test images
src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_test[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Calculate rotation and translation
rotation_rad = np.arctan2(M[1, 0], M[0, 0])
rotation_deg = np.degrees(rotation_rad)
translation = M[:, 2]

print("Estimated Rotation (degrees):", rotation_deg)
print("Estimated Translation:", translation)

# Display images and annotations
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
axes[0].imshow(template_image)
axes[0].set_title('Template Image')
show_anns(template_segmented, axes[0])

axes[1].imshow(test_image)
axes[1].set_title('Test Image with Bounding Box')

# Draw bounding box around the object in the test image
h, w, _ = template_image.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
test_image_with_bbox = cv2.polylines(test_image.copy(), [np.int32(dst)], True, (255, 0, 0), 2)

# Add rotation and translation text on the bounding box
cv2.putText(test_image_with_bbox, f'Rotation: {rotation_deg:.2f} degrees', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (255, 255, 255), 3)
cv2.putText(test_image_with_bbox, f'Translation: {translation}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2,
            (255, 255, 255), 3)

axes[1].imshow(test_image_with_bbox)
axes[1].set_title('Test Image with Bounding Box and Annotations')

axes[2].imshow(matched_img)
axes[2].set_title('Matched Keypoints')
plt.show()
