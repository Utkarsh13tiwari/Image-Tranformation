# Object Transformation Estimation using Template Matching and SAM Segmentation

This script demonstrates the process of estimating the transformation (rotation and translation) between a template image and a test image using template matching and SAM segmentation. It uses the ORB feature detector to compute keypoints and descriptors for both the template and test images. The keypoints are matched between the images using a brute-force matcher. The segmented masks generated by the SAM segmentation algorithm are used to compute the keypoints and descriptors.

## Prerequisites
- Python 3
- OpenCV
- NumPy
- PyTorch
- Matplotlib
- segment-anything
## Installation

1. Clone this repository to your local machine.
2. Install segment-anything using:
```
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```
3. Install the weight for segment-anything model using:
```
pip install 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
```
## Usage

1. Place your template image (`input.jpg`) and test image (`test.jpg`) in the root directory.
2. Run the script terminal or either IDE, there is agrs to be pass.


## Output

The script generates a visualization of the following:

- Template image with segmented mask overlaid.
- Test image with segmented mask overlaid, bounding box, and transformation annotations.
- Matched keypoints between the template and test images.
- Matched keypoints between the original template and test images.

## Results Interpretation

The script estimates the rotation and translation between the template and test images based on the matched keypoints. The transformation annotations are added to the test image to visualize the estimated transformation.

