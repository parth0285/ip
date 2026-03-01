import cv2
import numpy as np

# ----------------------------
# Load Image
# ----------------------------
img = cv2.imread("/Users/parthbole/Desktop/IP/ImageSmoothing/img.png")
if img is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----------------------------
# Apply Filters
# ----------------------------
mean_img = cv2.blur(gray, (5,5))

weighted_kernel = np.array([[1,2,1],
                            [2,4,2],
                            [1,2,1]], np.float32)
weighted_kernel /= weighted_kernel.sum()
weighted_img = cv2.filter2D(gray, -1, weighted_kernel)

gaussian_img = cv2.GaussianBlur(gray, (5,5), 1)

# Convert all to BGR for colored text
def to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

gray = to_bgr(gray)
mean_img = to_bgr(mean_img)
weighted_img = to_bgr(weighted_img)
gaussian_img = to_bgr(gaussian_img)

# ----------------------------
# Add Centered Label
# ----------------------------
def add_label(image, text):
    h, w, c = image.shape
    label_height = 60
    
    label = np.ones((label_height, w, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (label_height + text_size[1]) // 2
    
    cv2.putText(label, text, (text_x, text_y),
                font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
    
    return np.vstack((image, label))

# ----------------------------
# Add Labels
# ----------------------------
orig_l = add_label(gray, "Original")
mean_l = add_label(mean_img, "Mean Filter")
weighted_l = add_label(weighted_img, "Weighted Filter")
gaussian_l = add_label(gaussian_img, "Gaussian Filter")

# Combine horizontally
final = np.hstack((orig_l, mean_l, weighted_l, gaussian_l))

cv2.imshow("Image Smoothing Comparison", final)
cv2.waitKey(0)
cv2.destroyAllWindows()