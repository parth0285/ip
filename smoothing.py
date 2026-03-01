import cv2
import numpy as np

# ----------------------------
# Load Image
# ----------------------------
img = cv2.imread("/workspaces/ip/img.png")
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

# ----------------------------
# Convert to BGR for text
# ----------------------------
def to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

gray = to_bgr(gray)
mean_img = to_bgr(mean_img)
weighted_img = to_bgr(weighted_img)
gaussian_img = to_bgr(gaussian_img)

# ----------------------------
# Add Image Label
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

orig_l = add_label(gray, "Original")
mean_l = add_label(mean_img, "Mean Filter")
weighted_l = add_label(weighted_img, "Weighted Filter")
gaussian_l = add_label(gaussian_img, "Gaussian Filter")

# ----------------------------
# Combine Images
# ----------------------------
final_images = np.hstack((orig_l, mean_l, weighted_l, gaussian_l))

# ----------------------------
# Add Centered Comparison Table
# ----------------------------
table_height = 260
h, w, c = final_images.shape

table = np.ones((table_height, w, 3), dtype=np.uint8) * 255

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
thickness = 2

# Table width (75% of image width)
table_width = int(w * 0.75)
start_x = (w - table_width) // 2
end_x = start_x + table_width

# Outer border
cv2.rectangle(table, (start_x, 40), (end_x, table_height-40), (0,0,0), 2)

# Column positions
col1 = start_x + table_width // 6
col2 = start_x + table_width // 2
col3 = start_x + 5 * table_width // 6

# Header
cv2.putText(table, "Filter", (col1-60, 90), font, font_scale, (0,0,0), thickness)
cv2.putText(table, "Blur Level", (col2-80, 90), font, font_scale, (0,0,0), thickness)
cv2.putText(table, "Edge Preservation", (col3-120, 90), font, font_scale, (0,0,0), thickness)

# Header underline
cv2.line(table, (start_x, 110), (end_x, 110), (0,0,0), 2)

rows = [
    ("Mean", "High", "Low"),
    ("Weighted", "Medium", "Medium"),
    ("Gaussian", "Controlled", "High")
]

y = 160
for r in rows:
    cv2.putText(table, r[0], (col1-60, y), font, font_scale, (0,0,0), thickness)
    cv2.putText(table, r[1], (col2-80, y), font, font_scale, (0,0,0), thickness)
    cv2.putText(table, r[2], (col3-120, y), font, font_scale, (0,0,0), thickness)
    y += 45

# ----------------------------
# Final Output
# ----------------------------
final = np.vstack((final_images, table))

cv2.imwrite("result.png", final)
print("Result saved as result.png")