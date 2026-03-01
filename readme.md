🖼️ Image Smoothing Using Spatial Filters (OpenCV)
📌 Project Overview
This project demonstrates image smoothing techniques using three distinct spatial filtering methods in OpenCV. Image smoothing (or blurring) is a fundamental preprocessing step in computer vision used to reduce noise and prepare images for higher-level tasks like edge detection.

🎯 Aim
To implement and compare different smoothing filters to analyze their impact on noise reduction, blur intensity, and edge preservation.

🧠 What is Image Smoothing?
Image smoothing works by convolving a kernel over an image. For each pixel, the filter considers its neighbors and reassigns its value based on a mathematical calculation.

Key Objectives:

Noise Reduction: Removing "salt-and-pepper" or Gaussian noise.

Detail Suppression: Removing minor textures to focus on larger objects.

Preprocessing: Smoothing often precedes edge detection to prevent false positives.

🔍 Filters Comparison
1️⃣ Mean Filter (Averaging)

The simplest form of blurring. Every pixel under the kernel area is averaged to produce the new pixel value.

Function: cv2.blur(src, ksize)

Logic: All elements in the window have equal weight.

Pros: Very fast; effective for general noise.

Cons: Significantly blurs edges; creates a "blocky" look at high kernel sizes.

2️⃣ Weighted Filter

Unlike the mean filter, the weighted filter assigns higher importance to the pixels closer to the center of the kernel, resulting in a more refined blur.

Function: cv2.filter2D(src, ddepth, kernel)

Logic: Uses a custom normalized kernel where the center is the highest value.

Pros: Better edge preservation than the mean filter.

Cons: Requires manual kernel definition.

3️⃣ Gaussian Filter

The Gaussian filter uses a bell-shaped distribution. Pixels further from the center have significantly less influence on the result.

Function: cv2.GaussianBlur(src, ksize, sigmaX)

Pros: Most "natural" looking blur; excellent for removing Gaussian noise while keeping edges relatively sharp.

Cons: Computationally more expensive than a simple mean filter.