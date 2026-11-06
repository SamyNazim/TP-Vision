import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('algiers.jpg')

if img is None:
    raise FileNotFoundError("Image file not found. Please check the path and filename.")

# Create noisy image
noise = np.random.normal(0, 25, img.shape)

noisy = img.astype(np.float32) + noise
noisy = np.clip(noisy, 0, 255).astype(np.uint8)

# Mean Filter
h, w, c = img.shape
img_res_mean = np.zeros((h, w, c), dtype=np.uint8)

k = int(input("Enter the kernel size (odd number): "))
if k % 2 == 0:
    raise ValueError("Kernel size must be an odd number.")
p = k // 2

for i in range(p, h-p):
    for j in range(p, w-p):
        neighborhood = noisy[i-p:i+p+1, j-p:j+p+1]
        mean = np.mean(neighborhood, axis=(0, 1))
        img_res_mean[i, j] = mean

# Median Filter
img_res_median = np.zeros((h, w, c), dtype=np.uint8)

for i in range(p, h-p):
    for j in range(p, w-p):
        neighborhood = noisy[i-p:i+p+1, j-p:j+p+1]
        median = np.median(neighborhood, axis=(0, 1))
        img_res_median[i, j] = median

# Histograms 

def calculate_histogram(image):
    hist = np.zeros(256, dtype=int)
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            intensity = image[i, j]
            hist[intensity] += 1
    return hist

# Gray scale histogram
gray = (0.299 * img[:, :, 0] +
        0.587 * img[:, :, 1] +
        0.114 * img[:, :, 2]).astype(np.uint8)

hist = calculate_histogram(gray)

# Darkened histogram

gray2 = (gray / 2).astype(np.uint8)

hist2 = calculate_histogram(gray2)

# Brightened histogram

gray3 = np.clip(gray.astype(np.int16) + 128, 0, 255).astype(np.uint8)

hist3 = calculate_histogram(gray3)

# Equalized histogram

gray_equalized = (gray / 2 + 50).astype(np.uint8)

hist_equalized = calculate_histogram(gray_equalized)


# Display images

# Display original image, noise and noisy image
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noise)
plt.title("Random Noise")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(noisy)
plt.title("Noisy Image")
plt.axis("off")

plt.show()

# Display original, mean filtered and median filtered images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_res_mean)
plt.title("Mean Filtered Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_res_median)
plt.title("Median Filtered Image")
plt.axis("off")

plt.show()

# Display histograms

plt.figure(figsize=(12, 8))

# Grayscale histogram
plt.subplot(2, 2, 1)
plt.plot(hist)
plt.title("Grayscale Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim(0, 255)

# Darkened histogram
plt.subplot(2, 2, 2)
plt.plot(hist2)
plt.title("Darkened Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim(0, 255)

# Brightened histogram
plt.subplot(2, 2, 3)
plt.plot(hist3)
plt.title("Brightened Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim(0, 255)

# Equalized histogram
plt.subplot(2, 2, 4)
plt.plot(hist_equalized)
plt.title("Equalized Image Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.xlim(0, 255)

plt.tight_layout()
plt.show()