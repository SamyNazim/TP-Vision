import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('green.jpg',cv2.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError("Image file not found. Please check the path and filename.")

h,w,c = img.shape
img_res = np.zeros((h,w,c), dtype=np.uint8)

# Copying pixel values from the original image to the new image
for y in range(h):    
    for x in range(w):
        for k in range(c):
            img_res[y,x,0] = img[y,x,0]
            img_res[y,x,1] = img[y,x,1]
            img_res[y,x,2] = img[y,x,2]

# Inverting pixel values to create a negative image
#for y in range(h):
#    for x in range(w):
#        for k in range(c):
#            img_res[y,x,0] = 255 - img[y,x,0] # Inverting the blue channel
#            img_res[y,x,1] = 255 - img[y,x,1] # Inverting the green channel
#            img_res[y,x,2] = 255 - img[y,x,2] # Inverting the red channel

# Inverting pixel values to create a negative image (optimized version)
#for y in range(h):
#    for x in range(w):
#        img_res[y,x,:] = 255 - img_res[y,x,:] # Inverting all channels at once

# Alternatively, you can use NumPy's vectorized operations to invert the image without explicit loops:
img_res = 255 - img

img_res[:] = 255 - img[:]

#for y in range(h):    
#    for x in range(w):
#        for k in range(c):
#            img_res[y,x,0] = img[r_greyscale,0]
#            img_res[y,x,1] = img[g_greyscale,1]
#            img_res[y,x,2] = img[b_greyscale,2]

# Converting the image to greyscale by averaging the RGB channels
#for y in range(h):
#    for x in range(w):
#        img_res[y,x] = (img[y,x,2] + img[y,x,1] + img[y,x,0]) / 3

img_res = ((1/3) * (img[:,:,0] + img[:,:,1] + img[:,:,2])) / 255

# Displaying image properties
print("Shape:", img.shape)
print("Data type:", img.dtype)
print("Number of dimensions:", img.ndim)

cv2.imshow('Image', img)
#cv2.imshow('Image Resized', img_res)

# Wait for a key press indefinitely or for a specified amount of time in milliseconds
#cv2.waitKey(0)
# Close all OpenCV windows
#cv2.destroyAllWindows()

# Afficher avec matplotlib
#plt.imshow(img)
#plt.show()

# Matplotlib n'affiche pas correctement car il utilise RGB et cv2 utilise GBR (bleu et rouge inversés)

# Correction de l'affichage matplotlib: Conversion de BGR vers RGB

#imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.imshow(imgRGB)
#plt.show()

# Conversion BGR à HSV

imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#plt.imshow(imgHSV)
#plt.show()

lower_green = np.array([35,50,50])
upper_green = np.array([85,255,255])

mask = cv2.inRange(
    imgHSV,
    lower_green,
    upper_green
)

cv2.imshow("Mask", mask)

result = np.zeros_like(img)
result[mask > 0] = img[mask > 0]

cv2.imshow("Result", result)
cv2.waitKey(0)