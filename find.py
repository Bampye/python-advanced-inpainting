import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

img_rgb = cv2.imread('i.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
test = cv2.Canny(img_rgb, 30, 200)

img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
# img_gray = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
# ret,img_gray = cv2.threshold(img_gray,127,255,1)
# ret,img_gray = cv2.threshold(img_gray,127,255,1)
img_gray = cv2.Canny(img_gray, 30, 200)
# img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

cv2.imshow("img_bgr", img_rgb)
cv2.imshow("img_gray", img_gray)



template_bgr = cv2.imread('_/t.jpg')
template_ori = cv2.imread('_/t.jpg',0)
template = cv2.Canny(template_ori, 30, 200)
w, h = template.shape[::-1]

# Apply template Matching
# res = cv2.matchTemplate(test,template,cv2.TM_CCOEFF_NORMED)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# top_left = max_loc
# bottom_right = (top_left[0] + w, top_left[1] + h)

# cv2.rectangle(img_rgb,top_left, bottom_right, (0,255,0), 2)

# cv2.imshow('img_rgb', img_rgb)
# cv2.waitKey(0)

img2 = img_rgb.copy()

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED']

for meth in methods:
	img_gray1 = img2.copy()
	img_gray = cv2.Canny(img_gray1, 30, 200)

	method = eval(meth)

	# Apply template Matching
	res = cv2.matchTemplate(img_gray,template,method)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	print("matchTemplate: " , max_val)


	# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)

	cv2.rectangle(img_gray1,top_left, bottom_right, (0,0,255), 2)

	cv2.imshow(meth, img_gray1)

	print("Method: " , meth)
	# print("min_val: " , min_val)
	# print("max_val: " , max_val)
	# print("min_loc: " , min_loc)
	# print("max_loc: " , max_loc)

	box = img2[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
	box_gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
	box_blur = cv2.GaussianBlur(box_gray, (3, 3), 0)
	box_canny = cv2.Canny(box_blur, 30, 200)


	# load the image, convert it to grayscale, and blur it slightly
	gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)

	# apply Canny edge detection using a wide threshold, tight
	# threshold, and automatically determined threshold
	wide = cv2.Canny(blurred, 10, 200)
	tight = cv2.Canny(blurred, 225, 250)
	# auto = auto_canny(blurred)

	# compute the median of the single channel pixel intensities
	v = np.median(blurred)
	sigma=0.33

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(blurred, lower, upper)

	# cv2.imshow("edged_canny" + meth, edged)
	cv2.imshow("edged_canny" + meth, np.hstack([wide, tight, edged]))



	i1 = template
	i2 = box_canny

	cv2.imshow("template" + meth, i1)
	cv2.imshow("cropped" + meth, i2)

	#

	dist_ncc = np.sum( (i1 - np.mean(i1)) * (i2 - np.mean(i2)) ) / ((i1.size - 1) * np.std(i1) * np.std(i2) )
	print("dist_ncc: " ,dist_ncc)

cv2.waitKey(0)
