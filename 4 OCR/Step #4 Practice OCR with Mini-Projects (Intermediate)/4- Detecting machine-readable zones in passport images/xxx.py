#python xxx.py -i images
# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())
# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))





# loop over the input image paths
for imagePath in paths.list_images(args["images"]):
	# load the image, resize it, and convert it to grayscale
	image = cv2.imread(imagePath)
	image = imutils.resize(image, height=600)
	cv2.imshow("image",image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imshow("gray",gray)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# smooth the image using a 3x3 Gaussian, then apply the blackhat
	# morphological operator to find dark regions on a light background
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
	cv2.imshow("blackhat",blackhat)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# compute the Scharr gradient of the blackhat image and scale the
	# result into the range [0, 255]
	gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
	cv2.imshow("gradX",gradX)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# apply a closing operation using the rectangular kernel to close
	# gaps in between letters -- then apply Otsu's thresholding method
	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	cv2.imshow("gradX",gradX)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	cv2.imshow("thresh",thresh)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# perform another closing operation, this time using the square
	# kernel to close gaps between lines of the MRZ, then perform a
	# series of erosions to break apart connected components
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
	cv2.imshow("thresh",thresh)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	thresh = cv2.erode(thresh, None, iterations=4)
	cv2.imshow("thresh",thresh)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# during thresholding, it's possible that border pixels were
	# included in the thresholding, so let's set 5% of the left and
	# right borders to zero
	p = int(image.shape[1] * 0.05)
	thresh[:, 0:p] = 0
	thresh[:, image.shape[1] - p:] = 0
	cv2.imshow("thresh",thresh)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# find contours in the thresholded image and sort them by their
	# size
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	# loop over the contours
	for c in cnts:
		# compute the bounding box of the contour and use the contour to
		# compute the aspect ratio and coverage ratio of the bounding box
		# width to the width of the image
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
		crWidth = w / float(gray.shape[1])
		# check to see if the aspect ratio and coverage width are within
		# acceptable criteria
		if ar > 5 and crWidth > 0.75:
			# pad the bounding box since we applied erosions and now need
			# to re-grow it
			pX = int((x + w) * 0.03)
			pY = int((y + h) * 0.03)
			(x, y) = (x - pX, y - pY)
			(w, h) = (w + (pX * 2), h + (pY * 2))
			# extract the ROI from the image and draw a bounding box
			# surrounding the MRZ
			roi = image[y:y + h, x:x + w].copy()
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
			break
	# show the output images
	cv2.imshow("Image", image)
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)
	
	
