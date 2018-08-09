#Created: Stanley Markman 8/9/18 WCMC
#Demonstration of sample code to read from MetaSUB sample barcode images
import numpy as np
import argparse
import imutils
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True,
    	help = "path to the image file")
    args = vars(ap.parse_args())

    # load the image and convert it to grayscale
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (50, 50))
    (_, thresh) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)

    # draw a bounding box arounded the detected barcode
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    showImageWindow(image)


#Shows image at half size, waits for any keystroke
def showImageWindow(im):
    cv2.namedWindow("Barcode", cv2.WINDOW_NORMAL)
    imS = cv2.resize(im, (0,0), fx=0.5, fy=0.5)
    cv2.imshow("Barcode", imS)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()