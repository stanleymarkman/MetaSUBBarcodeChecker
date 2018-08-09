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

    showImageWindow(gradient)

def showImageWindow(im):
    cv2.namedWindow("Barcode", cv2.WINDOW_NORMAL)
    imS = cv2.resize(im, (0,0), fx=0.5, fy=0.5) 
    cv2.imshow("Barcode", imS)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
