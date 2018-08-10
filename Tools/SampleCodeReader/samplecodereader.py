#Created: Stanley Markman 8/9/18 WCMC
#Demonstration of sample code to read from MetaSUB sample barcode images
import numpy as np
import argparse
import imutils
import cv2
import pytesseract

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True,
    	help = "path to the image file")
    args = vars(ap.parse_args())

    # load the image
    image = cv2.imread(args["image"])
    
    #Attempt to create a bounding box using blur factor of 50 and threshold of 100

    box = attemptBoundingBox(image, 50, 90)

    print(box)


    cropped = crop_minAreaRect(image, box)
    cropped = imutils.rotate_bound(cropped, -90)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    (_, cropped) = cv2.threshold(cropped, 74, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    closed = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel)

    closed = closed[250:]

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 2)
    closed = cv2.dilate(closed, None, iterations = 3)
    print("PRESS ANY KEY TO CLOSE IMAGE")
    showImageWindow(closed)
    print("PERFORMING OCR...")
    print(pytesseract.image_to_string(closed))


def attemptBoundingBox(image, blurFactor, threshold):
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
    blurred = cv2.blur(gradient, (blurFactor, blurFactor))
    (_, thresh) = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

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
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    showImageWindow(image)

    return rect

#Shows image at half size, waits for any keystroke
def showImageWindow(im):
    cv2.namedWindow("Barcode", cv2.WINDOW_NORMAL)
    imS = cv2.resize(im, (0,0), fx=1, fy=1)
    cv2.imshow("Barcode", imS)
    cv2.waitKey(0)

#crop image to bounding box
def bboxcrop(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin:ymax+1, xmin:xmax+1]

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop


if __name__ == "__main__":
    main()
