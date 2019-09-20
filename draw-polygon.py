# Image annotation for creating training and test sets

# import the necessary packages
import argparse
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
num = 3
refPt = []
stored = []


def click_and_crop(event, x, y, flags, param):
    global refPt, stored

    # check to see if the left mouse button was released
    if event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        if len(refPt) != 1:
            cv2.line(image, refPt[-2], refPt[-1], (0, 255, 0), 1)
    if event == cv2.EVENT_MBUTTONUP:
        stored.append(refPt)
        refPt = []
    
    

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
# clone = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
        refPt = []
        stored = []

    # if the 'q' key is pressed, break from the loop
    elif key == ord("q"):
        break


outfile = open(args["image"].split(".")[0] + ".txt", "w")
for i in stored:
    for j in i:
        outfile.write("(%d, %d)" %(j[1],j[0])) # have to flip the x and y coordinates, as they are opposite in skimage (numpy) and opencv2
        outfile.write("\t")
    outfile.write("\n")
outfile.close()


# close all open windows
cv2.destroyAllWindows()
