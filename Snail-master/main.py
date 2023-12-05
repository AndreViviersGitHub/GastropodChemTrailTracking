from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from matplotlib import pyplot as plt
import uuid
from SnailModel import *
from Snail import *
import matplotlib.image as mpimg
import math
import tkinter as tk
from tkinter import Button

def predictcontourimg(model, labels, img, c):
    x, y, w, h = cv2.boundingRect(c)
    cropped = img[y:y+h, x:x+w]
    resized_img = cv2.resize(cropped, (64, 64), interpolation=cv2.INTER_CUBIC)
    test_image = resized_img[..., ::-1]
    return predictfromimg(model, test_image, labels)



def drawLine(img, localPts):
    if len(localPts) == 0 or len(localPts) == 1:
        print("list is empty pts")
        return

    x, y = zip(*localPts)
    print(localPts)
    print(x)
    print(y)
    plt.imshow(img)
    plt.axis('off')
    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    # Scatter the individual points
    plt.scatter(x, y, marker=".", color="red", s=20)

    # Draw a line connecting the points
    plt.plot(x, y, color="red", linewidth=1.5)  # Adjust color and linewidth as needed

    plt.show()


def ShowCtrImage(img, c, imgName):
    # x, y, w, h = cv2.boundingRect(c)
    # ctrImage = img[y:y+h, x:x+w]
    # cv2.imshow('img_{}.jpg', ctrImage)

    rect = cv2.boundingRect(c)
    x, y, w, h = rect
    # box = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cropped = img[y: y + h, x: x + w]
    res = cv2.resize(cropped, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("Show Boxes", res)
    path = 'E:\\Snail Images\\RandomBS\\'
    cv2.imwrite(path + imgName + '.jpg', res)


vid_path = "E:\\Snail Runs - Testing\\"
area = "CatBay\\TA\\"
runrep = "2) Cat Bay 12 Jul 2012 T+A - Run 3"
extension = ".mp4"

vid_path_completed = vid_path + area + runrep + extension

def TrackingObjects():

    labels = getlabels()
    model = loadModel()
    #print(labels)
    #train_loss, train_acc, valid_loss, valid_acc, model = trainModel()
    #save_combined_plot(train_loss, train_acc, valid_loss, valid_acc)
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()  # "E:\\snails\\Recordings\\Charters Creek\\10) 26 Mar A+A Cat Bay\\VTS_01_1.VOB",
    ap.add_argument("-v", "--video", default=vid_path_completed,
                    help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=64,
                    help="max buffer size")
    args = vars(ap.parse_args())
    runImg = mpimg.imread('E:\\snails\\Pathways\\July 2012 - Catalina Bay\\T-A\\Run5Rep3!.jpg')
    # define the lower and upper boundaries of the snail
    # ball in the HSV color space, then initialize the
    # list of tracked points
    # 001837 - Snail Upper - 0 24 55
    # 1F3B5F - Snail Lower - 31 59 95

    pts = deque(maxlen=args["buffer"])
    print(args["video"])
    print(args["buffer"])

    vs = cv2.VideoCapture(args["video"])

    # keep looping
    previous_frame = None
    frame_count = 0
    cv2.namedWindow('Thresholds')
    cv2.createTrackbar('LH', 'Thresholds', 0, 255, nothing)
    cv2.createTrackbar('LS', 'Thresholds', 0, 255, nothing)
    cv2.createTrackbar('LV', 'Thresholds', 29, 255, nothing)
    #cv2.createTrackbar('UH', 'Thresholds', 255, 255, nothing)
    #cv2.createTrackbar('US', 'Thresholds', 255, 255, nothing)
    #cv2.createTrackbar('UV', 'Thresholds', 255, 255, nothing)
    cv2.createTrackbar('DI1', 'Thresholds', 0, 255, nothing)
    #cv2.createTrackbar('DI2', 'Thresholds', 5, 255, nothing)
    cv2.createTrackbar('contourSizeMIN', 'Thresholds', 50, 1000, nothing)
    #cv2.createTrackbar('contourSizeMAX', 'Thresholds', 100, 1000, nothing)

    nocontourimg = None
    frameImg = None
    img_rgb = None
    stop = True
    whiteFrame = None
    exportCount = 40000
    snail = None
    video_flag = args.get("video", False)
    while stop:
        frame_count = frame_count + 1
        if ((frame_count % 100) == 0):
            # grab the current frame
            frame = vs.read()

            if whiteFrame is None:
                height, width = int(vs.get(4)), int(vs.get(3))
                whiteFrame = np.zeros((height, width, 4), np.uint8)
                whiteFrame[:, :] = [0, 0, 0, 255]

            # handle the frame from VideoCapture or VideoStream
            frame = frame[1] if video_flag else frame
            if frame is None:
                break


            #lh = cv2.getTrackbarPos('LH', 'Thresholds')
            #ls = cv2.getTrackbarPos('LS', 'Thresholds')
            lv = cv2.getTrackbarPos('LV', 'Thresholds')
            # uh = cv2.getTrackbarPos('UH', 'Thresholds')
            # us = cv2.getTrackbarPos('US', 'Thresholds')
            # uv = cv2.getTrackbarPos('UV', 'Thresholds')
            di1 = cv2.getTrackbarPos('DI1', 'Thresholds')
            #di2 = cv2.getTrackbarPos('DI2', 'Thresholds')

            #contoursizemin = cv2.getTrackbarPos('contourSizeMIN', 'Thresholds')
            #contoursizemax = cv2.getTrackbarPos('contourSizeMAX', 'Thresholds')
            # resize the frame, blur it, and convert it to the HSV
            # color space
            nocontourimg = frame
            blankImg = frame
            #cv2.imshow("nocontourimg", nocontourimg)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Blur the image
            prepared_frame = cv2.GaussianBlur(src=img_rgb, ksize=(5, 5), sigmaX=0)
            #cv2.imshow("blur", prepared_frame)
            #Grayscale the image
            prepared_frame = cv2.cvtColor(prepared_frame, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("grayscale", prepared_frame)
            #Erode it (Increaseing the gap between poinnts.
            prepared_frame = cv2.erode(prepared_frame,  np.ones((di1, di1), "uint8"), iterations=5)
            #cv2.imshow("erode", prepared_frame)

            #define range that we are looking for in grayscale.
            snailRange = cv2.inRange(prepared_frame, lv, 255)
            #cv2.imshow("snailRange", snailRange)
            #Reduce errosion by dilating.
            kernal = np.ones((2, 2), "uint8")
            prepared_frame = cv2.dilate(snailRange, kernal)
            #cv2.imshow("dilate", prepared_frame)
            #cv2.imshow("dilate", prepared_frame)

            if (previous_frame is None):
                # First frame; there is no previous one yet
                previous_frame = prepared_frame
                continue

            # USE THIS WHEN NOT WANTING TO TRACK DIFFERENCES
            NoDifferenceFrame = cv2.bitwise_not(prepared_frame)
            #cv2.imshow("invert", NoDifferenceFrame)

            # USE THIS WHEN ONLY WANTING TO TRACK WHEN DIFFERENCE BETWEEN FRAMES
            #diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
            #cv2.imshow("difference", diff_frame)
            previous_frame = prepared_frame

            contours, _ = cv2.findContours(NoDifferenceFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if 50 < cv2.contourArea(contour) < 2000:
                    M = cv2.moments(contour)
                    cX = int(M["m10"] / M["m00"]) if M["m00"] else 0
                    cY = int(M["m01"] / M["m00"]) if M["m00"] else 0
                    adjusted_cY = cY
                    adjusted_cX = cX

                    pred = predictcontourimg(model, labels, blankImg, contour)
                    if pred == "snail":
                        if snail is None:
                            snail = Snail(adjusted_cX, adjusted_cY, calculate_orientation(M))

                        #print(snail.calculate_distance(cX, adjusted_cY))
                        #circle_position = get_circle_position(cX, cY, snail.get_rotation())
                        # Draw the circle at the adjusted position
                        cv2.circle(nocontourimg, (adjusted_cX, adjusted_cY), 5, (0, 0, 255), -1)
                        x, y, w, h = cv2.boundingRect(contour)
                        h = 15
                        y = y + 30
                        w = 15

                        #cv2.rectangle(nocontourimg, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle around contour

                        #print(snail.calculate_distance(adjusted_cX, adjusted_cY))
                        if 18 < snail.calculate_distance(adjusted_cX, adjusted_cY) <= 40:
                            snail.update_coordinates(adjusted_cX, adjusted_cY)
                            snail.update_rotation_and_position(calculate_orientation(M), 0)
                            #cv2.drawContours(nocontourimg, [contour], -1, (0, 255, 0), 2)
                            # Calculate circle's position based on the snail's rotation

                            #cv2.circle(nocontourimg, (cX, adjusted_cY), 5, (0, 0, 255), -1)

            #print(snail.get_all_positions())
            #cv2.imshow('Contour Detection', nocontourimg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("KEYPRESSED")
                PosProcessingUI(whiteFrame, snail)

    points = snail.get_all_positions()
    PosProcessingUI(whiteFrame, snail)


def rotation_Line(frame, snail):
    significant_positions = snail.significant_rotation_positions()
    print("Significant Positions:", significant_positions)
    drawLine(frame, significant_positions)

def all_points(frame, snail):
    drawLine(frame, snail.get_all_positions())

def PosProcessingUI(frame, snail):
    root = tk.Tk()
    root.title("UI for Snail Processing")

    # Add a button to the window and attach the on_button_press function
    button = Button(root, text="Process Significant Rotations", command=lambda: rotation_Line(frame,snail))
    button.pack(pady=20)

    button2 = Button(root, text="Another Action", command=lambda: all_points(frame,snail))
    button2.pack(pady=20)

    # Run the tkinter main loop
    root.mainloop()

def normalize_orientation(orientation_degrees):
    orientation_radians = math.radians(orientation_degrees % 360)
    return orientation_radians


def get_circle_position(cX, cY, orientation, distance=10):
    normalized_orientation = normalize_orientation(orientation)
    offsetX = distance * math.cos(normalized_orientation)
    offsetY = distance * math.sin(normalized_orientation)

    # Adjust cX and cY to get the position of the circle
    circleX = cX + offsetX
    circleY = cY - offsetY  # minus because image coordinate system is top-left

    return (int(circleX), int(circleY))


def calculate_orientation(M):
    """Calculate the orientation of a contour using its moments."""
    delta_x2 = M['mu20']
    delta_y2 = M['mu02']
    delta_xy = M['mu11']
    angle_rad = 0.5 * math.atan2(2 * delta_xy, delta_x2 - delta_y2)
    angle_deg = math.degrees(angle_rad)

    # Adjust to ensure the angle is positive
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg

def nothing(pos):
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    TrackingObjects()
