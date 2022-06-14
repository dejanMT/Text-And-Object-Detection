import cv2
import math
import numpy as np
import easyocr
from PIL import Image
import matplotlib.pyplot as plt
import keyboard

cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

path = 'Images/stopSign2.jpg'

image = cv2.imread(path)
im = np.array(Image.open(path))

def arrayImage(path):
        return cv2.imread(path)


#Read text from image function
def textRecognition(imgPath):
    img = cv2.imread(imgPath)
    try:
        # perform character recognition
        reader = easyocr.Reader(['en'])
        result = reader.readtext(img)

        # loop over the results
        for (bbox, text, prob) in result:
            # display the OCR'd text and associated probability
            # unpack the bounding box
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            # Draw rectange around text
            cv2.rectangle(img, tl, br, (245, 224, 66), 2)
            cv2.putText(img, text, (tl[0] + 10, tl[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 224, 66), 2)

        cv2.imshow("Test", img)


        for item in result:
            if item[1] == "" or item[1] == " ":
                print("Unable to read text from image")
            else:
                print(item[1])
    except:
        print("Unable to read text from image")


#Detect Objects Function
def objectTextRecognition(imgPath):
    try:
        img = cv2.imread(imgPath)

        #Text Recognition
        try:
            # perform character recognition
            reader = easyocr.Reader(['en'])
            result = reader.readtext(img)

            # loop over the results
            for (bbox, text, prob) in result:
                # display the OCR'd text and associated probability
                # unpack the bounding box
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))
                # Draw rectange around text
                cv2.rectangle(img, tl, br, (245, 224, 66), 2)
                cv2.putText(img, text, (tl[0] + 10, tl[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 224, 66), 2)

            for item in result:
                if item[1] == "" or item[1] == " ":
                    print("Unable to read text from image")
                else:
                    print("text Found: " + item[1])
        except:
            print("Unable to read text from image")

        #Object Recognition
        classNames = []
        classFile = 'Config/coco.names'
        #Extract the words from coco.names and save them to the classNames array
        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = 'Config/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPaths = 'Config/frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightsPaths, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0/ 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        classIds, confs, bbox = net.detect(img, confThreshold = 0.5)
        #print(classIds, bbox)

        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255,0), thickness=2)
            #Since arrays start from 0 we have to -1
            cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        print("Object found: " + classNames[classId-1])
        #Uncomment the following line to see objectRecognition 's "proccess"
        cv2.imshow("Output", img)
    except:
        print("Unable to recognition object/s, make sure the image is clear and not obstructed.")


def nothing(x):  # mock function does nothing , used for the trackbar
    pass




cv2.namedWindow("TrackBar")

im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
img_bgr_copy = im_bgr.copy()

cv2.createTrackbar('Blurring', 'TrackBar', 1, 100, nothing)
Pass = False

while True:
    cv2.imshow('TrackBar', img_bgr_copy)
    factor = cv2.getTrackbarPos('Blurring', 'TrackBar')

    if factor > 0:
        factor = math.ceil(factor * 0.1)
        img_bgr_copy = cv2.blur(im_bgr,(factor, factor))
        cv2.imshow('TrackBar', img_bgr_copy)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    if cv2.getWindowProperty('TrackBar', cv2.WND_PROP_VISIBLE) < 1:
        break

    cv2.imwrite('Images/Cropped/bluredImage.jpg', img_bgr_copy)

    if keyboard.is_pressed("q"):

        img = arrayImage('Images/Cropped/bluredImage.jpg')

        # Filter the image - Noise Reduction
        b, g, r = cv2.split(img)  # get b,g,r
        rgb_img = cv2.merge([r, g, b])  # switch it to rgb

        # Denoising
        dst = cv2.fastNlMeansDenoisingColored(rgb_img, None, 10, 10, 7, 21)
        b, g, r = cv2.split(dst)  # get b,g,r
        rgb_dst = cv2.merge([r, g, b])  # switch it to rgb

        # Filter the image to grayscale and copy the image
        gs = cv2.cvtColor(rgb_dst, cv2.COLOR_RGB2GRAY)
        oriImage = gs.copy()


        def mouse_crop(event, x, y, flags, param):

            # grab references to the global variables
            global x_start, y_start, x_end, y_end, cropping

            # if the left mouse button was DOWN, start RECORDING
            # (x, y) coordinates and indicate that cropping is being
            if event == cv2.EVENT_LBUTTONDOWN:
                x_start, y_start, x_end, y_end = x, y, x, y
                cropping = True

            # Mouse is Moving
            elif event == cv2.EVENT_MOUSEMOVE:
                if cropping == True:
                    x_end, y_end = x, y

            # if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates
                x_end, y_end = x, y
                cropping = False  # cropping is finished

                refPoint = [(x_start, y_start), (x_end, y_end)]

                if len(refPoint) == 2:  # when two points were found
                    roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                    #cv2.imshow("Cropped", roi)
                    cv2.imwrite('Images/Cropped/croppedImage.jpg', roi)
                    objectTextRecognition('Images/Cropped/croppedImage.jpg')

        cv2.setMouseCallback("TrackBar", mouse_crop)

        while True:
            i = img_bgr_copy.copy()

            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("TrackBar", i)

            cv2.waitKey(1)

            if cv2.getWindowProperty('TrackBar', cv2.WND_PROP_VISIBLE) < 1:
                quit()

cv2.destroyAllWindows()
# Made By: dejan_mt
