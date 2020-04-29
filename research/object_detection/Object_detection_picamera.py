######## Picamera Card Detection Using Tensorflow ########
######## Based off of Evan Juras's Object Detection Using Tensorflow Classifier #########
# Description:
# This program uses a TensorFlow classifier to perform card detection.
# It loads the classifier uses it to perform card detection on frames from a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. 

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py


# Import packages
import io
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import sys
import serial
from threading import Timer
import time


# create and open a serial port
sp = serial.Serial('/dev/ttyUSB0', 9600)

# set the arm to default centered position
def arm_calibrate():
    print("calibrating arm")
    #sp.write("#0 P1500\r".encode())                                                                                                                                      #sp.write("#1 P1700\r".encode())
    sp.write("#2 P200\r".encode())
    sp.write("#3 P1300\r".encode())
    sp.write("#4 P1500\r".encode())
    #sp.write("#4 P1500\r".encode())

def arm_slap():
    print("slapping!")
    # USE most left sp.write("#1 P700\r".encode())                                                                                                                        # most right sp.write("#1 P2300\r".encode())                                                                                                                          #sp.write("#1 P1500\r".encode())                                                                                                                                      #sp.write("#1 P900\r".encode())
    sp.write("#2 P1700 T500\r".encode())


    #sp.write("#4 P1500 S1000 T5000\r".encode())                                                                                                                          #sp.write("#5 P1500 S1000 T5000\r".encode())                                                                                                                          #sp.write("#3 P1500 S1000 T5000\r".encode())                                                                                                                          #sp.write("#3 P1500 S1000 T5000\r".encode())                                                                                                                          #sp.write("#4 P1500\r".encode())                                                                                                                                      #sp.write("#5 P1500\r".encode())
    time.sleep(5)

    # recalibrate
    print("recalibrating arm")
    #sp.write("#0 P1500\r".encode())                                                                                                                                      #sp.write("#1 P1700\r".encode())
    sp.write("#2 P200\r".encode())
    sp.write("#3 P1300\r".encode())
    sp.write("#4 P1500\r".encode())
    time.sleep(5)



# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

print("naming model...")

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'card_model'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','card_labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 13

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print("Loading tf model...")

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.

print("Starting camera loop...")
# Initialize Picamera and grab reference to the raw capture
camera = PiCamera()
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.framerate = 10 #10 frames per second
rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
rawCapture.truncate(0)
    
counter = 0
bottomCard = None
prevCard = None
prevprevCard = None


def detect():

    print("detected")
    
    stream = io.BytesIO()
    camera.capture(stream, format = 'jpeg')
    data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    frame =cv2.imdecode(data,1)
    
    global bottomCard
    global prevCard
    global prevprevCard
    global counter
    global frame_rate_calc
    global freq
    global font
 

    # for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

    t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
   # frame = np.copy(frame1.array)
    frame.setflags(write=1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
            
    # add game logic
    # print classes
    print(np.squeeze(classes[0, 0]).astype(np.int32)) # should only have one card and they are strings
    print(np.squeeze(scores[0, 0]))
        
    if num != 0:#if a card is detected
        currCard = classes[0,0]


        if bottomCard == None:
            #first card placed down
            bottomCard = classes[0, 0]
            print("set bottom card")
        elif currCard == prevCard:
            print("doubles")
            arm_slap()
            # reset
            counter = 0
            bottomCard = None
            prevCard = None
            prevprevCard = None
            
        elif counter >= 2 and currCard == prevprevCard:
            print("sandwich")
            arm_slap()
            # reset
            counter = 0
            bottomCard = None
            prevCard = None
            prevprevCard = None

            
        elif currCard == bottomCard:
            print("top bottom")
            arm_slap()
            # reset
            counter = 0
            bottomCard = None
            prevCard = None
            prevprevCard = None

        counter += 1
        if counter > 1:
            prevprevCard = prevCard
        prevCard = classes[0, 0]

            
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
              line_thickness=8,
              min_score_thresh=0.40)

    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    # cv2.imshow('Object detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1


    rawCapture.truncate(0)

    print("resetting timer")
    t = Timer(5.0, detect)
    t.start()
    

t = Timer(5.0, detect)
t.start()
print("first time timer")
arm_calibrate()

# Press 'q' to quit
if cv2.waitKey(1) == ord('q'):
    camera.close()
    cv2.destroyAllWindows()

