#!/usr/bin/python3
from threading import Condition
from time import sleep
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from picamera2 import MappedArray, Picamera2, Preview
import cv2
picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

# init variables 
frameNum = 75
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vidName = "test.h264"
padding = 40
lastKnownCrop = (300,200, 250, 250)

def faceDetection(img):
    return faceCascade.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80),
        maxSize=(img.shape[0] - 2 * padding, img.shape[1] - 2 * padding)
    )

def processImage(img):
    global lastKnownCrop
    rects = faceDetection(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # add error handling for multiple faces 
    for (x,y,w,l) in rects: 
        cv2.rectangle(img, (x, y), (x+w, y+l), (255, 0, 0), 3) 
        print("rectCoords", x,y,w,l)
        lastKnownCrop = (x,y,w,l)
    return img[lastKnownCrop[1]-padding: lastKnownCrop[1]+lastKnownCrop[3]+padding,
                lastKnownCrop[0]-padding: lastKnownCrop[0]+lastKnownCrop[2]+padding]

class FileOutputStop(FileOutput):
    def __init__(self, file=None, pts=None, split=None):
        super().__init__(file, pts, split)
        self.frame_counter = 0
        print("recording started")
        self.cond = Condition()

    def _write(self, frame, timestamp=None):
        self.frame_counter += 1
        # Stop capturing when 75 frames is reached
        if self.frame_counter <= frameNum:
            super()._write(frame, timestamp)
            if self.frame_counter == frameNum:
                with self.cond:
                    self.stop()
                    self.cond.notify()


def recordVid():
    encoder = H264Encoder(10000000)
    encoder.output = FileOutputStop(vidName)
    picam2.start_encoder(encoder)
    picam2.start()

    # Wait for 75 frames to be captured
    with encoder.output.cond:
        encoder.output.cond.wait()
    
    print("captured") 
    picam2.stop()


def generateFaceVideo():
    # start going through the saved video 
    cap = cv2.VideoCapture(vidName)

    size = (int(cap.get(3) / 2), int(cap.get(4) / 2))
    print("video size:", size)

    # videowriter initializing
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MPEG'), 30, size)

    frameNum = 1
    while(True):
        # get start reading the video  
        ret, frame = cap.read() 
        if (ret):
            # track the frames
            print("frame:", frameNum)
            frameNum = frameNum+1

            # resize the frames and save the current frame 
            cv2.imwrite("currentFrame.png", frame)

            # process the frames and save that 
            frame = processImage(frame)
            print("frameSize", frame.shape)
            print("setSize", size)
            frame = cv2.resize(frame, size)

            cv2.imwrite("currentProcessedFrame.png", frame)

            # add to the video 
            out.write(frame)

        else: 
            break
    cv2.destroyAllWindows() 
    out.release()
    cap.release()
    print("released")

recordVid()
generateFaceVideo()