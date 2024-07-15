#!/usr/bin/python3
from threading import Condition
import time
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from picamera2 import Picamera2
import cv2
import cropFuncs


picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

# init variables 
frameNum = 290
vidName = "recordedVideo.h264"
processedFaceVidName = "processedVideo.avi"

class FileOutputStop(FileOutput):
    def __init__(self, file=None, pts=None, split=None):
        super().__init__(file, pts, split)
        self.frame_counter = 0
        print(f"recording started, capturing {frameNum} frames")
        self.cond = Condition()

    def _write(self, frame, timestamp=None):
        self.frame_counter += 1
        # Stop capturing when frameNum of  frames is reached
        if self.frame_counter <= frameNum:
            super()._write(frame, timestamp)
            # print(f"frame number {self.frame_counter} written")
            if self.frame_counter == frameNum:
                with self.cond:
                    self.stop()
                    self.cond.notify()

def recordVid(numFrames=frameNum):
    frameNum = numFrames
    encoder = H264Encoder(10000000)
    encoder.output = FileOutputStop(vidName)
    picam2.start_encoder(encoder)
    picam2.start()

    # Wait for {numFrames} frames to be captured
    with encoder.output.cond:
        encoder.output.cond.wait()
    
    print("video recorded") 
    picam2.stop()

def generateFaceVideo():
    # start going through the saved video 
    cap = cv2.VideoCapture(vidName)

    size = (120, 40)

    print("video sizes:", size)

    # videowriter initializing
    out = cv2.VideoWriter(processedFaceVidName, cv2.VideoWriter_fourcc(*'MPEG'), 30, size)

    frameNum = 1
    while(True):
        # get start reading the video  
        ret, frame = cap.read() 
        if (ret):
            # track the frames
            # print("frame:", frameNum)
            frameNum = frameNum+1

            #  save the current frame 
            # cv2.imwrite("currentFrame.png", frame)

            # process the frames and save that 
            frame = cropFuncs.cropForMouth(frame)
            frame = cv2.resize(frame, size)
            # cv2.imwrite("currentProcessedFrame.png", frame)

            # write to the videos
            out.write(frame)

        else: 
            break
    cv2.destroyAllWindows() 

    out.release()
    cap.release()
    print("all captures released")

# start = time.time()
# recordVid()
# # print(f"video recorded, saved as {vidName}")
# # print(f"took {start - time.time()} seconds")

# start = time.time()
# generateFaceVideo()
# print(f"video recorded, saved as {processedFaceVidName}")
# print(f"took {start - time.time()} seconds")
