
import cv2 
# import tflite_runtime.interpreter as tflite

import numpy as np 
import cropFuncs
import tensorflow as tf
import time 

def loadVideo(path): 
    cap = cv2.VideoCapture(path)
    global lastKnownCrop
    processedFrames = []
    # for each frame 
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        _, frame = cap.read()
        
        # in case a frame is missing, just continue
        if frame is None or frame.shape[0] == 0: 
            continue

        grayFrame = tf.image.rgb_to_grayscale(frame)
        processedFrames.append(grayFrame)
    cap.release()    

    mean = tf.math.reduce_mean(processedFrames, keepdims=True)
    std = tf.math.reduce_std(tf.cast(processedFrames, tf.float32), keepdims=True)
    frames = tf.cast(processedFrames, tf.float32)
    normalizedFrames = (tf.cast(frames, tf.float32) - tf.cast(mean, tf.float32)) / tf.cast(std, tf.float32)
    return normalizedFrames

def padTensor(tensor, target_shape, padding_value=0):
  pad_width = [(0, max(target_shape[i] - tensor.shape[i], 0)) for i in range(len(target_shape))]
  return np.pad(tensor, pad_width, mode='constant', constant_values=padding_value)

def makePrediction(vidPath):
    # vid processing 
    inputVid = loadVideo(vidPath)
    paddedVid  = padTensor(inputVid, target_shape = (290, 40, 120, 1))
    batchedPaddedVid = paddedVid.reshape((1,) + paddedVid.shape)  # Add a dimension of size 1
    
    # making a prediction 
    inputData = np.array(batchedPaddedVid, dtype=np.float32) # make data into a np arr
    interpreter.set_tensor(0, inputData)
    interpreter.invoke() # "prediction" 
    
    # decode the final ctc decoded tensor 
    outputDetails = interpreter.get_output_details()
    outputData = interpreter.get_tensor(outputDetails[0]['index'])

    decoded = tf.keras.backend.ctc_decode(outputData, input_length=[290], greedy=True)[0][0].numpy()
    return  tf.strings.reduce_join(numToChar(decoded[0])).numpy().decode('utf-8')

vocab = [x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ "]
charToNum = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
numToChar = tf.keras.layers.StringLookup(vocabulary=charToNum.get_vocabulary(), oov_token="", invert=True)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="GDRIVE/finalModelCUSTOM.tflite")
interpreter.allocate_tensors()


# path = "../lipreading/GDRIVE/customVids/5570920046221178499_00015.mp4"
# print(makePrediction(path))