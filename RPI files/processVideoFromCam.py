import cv2 
# reading the input 
cap = cv2.VideoCapture("test.h264") 
size = (int(cap.get(3)), int(cap.get(4)))
newVidName = "sample.avi"
out = cv2.VideoWriter(newVidName, cv2.VideoWriter_fourcc(*'MPEG'), 30, size) 
# init variables 
frameNum = 290
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vidName = "test.h264"
padding = 5
lastKnownCrop = (10, )
def faceDetection(img):
    return faceCascade.detectMultiScale(
        img,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

def processImage(img):
    rects = faceDetection(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # add error handling for multiple faces 

    cv2.rectangle(img, (200, 200), (550, 600), (255, 0, 0), 3) 

    for (x,y,w,l) in rects: 
        cv2.rectangle(img, (x, y), (x+w, y+l), (255, 0, 0), 3) 
        
        return img[y-padding: y+l+padding, x-padding: x+w+padding]
    return img[40: -40, 50:-50]

while(True): 
    ret, frame = cap.read() 
    if(ret): 
        
        # adding rectangle on each frame 
        
        newFrame = processImage(frame)


        # writing the new frame in output 
        out.write(frame) 
    else:
        break

cv2.destroyAllWindows() 
print(f"new file saved as {newVidName}")
out.release() 
cap.release() 


