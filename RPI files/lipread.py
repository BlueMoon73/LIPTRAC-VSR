import vidProcessing
from modelPredictions import makePrediction
import time 

numFrames =  75
start1 = time.time()
start2 = time.time()

print("=" * 100)
print("=" * 100)
vidProcessing.frameNum = numFrames
print(vidProcessing.frameNum)
vidProcessing.recordVid()
print(f"Took {time.time() - start1} seconds to record")
start1 = time.time()
print("~" * 75)

vidProcessing.generateFaceVideo()
print(f"Took {time.time() - start1} seconds to generate face video")
start1 = time.time()
print("~" * 75)


path = f"../lipreading/{vidProcessing.processedFaceVidName}"
prediction = makePrediction(path)
print(prediction)
print(f"Took {time.time() - start1} seconds to generate a prediction")


print(f"Took {time.time() - start2 - float(numFrames / 30)} seconds to do everything")
print("=" * 100)
print("=" * 100)


#  source ~/.python-tf/bin/activate