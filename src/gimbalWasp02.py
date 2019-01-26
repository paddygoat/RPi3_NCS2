import cv2
import time
import numpy
import random
from multiprocessing import Process
from multiprocessing import Queue
from picamera.array import PiRGBArray
from picamera import PiCamera
from adafruit_servokit import ServoKit

kit = ServoKit(channels=8)
kit.servo[0].angle = 110    # y
kit.servo[1].angle = 90    # x 
#hacked from:
#https://software.intel.com/articles/OpenVINO-Install-RaspberryPI
#https://opencv2-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
#https://github.com/PINTO0309/MobileNet-SSD-RealSense/blob/master/SingleStickSSDwithUSBCamera_OpenVINO_NCS2.py
#https://raspberrypi.stackexchange.com/questions/87062/overhead-counter

#Les Wright Dec 24 2018 
#modified to support picam 30 Dec 2018
#Robot code incorportated on 17 Jan 2019
count = 0
count2 = 0
def motion(xminQueue,xmaxQueue,yminQueue,ymaxQueue):
    
    def left(stime):
        myXAngle = myXAngle -1
        kit.servo[1].angle = myXAngle
        sustain(stime)

    def right(stime):
        myXAngle = myXAngle +1
        kit.servo[1].angle = myXAngle

    def sustain(stime):
        time.sleep(stime) 
        stop()

    def stop():
        kit.servo[1].angle = 90

    def hunt():
            right(0.2)
            stop()

    stop()

    start = time.time() #start a timer
    myXAngle = 90
    myYAngle = 110
    while True:
        if not xminQueue.empty():
            xmin = xminQueue.get()
            xmax = xmaxQueue.get()
            #print(str(xmin)+' '+str(xmax))

            midpointX = (xmin+xmax)/2
            width = xmax-xmin

            #print("W:"+str(width))
            
            #stime = 1        # seconds
            #sustain(stime)
            #stime = abs(150-midpointX)/3000

            #print(str(stime))
            #align midoint with middle of the frame

            if midpointX < 140:
                myXAngle = myXAngle +0.1
                kit.servo[1].angle = myXAngle

            if midpointX > 160:
                myXAngle = myXAngle -0.1
                kit.servo[1].angle = myXAngle
                
            start = time.time() #reset the timer
            
        if not yminQueue.empty():
            ymin = yminQueue.get()
            ymax = ymaxQueue.get()
            #print(str(xmin)+' '+str(xmax))

            midpointY = (ymin+ymax)/2
            width = ymax-ymin
            #print("M:"+str(midpointY))
            #print("W:"+str(width))
            #print("Mx: "+str(midpointX) + "   My: " +str(midpointY)  )
            #stime = abs(150-midpointY)/3000
            #sustain(stime)
            #print(str(stime))
            #align midoint with middle of the frame

            if midpointY < 140:
                myYAngle = myYAngle -0.2
                kit.servo[0].angle = myYAngle

            if midpointY > 160:
                myYAngle = myYAngle +0.2
                kit.servo[0].angle = myYAngle
                
            start = time.time() #reset the timer


        if xminQueue.empty():
            seconds = time.time()-start
            if seconds > 0.8: #if we are empty for longer than 0.8 sec, we probably lost the target...
                #print('Hunting...')
                #hunt()
                start = time.time() #reset the timer
    


# initialize the input queue (frames), output queue (out),
# and the list of actual detections returned by the child process
xminQueue = Queue(maxsize=1)
xmaxQueue = Queue(maxsize=1)
yminQueue = Queue(maxsize=1)
ymaxQueue = Queue(maxsize=1)
# construct a child process indepedent from our main process

print("[INFO] starting motion handling process...")
p2 = Process(target=motion, args=(xminQueue,xmaxQueue,yminQueue,ymaxQueue))
p2.daemon = True
p2.start()

labels_file = 'models/labels.txt'
with open(labels_file, 'r') as f:
    labels = [x.strip() for x in f]
#print(labels)
    
# Note cv2.dnn.blobFromImage, the size is present in the XML files, we could write a preamble to go get that data,
# Then we dont have to explicitly set it!
# Load the model
net = cv2.dnn.readNet('models/wasp.xml', 'models/wasp.bin')


# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

#Misc vars
font = cv2.FONT_HERSHEY_SIMPLEX
frameWidth = 320
frameHeight = 240
framesPerSec = 24
##frameWidth = 640
##frameHeight = 480
##framesPerSec = 12
secPerFrame = 0.0
detections = 0.0

confThreshold = 0.5



#initialize the camera and grab a reference to the raw camera capture
#well this is interesting, we can closely match the input of the network!
#this 'seems' to have improved accuracy!
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=(320, 240)) 

# allow the camera to warmup
time.sleep(0.1)


#print(labels)


#define the function that handles our processing thread
def classify_frame(net, inputQueue, outputQueue):
# keep looping
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            frame = inputQueue.get()
            resframe = cv2.resize(frame, (300, 300))
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
            #net.setInput(blob)
            #blob = cv2.dnn.blobFromImage(resframe, 0.007843, size=(300, 300),\
            #mean=(127.5,127.5,127.5), swapRB=False, crop=False)
            net.setInput(blob)
            out = net.forward()
            # write the detections to the output queue
            outputQueue.put(out)
            

# initialize the input queue (frames), output queue (out),
# and the list of actual detections returned by the child process
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
out = None

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting inference process...")
p = Process(target=classify_frame, args=(net,inputQueue,outputQueue,))
p.daemon = True
p.start()

print("[INFO] starting capture...(loading model) .....")

#time the frame rate....
start = time.time()
frames = 0
#confidence = 0.00
for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    # Capture frame-by-frame
    frame = frame.array

    # if the input queue *is* empty, give the current frame to
    # classify
    if inputQueue.empty():
        inputQueue.put(frame)

    # if the output queue *is not* empty, grab the detections
    if not outputQueue.empty():
        out = outputQueue.get()


    # check to see if 'out' is not empty
    if out is not None:
    # loop over the detections
        # Draw detections on the frame
        for detection in out.reshape(-1, 7):

            confidence = float(detection[2])
            obj_type = int(detection[1]-1)

            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            #bottle = 4, person = 14 , dog = 11

            #if obj_type == 1: #Our object
            obj_type == 1 #Our object
            if confidence > confThreshold:
                    #bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 255))

                    #label
                cv2.rectangle(frame, (xmin-1, ymin-1),\
                (xmin+70, ymin-10), (0,255,255), -1)
                    #labeltext
                cv2.putText(frame,labels[obj_type]+' '+str(round(confidence,2)),\
                (xmin,ymin-2), font, 0.3,(0,0,0),1,cv2.LINE_AA)
                detections += 1


                xmaxQueue.put(xmax)
                xminQueue.put(xmin)
                ymaxQueue.put(ymax)
                yminQueue.put(ymin)


    # Display the resulting frame
    cv2.putText(frame,'Threshold: '+str(round(confThreshold,1)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 0, 0), 1, cv2.LINE_AA)
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame',frameWidth,frameHeight)
    cv2.imshow('frame',frame)
    
    frames+=1
    
    #for num in range(10):    #to iterate on the factors of the number
    count = count +1
    
    if count == 50:         #to determine the first factor
         end = time.time()
         count2 = count2 +1
         seconds = end-start
         fps = frames/seconds
         print("Avg Frames Per Sec: "+str(fps))
         dts = detections/seconds
         print("Avg detections Per Sec: "+str(dts))
         #print("Current confidence: "+ str(confidence))
         if dts == 0:
             print(".... loading model ....."+str(count2))
         else: i = 0
         print("")
         count = 0

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
    keyPress = cv2.waitKey(1)

    if keyPress == 113:
            break

    if keyPress == 82:
        confThreshold += 0.1

    if keyPress == 84:
        confThreshold -= 0.1

    if confThreshold >1:
        confThreshold = 1
    if confThreshold <0:
        confThreshold = 0


end = time.time()
seconds = end-start
fps = frames/seconds
print("Avg Frames Per Sec: "+str(fps))
dts = detections/seconds
print("Avg detections Per Sec: "+str(dts))


cv2.destroyAllWindows()
GPIO.cleanup()


