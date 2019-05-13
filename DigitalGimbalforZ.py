def locateFeatures(inImg):
    
    fast = cv2.FastFeatureDetector_create()# find and draw the keypoints
    fast.setThreshold(60)
    kp  = fast.detect(inImg,None)
    l=len(kp)
    a = np.empty((l,1,2),dtype = np.float32)
    for i in range(l):
        x = kp[i].pt[0]
        y = kp[i].pt[1]
        a[i] = [x,y]
    return a
            
        
def clacFlow(good_old,good_new):
    
    global x,y,Q
    
    transformation = cv2.estimateRigidTransform(good_old, good_new, False)
    
    if(transformation != None):
        #transformation = np.array([[1,1,1],[1,1,1]],dtype = int)
        scaling = np.sqrt(pow(transformation[0,0],2) + pow(transformation[0,1],2))  
        rotation = np.arctan2(transformation[1,0],transformation[0,0])
            
        translation = np.sqrt(pow(transformation[0,2],2) + pow(transformation[1,2],2))
        
        Q = Q + np.arctan2(transformation[1,0],transformation[0,0])
        x = x + transformation[0,2]
        y = y + transformation[1,2]
        
        status = 1
    else:
        print "lost feature continuity. Please restart"
        status = 0    
    return x,y,0,Q,status
   
    
    
def clacFlowS(good_old,good_new):
    
    global xS,yS,qS
    
    transformation = cv2.estimateRigidTransform(good_old, good_new, False)
    
    if(transformation != None):
        #transformation = np.array([[1,1,1],[1,1,1]],dtype = int)
        scaling = np.sqrt(pow(transformation[0,0],2) + pow(transformation[0,1],2))  
        rotation = np.arctan2(transformation[1,0],transformation[0,0])
            
        translation = np.sqrt(pow(transformation[0,2],2) + pow(transformation[1,2],2))
        
        qS = qS + np.arctan2(transformation[1,0],transformation[0,0])
        xS = xS + transformation[0,2]
        yS = yS + transformation[1,2]
        return xS,yS,0,qS
    
    
#________________________________________Main Code Starts__________________
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import imutils as im

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()


camera.framerate = 49 # this parameter value depends upon the frequency of the artificial lighting, in my case it was 50hz and 49 worked fine.
#camera.start_preview()
rawCapture = PiRGBArray(camera,)
camera.resolution = (1280,720)
# allow the camera to warmup
time.sleep(0.01)

blank = np.zeros((368,368),np.uint8)
 
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 15,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

u = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    Sframe =cv2.resize(frame.array,(320,180))
    old_frame = Sframe 
    
    rawCapture.truncate(0)
    u = u+1
    #print("starting")
    if (u == 10 ):
        break
# Take first frame and find corners in it
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    Sframe =cv2.resize(frame.array,(320,180))
    old_frame = Sframe
    rawCapture.truncate(0)
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = locateFeatures(old_gray)
    print len(p0)
    if len(p0)<30:
        continue
    break


m=1
x,y,z_old,Q = 0,0,0,0
xS,yS,qS =0,0,0
# variable for change in the angle 
old_angX = 0.0
old_angY = 0.0
angX = 0.0
angY = 0.0

# capture frames from the camera
e = 0
f = 0
xdis = 0
q0 = 0

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    startTime = time.time()
    Sframe =cv2.resize(frame.array,(320,180))
    image = Sframe
     
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray,(5,5),0.3)
    
    
    # calculate optical flow
    if(p0.size != 0):
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        offset = np.array([160,90])
        modifiedOld = good_old - offset
        modifiedNew = good_new - offset
        
        
        X,Y,Z,q,status = clacFlow(modifiedOld,modifiedNew)
        if (status == 0):
            rawCapture.truncate(0)
            continue
        q1 = q
        
        a = q*180/(np.pi)
        print "Angle of rotation is",a
        '''
        print("x",X*Z*0.01)# odometry x
        print("y",Y*Z*0.01)# odometry y
        print("z",Z)
        print("Q",q)
        print "*******************************"
        
        print("xS",xS*Z*0.01)# odometry x
        print("yS",yS*Z*0.01)# odometry y
        print("zS",Z)
        print("qS",qS)
        '''
    else:
        rawCapture.truncate(0)
        continue
    print ("______________________________________-")
    
    blank[94:274,24:344] = frame_gray
    
    rot = im.rotate(blank,angle = a)
    cv2.circle(image,(160,90),1,(0,255,0),-1)
    cv2.imshow('frame',image)
    cv2.imshow('stab',rot)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()

    if(m%10==0):
        pNew = locateFeatures(old_gray)
        print "length of p0 is ",len(pNew)
        if len(pNew <20):
            m=m+1
            rawCapture.truncate(0)
            continue
        else:
            m=m+1
            p0=pNew
        
        # print "yes"

    else:
        p0 = good_new.reshape(-1,1,2)
    print m
    m=m+1
    key = cv2.waitKey(1) & 0xFF
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    timeElapsed = time.time() -startTime
    print("FrameRate",1/timeElapsed)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()