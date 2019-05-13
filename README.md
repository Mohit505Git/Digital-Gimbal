# Digital-Gimbal
This project contains algorithm for optical Image Stabilisation or Digital Gimbal using optical flow and obtaining rotation angle using the rigid transform between two consecutive video frame. First version contains the script for a raspberry pi with a raspberry pi camera.
# Digital gimbal for rotation about z axis for a raspberry pi using a raspberry pi camera.

I was inspired to work on this project mainly because of the application and the need for digital image stabilisation on aerial robotics platforms.
A mechanical gimbal is a very bulky and redundant solution for the above purpose. Also, mechanical gimbals are hard to mount on small platforms and consume power which can be used to enhance the drone flight time.

### Prerequisites and Installing

1) One must have a raspberry pi and a raspberry pi camera attached to it(I would recommend version 2 of the camera for best results)
2) Also install Opencv, numpy, imutils,as essential modules for the raspberry pi camera.
  * You can get numpy and imutils using pip.
  * To get basic camera feed see the following and refer to 
  https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/   
```
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
rawCapture = PiRGBArray(camera)
 
# allow the camera to warmup
time.sleep(0.1)
 
# grab an image from the camera
camera.capture(rawCapture, format="bgr")
image = rawCapture.array
 
# display the image on screen and wait for a keypress
cv2.imshow("Image", image)
cv2.waitKey(0)
```

# Important
Camera framerate parameter value depends upon the frequency of the artificial lighting, in my case it was 50hz and 49 worked fine.
Check the frequency for your conutry.
```
camera.framerate = 49
```

That's all
You will see the stabilised image of "Stab" window.

Have a nice day.

For more knowlegde, search the functions used in opencv docs.
## Author

* **Mohit Singh**
