from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=10,
    help="max buffer size")
args = vars(ap.parse_args())

greenLower = (21, 58, 113)
greenUpper = (44, 255, 249)

counter = 0
(dX, dY) = (0, 0)

oldy=0
newy=0
flagnew=""
flagold=""

direction = ""
n=0
dirY = ""
dir="null"

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])
    
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('output.avi',fourcc, 5.0, (500,909),True)
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break
    frame = imutils.resize(frame, width=500)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    
    flagold = flagnew
    oldy = newy
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            newy=y
            
      
    if(newy > oldy):
        flagnew="down"
    else:
        flagnew="up"
        
    if(flagnew=="up" and flagold=="down"):
        n+=1
        
    
    
    direction = "{}".format(n)
        

# =============================================================================
#     for i in np.arange(1, len(pts)):
#         if pts[i - 1] is None or pts[i] is None:
#             continue
# 
#         if counter >= 10 and i == 1 and pts[-10] is not None:
#             dY = pts[-10][1] - pts[i][1]
#             
#             dirY_old = dirY
#             
#             if np.abs(dY) > 10:
#                 dirY = "North" if np.sign(dY) == 1 else "South"
#             
#             if dirY != dirY_old:
#                 n+=1
# 
#             # handle when both directions are non-empty
#             if dirY != "":
#                 direction = "{} , {}".format(dirY,n)
#         thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
#         cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
# =============================================================================

    # show the movement deltas and the direction of movement on
    # the frame
    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 3)
    cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)

    # show the frame to our screen and increment the frame counter
    cv2.imshow("Frame", frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

# otherwise, release the camera
else:
    vs.release()

# close all windows
out.release()
cv2.destroyAllWindows()
