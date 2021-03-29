# importing Libraries

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
	# from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))  
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)
    # initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
	    confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
	    if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
		    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		    (startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
		    (startX, startY) = (max(0, startX), max(0, startY))
		    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
		    face = frame[startY:endY, startX:endX]
		    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		    face = cv2.resize(face, (224, 224))
		    face = img_to_array(face)
		    face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
		    faces.append(face)
		    locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
    if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
    	faces = np.array(faces, dtype="float32")
    	preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
    return (locs, preds)

	
    
# load face detector model from files
prototxtpath = r"C:\Users\anurag shukla\facemask\face_detector\deploy.prototxt"
weightpath = r"C:\Users\anurag shukla\facemask\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtpath, weightpath)

# initialize trained model from 

maskNet = load_model("mask_detector.model")

# initialize video stream
print("activating camera...")
vs = VideoStream(src=0).start()

# looping over the frames

while True:
    frame=vs.read()
    frame = imutils.resize(frame, width=500)
    
    
    # detect faces in the frame and determing if they are wearing mask 
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    # loopover the detected face location and corresponding location
    
    for(box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutmask) = pred
        
        # determine the class label and color we'll use to draw
		# the bounding box and text
        label = "Mask! sahi baat..." if mask > withoutmask else "No Mask! mask pehen varna 2000 ka challan katega"
        color = (0, 255, 0) if label == "Mask! sahi baat..." else (0, 0, 255)

		# include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)
	    

		# display the label and bounding box rectangle on the output
		# frame
        cv2.putText(frame, label, (startX, startY - 10),
	     	cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	    

	# show the output frame
    cv2.imshow("Frame", frame)
	

	# if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) == 13:
        
        break

vs.stop()
cv2.destroyAllWindows()    
    

