import cv2
import numpy as np
import time

def video_start(device = 0, tuResolution =(320, 240), nFramePerSecond = 30):
	""" Returns videocapture object/stream
	Parameters:
		device: 0 for the primary webcam, 1 for attached webcam
	"""
	
	# try to open webcam device
	oStream = cv2.VideoCapture(device) 
	if not oStream.isOpened():
		# try again with inbuilt camera
		print("Try to initialize inbuilt camera ...")
		device = 0
		oStream = cv2.VideoCapture(device)
		if not oStream.isOpened(): raise ValueError("Could not open webcam")

	# set camera resolution
	nWidth, nHeight = tuResolution
	oStream.set(3, nWidth)
	oStream.set(4, nHeight)

	# try to set camera frame rate
	oStream.set(cv2.CAP_PROP_FPS, nFramePerSecond)

	print("Initialized video device %d, with resolution %s and target frame rate %d" % \
		(device, str(tuResolution), nFramePerSecond))

	return oStream

def video_capture(oStream, tuRectangle = (224, 224), nTimeDuration =4 ):
	liFrames = []
	fTimeStart = time.time()

	# loop over frames from the video file stream
	while True:
		# grab the frame from the threaded video file stream
		(bGrabbed, arFrame) = oStream.read()
		liFrames.append(arFrame)

		fTimeElapsed = time.time() - fTimeStart

		# paint rectangle & text, show the frame
		cv2.imshow("Video", arFrame)

		# stop after nTimeDuration sec
		if fTimeElapsed >= nTimeDuration: break

		# Press 'q' for early exit
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'): break
		cv2.waitKey(1)

	return fTimeElapsed, np.array(liFrames)

if __name__ == "__main__":
   oStream = video_start()
   tm, video_frames = video_capture(oStream)
   print(tm, video_frames.shape)