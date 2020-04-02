import cv2
import numpy as np
import time

from .utility_functions import images_normalize



def video_start(device = 0, tuResolution =(320, 300), nFramePerSecond = 30):
	"""
	Returns videocapture object/stream
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

def video_capture(oStream, tuRectangle = (320, 300), nTimeDuration =4 ):
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

def capture_frames(tuResolution=(320, 300)):
	"""
	capture live frames from webcam.
	"""
	# construct videoCapture object
	oStream = video_start(tuResolution=(320, 300))
	# capture live feeds from webcam
	tm, video_frames = video_capture(oStream)
	   
	return tm, video_frames
	
def predict_from_camera(trained_model, nTargetFrames, nHeight, nWidth, bRescale=True):
	"""
	perform predictions using live stream from webcam
	"""
	# infinite loop
	while True:
		# capture live feed
		tuResolution = (nHeight, nWidth)
		tm, video_frames = capture_frames(tuResolution=tuResolution)
		# process frames
		video_frames = images_normalize(video_frames, nTargetFrames, nHeight, nWidth, bRescale=bRescale)
		# predict from live feeds
		prediction = trained_model.predict([video_frames])

		# print outcome
		print(prediction[0])

# if __name__ == "__main__":
#    tm, video_frames = capture_frames()
#    print(tm, video_frames.shape)