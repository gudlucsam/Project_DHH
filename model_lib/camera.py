import cv2

class VideoCamera(object):
    def __init__(self, device = 0, tuResolution =(320, 300), nFramePerSecond = 30):
        self.frames = []
        self.video = cv2.VideoCapture(device)
        if not self.video.isOpened():
            # try again with inbuilt camera
            print("Try to initialize inbuilt camera ...")
            device = 0
            self.video = cv2.VideoCapture(device)
            if not self.video.isOpened(): raise ValueError("Could not open webcam")

        # set camera resolution
        nWidth, nHeight = tuResolution
        self.video.set(3, nWidth)
        self.video.set(4, nHeight)

        # try to set camera frame rate
        self.video.set(cv2.CAP_PROP_FPS, nFramePerSecond)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), image