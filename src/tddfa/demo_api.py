import cv2

from tddfa.apis.facial_landmark_detection import FacialLandmarkDetector
from tddfa.utils.config import LandmarkDetectorConfig
from tddfa.utils.functions import cv_draw_landmark

detector = FacialLandmarkDetector(LandmarkDetectorConfig.default_config(), use_onnx=True)
file = "/Users/mvu/Documents/PhD/LeftVideoSN001_comp.avi"


def read_cv2(vid_file):
    vidcap = cv2.VideoCapture(vid_file)
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            yield image


for frame, landmarks in detector.tracking_and_detect(stream_reader=read_cv2(file)):

    cv2.imshow('image', cv_draw_landmark(frame, landmarks[0]))
    k = cv2.waitKey(20)
    if k & 0xff == ord('q'):
        break