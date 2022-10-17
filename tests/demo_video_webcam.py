import argparse

import cv2

from tddfa.apis.facial_landmark_detection import FacialLandmarkDetector
from tddfa.utils.config import LandmarkDetectorConfig
from tddfa.utils.functions import cv_draw_landmark
from tddfa.utils.render import render

parser = argparse.ArgumentParser(description='The smooth demo of webcam and video')
parser.add_argument('-i', '--input', type=str, default=0, help='Path to video file')
parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
parser.add_argument('--onnx', action='store_true', default=False)
args = parser.parse_args()


def read_cv2(vid_file):
    vidcap = cv2.VideoCapture(vid_file)
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            yield image


video_reader = read_cv2(args.input)
dense = False if args.opt == '2d_sparse' else True
detector = FacialLandmarkDetector(LandmarkDetectorConfig.default_config(), use_onnx=args.onnx)
for frame, landmarks in detector.tracking_and_detect(video_reader, dense_flag=dense):
    if args.opt in ['2d_sparse', '2d_dense']:
        img_draw = cv_draw_landmark(frame, landmarks[0])
    elif args.opt == '3d':
        img_draw = render(frame, landmarks, detector.tddfa.tri, alpha=0.7)

    cv2.imshow('image', img_draw)
    k = cv2.waitKey(20)
    if k & 0xff == ord('q'):
        break
