# coding: utf-8

__author__ = 'cleardusk'

import os
from collections import deque

import imageio
import numpy as np

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.config import LandmarkDetectorConfig


class FacialLandmarkDetector:

    def __init__(self, config: LandmarkDetectorConfig, use_onnx=False, gpu_mode=False, box_valid_threshold=2020):
        cfg = config.config
        self.box_valid_threshold = box_valid_threshold
        if use_onnx:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'

            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX

            self.face_boxes = FaceBoxes_ONNX()
            self.tddfa = TDDFA_ONNX(**cfg)
        else:
            self.tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
            self.face_boxes = FaceBoxes()

    # def single_image_detect(self, image):
    #

    def is_valid_boxes(self, roi_boxes):
        if len(roi_boxes) == 0:
            return False
        roi_box = roi_boxes[0]
        return abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) >= self.box_valid_threshold

    def tracking_and_detect(self, stream='<video0>', n_pre=1, n_next=1, dense_flag=False):
        """
        Grabs frames from stream source and performs the facial landmark detection
        Supports only one face in camera/video

        @param dense_flag:
        @param stream: Source of the stream, can be video file or camera
        @param n_pre: number of frames look back for smoothing
        @param n_next: number of frames look ahead for smoothing
        """
        # Given a camera
        # before run this line, make sure you have installed `imageio-ffmpeg`
        reader = imageio.get_reader(stream)

        # the simple implementation of average smoothing by looking ahead by n_next frames
        queue_ver, queue_frame = deque(), deque()

        # run
        ver = []

        for i, frame in enumerate(reader):
            frame_bgr = frame[..., ::-1]  # RGB->BGR
            first_frame_detected = False

            if len(ver) == 0:
                # the first frame, detect face, here we only use the first face, you can change depending on your need
                boxes = self.face_boxes(frame_bgr)
                boxes = [] if len(boxes) == 0 else [boxes[0]]
                param_lst, roi_box_lst = self.tddfa(frame_bgr, boxes)
                ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
                first_frame_detected = True

            param_lst, roi_box_lst = self.tddfa(frame_bgr, ver, crop_policy='landmark')
            if not first_frame_detected and not self.is_valid_boxes(roi_box_lst):
                boxes = self.face_boxes(frame_bgr)  # Re-detect in case box is invalid
                boxes = [] if len(boxes) == 0 else [boxes[0]]
                param_lst, roi_box_lst = self.tddfa(frame_bgr, boxes)

            ver = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

            if len(ver) == 0:
                yield frame_bgr, ver
                continue

            current_ver = ver[0]

            if first_frame_detected:
                queue_ver.clear()   # Clear previous landmark vector
                queue_frame.clear()     # Clear previous frames
                for _ in range(n_pre + n_next):
                    queue_ver.append(current_ver.copy())
                    queue_frame.append(frame_bgr.copy())

            queue_ver.append(current_ver.copy())
            queue_frame.append(frame_bgr.copy())

            ver_ave = np.mean(queue_ver, axis=0)
            yield frame_bgr[n_pre], [ver_ave]

            queue_ver.popleft()
            queue_frame.popleft()

        for _ in range(n_pre):
            queue_ver.popleft()
            queue_frame.popleft()

        while queue_ver:
            yield queue_frame.popleft(), [queue_ver.popleft()]
