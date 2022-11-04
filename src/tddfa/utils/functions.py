# coding: utf-8

__author__ = 'cleardusk'

import math

import numpy as np
import cv2
from math import sqrt
import matplotlib.pyplot as plt

from tddfa.utils import gpa

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos:]


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)

    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box


def plot_image(img):
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    plt.imshow(img[..., ::-1])
    plt.show()


def draw_landmarks(img, pts, style='fancy', wfp=None, show_flag=False, **kwargs):
    """Draw landmarks using matplotlib"""
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))
    plt.imshow(img[..., ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    dense_flag = kwargs.get('dense_flag')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        if dense_flag:
            plt.plot(pts[i][0, ::6], pts[i][1, ::6], 'o', markersize=0.4, color='c', alpha=0.7)
        else:
            alpha = 0.8
            markersize = 4
            lw = 1.5
            color = kwargs.get('color', 'w')
            markeredgecolor = kwargs.get('markeredgecolor', 'black')

            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

            # close eyes and mouths
            plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                 color=color, lw=lw, alpha=alpha - 0.1)
            plot_close(41, 36)
            plot_close(47, 42)
            plot_close(59, 48)
            plot_close(67, 60)

            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

                plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                         color=color,
                         markeredgecolor=markeredgecolor, alpha=alpha)
    if wfp is not None:
        plt.savefig(wfp, dpi=150)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plt.show()


def cv_draw_landmark(img_ori, pts, box=None, color=GREEN, size=1):
    img = img_ori.copy()
    n = pts.shape[1]
    if n <= 106:
        for i in range(n):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, -1)
    else:
        sep = 1
        for i in range(0, n, sep):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, 1)

    if box is not None:
        left, top, right, bottom = np.round(box).astype(np.int32)
        left_top = (left, top)
        right_top = (right, top)
        right_bottom = (right, bottom)
        left_bottom = (left, bottom)
        cv2.line(img, left_top, right_top, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, right_top, right_bottom, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, right_bottom, left_bottom, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, left_bottom, left_top, BLUE, 1, cv2.LINE_AA)

    return img


def read_cv2(vid_file=0):
    """
    Simple wrapper function for getting frames from webcam or video file
    Args:
        vid_file: path to the video file. Passing 0 means using webcam instead
    """
    vidcap = cv2.VideoCapture(vid_file)
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            yield image


def get_landmark_most_points(landmarks):
    min_x, min_y, max_x, max_y = 9999, 9999, 0, 0
    for landmark in landmarks:
        if min_x > landmark[0]:
            min_x = landmark[0]
        if max_x < landmark[0]:
            max_x = landmark[0]
        if min_y > landmark[1]:
            min_y = landmark[1]
        if max_y < landmark[1]:
            max_y = landmark[1]
    return min_x, min_y, max_x, max_y


def to_66_points_standard(lm_68_points):
    """
    Convert the 68 points landmark (3, 68) to standard 66 points landmark (66, 2)
    @param lm_68_points:
    @return:
    """
    landmark_68_points = lm_68_points.transpose()[:, 0:2]
    landmark_66_points = np.delete(landmark_68_points, [60, 64], 0)
    return landmark_66_points


class CentralCrop(object):
    """Crop the image in a sample.
        Make sure the head is in the central of image
    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size, zoom_percent=0.15, close_top=0.6, left=0.5):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.zoom_percent = zoom_percent
        self.close_top = close_top
        self.left = left

    def __call__(self, image, landmarks):
        """
        Crop an image based on its facial landmarks
        @param image: np image
        @param landmarks: 2D landmarks with shape (n_points,2)
        @return: cropped image, adjusted landmarks
        """
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        min_x, min_y, max_x, max_y = get_landmark_most_points(landmarks)
        face_size = max_x - min_x
        if self.zoom_percent == -1:
            distance = new_h
        else:
            gap = face_size * self.zoom_percent
            distance = int(face_size + gap * 2)
            if distance > min(w, h):
                distance = min(w, h)

        x = int(min_x - (distance - (max_x - min_x)) * self.left)
        if x > min_x:
            x = math.floor(min_x)
        if x < 0:
            x = 0
        if x + distance < max_x:
            x = int(math.ceil(max_x) - distance)
        if x + distance > w:
            x = w - distance
        y = int(min_y - (distance - (max_y - min_y)) * self.close_top)
        if y > min_y:
            y = math.floor(min_y)
        if y < 0:
            y = 0
        # if y + distance < max_y:
        #     y = int(math.ceil(max_y) - distance)

        # if y + distance > h:
        #     y = h - distance

        image = image[y: y + distance, x: x + distance].copy()
        height, width = image.shape[:2]
        out_img = np.zeros((distance, distance, 3), dtype=np.uint8)
        out_img[0:height, 0:width] = image

        landmarks = landmarks - np.array([x, y])

        image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_AREA)

        landmarks *= [new_w / distance, new_h / distance]

        return image, landmarks


class GPAAlignment:
    """
    Employ Generalised Procrusters analysis to align both the image and the landmarks, based on the reference
    mean shape. See utils/gpa.py for how to generate the mean shape
    """

    def __init__(self, mean_shape):
        self.mean_shape = mean_shape

    def __call__(self, image, landmarks):
        """
        @param image: np image
        @param landmarks: 2D landmarks with shape (n_points,2)
        @return: aligned image, adjusted landmarks
        """
        _, aligned_landmarks, tform = gpa.procrustes(self.mean_shape, landmarks)

        rotate = np.identity(3)
        rotate[:2, :2] = tform['rotation'].transpose()
        translate = np.identity(3)
        translate[:2, -1] = tform['translation']

        scale = np.identity(3)
        scale[0, 0] = tform['scale']
        scale[1, 1] = tform['scale']

        m_translate = np.dot(translate, scale)
        m = np.dot(m_translate, rotate)

        rows, cols, _ = image.shape
        im2 = cv2.warpAffine(image, m[:-1, :], (cols, rows), flags=cv2.INTER_CUBIC)
        im2 = np.clip(im2, 0, 255)

        return im2, aligned_landmarks
