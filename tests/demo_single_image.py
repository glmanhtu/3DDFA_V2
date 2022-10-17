import argparse
import os.path
from pathlib import Path

import cv2

from tddfa.apis.facial_landmark_detection import FacialLandmarkDetector
from tddfa.utils.config import LandmarkDetectorConfig
from tddfa.utils.depth import depth
from tddfa.utils.functions import get_suffix, draw_landmarks
from tddfa.utils.pncc import pncc
from tddfa.utils.pose import viz_pose
from tddfa.utils.render import render
from tddfa.utils.serialization import ser_to_ply, ser_to_obj
from tddfa.utils.tddfa_util import str2bool
from tddfa.utils.uv import uv_tex

parser = argparse.ArgumentParser(description='The smooth demo of webcam and video')
parser.add_argument('-f', '--img_fp', type=str, default=0, help='Path to input image')
parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
                    choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
parser.add_argument('--output', type=str, default='results', help='Output folder')
parser.add_argument('--onnx', action='store_true', default=False)

args = parser.parse_args()


# Visualization and serialization
dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
old_suffix = get_suffix(args.img_fp)
new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

img = cv2.imread(args.img_fp)
detector = FacialLandmarkDetector(LandmarkDetectorConfig.default_config(), use_onnx=args.onnx)
boxes, ver_lst, param_lst = detector.single_image_detect(img, dense_flag=dense_flag)

wfp = os.path.join(args.output, Path(args.img_fp).stem + new_suffix)

if args.opt == '2d_sparse':
    draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
elif args.opt == '2d_dense':
    draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
elif args.opt == '3d':
    render(img, ver_lst, detector.tddfa.tri, alpha=0.6, show_flag=args.show_flag, wfp=wfp)
elif args.opt == 'depth':
    # if `with_bf_flag` is False, the background is black
    depth(img, ver_lst, detector.tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
elif args.opt == 'pncc':
    pncc(img, ver_lst, detector.tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
elif args.opt == 'uv_tex':
    uv_tex(img, ver_lst, detector.tddfa.tri, show_flag=args.show_flag, wfp=wfp)
elif args.opt == 'pose':
    viz_pose(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=wfp)
elif args.opt == 'ply':
    ser_to_ply(ver_lst, detector.tddfa.tri, height=img.shape[0], wfp=wfp)
elif args.opt == 'obj':
    ser_to_obj(img, ver_lst, detector.tddfa.tri, height=img.shape[0], wfp=wfp)
else:
    raise ValueError(f'Unknown opt {args.opt}')
