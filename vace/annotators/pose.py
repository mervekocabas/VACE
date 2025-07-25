# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import cv2
import torch
import numpy as np
import pandas as pd
from .dwpose import util
from .dwpose.wholebody import Wholebody, HWC3, resize_image
from .utils import convert_to_numpy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def draw_pose(pose, H, W, use_hand=False, use_body=False, use_face=False):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    if use_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if use_hand:
        canvas = util.draw_handpose(canvas, hands)
    if use_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


class PoseAnnotator:
    def __init__(self, cfg, device=None):
        onnx_det = cfg['DETECTION_MODEL']
        onnx_pose = cfg['POSE_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.pose_estimation = Wholebody(onnx_det, onnx_pose, device=self.device)
        self.resize_size = cfg.get("RESIZE_SIZE", 1024)
        self.use_body = cfg.get('USE_BODY', True)
        self.use_face = cfg.get('USE_FACE', True)
        self.use_hand = cfg.get('USE_HAND', True)

    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image, frame_id=0, input_filename=None):
        image = convert_to_numpy(image)
        input_image = HWC3(image[..., ::-1])
        return self.process(resize_image(input_image, self.resize_size), image.shape[:2], frame_id, input_filename)

    def process(self, ori_img, ori_shape, frame_id=0, input_filename=None):
        ori_h, ori_w = ori_shape
        ori_img = ori_img.copy()
        H, W, C = ori_img.shape
        with torch.no_grad():
            candidate, subset, det_result = self.pose_estimation(ori_img)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1
            
            foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)
            
            self.save_to_csv(pose, float(W), float(H), frame_id, input_filename, self.use_hand)
            
            
            ret_data = {}
            if self.use_body:
                detected_map_body = draw_pose(pose, H, W, use_body=True)
                detected_map_body = cv2.resize(detected_map_body[..., ::-1], (ori_w, ori_h),
                                               interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_body"] = detected_map_body

            if self.use_face:
                detected_map_face = draw_pose(pose, H, W, use_face=True)
                detected_map_face = cv2.resize(detected_map_face[..., ::-1], (ori_w, ori_h),
                                               interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_face"] = detected_map_face

            if self.use_body and self.use_face:
                detected_map_bodyface = draw_pose(pose, H, W, use_body=True, use_face=True)
                detected_map_bodyface = cv2.resize(detected_map_bodyface[..., ::-1], (ori_w, ori_h),
                                                   interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_bodyface"] = detected_map_bodyface

            if self.use_hand and self.use_body and self.use_face:
                detected_map_handbodyface = draw_pose(pose, H, W, use_hand=True, use_body=True, use_face=True)
                detected_map_handbodyface = cv2.resize(detected_map_handbodyface[..., ::-1], (ori_w, ori_h),
                                                       interpolation=cv2.INTER_LANCZOS4 if ori_h * ori_w > H * W else cv2.INTER_AREA)
                ret_data["detected_map_handbodyface"] = detected_map_handbodyface

            # convert_size
            if det_result.shape[0] > 0:
                w_ratio, h_ratio = ori_w / W, ori_h / H
                det_result[..., ::2] *= h_ratio
                det_result[..., 1::2] *= w_ratio
                det_result = det_result.astype(np.int32)
            
            return ret_data, det_result
        
    def save_to_csv(self, pose, W, H, frame_id, input_filename=None, use_hand=False):
        # Get the body data from the pose dictionary
        body = pose['bodies']['candidate']  # shape: (72, 2) - 4 people * 18 keypoints
        face = pose['faces']
        if use_hand:
            hands = pose['hands']
        subset = pose['bodies']['subset']   # shape: (4, 18) - 4 people, 18 keypoints each
        
        # Create data for DataFrame
        data = []
        num_people = subset.shape[0]  # from subset shape
        num_keypoints = subset.shape[1]  # from subset shape
        num_facepoints = face.shape[1]
        num_facepeople = face.shape[0]
        if use_hand:
            num_handpoints = hands.shape[1]
            num_handpeople = hands.shape[0]
        
        for person_id in range(num_people):
            for keypoint_id in range(num_keypoints):
                # Get x,y coordinates for this person's keypoint
                idx = person_id * num_keypoints + keypoint_id
                x, y = body[idx]
                subset_val = subset[person_id, keypoint_id]
                
                data.append([
                    frame_id,  # frame number
                    person_id,  # body number (0-3)
                    keypoint_id,  # keypoint number (0-17)
                    x,  # keypoint x
                    y,  # keypoint y
                    subset_val,  # subset value
                ])
        
        if use_hand:   
            # Each person has 2 hands: left and right
            num_handpeople = hands.shape[0] // 2
            num_handpoints = hands.shape[1]

            for person_id in range(num_handpeople):
                left_hand = hands[2 * person_id]
                right_hand = hands[2 * person_id + 1]

                # Concatenate both hands into 42 keypoints
                full_hand = np.vstack([left_hand, right_hand])  # shape: (42, 2)
                for keypoint_id, (x, y) in enumerate(full_hand):
                    data.append([
                        frame_id,        # frame number
                        person_id,       # body_id
                        keypoint_id + 95,     # keypoint number (0–41)
                        x,
                        y,
                    ])

        for person_id in range(num_facepeople):
            for keypoint_id in range(num_facepoints):
                # Get x,y coordinates for this person's keypoint
                face_x, face_y = face[person_id][keypoint_id]
                data.append([
                    frame_id,  # frame number
                    person_id,  # body number 
                    keypoint_id + 137,  
                    face_x, #keypoint face x
                    face_y, #keypoint face y
                ])
                
        # Create DataFrame
        df = pd.DataFrame(data, columns=['frame_id', 'body_id', 'keypoint_id', 'x', 'y', 'subset'])
        
        # Save to CSV
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'dwpose', 'pose_data_diffsynth')
        os.makedirs(output_dir, exist_ok=True)
        # Create filename with input filename if provided
        if input_filename:
            input_basename = os.path.splitext(os.path.basename(input_filename))[0]
            output_file = os.path.join(output_dir, f'{input_basename}.csv')
        else:
            output_file = os.path.join(output_dir, f'pose_keypoints.csv')
        
        # Append to existing file or create new one
        if os.path.exists(output_file):
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, index=False)
            print(f"Created new pose keypoints file at {output_file}")

class PoseBodyFaceAnnotator(PoseAnnotator):
    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.use_body, self.use_face, self.use_hand = True, True, True
    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image, frame_id=0, input_filename=None):
        ret_data, det_result = super().forward(image, frame_id, input_filename)
        return ret_data['detected_map_bodyface']


class PoseBodyFaceVideoAnnotator(PoseBodyFaceAnnotator):
    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.use_body, self.use_face, self.use_hand = True, True, True
    @torch.no_grad()
    @torch.inference_mode
    def forward(self, frames, input_filename):
        ret_frames = []
        for frame_id, frame in enumerate(frames):
            anno_frame = super().forward(np.array(frame), frame_id, input_filename)
            ret_frames.append(anno_frame)
        return ret_frames

class PoseBodyAnnotator(PoseAnnotator):
    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        self.use_body, self.use_face, self.use_hand = True, False, False
    @torch.no_grad()
    @torch.inference_mode
    def forward(self, image, frame_id=0, input_filename=None):
        ret_data, det_result = super().forward(image, frame_id, input_filename)
        return ret_data['detected_map_body']


class PoseBodyVideoAnnotator(PoseBodyAnnotator):
    def forward(self, frames, input_filename):
        self.use_body, self.use_face, self.use_hand = True, True, True
        ret_frames = []
        for frame_id, frame in enumerate(frames):
            anno_frame = super().forward(np.array(frame), frame_id, input_filename)
            ret_frames.append(anno_frame)
        return ret_frames