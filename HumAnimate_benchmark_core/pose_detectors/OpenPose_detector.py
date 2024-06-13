from pathlib import Path
from PIL import Image
from tqdm import tqdm
from utils import get_config
from TextPoseVidGen_benchmark.pose_detectors.openpose_model import Body

import cv2
import mmpose
import numpy as np
import sys
import torch
import torchvision


class OpenPose_PoseDetector(object):
    def __init__(self, config):
        base_path = Path(config["ckpts"]["base_path"])/config["ckpts"]["ckpt_dir"]/config["ckpts"]["pose_detectors"]["ckpt_dir"]
        model_path = base_path/config["ckpts"]["pose_detectors"]["openpose_pose_weights"]
        self.device = torch.device("cuda:{}".format(config["gpu_id"]) if torch.cuda.is_available() else "cpu")
        self.pose_model = Body(str(model_path)).to(self.device)
        self.keypoint_bodypart_map = {
            0 : "nose",
            1 : "torso",
            2 : "left_shoulder",
            3 : "left_elbow",
            4 : "left_wrist",
            5 : "right_shoulder",
            6 : "right_elbow",
            7 : "right_wrist",
            8 : "left_hip",
            9 : "left_knee",
            10 : "left_ankle",
            11 : "right_hip",
            12 : "right_knee",
            13 : "right_ankle",
            14 : "left_eye",
            15 : "right_eye",
            16 : "left_ear",
            17 : "right_ear",
        }
        print("Loaded OpenPose model")

    def map_openpose_keypoints_to_common_keypoints(self, openpose_body_keypoints, frame_dim):
        H, W, _ = frame_dim
        bodies = list()
        for bodyresults in openpose_body_keypoints:
            body = dict()
            for ndx, keypoint in enumerate(bodyresults.keypoints):
                if keypoint is not None:
                    x = keypoint.x / W
                    y = keypoint.y / H
                    score = keypoint.score
                    norm_keyp = np.array([x, y, score])
                    body[self.keypoint_bodypart_map[ndx]] = norm_keyp
                else:
                    body[self.keypoint_bodypart_map[ndx]] = None
            bodies.append(body)
        return bodies

    @torch.no_grad()
    def detect_pose_keypoints_from_frame(self, frame, resize):
        frame = np.asarray(frame.resize((512,512)) if resize else frame)
        frame = frame[:, :, ::-1].copy() # RGB2BGR
        candidates, subset = self.pose_model(frame)
        bodies = self.pose_model.format_body_result(candidates, subset)
        bodies = self.map_openpose_keypoints_to_common_keypoints(bodies, frame.shape)
        response = {
            "bodies": bodies,
            "H": frame.shape[0],
            "W": frame.shape[1],
        }
        return response

    def detect_pose_keypoints_from_video(self, vid_frames, resize=True):
        vid_keypoints = list()
        for frame in tqdm(vid_frames):
            frame_keypoints = self.detect_pose_keypoints_from_frame(Image.fromarray(frame), resize)
            vid_keypoints.append(frame_keypoints)
        return vid_keypoints


if __name__ == "__main__":
    config = get_config("configs/baseline.yaml")
    obj = OpenPose_PoseDetector(config)

    # from decord import VideoReader
    # vid = VideoReader("pose_vids/real/jimmydance.mp4")
    # vidfr = vid.get_batch(list(range(len(vid)))).asnumpy()
    # vkp = obj.detect_pose_keypoints_from_video(vidfr)
