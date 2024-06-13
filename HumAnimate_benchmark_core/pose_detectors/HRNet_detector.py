from mmpose.apis import MMPoseInferencer
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from utils import get_config

import numpy as np
import torch


class HRNet_PoseDetector(object):
    def __init__(self, config):
        base_path = Path(config["ckpts"]["base_path"])/config["ckpts"]["ckpt_dir"]/config["ckpts"]["pose_detectors"]["ckpt_dir"]
        pose_config = str(base_path/config["ckpts"]["pose_detectors"]["pose_estimator_config"])
        pose_checkpoint = str(base_path/config["ckpts"]["pose_detectors"]["pose_estimator_weights"])
        det_config = str(base_path/config["ckpts"]["pose_detectors"]["person_det_config"])
        det_checkpoint = str(base_path/config["ckpts"]["pose_detectors"]["person_det_weights"])
        self.device = torch.device("cuda:{}".format(config["gpu_id"]) if torch.cuda.is_available() else "cpu")
        self.pose_model = MMPoseInferencer(
                            pose2d=pose_config,
                            pose2d_weights=pose_checkpoint,
                            det_model=det_config,
                            det_weights=det_checkpoint,
                            det_cat_ids=(0,),
                            device=self.device
                        )
        self.keypoint_bodypart_map = {
            0 : "nose",
            1 : "right_eye",
            2 : "left_eye",
            3 : "right_ear",
            4 : "left_ear",
            5 : "right_shoulder",
            6 : "left_shoulder",
            7 : "right_elbow",
            8 : "left_elbow",
            9 : "right_wrist",
            10 : "left_wrist",
            11 : "right_hip",
            12 : "left_hip",
            13 : "right_knee",
            14 : "left_knee",
            15 : "right_ankle",
            16 : "left_ankle"
        }
        print("Loaded HRNet model")

    def map_hrnet_keypoints_to_common_keypoints(self, hrnet_body_keypoints, frame_dim):
        H, W, _ = frame_dim
        bodies = list()
        for keypoints in hrnet_body_keypoints:
            body = dict()
            for ndx, keypoint in enumerate(keypoints["keypoints"]):
                if keypoint is not None:
                    x = keypoint[0]/W
                    y = keypoint[1]/H
                    score = keypoints["keypoint_scores"][ndx]
                    norm_keyp = np.array([x, y, score])
                    body[self.keypoint_bodypart_map[ndx]] = norm_keyp
                else:
                    body[self.keypoint_bodypart_map[ndx]] = None
            body["torso"] = (body["left_shoulder"] + body["right_shoulder"])/2
            bodies.append(body)
        return bodies

    @torch.no_grad()
    def detect_pose_keypoints_from_frame(self, frame, resize):
        frame = np.asarray(frame.resize((512,512)) if resize else frame)
        bodies = next(self.pose_model(frame, show=False, return_vis=True))["predictions"][0]
        bodies = self.map_hrnet_keypoints_to_common_keypoints(bodies, frame.shape)
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
    config = get_config("configs/pose_detector_config.yaml")
    obj = HRNet_PoseDetector(config)

    # from decord import VideoReader
    # vid = VideoReader("pose_vids/real/jimmydance.mp4")
    # vidfr = vid.get_batch(list(range(len(vid)))).asnumpy()
    # vkp = obj.detect_pose_keypoints_from_video(vidfr)
