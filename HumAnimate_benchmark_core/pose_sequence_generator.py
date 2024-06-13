from TextPoseVidGen_benchmark.pose_detectors.HRNet_detector import HRNet_PoseDetector
from TextPoseVidGen_benchmark.pose_detectors.OpenPose_detector import OpenPose_PoseDetector
from TextPoseVidGen_benchmark.pose_visualizer import PoseVisualizer
from decord import VideoReader

import gc
import numpy as np
import os
import pickle


class PoseSequenceGenerator(object):
    pose_detectors = {
        "HRNet": HRNet_PoseDetector,
        "OpenPose": OpenPose_PoseDetector,
    }
    def __init__(self, config, load_detector=True):
        self.config = config
        self.pose_model = self.pose_detectors[config["pose_model"]](config) if load_detector else None
        self.visualizer = self.get_visualizer()
        self.keypoints = None
        self.frames = None

    def get_visualizer(self):
        visualizer = None
        if "default_canvas_dim" in self.config.keys():
            visualizer = PoseVisualizer(self.config["default_canvas_dim"], self.config["pose_det_confidence"])
        return visualizer

    def get_pose_sequence(self, figure_type, overlay_over_frames=False):
        valid_pose_vis_types = ["openpose_keypose", "hrnet_stickfigure", "keypoint_arrows"]
        pose_frames = None
        if figure_type == "openpose_keypose":
            pose_frames = self.visualizer.get_openpose_keypoint_figure(self.keypoints, self.frames, overlay_over_frames)
        elif figure_type == "hrnet_stickfigure":
            pose_frames = self.visualizer.get_hrnet_stick_figure(self.keypoints, self.frames, overlay_over_frames)
        elif figure_type == "keypoint_arrows":
            pose_frames = self.visualizer.get_moving_keypoints_arrow_figure(self.keypoints, self.frames, overlay_over_frames)
        else:
            print("Invalid `figure_type`, select one from {}".format(valid_pose_vis_types))
        return pose_frames

    def collect_pose_keypoints(self, vid, resize_frames=False):
        frames = self.get_frames(vid) if isinstance(vid, str) else vid[:]
        vid_keypoints = self.pose_model.detect_pose_keypoints_from_video(frames, resize_frames)
        self.keypoints = vid_keypoints
        self.frames = frames
        print("Updated keypoint cache")

    def get_frames(self, filepath):
        extn = filepath.split(".")[-1]
        frames = None
        if extn == "npy":
            frames = np.load(filepath)
        else:
            vid = VideoReader(filepath)
            frames = vid.get_batch(list(range(len(vid)))).asnumpy()
        return frames

    def set_visualizer_canvas_shape(self, canvas_shape):
        self.visualizer.default_canvas_dim = canvas_shape

    def override_keypoints(self, new_keypoints):
        self.keypoints = new_keypoints

    def get_keypoints_and_frames(self):
        return (self.keypoints, self.frames)

    def save(self, outpath, save_filename, save_frames=True):
        cache = self.keypoints
        with open(str(outpath/(save_filename+".pkl")), 'wb') as f:
            pickle.dump(cache, f)
        if save_frames:
            np.save(str(outpath/(save_filename+"__frames")), self.frames)
        print("Saved keypoints and frames({}) at {}.(pkl|npy)".format(save_frames, str(outpath/save_filename)))

    def load(self, filepath, filename):
        with open(str(filepath/(filename+".pkl")), 'rb') as f:
            self.keypoints = pickle.load(f)
        if os.path.isfile(str(filepath/(filename+"__frames.npy"))):
            self.frames = np.load(str(filepath/(filename+"__frames.npy")))
        else:
            print("{} not found. Populating zero frames".format(str(filepath/(filename+"__frames.npy"))))
            shape = [len(self.keypoints)] + list(self.visualizer.default_canvas_dim)
            self.frames = np.zeros(shape, dtype=np.uint8)
        print("Loaded cached keypoints and frames from {}".format(filepath/filename))

    def clear(self):
        self.keypoints = None
        self.frames = list()
        gc.collect()
        print("Flushed the keypoints and video frames")
