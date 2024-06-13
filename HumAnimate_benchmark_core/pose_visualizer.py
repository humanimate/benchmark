import cv2
import math
import numpy as np


class PoseVisualizer(object):
    def __init__(self, canvas_dim=(512, 512, 3), keyp_confidence_thresh=0.1):
        self.default_canvas_dim = canvas_dim
        self.confidence_thresh = keyp_confidence_thresh

    def get_hrnet_stick_figure(self, keypoints, frames, overlay_pose_over_frame=False):
        assert keypoints is not None, "Execute cache_pose_keypoints(vid) first."
        orange=(255,153,51)
        blue=(0,128,255)
        green=(0,255,0)
        skeleton = [
            ("left_eye", "right_eye", orange),("nose", "right_eye", orange), ("right_eye", "right_ear", orange), ("nose", "left_eye", orange), ("left_eye", "left_ear", orange),
            ("right_shoulder", "right_ear", orange),("left_shoulder", "left_ear", orange), ("right_shoulder", "left_shoulder", orange), ("right_shoulder", "right_elbow", green),
            ("left_shoulder", "left_elbow",blue), ("right_elbow", "right_wrist",green), ("left_elbow", "left_wrist",blue), ("right_shoulder", "right_hip",orange),
            ("left_shoulder", "left_hip", orange), ("right_hip", "left_hip", orange), ("right_hip", "right_knee",green),
            ("left_hip", "left_knee",blue), ("right_knee", "right_ankle",green), ("left_knee", "left_ankle",blue)
        ]
        pose_frames = list()
        for ndx, frame_data in enumerate(keypoints):
            canvas = frames[ndx].copy() if overlay_pose_over_frame else np.zeros(self.default_canvas_dim, np.uint8)
            H, W, _ = canvas.shape
            for body in frame_data["bodies"]:
                for keyp1, keyp2, color in skeleton:
                    if (body[keyp1] is not None and body[keyp2] is not None) and (body[keyp1][2] >= self.confidence_thresh and body[keyp2][2] >= self.confidence_thresh):
                        x1, y1 = body[keyp1][0]*W, body[keyp1][1]*H
                        x2, y2 = body[keyp2][0]*W, body[keyp2][1]*H
                        pt1 = (round(x1), round(y1))
                        pt2 = (round(x2), round(y2))
                        cv2.line(canvas, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
            pose_frames.append(canvas)
        pose_frames = np.stack(pose_frames)
        return pose_frames

    def get_openpose_keypoint_figure(self, keypoints, frames, overlay_pose_over_frame=False):
        assert keypoints is not None, "Execute cache_pose_keypoints(vid) first."
        skeleton = [
            ('torso', 'left_hip', (0, 255, 0)), ('left_hip', 'left_knee', (0, 255, 85)), ('left_knee', 'left_ankle', (0, 255, 170)), ('torso', 'right_hip', (0, 255, 255)),
            ('right_hip', 'right_knee', (0, 170, 255)), ('right_knee', 'right_ankle', (0, 85, 255)), ('torso', 'left_shoulder', (255, 0, 0)), ('torso', 'right_shoulder', (255, 85, 0)),
            ('left_shoulder', 'left_elbow', (255, 170, 0)),('left_elbow', 'left_wrist', (255, 255, 0)), ('right_shoulder', 'right_elbow', (170, 255, 0)), ('right_elbow', 'right_wrist', (85, 255, 0)),
            ('torso', 'nose', (0, 0, 255)), ('nose', 'left_eye', (85, 0, 255)), ('left_eye', 'left_ear', (170, 0, 255)), ('nose', 'right_eye', (255, 0, 255)), ('right_eye', 'right_ear', (255, 0, 170))
        ]
        keypoint_markers = [
            ('nose', (255, 0, 0)), ('torso', (255, 85, 0)), ('left_shoulder', (255, 170, 0)), ('left_elbow', (255, 255, 0)), ('left_wrist', (170, 255, 0)),
            ('right_shoulder', (85, 255, 0)), ('right_elbow', (0, 255, 0)), ('right_wrist', (0, 255, 85)), ('left_hip', (0, 255, 170)), ('left_knee', (0, 255, 255)),
            ('left_ankle', (0, 170, 255)), ('right_hip', (0, 85, 255)), ('right_knee', (0, 0, 255)), ('right_ankle', (85, 0, 255)), ('left_eye', (170, 0, 255)),
            ('right_eye', (255, 0, 255)), ('left_ear', (255, 0, 170)), ('right_ear', (255, 0, 85))
        ]
        pose_frames = list()
        for ndx, frame_data in enumerate(keypoints):
            canvas = frames[ndx].copy() if overlay_pose_over_frame else np.zeros(self.default_canvas_dim, np.uint8)
            H, W, _ = canvas.shape
            for body in frame_data["bodies"]:
                for keyp1, keyp2, color in skeleton:
                    if (body[keyp1] is not None and body[keyp2] is not None) and (body[keyp1][2] >= self.confidence_thresh and body[keyp2][2] >= self.confidence_thresh):
                        x1, y1 = body[keyp1][0]*W, body[keyp1][1]*H
                        x2, y2 = body[keyp2][0]*W, body[keyp2][1]*H
                        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                        angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
                        polygon = cv2.ellipse2Poly((round((x1+x2)/2), round((y1+y2)/2)), (round(length / 2), 4), round(angle), 0, 360, 1)
                        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
                for keyp_name, color in keypoint_markers:
                    if (body[keyp_name] is not None) and (body[keyp_name][2] >= self.confidence_thresh):
                        x, y = body[keyp_name][0]*W, body[keyp_name][1]*H
                        cv2.circle(canvas, (round(x), round(y)), 4, color, thickness=-1)
            pose_frames.append(canvas)
        pose_frames = np.stack(pose_frames)
        return pose_frames

    def get_moving_keypoints_arrow_figure(self, keypoints, frames, overlay_arrows_over_image=False):
        assert keypoints is not None, "Execute cache_pose_keypoints(vid) first."
        arrow_color = (0, 255, 0)
        arrow_frames = list()
        for frndx in list(range(len(keypoints)))[1:]:
            prev = keypoints[frndx-1]
            curr = keypoints[frndx]
            canvas = frames[frndx].copy() if overlay_arrows_over_image else np.zeros(self.default_canvas_dim, np.uint8)
            H, W, _ = canvas.shape
            for keyp1, keyp2 in zip(prev["bodies"][0].items(), curr["bodies"][0].items()):
                if ((keyp1[1] is not None) and (keyp2[1] is not None)) and (keyp1[1][2] >= self.confidence_thresh and keyp2[1][2] >= self.confidence_thresh):
                    x1, y1 = keyp1[1][0]*W, keyp1[1][1]*H
                    x2, y2 = keyp2[1][0]*W, keyp2[1][1]*H
                    cv2.arrowedLine(canvas, (round(x1), round(y1)), (round(x2), round(y2)), arrow_color, thickness=1, tipLength=0.2)
            arrow_frames.append(canvas)
        arrow_frames = np.stack(arrow_frames)
        return arrow_frames
