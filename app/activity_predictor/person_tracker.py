from collections import deque

from config.app_config import BODY_LABELS, BODY_IDX
import numpy as np
import sys

sys.path.append('../../')


class PersonTracker:
    """
    Class for store detected objects movements
    """
    def __init__(self, obj_meta):
        self.obj_meta = obj_meta
        self.states = deque(maxlen=10)
        self.activity = None
        return

    def update_pose(self, pose_dict, tracker_bbox_info):
        """
        Append ingo from current frame to object's states list
        Args:
            pose_dict:
            tracker_bbox_info:

        Returns:
            None
        """
        ft_vec = np.zeros(2 * len(BODY_LABELS))
        centroid = tuple(map(int, (tracker_bbox_info.top + tracker_bbox_info.height / 2,
                                   tracker_bbox_info.left + tracker_bbox_info.width / 2)))
        for ky in pose_dict:
            idx = BODY_IDX[ky]

            ft_vec[2 * idx: 2 * (idx + 1)] = 2 * (np.array(pose_dict[ky]) - np.array(centroid)) / np.array((
                tracker_bbox_info.height, tracker_bbox_info.width))
        self.states.append(ft_vec)

    def annotate(self):
        """
        Draw label with detected activity
        Returns:
            None
        """
        txt_params = self.obj_meta.text_params

        txt_params.x_offset = int(self.obj_meta.rect_params.left)
        txt_params.y_offset = max(0, int(self.obj_meta.rect_params.top) - 10)
        txt_params.display_text = self.activity if self.activity else 'None'
        # Font , font-color and font-size
        txt_params.font_params.font_name = "Serif"
        txt_params.font_params.font_size = 10
        txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        txt_params.set_bg_clr = 1
        txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
