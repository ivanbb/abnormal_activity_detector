from collections import deque

from utils import id_gen
from constants import BODY_LABELS, BODY_IDX
import numpy as np


class PersonTracker:
    def __init__(self, obj_meta):
        self.obj_meta = obj_meta
        self.states = deque(maxlen=10)
        self.activity = None
        return

    def update_pose(self, pose_dict, tracker_bbox_info):
        ft_vec = np.zeros(2 * len(BODY_LABELS))
        centroid = tuple(map(int, (tracker_bbox_info.top + tracker_bbox_info.height / 2,
                                   tracker_bbox_info.left + tracker_bbox_info.width / 2)))
        for ky in pose_dict:
            idx = BODY_IDX[ky]

            ft_vec[2 * idx: 2 * (idx + 1)] = 2 * (np.array(pose_dict[ky]) - np.array(centroid)) / np.array((
                tracker_bbox_info.height, tracker_bbox_info.width))
        self.states.append(ft_vec)
        return

    def annotate(self):
        self.obj_meta.obj_label = self.activity
        # if (dmeta.num_labels == 16):
        #     dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)
        #     pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

        # # Set display text for the object.
        # txt_params = dmeta.text_params[dmeta.num_labels]
        # if txt_params.display_text:
        #     pyds.free_buffer(txt_params.display_text)
        #
        # txt_params.x_offset = int(rect_params.left)
        # txt_params.y_offset = max(0, int(rect_params.top) - 10)
        # txt_params.display_text = self.activity
        # # Font , font-color and font-size
        # txt_params.font_params.font_name = "Serif"
        # txt_params.font_params.font_size = 10
        # # set(red, green, blue, alpha); set to White
        # txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        #
        # # Text background color
        # txt_params.set_bg_clr = 1
        # # set(red, green, blue, alpha); set to Black
        # txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # dmeta.num_labels = dmeta.num_labels + 1

        # pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)
