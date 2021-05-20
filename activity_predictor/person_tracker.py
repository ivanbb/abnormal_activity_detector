from collections import deque

from utils import id_gen
from constants import BODY_LABELS, BODY_IDX

class PersonTracker(object):
    def __init__(self):
        self.id = id_gen() #int(time.time() * 1000)
        self.q = deque(maxlen=10)
        return

    def set_bbox(self, bbox):
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.h = 1e-6 + x2 - x1
        self.w = 1e-6 + y2 - y1
        self.centroid = tuple(map(int, ( x1 + self.h / 2, y1 + self.w / 2)))
        return

    def update_pose(self, pose_dict):
        ft_vec = np.zeros(2 * len(BODY_LABELS))
        for ky in pose_dict:
            idx = BODY_IDX[ky]
            ft_vec[2 * idx: 2 * (idx + 1)] = 2 * (np.array(pose_dict[ky]) - np.array(self.centroid)) / np.array((self.h, self.w))
        self.q.append(ft_vec)
        return

    def annotate(self, frame_meta, frame_width, frame_height):
        bmeta = frame_meta.base_meta.batch_meta
        dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)

        x1, y1, x2, y2 = self.bbox

        print("x1, y1, x2, y2 : {0}, {1}, {2}, {3}, {4}".format(x1, y1, x2, y2, self.activity))

        if (dmeta.num_rects == 16):
            dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

        rect_params = dmeta.rect_params[dmeta.num_rects]
        rect_params.left = int(x1)
        rect_params.top = int(y1)
        rect_params.width = int(x2-x1)
        rect_params.height = int(y2-y1)
        dmeta.num_rects = dmeta.num_rects + 1

        # Red border of width 3
        rect_params.border_width = 3
        rect_params.border_color.set(1, 0, 0, 1)



        if (dmeta.num_labels == 16):
            dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

    # Set display text for the object.
        txt_params = dmeta.text_params[dmeta.num_labels]
        if txt_params.display_text:
            pyds.free_buffer(txt_params.display_text)

        txt_params.x_offset = int(rect_params.left)
        txt_params.y_offset = max(0, int(rect_params.top) - 10)
        txt_params.display_text = self.activity
        # Font , font-color and font-size
        txt_params.font_params.font_name = "Serif"
        txt_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        txt_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        dmeta.num_labels = dmeta.num_labels + 1

        pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)