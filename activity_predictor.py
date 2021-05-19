from operator import itemgetter
from scipy.optimize import linear_sum_assignment
import string
import random
import numpy as np
from collections import deque
import pyds
from utils import id_gen

trackers = []

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



def IOU(boxA, boxB):
    # pyimagesearch: determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_bbox(kp_list):
    bbox = []
    for aggs in [min, max]:
        for idx in range(2):
            bound = aggs(kp_list, key=itemgetter(idx))[idx]
            bbox.append(bound)
    return bbox

def tracker_match(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatched trackers, unmatched detections.
    https://towardsdatascience.com/computer-vision-for-tracking-8220759eee85
    '''

    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        for d,det in enumerate(detections):
            IOU_mat[t,d] = IOU(trk,det)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    print("IOU_mat: {0}".format(IOU_mat))
    matched_idx = linear_sum_assignment(-IOU_mat)
    matched_idx = np.asarray(matched_idx)
    matched_idx = np.transpose(matched_idx)
    print("matched_idx: {0}".format(matched_idx))
    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0],m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def predict_activity(secondary_model, pose_list, frame_meta, frame_width, frame_height):
    bboxes = []

    for body in pose_list:
            bbox = get_bbox(list(body.values()))
            bboxes.append((bbox, body))

    print("pose_list {0}".format(pose_list))

    track_boxes = [tracker.bbox for tracker in trackers]
    matched, unmatched_trackers, unmatched_detections = tracker_match(track_boxes, [b[0] for b in bboxes])

    for idx, jdx in matched:
            trackers[idx].set_bbox(bboxes[jdx][0])
            trackers[idx].update_pose(bboxes[jdx][1])

    for idx in unmatched_detections:
        try:
            trackers.pop(idx)
        except:
            pass

    for idx in unmatched_trackers:
        person = PersonTracker()
        person.set_bbox(bboxes[idx][0])
        person.update_pose(bboxes[idx][1])
        trackers.append(person)

    for tracker in trackers:
        print(len(tracker.q))
        if len(tracker.q) >= 3:
            sample = np.array(list(tracker.q)[:3])
            sample = sample.reshape(1, pose_vec_dim, window)
            pred_activity = motion_dict[np.argmax(secondary_model.predict(sample)[0])]
            tracker.activity = pred_activity
            tracker.annotate(frame_meta, frame_width, frame_height)
            print(pred_activity)


