from operator import itemgetter
from scipy.optimize import linear_sum_assignment
import string
import random
import numpy as np
from collections import deque
import pyds
from person_tracker import PersonTracker
from constants import window, motion_dict, pose_vec_dim 

trackers = []


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


