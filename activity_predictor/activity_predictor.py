from operator import itemgetter
from scipy.optimize import linear_sum_assignment
import string
import random
import numpy as np
from collections import deque
import pyds
from utils import id_gen
import tensorflow as tf

from person_tracker import PersonTracker


class ActivityPredictor:
    def __init__(
            self,
            model_path='models/lstm_spin_squat.h5',
            window=3,
            pose_vec_dim=36,
            motion_dict=None
    ):
        self.motion_dict = {0: 'spin', 1: 'squat'} if motion_dict is None else motion_dict
        self.window = window
        self.pose_vec_dim = pose_vec_dim
        self.secondary_model = tf.keras.models.load_model(model_path)
        self.tracked_objects = []
        self.untracked_objects = {}
        print("Activity predictor initialised successfully")

    def add_untracked_pose_dict(self, rect_params, pose_dict):
        self.untracked_objects[rect_params] = pose_dict

    def __remove_not_relevant_trackers(self, objects_meta):
        relevant_ids = [obj_meta.object_id for obj_meta in objects_meta]
        self.tracked_objects = list(filter(lambda x: (x.obj_meta.object_id in relevant_ids), self.tracked_objects))

    def update_person_trackers(self, objects_meta):
        for obj_meta in objects_meta:
            found_person_tracker = list(filter(
                lambda tracker: (tracker.obj_meta.object_id == obj_meta.object_id),
                self.tracked_objects
            ))
            if len(found_person_tracker):
                person_tracker = found_person_tracker[0]
            else:
                person_tracker = PersonTracker(obj_meta)
                self.tracked_objects.append(person_tracker)

            try:
                rect_params = self.untracked_objects[obj_meta.rect_params]
            except KeyError:
                print("Update person tracker error: "
                      "Unable to find pose_dict for specified rect_params. id: {0}".format(obj_meta.object_id))
                continue

            person_tracker.update_pose(rect_params, obj_meta.rect_params)
            person_tracker.obj_meta = obj_meta
            self.untracked_objects.pop(obj_meta.rect_params)
            print("Tracker {0} updated".format(person_tracker.obj_meta.object_id))

        self.__remove_not_relevant_trackers(objects_meta)

    def predict_activity(self):
        for tracker in self.tracked_objects:
            print(len(tracker.states))
            if len(tracker.states) >= 3:
                sample = np.array(list(tracker.states)[:3])
                sample = sample.reshape(1, self.pose_vec_dim, self.window)
                predicted_activity = self.motion_dict[int(np.argmax(self.secondary_model.predict(sample)[0]))]
                tracker.activity = predicted_activity
                tracker.annotate()
                print(predicted_activity)


