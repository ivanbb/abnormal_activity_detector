from operator import itemgetter
from scipy.optimize import linear_sum_assignment
import string
import random
import numpy as np
from collections import deque
import tensorflow as tf

from activity_predictor.person_tracker import PersonTracker


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
        self.tracked_objects = {}
        self.untracked_objects = {}
        print("Activity predictor initialised successfully")

    def add_untracked_pose_dict(self, component_id, pose_dict):
        #rect_str = "{0}, {1}, {2}, {3}".format(int(rect_params.left), int(rect_params.top), int(rect_params.width), int(rect_params.height))
        if component_id not in self.untracked_objects:
            self.untracked_objects[component_id] = pose_dict
            print("rect params added: {0}".format(component_id))

    def __remove_not_relevant_trackers(self, objects_meta):
        relevant_ids = [obj_meta.object_id for obj_meta in objects_meta]
        #self.tracked_objects = list(filter(lambda x: (x.obj_meta.object_id in relevant_ids), self.tracked_objects))
        trackers_id_to_delete = []
        for id in self.tracked_objects:
            if (id not in relevant_ids):
                print("delete tracker {0}".format(id))
                trackers_id_to_delete.append(id)
        for id in trackers_id_to_delete: del self.tracked_objects[id]

    def update_person_trackers(self, objects_meta):
        for obj_meta in objects_meta:
            #found_person_tracker = list(filter(
            #    lambda tracker: (tracker.obj_meta.object_id == obj_meta.object_id),
            #    self.tracked_objects
            #))
            for trk in self.tracked_objects:
                print("tracker id: {0} obj id: {1}".format(trk, obj_meta.object_id))
            try:
                person_tracker = self.tracked_objects[obj_meta.object_id]
            except KeyError:
                person_tracker = PersonTracker(obj_meta)
                print("Tracker {0} created".format(person_tracker.obj_meta.object_id))
                self.tracked_objects[obj_meta.object_id] = person_tracker
            try:
                #rect_str = "{0}, {1}, {2}, {3}".format(int(obj_meta.rect_params.left), int(obj_meta.rect_params.top),
                #                                       int(obj_meta.rect_params.width), int(obj_meta.rect_params.height))
                rect_params = self.untracked_objects[obj_meta.unique_component_id]
            except KeyError:
                print("Update person tracker error: "
                      "Unable to find pose_dict for specified rect_params. id: {0}".format(obj_meta.object_id))
                print("component_id: {0}".format(obj_meta.unique_component_id))
                continue

            person_tracker.update_pose(rect_params, obj_meta.rect_params)
            person_tracker.obj_meta = obj_meta
            print("Tracker {0} updated".format(person_tracker.obj_meta.object_id))

        self.untracked_objects = {}
        self.__remove_not_relevant_trackers(objects_meta)

    def predict_activity(self, frame_meta):
        for tracker_id in self.tracked_objects:
            tracker = self.tracked_objects[tracker_id]
            #print(len(tracker.states))
            if len(tracker.states) >= 3:
                sample = np.array(list(tracker.states)[:3])
                sample = sample.reshape(1, self.pose_vec_dim, self.window)
                predict = self.secondary_model.predict(sample);
                print("predict: {0}".format(predict))
                predicted_activity = self.motion_dict[int(np.argmax(predict[0]))]
                tracker.activity = predicted_activity
                tracker.annotate(frame_meta)
                print(predicted_activity)


