import numpy as np
import tensorflow as tf
from pickle import load

from activity_predictor.person_tracker import PersonTracker


class ActivityPredictor:
    """
    Human behaviour classifier
    """
    def __init__(
            self,
            model_path='models/lstm_model.h5',
            scalar_path='models/min_max_scalar.pickle',
            window=3,
            pose_vec_dim=36,
            motion_dict=None,
            prediction_threshold=0.80
    ):
        self.__prediction_threshold = prediction_threshold
        self.__motion_dict = {0: 'spin', 1: 'squat'} if motion_dict is None else motion_dict
        self.__window = window
        self.__pose_vec_dim = pose_vec_dim
        self.__secondary_model = tf.keras.models.load_model(model_path)
        self.__scalar = load(open(scalar_path, 'rb'))
        self.__tracked_objects = {}
        self.__untracked_objects = {}
        print("Activity predictor initialised successfully")

    def add_untracked_pose_dict(self, component_id, pose_dict):
        """
        Add rect params data for found object to store property
        for link them with object from tracker later
        Args:
            component_id: unique detected object's component id
            pose_dict:

        Returns:
            None
        """
        if component_id not in self.__untracked_objects:
            self.__untracked_objects[component_id] = pose_dict
            print("rect params added: {0}".format(component_id))

    def __remove_not_relevant_trackers(self, objects_meta):
        """
        Remove PersonTracker objects linked with Detected objects removed from frame
        Args:
            objects_meta:

        Returns:
            None
        """
        relevant_ids = [obj_meta.object_id for obj_meta in objects_meta]
        trackers_id_to_delete = []

        for obj_id in self.__tracked_objects:
            if obj_id not in relevant_ids:
                print("delete tracker {0}".format(obj_id))
                trackers_id_to_delete.append(obj_id)

        for obj_id in trackers_id_to_delete:
            del self.__tracked_objects[obj_id]

    def update_person_trackers(self, objects_meta):
        """
        Manage person trackers
        Args:
            objects_meta:

        Returns:

        """
        for obj_meta in objects_meta:
            for trk in self.__tracked_objects:
                print("tracker id: {0} obj id: {1}".format(trk, obj_meta.object_id))
            try:
                person_tracker = self.__tracked_objects[obj_meta.object_id]
            except KeyError:
                person_tracker = PersonTracker(obj_meta)
                print("Tracker {0} created".format(person_tracker.obj_meta.object_id))
                self.__tracked_objects[obj_meta.object_id] = person_tracker
            try:
                rect_params = self.__untracked_objects[obj_meta.unique_component_id]
            except KeyError:
                print("Update person tracker error: "
                      "Unable to find pose_dict for specified rect_params. id: {0}".format(obj_meta.object_id))
                print("component_id: {0}".format(obj_meta.unique_component_id))
                continue

            person_tracker.update_pose(rect_params, obj_meta.rect_params)
            person_tracker.obj_meta = obj_meta
            print("Tracker {0} updated".format(person_tracker.obj_meta.object_id))

        self.__untracked_objects = {}
        self.__remove_not_relevant_trackers(objects_meta)

    def predict_activity(self):
        """
        Inference human behaviour classifier
        Returns:
            None
        """
        for tracker_id in self.__tracked_objects:
            tracker = self.__tracked_objects[tracker_id]
            if len(tracker.states) >= self.__window:
                sample = np.array(list(tracker.states)[:self.__window])
                sample = self.__scalar.transform(sample)
                sample = sample.reshape(1, self.__window, self.__pose_vec_dim)
                prediction_vector = self.__secondary_model.predict(sample)[0]
                predicted_class = int(np.argmax(prediction_vector))

                predicted_activity = self.__motion_dict[predicted_class] \
                    if prediction_vector[predicted_class] >= self.__prediction_threshold \
                    else None
                tracker.activity = predicted_activity
                tracker.annotate()
