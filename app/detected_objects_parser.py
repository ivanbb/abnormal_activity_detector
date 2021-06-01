from operator import itemgetter

from trt_pose.parse_objects import ParseObjects
from trt_pose.coco import coco_category_to_topology
import pyds
from ctypes import *
import numpy as np
from torch import Tensor
from config.app_config import UNTRACKED_OBJECT_ID
import json


class BodyPartsParser:
    """
    Parse trt_pose output and return detected objets
    """
    def __init__(self):
        with open('config/human_pose.json', 'r') as f:
            human_pose = json.load(f)

        self.__topology = coco_category_to_topology(human_pose)
        self.__parse_objects = ParseObjects(self.__topology)

    @property
    def topology(self):
        return self.__topology

    def parse_objects_from_tensor_meta(self, tensor_meta):
        cmap_layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
        paf_layer = pyds.get_nvds_LayerInfo(tensor_meta, 1)

        cmap_dims = pyds.get_nvds_LayerInfo(tensor_meta, 0).dims
        cmap_data_ptr = cast(pyds.get_ptr(cmap_layer.buffer), POINTER(c_float))

        cmap_data = np.ctypeslib.as_array(cmap_data_ptr, shape=(cmap_dims.d[0], cmap_dims.d[1], cmap_dims.d[2]))
        cmap_data = Tensor([cmap_data])

        paf_dims = pyds.get_nvds_LayerInfo(tensor_meta, 1).dims
        paf_data_ptr = cast(pyds.get_ptr(paf_layer.buffer), POINTER(c_float))

        paf_data = np.ctypeslib.as_array(paf_data_ptr, shape=(paf_dims.d[0], paf_dims.d[1], paf_dims.d[2]))
        paf_data = Tensor([paf_data])

        counts, objects, peaks = self.__parse_objects(cmap_data, paf_data)

        return counts, objects, peaks


def get_bbox(kp_list):
    """
    Get boxes for found objects
    Args:
        kp_list: list on body parts found in image

    Returns:
        bbox: boundary boxes coordinates
    """
    bbox = []
    for aggs in [min, max]:
        for idx in range(2):
            bound = aggs(kp_list, key=itemgetter(idx))[idx]
            bbox.append(bound)
    return bbox


def create_frame_objects(body_list):
    """
    Create NvDsInferObjectDetectionInfo for found bodies objects
    Args:
        body_list:

    Returns:
        objects_list
    """
    objects_list = []
    for body in body_list:
        bbox = get_bbox(list(body.values()))
        res = pyds.NvDsInferObjectDetectionInfo()

        rect_x1_f, rect_y1_f, rect_x2_f, rect_y2_f = bbox
        res.left = rect_x1_f
        res.top = rect_y1_f
        res.width = rect_x2_f - rect_x1_f
        res.height = rect_y2_f - rect_y1_f
        objects_list.append({'frame_object': res, 'body': body})
    return objects_list


def add_obj_meta_to_frame(frame_object, batch_meta, frame_meta):
    """
    Inserts an object into the metadata for tracking
    Args:
        frame_object:
        batch_meta:
        frame_meta:

    Returns:
        obj_meta
    """
    # this is a good place to insert objects into the metadata.
    # Here's an example of inserting a single object.
    obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    # Set bbox properties. These are in input resolution.
    rect_params = obj_meta.rect_params
 
    rect_params.left = int(frame_object.left)
    rect_params.top = int(frame_object.top)
    rect_params.width = int(frame_object.width)
    rect_params.height = int(frame_object.height)

    # Red border of width 3
    rect_params.border_width = 3
    rect_params.border_color.set(1, 0, 0, 1)

    # There is no tracking ID upon detection. The tracker will
    # assign an ID.
    obj_meta.object_id = UNTRACKED_OBJECT_ID

    # Inser the object into current frame meta
    # This object has no parent
    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

    return obj_meta
