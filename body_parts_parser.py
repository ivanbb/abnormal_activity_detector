from trt_pose.parse_objects import ParseObjects
from trt_pose.coco import coco_category_to_topology
import pyds
from ctypes import *
import numpy as np
from torch import Tensor
import json


class BodyPartsParser:
    def __init__(self):
        with open('human_pose.json', 'r') as f:
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
