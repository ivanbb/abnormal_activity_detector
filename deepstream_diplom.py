#!/usr/bin/env python3

################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import sys

from activity_predictor.activity_predictor import ActivityPredictor

sys.path.append('../')
import gi
import configparser

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import sys
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import json
import trt_pose.coco
from pprint import pprint
from torch.utils import dlpack
import torch
import numpy as np
from operator import itemgetter
# import tensorflow as tf
# from activity_predictor import *
from exceptions import CreatePipelineException
from detected_objects_parser import BodyPartsParser, create_frame_objects, add_obj_meta_to_frame
from constants import *

import pyds

fps_streams = {}

trackers = []

body_parts_parser = BodyPartsParser()
activity_predictor = ActivityPredictor(window=POSE_PREDICT_WINDOW, pose_vec_dim=POSE_VEC_DIM, motion_dict=MOTION_DICT)


def create_display_meta(objects, count, normalized_peaks, frame_meta, frame_width, frame_height):
    bmeta = frame_meta.base_meta.batch_meta
    dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)
    pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

    peaks_count = body_parts_parser.topology.shape[0]
    body_list = []
    for i in range(count):
        obj = objects[0][i]
        body_dict = {}
        parts_count = obj.shape[0]
        for j in range(parts_count):
            peak_idx = int(obj[j])
            if peak_idx >= 0:
                peak = normalized_peaks[0][j][peak_idx]
                x = round(float(peak[1]) * frame_width)
                y = round(float(peak[0]) * frame_height)
                if dmeta.num_circles == MAX_ELEMENTS_IN_DISPLAY_META:
                    dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)
                    pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

                cparams = dmeta.circle_params[dmeta.num_circles]
                cparams.xc = x
                cparams.yc = y
                cparams.radius = 8
                cparams.circle_color.set(244, 67, 54, 1)
                cparams.has_bg_color = 1
                cparams.bg_color.set(0, 255, 0, 1)
                dmeta.num_circles = dmeta.num_circles + 1
                body_dict[BODY_LABELS[j]] = (x, y)
        body_list.append(body_dict)

        for peak_idx in range(peaks_count):
            c_a = body_parts_parser.topology[peak_idx][2]
            c_b = body_parts_parser.topology[peak_idx][3]
            if obj[c_a] >= 0 and obj[c_b] >= 0:
                peak0 = normalized_peaks[0][c_a][obj[c_a]]
                peak1 = normalized_peaks[0][c_b][obj[c_b]]
                x0 = round(float(peak0[1]) * frame_width)
                y0 = round(float(peak0[0]) * frame_height)
                x1 = round(float(peak1[1]) * frame_width)
                y1 = round(float(peak1[0]) * frame_height)

                if dmeta.num_lines == MAX_ELEMENTS_IN_DISPLAY_META:
                    dmeta = pyds.nvds_acquire_display_meta_from_pool(bmeta)
                    pyds.nvds_add_display_meta_to_frame(frame_meta, dmeta)

                lparams = dmeta.line_params[dmeta.num_lines]
                lparams.x1 = x0
                lparams.y1 = y0
                lparams.x2 = x1
                lparams.y2 = y1
                lparams.line_width = 3
                lparams.line_color.set(0, 255, 0, 1)
                dmeta.num_lines = dmeta.num_lines + 1
    return body_list


def pgie_src_pad_buffer_probe(pad, info, u_data):
    """
    Process output from pose estimation model
    Apply postprocessing and locate the bodies parts
    Args:
        pad
        info
        u_data

    Returns:
        none
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_user = frame_meta.frame_user_meta_list

        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user_meta.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                continue

            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

            counts, objects, normalized_peaks = body_parts_parser.parse_objects_from_tensor_meta(tensor_meta)
            body_list = create_display_meta(objects, counts, normalized_peaks, frame_meta, TILED_OUTPUT_WIDTH,
                                            TILED_OUTPUT_HEIGHT)

            # predict_activity(secondary_model, body_list, frame_meta, MUXER_OUTPUT_WIDTH, MUXER_OUTPUT_HEIGHT)

            try:
                l_user = l_user.next
            except StopIteration:
                break

            frame_object_list = create_frame_objects(body_list)
            for frame_object in frame_object_list:
                obj_meta = add_obj_meta_to_frame(frame_object['frame_object'], batch_meta, frame_meta)
                activity_predictor.add_untracked_pose_dict(obj_meta.unique_component_id, frame_object['body'])
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def osd_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.glist_get_nvds_frame_meta()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list
        objects_meta = []
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.glist_get_nvds_object_meta(l_obj.data)
                print("object tracker id: {0}, left: {1}".format(obj_meta.object_id, obj_meta.rect_params.left))
                objects_meta.append(obj_meta)
            except StopIteration:
                break
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        activity_predictor.update_person_trackers(objects_meta)
        activity_predictor.predict_activity(frame_meta)

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        # display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        # display_meta.num_labels = 1
        # py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        # py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Vehicle_count={} Person_count={}".format(frame_number, num_rects, obj_counter[PGIE_CLASS_ID_VEHICLE], obj_counter[PGIE_CLASS_ID_PERSON])

        # Now set the offsets where the string should appear
        # py_nvosd_text_params.x_offset = 10
        # py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        # py_nvosd_text_params.font_params.font_name = "Serif"
        # py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        # py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        # py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        # py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        # print(pyds.get_string(py_nvosd_text_params.display_text))
        # pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if name.find("nvv4l2decoder") != -1:
        print("Setting drop frame interval\n")
        Object.set_property("drop-frame-interval", 7)


def cb_newpad(decodebin, decoder_src_pad, data):
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)

    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)

        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                raise CreatePipelineException("Failed to link decoder src pad to source bin ghost pad")
        else:
            raise CreatePipelineException("Decodebin did not pick nvidia decoder plugin")


def create_source_bin(index, uri):
    """
    Args:
        index: stream index
        uri: stream url

    Returns:
        nbin - source bin
    """
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)

    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        raise CreatePipelineException("Unable to create source bin")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = create_pipeline_element("uridecodebin", "uri-decode-bin")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        raise CreatePipelineException("Failed to add ghost pad in source bin \n")

    return nbin


def configure_tracker(tracker):
    """
    Set properties of tracker
    Args:
        tracker

    Returns:
        none

    """
    config = configparser.ConfigParser()
    config.read('config/tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width':
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height':
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id':
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file':
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file':
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process':
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame':
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)


def create_pipeline_element(element_type, element_name):
    """
    Create Gst pipeline element

    Args:
        element_type
        element_name

    Returns:
        element: Gst pipeline element

    """
    print("Creating {0} \n ".format(element_type))
    element = Gst.ElementFactory.make(element_type, element_name)
    if not element:
        raise CreatePipelineException("Unable to create {0}".format(element_type))
    return element


def create_pipeline(urls):
    """
    Create Pipeline element that will form a connection of other elements

    Args:
        urls: list of stream urls

    Returns:
        pipeline: Gst pipeline with added and linked elements

    """
    number_sources = len(urls)
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        raise CreatePipelineException()

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = create_pipeline_element("nvstreammux", "Stream-muxer")
    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin {0} \n ".format(i))
        uri_name = urls[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        pipeline.add(source_bin)

        pad_name = "sink_%u" % i
        sink_pad = streammux.get_request_pad(pad_name)
        if not sink_pad:
            raise CreatePipelineException("Unable to create sink pad bin")

        src_pad = source_bin.get_static_pad("src")
        if not src_pad:
            raise CreatePipelineException("Unable to create src pad bin")
        src_pad.link(sink_pad)

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    queue5 = Gst.ElementFactory.make("queue", "queue5")
    queue6 = Gst.ElementFactory.make("queue", "queue6")

    pgie = create_pipeline_element("nvinfer", "primary-inference")
    tracker = create_pipeline_element("nvtracker", "tracker")
    tiler = create_pipeline_element("nvmultistreamtiler", "nvtiler")
    nvvidconv = create_pipeline_element("nvvideoconvert", "convertor")
    nvosd = create_pipeline_element("nvdsosd", "onscreendisplay")

    nvosd.set_property('process-mode', OSD_PROCESS_MODE)
    nvosd.set_property('display-text', OSD_DISPLAY_TEXT)

    if is_aarch64():
        transform = create_pipeline_element("nvegltransform", "nvegl-transform")

    sink = create_pipeline_element("nveglglessink", "nvvideo-renderer")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    pgie.set_property('output-tensor-meta', True)
    pgie.set_property('config-file-path', "config/deepstream_pose_estimation_config.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print("WARNING: Overriding infer-config batch-size {0} with number of sources {1} \n".format(pgie_batch_size,
                                                                                                     number_sources))
        pgie.set_property("batch-size", number_sources)

    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos", 1)
    sink.set_property("sync", 0)
    configure_tracker(tracker)

    print("Adding elements to Pipeline \n")

    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)

    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(sink)

    # We link elements in the following order:
    # sourcebin -> streammux -> queue -> nvinfer -> queue -> nvtracker  -> queue ->  nvdsanalytics -> queue ->
    # nvtiler -> queue -> nvvideoconvert -> queue -> nvdsosd -> queue ->  sink
    print("Linking elements in the Pipeline \n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tracker)
    tracker.link(queue3)
    queue3.link(tiler)
    tiler.link(queue4)
    queue4.link(nvvidconv)
    nvvidconv.link(queue5)
    queue5.link(nvosd)
    if is_aarch64():
        nvosd.link(queue6)
        queue6.link(transform)
        transform.link(sink)
    else:
        nvosd.link(queue6)
        queue6.link(sink)

    # create an event loop and feed gstreamer bus messages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        raise CreatePipelineException("Unable to get src pad of pgie")

    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        raise CreatePipelineException("Unable to get osdsinkpad")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    return pipeline, loop


def main(args):
    """
    Initialise Gstreamer and start loop
    Args:
        args: command line arguments (list of stream urls)

    Returns:
        none
    """
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % args[0])
        sys.exit(1)

    sources_urls = []
    for i in range(0, len(args)):
        print(args[i])
        fps_streams["stream{0}".format(i)] = GETFPS(i)
        if i != 0:
            sources_urls.append(args[i])

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    try:
        pipeline, loop = create_pipeline(sources_urls)
    except CreatePipelineException as e:
        sys.stderr.write("Creating pipeline error:")
        sys.stderr.write(e.message)
        sys.exit(1)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(sources_urls):
        print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
