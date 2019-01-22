#!/usr/bin/env python

"""
Detector class to use TensorFlow detection API

The codes are from TensorFlow/Models Repo. I just transferred the code to
ROS.

Cagatay Odabasi
"""

import numpy as np
import tensorflow as tf
import time

import cv2

import rospkg
from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config




CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]
weights_path='CKPT/lanenet_vgg_2018-10-19-13-33-56.ckpt-200000'






class LaneProcessor(object):


    def minmax_scale(self,input_arr):

        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

        return output_arr




    def __init__(self, \
        model_name='lanenet_vgg_2018-10-19-13-33-56.ckpt-200000',\
        num_workers=-1
        ):

        super(LaneProcessor, self).__init__()

        self._model_name = model_name

        self._detection_graph = None

        self._sess = None

        self._num_workers = num_workers

        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()

        self._path_to_ckpt = rospack.get_path('vision_segment_enet_detect') + \
            '/src/CKPT' + '/' + \
            self._model_name

        # Prepare the model for detection
        self.prepare()




    def prepare(self):




        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        self.phase_tensor = tf.constant('test', tf.string)
        self.net = lanenet_merge_model.LaneNet(phase=self.phase_tensor, net_flag='vgg')
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='lanenet_model')
        self.cluster = lanenet_cluster.LaneNetCluster()
        self.postprocessor = lanenet_postprocess.LaneNetPoseProcessor()




        saver = tf.train.Saver()

        sess_config = tf.ConfigProto(device_count={'GPU': 0})


        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        self._sess = tf.Session(config=sess_config)







        saver.restore(sess=self._sess, save_path=self._path_to_ckpt)




    def process(self, image):


        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image - VGG_MEAN

        binary_seg_image, instance_seg_image = self._sess.run([self.binary_seg_ret, self.instance_seg_ret],
                                                        feed_dict={self.input_tensor: [image]})

        binary_seg_image[0] = self.postprocessor.postprocess(binary_seg_image[0])
        mask_image = self.cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])

        for i in range(4):
            instance_seg_image[0][:, :, i] = self.minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        # cv2.imshow('mask_image', mask_image[:, :, (2, 1, 0)])
        # cv2.imshow('src_image', image[:, :, (2, 1, 0)])
        # cv2.imshow('instance_image', embedding_image[:, :, (2, 1, 0)])
        #
        # cv2.waitKey(1)


        return mask_image,embedding_image




