#!/usr/bin/env python


# ROS
import rospy

import cv2

from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

from LaneProcessor4 import LaneProcessor



class LaneProcessorNode(object):
    """docstring for PeopleObjectDetectionNode."""
    def __init__(self):
        super(LaneProcessorNode, self).__init__()

        # init the node
        rospy.init_node('lane_processor', anonymous=False)

        # Create Detector
        self._lane_processor = LaneProcessor()

        self._bridge = CvBridge()

        self.sub_rgb = rospy.Subscriber('/image_raw',Image, self.rgb_callback, queue_size=1, buff_size=2**24)
        self.pub_image_segmented = rospy.Publisher( '/image_segmented', Image, queue_size=1)
        self.pub_image_segmented_blended = rospy.Publisher( '/image_segmented_blended', Image, queue_size=1)

        rospy.spin()



    def rgb_callback(self, data):
        """
        Callback for RGB images
        """
        print('rgb_callback')

        try:
            # .publish(self._cv_bridge.cv2_to_imgmsg(image_np, "bgr8"))
            # Convert image to numpy array
            cv_image = self._bridge.imgmsg_to_cv2(data, "bgr8")

            image_segmented,image_segmented_blended=self._lane_processor.process(cv_image)



            self.pub_image_segmented.publish(self._bridge.cv2_to_imgmsg(image_segmented, "bgr8"))

            self.pub_image_segmented_blended.publish(self._bridge.cv2_to_imgmsg(image_segmented_blended, "bgr8"))






        except CvBridgeError as e:
            print(e)



if __name__ == '__main__':
    node = LaneProcessorNode()
