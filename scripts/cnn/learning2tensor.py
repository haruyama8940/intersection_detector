#!/usr/bin/env python3

import roslib
roslib.load_manifest('intersection_detector')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from intersection_detect_mobilenetv2 import *
# from intersection_detect_LRCN import *
# from intersection_detect_LRCN_no_buffer import *
# from intersection_detect_LRCN_all import *
from bag2torch import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Int8,String
from std_srvs.srv import Trigger
from std_msgs.msg import Int8MultiArray
# from waypoint_nav.msg import cmd_dir_intersection
from scenario_navigation_msgs.msg import cmd_dir_intersection
# from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Empty
from std_srvs.srv import SetBool, SetBoolResponse
import os
import time
import sys
import tf
from nav_msgs.msg import Odometry

class intersection_detector_node:
    def __init__(self):
        rospy.init_node('intersection_detector_node', anonymous=True)
        self.class_num = 8
        self.b2t = bag_to_tensor()
        self.bridge = CvBridge()
        # self.intersection_pub = rospy.Publisher("passage_type",String,queue_size=1)
        self.intersection_pub = rospy.Publisher("passage_type",cmd_dir_intersection,queue_size=1)
        # self.image_sub = rospy.Subscriber("/camera_center/image_raw", Image, self.callback)
        # self.image_sub = rospy.Subscriber("/image_center", Image, self.callback)
        # self.image_sub = rospy.Subscriber("/image_left", Image, self.callback)
        self.image_sub = rospy.Subscriber("/image_right", Image, self.callback)
        # self.image_left_sub = rospy.Subscriber("/camera_left/rgb/image_raw", Image, self.callback_left_camera)
        # self.image_right_sub = rospy.Subscriber("/camera_right/rgb/image_raw", Image, self.callback_right_camera)
        self.srv = rospy.Service('/training_intersection', SetBool, self.callback_dl_training)
        self.loop_srv = rospy.Service('/loop_count', SetBool, self.callback_dl_training)
        
        # self.mode_save_srv = rospy.Service('/model_save_intersection', Trigger, self.callback_model_save)
        self.cmd_dir_sub = rospy.Subscriber("/cmd_dir_intersection", cmd_dir_intersection, self.callback_cmd,queue_size=1)
        self.min_distance = 0.0
        self.action = 0.0
        self.episode = 0
        # self.intersection =String()
        self.intersection = cmd_dir_intersection()
        self.path_pose = PoseArray()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.cv_left_image = np.zeros((480,640,3), np.uint8)
        self.cv_right_image = np.zeros((480,640,3), np.uint8)
        self.learning = True
        self.learning_tensor_flag = False
        self.select_dl = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        # self.path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/result'
        
        self.load_image_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/single/image/image_center/image.pt'
        self.load_label_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/single/label/center_label/label.pt'

        # self.load_image_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/single/image/image_left/image.pt'
        # self.load_label_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/single/label/left_label/label.pt'
        
        # self.load_image_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/single/image/image_right/image.pt'
        # self.load_label_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/single/label/right_label/label.pt'

        self.load_center_image_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/single/image/image_center/image.pt'
        self.load_center_label_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/single/label/center_label/label.pt'

        self.load_left_image_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/single/image/image_left/image.pt'
        self.load_left_label_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/single/label/left_label/label.pt'
        
        self.load_right_image_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/single/image/image_right/image.pt'
        
        self.load_right_label_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/single/label/right_label/label.pt'

        #self.load_path =roslib.packages.get_pkg_dir('intersection_detector') + '/data/model/lrcn/real/frame16/hz8/30ep/0626_left/model_gpu.pt'
        # self.load_path =roslib.packages.get_pkg_dir('intersection_detector') + '/data/model/lrcn/real/frame16/hz8/30ep/0626_right/model_gpu.pt'
        
        self.load_path =roslib.packages.get_pkg_dir('intersection_detector') + '/data/model/lrcn/real/frame16/hz8/30ep/0626_balance_right/model_gpu.pt'
        # self.load_path =roslib.packages.get_pkg_dir('intersection_detector') + '/data/model/single/0630_single_all/model_gpu.pt'

        self.save_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/model/lrcn/real/frame16/hz8/30ep/'
        self.previous_reset_time = 0
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_the = 0.0
        self.is_started = False
        self.cmd_dir_data = [0,0,0,0,0,0,0,0]
        self.intersection_list = ["straight_road","dead_end","corner_right","corner_left","cross_road","3_way_right","3_way_center","3_way_left"]
        self.start_time_s = rospy.get_time()
        # os.makedirs(self.path + self.start_time)

        self.target_dataset = 12000
        print("target_dataset :" , self.target_dataset)
        # with open(self.path + self.start_time + '/' +  'training.csv', 'w') as f:
        #     writer = csv.writer(f, lineterminator='\n')
        #     writer.writerow(['step', 'mode', 'loss', 'angle_error(rad)', 'distance(m)','x(m)','y(m)', 'the(rad)', 'direction'])

    def callback(self, data):
        try:
            # self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

    # def callback_left_camera(self, data):
    #     try:
    #         # self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #         self.cv_left_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
    #     except CvBridgeError as e:
    #         print(e)

    # def callback_right_camera(self, data):
    #     try:
    #         # self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #         self.cv_right_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
    #     except CvBridgeError as e:
    #         print(e)


    def callback_cmd(self, data):
        self.cmd_dir_data = data.intersection_label
        # rospy.loginfo(self.cmd_dir_data)
        # rospy.loginfo(self.cmd_dir_data)

    def callback_dl_training(self, data):
        resp = SetBoolResponse()
        # self.learning = data.data
        self.learning_tensor_flag = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp


    def loop(self):
        test_flag =True
        
        if self.learning_tensor_flag:
            # dataset ,dataset_num,train_dataset =self.dl.make_dataset(img,self.cmd_dir_data)
            # self.dl.training(train_dataset)
            # self.b2t.load(self.load_path)
            # print("load model: ",self.load_path)
            # print(self.load_image_path)
            # print(self.load_label_path)
            x_tensor,t_tensor = self.b2t.cat_tensor(self.load_center_image_path,self.load_left_image_path,self.load_right_image_path,
                                                    self.load_center_label_path,self.load_left_label_path,self.load_right_label_path)
            _,_ = self.b2t.cat_training(x_tensor,t_tensor)
            # _,_ = self.b2t.training(self.load_image_path,self.load_label_path)
        #     # self.b2t.save_bagfile(image_tensor,self.save_image_path,'/image.pt')
        #     # self.b2t.save_bagfile(label_tensor,self.save_label_path, '/label.pt')
            self.b2t.save(self.save_path)
            self.learning_tensor_flag = False
           
            os.system('killall roslaunch')
            sys.exit()
        else :
            print("please start learning")
        if test_flag:
            self.b2t.load(self.load_path)
            # # print("load model: ",self.load_path)
            print(self.load_image_path)
            print(self.load_label_path)
            accuracy = self.b2t.model_test(self.load_image_path,self.load_label_path)
            os.system('killall roslaunch and test model')
            sys.exit()


if __name__ == '__main__':
    rg = intersection_detector_node()
    # DURATION = 0.1
    # r = rospy.Rate(1 / DURATION)
    # r= rospy.Rate(5.0)
    r = rospy.Rate(8.0)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
