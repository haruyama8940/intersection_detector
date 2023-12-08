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
        self.image_sub = rospy.Subscriber("/image_center", Image, self.callback)
        # self.image_sub = rospy.Subscriber("/image_left", Image, self.callback)
        # self.image_sub = rospy.Subscriber("/image_right", Image, self.callback)
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
        self.save_tensor_flag = False
        self.cat_tensor_flag = True
        self.select_dl = False
        self.start_time = time.strftime("%Y%m%d_%H:%M:%S")
        # self.path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/result'
        # self.save_image_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/image/single/'
        # self.save_label_path = roslib.packages.get_pkg_dir('intersection_detector') + '/data/dataset/label/single/'
        self.save_image_path = '/home/rdclab/Data/tensor/intersection_detactor/dataset/single/light_dark/image/'
        self.save_label_path = '/home/rdclab/Data/tensor/intersection_detactor/dataset/single/light_dark/label/'

        self.load_image_tensor_1 = '/home/rdclab/Data/tensor/intersection_detactor/dataset/single/dark/image/center/image.pt'
        self.load_image_tensor_2 = '/home/rdclab/Data/tensor/intersection_detactor/dataset/single/dark/image/right/image.pt'
        self.load_image_tensor_3 = '/home/rdclab/Data/tensor/intersection_detactor/dataset/single/dark/image/left/image.pt' 
        self.load_label_tensor_1 = '/home/rdclab/Data/tensor/intersection_detactor/dataset/single/dark/label/center/label.pt'
        self.load_label_tensor_2 = '/home/rdclab/Data/tensor/intersection_detactor/dataset/single/dark/label/right/label.pt'
        self.load_label_tensor_3 = '/home/rdclab/Data/tensor/intersection_detactor/dataset/single/dark/label/left/label.pt'

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
        self.save_tensor_flag = data.data
        resp.message = "Training: " + str(self.learning)
        resp.success = True
        return resp


    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        # if self.cv_left_image.size != 640 * 480 * 3:
        #     return
        # if self.cv_right_image.size != 640 * 480 * 3:
        #     return
        img = resize(self.cv_image, (48, 64), mode='constant')
        # img = resize(self.cv_image, (224, 224), mode='constant')
        
        # rospy.loginfo("start")
        # r, g, b = cv2.split(img)
        # img = np.asanyarray([r,g,b])

        # img_left = resize(self.cv_left_image, (48, 64), mode='constant')
        #r, g, b = cv2.split(img_left)
        #img_left = np.asanyarray([r,g,b])

        # img_right = resize(self.cv_right_image, (48, 64), mode='constant')
        #r, g, b = cv2.split(img_right)
        #img_right = np.asanyarray([r,g,b])
        # cmd_dir = np.asanyarray(self.cmd_dir_data)
        ros_time = str(rospy.Time.now())

        
        # dataset ,dataset_num,train_dataset =self.dl.make_dataset(img,self.cmd_dir_data)
        image_tensor ,label_tensor =self.b2t.make_dataset(img,self.cmd_dir_data)
        # intersection, loss = self.dl.act_and_trains(img , self.cmd_dir_data)
        # intersection_left,loss_left = self.dl.act_and_trains(img_left,self.cmd_dir_data)
        # intersection_right , loss_right = self.dl.act_and_trains(img_right, self.cmd_dir_data)
                # end mode
        # intersection_name = self.intersection_list[intersection]
        # ans_intersection =self.intersection_list[self.cmd_dir_data.index(max(self.cmd_dir_data))]
        # self.intersection.intersection_name = self.intersection_list[intersection]
        # self.intersection_pub.publish(self.intersection)
        #print("learning: " + str(self.episode) + ", loss: " + str(loss) + ", label: " + str(intersection) + " , intersection_name: " + str(intersection_name)+" , answer_name: " + str(ans_intersection))
        # print("learning: " + str(self.episode) + ", loss: " + str(loss) + ", label: " + str(intersection) + " , intersection_name: " + str(intersection_name) +", correct label: " + str(self.cmd_dir_data))
        # self.episode += 1
        # line = [str(self.episode), "training", str(loss), str(angle_error), str(distance), str(self.pos_x), str(self.pos_y), str(self.pos_the), str(self.cmd_dir_data)]
        # with open(self.path + self.start_time + '/' + 'training.csv', 'a') as f:
        #     writer = csv.writer(f, lineterminator='\n')
        #     writer.writerow(line)
        if self.cat_tensor_flag:
            image_tensor,label_tensor=self.b2t.cat_tensor(self.load_image_tensor_1,self.load_image_tensor_2,self.load_image_tensor_3,
                                                          self.load_label_tensor_1,self.load_label_tensor_2,self.load_label_tensor_3)
            self.b2t.save_bagfile(image_tensor,self.save_image_path,'/image.pt')
            self.b2t.save_bagfile(label_tensor,self.save_label_path, '/label.pt')
            print(image_tensor.shape)
            print(label_tensor.shape)
            print(self.save_image_path)
            print(self.save_label_path)
            os.system('killall roslaunch')
            sys.exit()
        if self.save_tensor_flag:
            # dataset ,dataset_num,train_dataset =self.dl.make_dataset(img,self.cmd_dir_data)
            # self.dl.training(train_dataset)
            image_tensor ,label_tensor=self.b2t.make_dataset(img,self.cmd_dir_data)
            self.b2t.save_bagfile(image_tensor,self.save_image_path,'/image.pt')
            self.b2t.save_bagfile(label_tensor,self.save_label_path, '/label.pt')
            self.save_tensor_flag = False
            print(self.save_image_path)
            print(self.save_label_path)
            os.system('killall roslaunch')
            sys.exit()
        else :
            pass


if __name__ == '__main__':
    rg = intersection_detector_node()
    # DURATION = 0.1
    # r = rospy.Rate(1 / DURATION)
    # r= rospy.Rate(5.0)
    r = rospy.Rate(8.0)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()
