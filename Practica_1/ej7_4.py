#!/usr/bin/env python
from niryo_one_python_api.niryo_one_api import *
import rospy
import time
rospy.init_node('niryo_one_example_python_api')
n = NiryoOne()
n.calibrate_auto()
n.activate_learning_mode(False)
# desactiva el modo de aprendizaje
for i in range (0, 5):
    n.move_pose(-0.03, -0.156, 0.48, -0.58, -0.58, -0.145) # mueve el robot a P1
    time.sleep(3)
    pose_actuel_1 = n.get_arm_pose()
    print pose_actuel_1
    n.move_pose(-0.136, -0.133, 0.255, -0.081, 0.744, -2.535) # mueve el robot a P2
    time.sleep(3)
    pose_actuel_2 = n.get_arm_pose()
    print pose_actuel_2
n.activate_learning_mode(True)