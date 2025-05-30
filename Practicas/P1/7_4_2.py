#!/usr/bin/env python

"""
Programa para controlar la pinza del robot
"""

from niryo_one_python_api.niryo_one_api import *
import rospy
import time
rospy.init_node('niryo_one_example_python_api')

# Inicializamos el brazo
brazo = NiryoOne()

#Iniciamos el robot y hacemos la calibracion automatica
brazo = NiryoOne()
brazo.calibrate_auto()

# Intentamos mover la pinza
try:
    # Le indicamos al robot que cambie a la pinza normal
    brazo.change_tool(TOOL_GRIPPER_1_ID)

    #le damos un tiempo
    time.sleep(3)

    #mueve el brazo a la posicion P1
    brazo.move_pose(-0.03, -0.156, 0.48, -0.58, -0.58, -0.145) 
    time.sleep(3)
    pose_actuel_1 = brazo.get_arm_pose()
    print pose_actuel_1

    # Le indicamos al robot que abra la pinza y espero 2 segundos
    brazo.open_gripper(TOOL_GRIPPER_1_ID, 500)
    brazo.wait(2)

except NiryoOneException as e:      # En caso de error imprime el fallo
    print e

#Activamos el modo aprendizaje 
brazo.activate_learning_mode(True)