#!/usr/bin/env python3

import math
import sys
import xml.dom.minidom

import rospy
from px4_msgs.msg import ActuatorMotors
from px4_msgs.msg import ActuatorServos
from px4_msgs.msg import ControlAllocationMetaData
from sensor_msgs.msg import JointState

# For meta data visualization
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

'''
header:
  seq: 13360
  stamp:
    secs: 1698155524
    nsecs: 675285577
  frame_id: ''
name:
  - module1_body_gimbal_joint
  - module1_gimbal_actuator_joint
  - module1_actuator_prop1_joint
  - module1_actuator_prop2_joint
  - module2_body_gimbal_joint
  - module2_gimbal_actuator_joint
  - module2_actuator_prop1_joint
  - module2_actuator_prop2_joint
  - module3_body_gimbal_joint
  - module3_gimbal_actuator_joint
  - module3_actuator_prop1_joint
  - module3_actuator_prop2_joint
  - module4_body_gimbal_joint
  - module4_gimbal_actuator_joint
  - module4_actuator_prop1_joint
  - module4_actuator_prop2_joint
position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
velocity: []
effort: []
'''

def get_param(name, value=None):
    private = "~%s" % name
    if rospy.has_param(private):
        return rospy.get_param(private)
    elif rospy.has_param(name):
        return rospy.get_param(name)
    else:
        return value
    
class Republisher:
    def __init_urdf(self, robot):
        # https://github.com/ros/joint_state_publisher/blob/noetic-devel/joint_state_publisher/src/joint_state_publisher/__init__.py
        robot = robot.getElementsByTagName('robot')[0]
        # Find all non-fixed joints
        for child in robot.childNodes:
            if child.nodeType is child.TEXT_NODE:
                continue
            if child.localName == 'joint':
                jtype = child.getAttribute('type')
                if jtype in ['fixed', 'floating', 'planar']:
                    continue
                name = child.getAttribute('name')
                self.joint_list.append(name)
                if jtype == 'continuous':
                    minval = -math.pi
                    maxval = math.pi
                else:
                    try:
                        limit = child.getElementsByTagName('limit')[0]
                        minval = float(limit.getAttribute('lower'))
                        maxval = float(limit.getAttribute('upper'))
                    except:
                        rospy.logwarn("%s is not fixed, nor continuous, but limits are not specified!" % name)
                        continue

                safety_tags = child.getElementsByTagName('safety_controller')
                if self.use_small and len(safety_tags) == 1:
                    tag = safety_tags[0]
                    if tag.hasAttribute('soft_lower_limit'):
                        minval = max(minval, float(tag.getAttribute('soft_lower_limit')))
                    if tag.hasAttribute('soft_upper_limit'):
                        maxval = min(maxval, float(tag.getAttribute('soft_upper_limit')))

                mimic_tags = child.getElementsByTagName('mimic')
                if self.use_mimic and len(mimic_tags) == 1:
                    tag = mimic_tags[0]
                    entry = {'parent': tag.getAttribute('joint')}
                    if tag.hasAttribute('multiplier'):
                        entry['factor'] = float(tag.getAttribute('multiplier'))
                    if tag.hasAttribute('offset'):
                        entry['offset'] = float(tag.getAttribute('offset'))

                    self.dependent_joints[name] = entry
                    continue

                if name in self.dependent_joints:
                    continue

                zeroval = get_param("zeros/" + name)
                if not zeroval:
                    if minval > 0 or maxval < 0:
                        zeroval = (maxval + minval)/2
                    else:
                        zeroval = 0

                joint = {'min': minval, 'max': maxval, 'zero': zeroval}
                if self.pub_def_positions:
                    joint['position'] = zeroval
                if self.pub_def_vels:
                    joint['velocity'] = 0.0
                if self.pub_def_efforts:
                    joint['effort'] = 0.0

                if jtype == 'continuous':
                    joint['continuous'] = True
                self.free_joints[name] = joint

    def __init__(self) -> None:
        rospy.init_node('tvmd_republisher')
        self.last_timestamp = 0
        self.num_modules = 4

        self.free_joints = {}
        self.joint_list = [] # for maintaining the original order of the joints
        self.dependent_joints = get_param("dependent_joints", {})
        self.use_mimic = get_param('use_mimic_tags', True)
        self.use_small = get_param('use_smallest_joint_limits', True)

        self.pub_def_positions = get_param("publish_default_positions", True)
        self.pub_def_vels = get_param("publish_default_velocities", False)
        self.pub_def_efforts = get_param("publish_default_efforts", False)

        description = get_param('robot_description')
        if description is None:
            raise RuntimeError('The robot_description parameter is required and not set.')
        robot = xml.dom.minidom.parseString(description)
        self.__init_urdf(robot)

        # joint state message
        self.joint_msg = JointState()
        self.joint_msg.position = [0.0 for i in range(4 * self.num_modules)]
        # joint_names = ["_body_gimbal_joint", "_gimbal_actuator_joint", "_actuator_prop1_joint", "_actuator_prop2_joint"]
        # self.joint_msg.name = ["module{}{}".format(i+1, n) for i in range(self.num_modules) for n in joint_names]
        self.joint_msg.name = [str(name) for name in self.joint_list]
        self.msg_updating = False

        # joint state publisher
        self.pub = rospy.Publisher("joint_states", JointState, queue_size=1)

        # TODO: Load parameters from urdf file, and calculate the scales for joints

    def data_update(self, data, mode):
        servo_offset = 0 if mode == "servos" else 2
        for i in range(4):
            for j in range(2):
                # the order of the joints in the message is different from the order in the urdf file
                # the order in the message is: module1_body_gimbal_joint, module1_gimbal_actuator_joint, module1_actuator_prop1_joint, module1_actuator_prop2_joint
                # i.e., y, x, p1, p2
                joint_idx = 4*i + (1-j) + servo_offset
                ctrl_idx = 2*i + j
                joint_name = self.joint_list[joint_idx]
                joint = self.free_joints[joint_name]

                if mode == "servos":
                    factor = (joint['max'] - joint['min']) / 2
                    offset = (joint['max'] + joint['min']) / 2
                    
                    self.joint_msg.position[joint_idx] = data.control[ctrl_idx] * factor + offset
                else:
                    self.joint_msg.position[joint_idx] += data.control[ctrl_idx]

            # if mode == "servos":
            #     self.joint_msg.position[4*i+0] = data.control[2*i+0]
            #     self.joint_msg.position[4*i+1] = data.control[2*i+1]
            # else:
            #     self.joint_msg.position[4*i+2] += data.control[2*i+0]
            #     self.joint_msg.position[4*i+3] += data.control[2*i+1]
        
        upcoming_timestamp = data.timestamp

        # the packet is completed
        if upcoming_timestamp == self.last_timestamp:
            self.joint_msg.header.stamp = rospy.Time.now()
            self.pub.publish(self.joint_msg)

        self.last_timestamp = data.timestamp
    
    def motor_listener(self, data):        
        self.data_update(data, "motors")
    
    def servo_listener(self, data):
        self.data_update(data, "servos")
    
    def meta_data_listener(self, data):
        pass

    def run(self):
        # Allow Rviz to initialize robot model
        hz = get_param("rate", 10)  # 10hz
        r = rospy.Rate(hz)
        for k in range(50):
            self.joint_msg.header.stamp = rospy.Time.now()
            self.pub.publish(self.joint_msg)
            r.sleep()
        
        print("Start republishing")
        rospy.Subscriber("px4/actuator_motors", ActuatorMotors, self.motor_listener)
        rospy.Subscriber("px4/actuator_servos", ActuatorServos, self.servo_listener)
        rospy.Subscriber("px4/control_allocation_meta_data", ControlAllocationMetaData, self.meta_data_listener)
        rospy.spin()
        return

if __name__ == '__main__':
    rep = Republisher()
    rep.run()