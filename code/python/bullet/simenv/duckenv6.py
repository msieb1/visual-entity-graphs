
import pybullet as p
import os
from .kuka_iiwa import kuka_iiwa
from easydict import EasyDict
from utils import sysdatapath
from .env import Env, userdatapath
import time
from ipdb import set_trace
from os.path import join
import math
import numpy as np

class UserEnv(Env):
    """This class must be called 'UserEnv'"""
    def load(self):
        self.prelim = True
        h = EasyDict()  #collection of handler ids
        o = EasyDict()  #collection of objects
        objects = [p.loadURDF(sysdatapath("plane.urdf"), 0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000)]
        #objects = [p.loadURDF("samurai.urdf", 0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000)]
        objects = [p.loadURDF(sysdatapath("pr2_gripper.urdf"), 0.500000,0.300006,0.700000,-0.000000,-0.000000,-0.000031,1.000000)]
        h.pr2_gripper = objects[0]
        print ("pr2_gripper=")
        print (h.pr2_gripper)

        jointPositions=[ 0.550569, 0.000000, 0.549657, 0.000000 ]
        for jointIndex in range (p.getNumJoints(h.pr2_gripper)):
                p.resetJointState(h.pr2_gripper,jointIndex,jointPositions[jointIndex])
                p.setJointMotorControl2(h.pr2_gripper,jointIndex,p.POSITION_CONTROL,targetPosition=0,force=0)

        h.pr2_cid = p.createConstraint(h.pr2_gripper,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0.2,0,0],[0.500000,0.300006,0.700000])
        print ("pr2_cid")
        print (h.pr2_cid)

        h.pr2_cid2 = p.createConstraint(h.pr2_gripper,0,h.pr2_gripper,2,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        p.changeConstraint(h.pr2_cid2,gearRatio=1, erp=0.5, relativePositionTarget=0.5, maxForce=3)

        o.kukaobject = kuka_iiwa(userdatapath())
        h.kuka = o.kukaobject.kukaId
        p.resetBasePositionAndOrientation(h.kuka, [1.3, -0.2, 0.675], [0, 0, 0, 1])
        jointPositions=[ -0.000000, -0.000000, 0.000000, 1.570793, 0.000000, -1.036725, 0.000001, 0, 0, 0]
        for jointIndex in range (p.getNumJoints(h.kuka)):
                p.resetJointState(h.kuka,jointIndex,jointPositions[jointIndex])
                p.setJointMotorControl2(h.kuka,jointIndex,p.POSITION_CONTROL,jointPositions[jointIndex],0)
        o.kukaobject.kuka_reset()

        objects = [p.loadURDF(sysdatapath("table/table.urdf"), 1.000000,-0.200000,0.000000,0.000000,0.000000,0.707107,0.707107)]
        # objects = [p.loadURDF(sysdatapath("toys", "concave_box.urdf"), 1.050000,-0.500000,0.700000,0.000000,0.000000,0.707107,0.707107)]
        # h.bowlid = p.loadURDF(userdatapath('bowl.urdf'), 1.050000,-0.500000,0.700000,0.707107,0,0,0.707107) # original
        # h.duckid = p.loadURDF(sysdatapath("cube_small.urdf"), 1.050000,-0.500000,0.700000,0.707107,0,0,0.707107)
        # objects = [p.loadURDF(sysdatapath("sphere_small.urdf"), 0.850000,-0.400000,0.700000,0.000000,0.000000,0.707107,0.707107)]
        h.duckid = p.loadURDF(sysdatapath("duck_vhacd.urdf"),0.9650000,-0.100000, 0.675000,0.0,0.0,0.707107,0.707107)

        self.h = h
        self.o = o
      
        return h, o

    def objects(self):
        return (self.h.kuka, self.h.duckid)

    def reset(self,init_pose=None):
        target_joint_pos = [-0.052721409309395645, -0.43955548037340625, -0.18695574853350838, 1.7226026878269807, 0.094884109753701, -0.9882597351861896, 0.6465899024499098, 0.6466216250457015, 4.164223524983287e-20, 1.1322904343665968e-07]
        for i in range (p.getNumJoints(self.o.kukaobject.kukaId)):
            p.resetJointState(self.o.kukaobject.kukaId,i,target_joint_pos[i])
        #self.o.kukaobject.open_gripper()
        # rest_orn = [0.70603903128, 0.708148792076, 0, 0]
        # #rest_orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        # rest_pos = [0.9, -0.100000, 1.]
        # self.move_arm(rest_pos, rest_orn)
        self.o.kukaobject.open_gripper()

        orn = [0.70603903128, 0.708148792076, 0, 0]
        pos = [0.96, -0.100000,     0.95]
        self.o.kukaobject.moveKukaEndtoPos(pos, orn)

        # Reset object
        init = np.random.choice([0, -0.5, 0.5, 1])
        init *= math.pi
        init = -math.pi/2.0
        time.sleep(0.2)
        p.resetBasePositionAndOrientation(self.h.duckid, [0.9650000,-0.100000, 0.775000],
                                          p.getQuaternionFromEuler([math.pi/2, 0,init]))
                                           # p.getQuaternionFromEuler([math.pi/2, 0, 0]))
        time.sleep(0.2)
       # x = (np.random.rand(1) - 0.5)/5
        # y = (np.random.rand(1)- 0.5 )/5 
        # z = (np.random.rand(1) - 0.5)/5       
        # p.resetBasePositionAndOrientation(self.h.cubeid, [0.93 + x,-0.480000 + y, 0.65000 + z],
        #                                   p.getQuaternionFromEuler([math.pi/2, 0, 0]))
 

        orn = [0.70603903128, 0.708148792076, 0, 0]
        pos = [0.96, -0.100000, 0.90]
        self.o.kukaobject.moveKukaEndtoPos(pos, orn)
        time.sleep(0.4)
        #self.move_arm(pos, orn)
        self.o.kukaobject.close_gripper()
        time.sleep(0.5)
        #time.sleep(1)
        if init_pose is None:    
            #orn = p.getQuaternionFromEuler([0, -math.pi, 0])
            orn = [0.70603903128, 0.708148792076, 0, 0]
            
            #orn = p.getQuaternionFromEuler([-math.pi/2,0, math.pi/2])
            pos = [1.0, -0.500000, 0.9]
        else:
            orn = init_pose[-4:]
            pos = init_pose[:3]
        # self.o.kukaobject.moveKukaEndtoPos(pos, orn)
        #self.move_arm(pos, orn)

 


