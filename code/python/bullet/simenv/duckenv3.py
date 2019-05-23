# objects: duck (left), mug (right)

import pybullet as p
import os
from simenv.kuka_iiwa import kuka_iiwa
from easydict import EasyDict
from utils import sysdatapath
from simenv.env import Env, userdatapath


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
        # h.cubeid = p.loadURDF(sysdatapath("cube_small.urdf"), 1.050000,-0.500000,0.700000,0.707107,0,0,0.707107)
        # objects = [p.loadURDF(sysdatapath("sphere_small.urdf"), 0.850000,-0.400000,0.700000,0.000000,0.000000,0.707107,0.707107)]
        print("FILE PATH", sysdatapath("duck_blue.urdf"))
        h.duckid = p.loadURDF(sysdatapath("duck_green.urdf"), 0.950000,-0.200000,0.700000,0.000000,0.000000,0.707107,0.707107)
        h.mugid = p.loadURDF(userdatapath('cup.urdf'), 0.950000,0.100000,0.700000,0.707107,0,0,0.707107)
        # h.cubeid = p.loadURDF(sysdatapath("cube_small.urdf"), 0.950000,-0.100000,0.700000,0.000000,0.000000,0.707107,0.707107)
        self.h = h
        self.o = o
      
        return h, o

    def objects(self):
        return (self.h.kuka, self.h.duckid, self.h.mugid)

    def reset(self):
        self.o.kukaobject.kuka_reset()
        # p.resetBasePositionAndOrientation(self.h.bowlid, [1.050000,-0.500000,0.700000], [0.707107,0,0,0.707107])
        p.resetBasePositionAndOrientation(self.h.duckid, [0.950000,-0.200000,0.700000], [0.000000,0.000000,0.707107,0.707107])
        p.resetBasePositionAndOrientation(self.h.mugid, [0.950000,0.100000,0.700000], [0.707107,0,0,0.707107])
        # p.resetBasePositionAndOrientation(self.h.cubeid, [0.950000,-0.100000,0.700000], [0.000000,0.000000,0.707107,0.707107])
