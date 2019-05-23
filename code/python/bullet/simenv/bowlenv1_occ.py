# objects: bowl (right), ball (right), duck(right), cube (middle)

import pybullet as p
import os
from kuka_iiwa import kuka_iiwa as kuka_iiwa
from easydict import EasyDict
from utils import sysdatapath
from env import Env, userdatapath
from ipdb import set_trace

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
        h.bowlid = p.loadURDF(userdatapath('bowl.urdf'), 0.850000,-0.090000,0.700000,0.707107,0,0,0.707107)
        h.cubeid = p.loadURDF(sysdatapath("cube_small.urdf"), 0.950000,-0.100000,0.700000,0.000000,0.000000,0.707107,0.707107)
        # h.cube2id = p.loadURDF(userdatapath('cube_red.urdf'), 0.850000,-0.14000,0.700000,0.000000,0.000000,0.707107,0.707107)
        h.duckid = p.loadURDF(sysdatapath("duck_vhacd.urdf"), 0.817506798805501, -0.65142933733046, 0.6541068487469601,0.000000,0.000000,0.707107,0.707107)

        #objects = [p.loadURDF(sysdatapath("sphere_small.urdf"), 0.850000,-0.400000,0.700000,0.000000,0.000000,0.707107,0.707107)]
        h.duck2id = p.loadURDF(sysdatapath("duck_vhacd.urdf"),1.050000,-0.10000,0.700000,0.000000,0.000000,0.707107,0.707107)
        h.duck3id = p.loadURDF(sysdatapath("duck_vhacd.urdf"),.80000,-0.10000,0.700000,0.000000,0.000000,0.707107,0.707107)
        h.duck4id = p.loadURDF(sysdatapath("duck_vhacd.urdf"),.850000,-0.15000,0.700000,0.000000,0.000000,0.707107,0.707107)
        self.h = h
        self.o = o
        return h, o

    def objects(self):
        return (self.h.kuka, self.h.cubeid, self.h.bowlid)

    def reset(self, reset_condition=None):
        # Reset object
        self.o.kukaobject.kuka_reset()
        orn = [0.70603903128, 0.708148792076, 0, 0]
        if reset_condition is None:
            p.resetBasePositionAndOrientation(self.h.cubeid, [0.950000,-0.100000,0.700000], [0.000000,0.000000,0.707107,0.707107])
            p.resetBasePositionAndOrientation(self.h.bowlid, [1.050000,-0.500000,0.700000], [0.707107,0,0,0.707107])
            p.resetBasePositionAndOrientation(self.h.duckid, [0.90000,-0.100000,0.900000], [0.707107,0,0,0.707107])
            p.resetBasePositionAndOrientation(self.h.duck2id, [1.050000,-0.10000,0.700000], [0.707107,0,0,0.707107])
            p.resetBasePositionAndOrientation(self.h.duck3id, [1.050000,-0.10000,0.700000], [0.707107,0,0,0.707107])
            p.resetBasePositionAndOrientation(self.h.duck4id, [1.050000,-0.10000,0.700000], [0.707107,0,0,0.707107])

            pos = [0.950000,-0.100000,0.700000]
        else:
            for key, val in reset_condition.items():
                p.resetBasePositionAndOrientation(int(key), val[:3], val[3:])
                if int(key) == self.h.cubeid:
                    pos = val[:3]
            p.resetBasePositionAndOrientation(self.h.duckid, [0.817506798805501, -0.65142933733046, 0.6541068487469601], [0.707107,0,0,0.707107])
            p.resetBasePositionAndOrientation(self.h.duck2id, [1.050000,-0.10000,0.700000], [0.707107,0,0,0.707107])
            p.resetBasePositionAndOrientation(self.h.duck3id, [.80000,-0.10000,0.700000], [0.707107,0,0,0.707107])
            p.resetBasePositionAndOrientation(self.h.duck4id, [.850000,-0.15000,0.700000], [0.707107,0,0,0.707107])

        # target_joint_pos = [-0.052721409309395645, -0.43955548037340625, -0.18695574853350838, 1.7226026878269807, 0.094884109753701, -0.9882597351861896, 0.6465899024499098, 0.6466216250457015, 4.164223524983287e-20, 1.1322904343665968e-07]
        # for i in range (p.getNumJoints(self.o.kukaobject.kukaId)):
        #     p.resetJointState(self.o.kukaobject.kukaId,i,target_joint_pos[i])
        self.o.kukaobject.kuka_reset()

        self.o.kukaobject.open_gripper()
        # Hover Pose
        pos[2] += 0.2
        self.o.kukaobject.moveKukaEndtoPos(pos, orn)
        pos[2] -= 0.047
        # Grab object
        self.o.kukaobject.moveKukaEndtoPos(pos, orn)
        self.o.kukaobject.close_gripper()
        self.o.kukaobject.gripper_centered()




        # pos = [0.9032135605812073, -0.5798930525779724, 0.8500000357627868]
        # pos[2] += 0.25
        # # for kuka_sec in range(500):

        # # #     jointPoses = p.calculateInverseKinematics(
        # # #         self.o.kukaobject.kukaId,
        # # #         self.o.kukaobject.kukaEndEffectorIndex,
        # # #         pos,
        # # #         orn,
        # # #         lowerLimits=self.o.kukaobject.ll,
        # # #         upperLimits=self.o.kukaobject.ul,
        # # #         jointRanges=self.o.kukaobject.jr,
        # # #         restPoses=self.o.kukaobject.rp)
        # # #     for i in range(self.o.kukaobject.numJoints):
        # # #         p.setJointMotorControl2(bodyIndex=self.o.kukaobject.kukaId,jointIndex=i,controlMode=p.POSITION_CONTROL,\
        # # #           targetPosition=jointPoses[i],targetVelocity=0,force=self.o.kukaobject.maxForce,positionGain=0.03,velocityGain=1)
        # # #     p.setJointMotorControl2(bodyIndex=self.o.kukaobject.kukaId,jointIndex=7,controlMode=p.POSITION_CONTROL,\
        # # #               targetPosition=0,targetVelocity=0,force=self.o.kukaobject.maxForce,positionGain=0.03,velocityGain=1)
        # # #     time.sleep(0.001)
        # # target_joint_pos = [-0.052721409309395645, -0.43955548037340625, -0.18695574853350838, 1.7226026878269807, 0.094884109753701, -0.9882597351861896, 0.6465899024499098, 0.6466216250457015, 4.164223524983287e-20, 1.1322904343665968e-07]
        # for i in range (p.getNumJoints(self.o.kukaobject.kukaId)):
        #     p.setJointMotorControl2(self.o.kukaobject.kukaId,i,p.POSITION_CONTROL,targetPosition=target_joint_pos[i],force=self.o.kukaobject.maxForce)

        #     # self.o.kukaobject.open_gripper()
        # pos[2] -= 0.25
        # for kuka_sec in range(500):
        #     jointPoses = p.calculateInverseKinematics(
        #         self.o.kukaobject.kukaId,
        #         self.o.kukaobject.kukaEndEffectorIndex,
        #         pos,
        #         orn,
        #         lowerLimits=self.o.kukaobject.ll,
        #         upperLimits=self.o.kukaobject.ul,
        #         jointRanges=self.o.kukaobject.jr,
        #         restPoses=self.o.kukaobject.rp)
        #     for i in range(self.o.kukaobject.numJoints):
        #         p.setJointMotorControl2(bodyIndex=self.o.kukaobject.kukaId,jointIndex=i,controlMode=p.POSITION_CONTROL,\
        #           targetPosition=jointPoses[i],targetVelocity=0,force=self.o.kukaobject.maxForce,positionGain=0.03,velocityGain=1)
        #     p.setJointMotorControl2(bodyIndex=self.o.kukaobject.kukaId,jointIndex=7,controlMode=p.POSITION_CONTROL,\
        #               targetPosition=0,targetVelocity=0,force=self.o.kukaobject.maxForce,positionGain=0.03,velocityGain=1)
        #     time.sleep(0.001)

            
        # # target_joint_pos =[0.7285943515687326, -0.9477748830689287, -0.12030294597421631, 1.4033022633001055, 0.1364471977598385, -0.7988572538697987, 2.130515422546533, 3.094484075162794e-05, -0.0286964290384345, 0.02864940567748811]
        # for i in range (p.getNumJoints(self.o.kukaobject.kukaId)):
        #     p.setJointMotorControl2(self.o.kukaobject.kukaId,i,p.POSITION_CONTROL,targetPosition=target_joint_pos[i],force=self.o.kukaobject.maxForce)
        # self.o.kukaobject.close_gripper()
        # self.o.kukaobject.gripper_centered()
        # self.o.kukaobject.endEffectorPos = [0.9, -0.6, 0.7]
        # self.o.kukaobject.endEffectorAngle = 0
        # time.sleep(0.001)