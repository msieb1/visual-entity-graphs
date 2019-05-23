import pybullet as p
import numpy as np
import os
import math
from time import sleep
import scipy
from . import grisp_proposal
from ipdb import set_trace
# parameteres for kuka
#lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#restposes for null space
# rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0, 0, -0.05, 0.05]
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]# 0, -0.05, 0.05]

#joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
hand_height = 0.3


class kuka_iiwa:
    def __init__(self,
                 urdfRootPath='/home/msieb/libs/pybullet_data/data',
                 pos=[-0.2, 0.0, -.12],
                 orn=[0, 0, 0, 1.0],
                 render=True,
                 global_t=0):
        self.lf_id = 8
        self.rf_id = 9
        self.gripper_close_pose = 0.04
        self.maxForce = 200  #200.
        self.scale = 0.01
        self.object_gripper_offset = 0
        self.fingerTipForce = 20
        self.is_gripper_open = False
        print(urdfRootPath)
        print(os.path.join(urdfRootPath, "kuka_iiwa/kuka_with_gripper5.sdf"))
        self.kukaId = p.loadSDF(
            os.path.join(urdfRootPath, "kuka_iiwa/kuka_with_gripper5.sdf"),
            globalScaling=1)[0]
        self.kukaEndEffectorIndex = 7  #Link index of gripper base
        p.resetBasePositionAndOrientation(self.kukaId, pos, orn)
        self.numJoints = p.getNumJoints(self.kukaId)
        far = 1
        #near = 0.001
        #self.cam_viewMatrix = list(p.computeViewMatrix(cameraEyePosition=[0, 0, 0.5],
        #  cameraTargetPosition=[0.4, 0, 0.2], cameraUpVector = [0.0, 0.0, 1.0]))
        #self.cam_projMatrix = p.computeProjectionMatrixFOV(80, 1, near, far);
        #self.img_size_h = 200#700
        near = 0.1
        self.cam_viewMatrix = list(
            p.computeViewMatrix(
                cameraEyePosition=[-0.2, 0, 0.04],
                cameraTargetPosition=[0.4, 0, -0.1],
                cameraUpVector=[0.6, 0.0, 0.8]))
        self.cam_projMatrix = p.computeProjectionMatrixFOV(100, 1, near, far)
        self.img_size_h = 360  #700
        self.img_size_w = 360  #1280
        self.far = far
        self.near = near
        self.render = render
        self.global_t = global_t  #for debugging
        self.default = {}

    def save_default_pose(self, pos, orn, jointInfo):
        self.default['pos'], self.default['orn'] = self.getBasePositionAndOrientation(self.kukaId)
        self.default['joints'] = []
        for i in range(self.numJoints):
            self.default['joints'].append(p.getJointState(self.kukaId, i))

    def kuka_reset(self):
        if len(self.default) == 0:
            self.open_gripper()
            orn = p.getQuaternionFromEuler([0, -math.pi, 0])
            self.moveKukaEndtoPos([0.858, -0.152, 1.1615], orn)
        else:
            p.resetBasePositionAndOrientation(self.kukaId, self.default['pos'], self.default['orn'])
            for i in range(self.numJoints):
                p.resetJointState(self.kukaId, i, self.default['joints'][i])

    def get_observation(self, get_image=True):
        ori_pos = np.array(
            p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0])
        state_dict = dict()
        state_dict['gripper_pos'] = ori_pos

        if get_image:
            img_arr = p.getCameraImage(width=self.img_size_w,height=self.img_size_h, viewMatrix=self.cam_viewMatrix, \
                         projectionMatrix=self.cam_projMatrix)
            rgb_array = img_arr[2]

            seg = img_arr[4]
            depth = self.far * self.near / (
                self.far - (self.far - self.near) * img_arr[3])
            #print("max seg", np.amax(np.array(seg)))
            #print("save observations...", str(self.global_t))
            #import scipy.misc
            #scipy.misc.imsave("depth_" + str(self.global_t) + ".png", depth)
            #scipy.misc.imsave("rgb_" + str(self.global_t) + ".png", rgb_array[:,:,:3])
            self.global_t += 1
            #if compress:
            #  return rgb_array/255.0 - 0.5
            #return self.compress_state(rgb_array, ori_pos)
            #else:
            state_dict['rgb'] = rgb_array[:, :, :3]
            state_dict['seg'] = seg
            state_dict['depth'] = depth
        return state_dict

    def decompress_state(self, state):
        state_dict = dict()
        state_dict['gripper_pos'] = np.array(state[:3])
        w, h = state[3:5]
        state_dict['rgb'] = np.array(state[5:]).reshape((w, h, 3))
        return state_dict

    def compress_state(self, rgb_array, ori_pos):
        w, h, _ = rgb_array.shape
        state = list(ori_pos)
        state.extend([w, h])
        state.extend(list(rgb_array.flatten()))
        return state

    def apply_action(self, action):
        dist = 0.1
        #print('apply action', action)
        if action == 'x+':
            self.move_gripper([dist, 0, 0])
        elif action == 'x-':
            self.move_gripper([-dist, 0, 0])
        elif action == 'y+':
            self.move_gripper([0, dist, 0])
        elif action == 'y-':
            self.move_gripper([0, -dist, 0])
        elif action == 'release':
            self.move_gripper([0, 0, -0.15])
            self.open_gripper()
        else:
            assert (1 == 2), 'action not supported' + action

    def move_gripper(self, delta):
        state_dict = self.get_observation(get_image=False)
        newxy = stat, e_dict['gripper_pos']
        # print('delta', delta, newxy)
        for i in range(3):
            newxy[i] += delta[i]
        #print('newxy',newxy)
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        self.moveKukaEndtoPos(newxy, orn)

    def apply_actions2(self, action, objects, object_info, noise=0):
        self.object_info = object_info
        if action.startswith("grab:"):
            return self.grab(objects[action.split(":")[1]])
        elif action.startswith("release to container:"):
            container = action.split(":")[1]
            self.release_to_container(container)
        elif action.startswith("stack on:"):
            obj2 = action.split(":")[1]
            self.stack_on(obj2)
        elif action.startswith("approach:"):
            object_name = action.split(":")[1]
            return self.moveto(objects[object_name], noise=noise)

        elif action.startswith("random move:"):
            rand = np.random.rand(3)
            # don't care about the z axis
            #rand = [max(rand[i] * objects[i], 0) for i in range(2)] + [0.0]
            rand = [rand[i] * objects[i] for i in range(2)] + [0.0]

            rd_sign = np.random.randint(2, size=3) * 2 - 1
            move = [rd_sign[i] * rand[i] for i in range(3)]
            #print('rand', rand, 'rd_sign', rd_sign, 'move', move)
            self.move_gripper(move)

        elif action.startswith("release:"):
            #_, pos = action.split(":")
            #pos = [float(ppos)*self.scale for ppos in pos.split(",")]
            return self.release()
            #print("<<<<<<<<<<<<<pos", pos)

        elif action.startswith("move_to:"):
            _, pos = action.split(":")
            pos = [float(ppos) * self.scale for ppos in pos.split(",")]
            return self.move_to_loc(pos)

        else:
            assert 1 == 2, "no such action:" + action

    def grab(self, obj_id):
        commend = []
        ori_pos = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]
        pixel2PosRatio = 300.0
        far = 5
        near = 0.01
        #print("ori pos", ori_pos, self.is_gripper_open)
        cam1_viewMatrix = list(
            p.computeViewMatrix(
                cameraEyePosition=[ori_pos[0], ori_pos[1], 2.0],
                cameraTargetPosition=[ori_pos[0], ori_pos[1], -1],
                cameraUpVector=[0.0, 1.0, 0.0]))
        cam1_projMatrix = p.computeProjectionMatrixFOV(30, 1, near, far)
        img_size = 277
        img_arr1 = p.getCameraImage(width=img_size,height=img_size, viewMatrix=cam1_viewMatrix, \
                                    projectionMatrix=cam1_projMatrix)
        depth = far * near / (far - (far - near) * img_arr1[3])
        mask = img_arr1[4]
        scipy.misc.imsave("kuka.png", depth)
        #scipy.io.savemat("rgb_depth.mat", mdict={'rgb':img_arr1[2][:,:,:3], 'depth':depth, 'mask': img_arr1[4]})
        #center, angle, _, gripper_pos  = grisp_proposal.depth2prop(depth)
        center, angle, _, gripper_pos = grisp_proposal.mask2prop(mask, obj_id)

        [x1, y1, x2, y2] = [ori_pos[0] + gripper_pos[0]/pixel2PosRatio, ori_pos[1] - gripper_pos[1]/pixel2PosRatio,\
                            ori_pos[0] + gripper_pos[2]/pixel2PosRatio, ori_pos[1] - gripper_pos[3]/pixel2PosRatio]

        #p.addUserDebugLine([x1, y1, 0.1], [x2, y2, 0.1], [0,0,0.2],10, 100)
        #p.addUserDebugLine([x1, y1, 0.1], [x2 + 0.4*(x2-x1), y2 + 0.4*(y2-y1), 0.1], [0,0,0.2],10, 100)

        new_x = ori_pos[0] + center[0] / pixel2PosRatio
        new_y = ori_pos[1] - center[1] / pixel2PosRatio
        #print("newx and newy:", center, new_x, new_y, angle)

        newxy = [new_x, new_y, ori_pos[2]]
        orn = p.getQuaternionFromEuler([0, -math.pi, angle])

        self.moveKukaEndtoPos(newxy, orn)
        commend.append(self.record_seq(newxy, orn))
        ori_pos = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]
        #print("after_pos", ori_pos)
        #img_arr1 = p.getCameraImage(width=img_size,height=img_size, viewMatrix=cam1_viewMatrix, \
        #                            projectionMatrix=cam1_projMatrix)
        #scipy.misc.imsave("after_move.png", img_arr1[2][:,:,:3])

        newxy[2] -= 0.65
        self.moveKukaEndtoPos(newxy, orn)
        commend.append(self.record_seq(newxy, orn))
        newxy[2] -= 0.65
        self.moveKukaEndtoPos(newxy, orn)
        commend.append(self.record_seq(newxy, orn))
        self.close_gripper()
        commend.append("close")

        newxy[2] += 0.2
        self.moveKukaEndtoPos(newxy, orn)
        commend.append(self.record_seq(newxy, orn))

        #orn = p.getQuaternionFromEuler([0,-math.pi, 0])
        #self.moveKukaEndtoPos(newxy, orn)
        #commend.append(self.record_seq(newxy, orn))

        ori_pos = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]
        object_pos, _ = p.getBasePositionAndOrientation(obj_id)
        self.object_gripper_offset = np.array(ori_pos) - np.array(object_pos)
        return commend

    def get_gripper_pose(self):
        pack = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)
        pose = list(pack[0]) + list(pack[1])
        return pose

    def release(self):
        commend = []
        #pos[2] += 0.8
        #middle_point = (np.array(list(ori_pos[0])) + np.array(pos))/2
        #self.moveKukaEndtoPos(middle_point, ori_pos[1])
        #print("ori pos", ori_pos, "target", pos)
        #self.moveKukaEndtoPos(pos, ori_pos[1])
        self.open_gripper()
        commend.append("open")
        ori_pos = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)
        ori_pos_ = list(ori_pos[0])
        ori_pos_[2] += 0.5
        self.moveKukaEndtoPos(ori_pos_, ori_pos[1])
        commend.append(self.record_seq(ori_pos_, ori_pos[1]))
        ori_pos_[2] += 1.5
        self.moveKukaEndtoPos(ori_pos_, ori_pos[1])
        commend.append(self.record_seq(ori_pos_, ori_pos[1]))
        return commend

    def record_seq(self, pos, orn):
        return ",".join([str(x) for x in list(pos) + list(orn)])

    def move_to_loc(self, pos):
        commend = []
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        orn = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[1]

        self.moveKukaEndtoPos(pos, orn)
        commend.append(self.record_seq(pos, orn))
        return commend

    def release_to_container(self, container=None):
        self.open_gripper()

    def stack_on(self, obj_down=None):
        ori_pos = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)[0]
        ori_pos = list(ori_pos)
        ori_pos[2] = 0.2

        orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        self.moveKukaEndtoPos(ori_pos, orn)
        self.open_gripper()
        ori_pos[2] = hand_height
        self.moveKukaEndtoPos(ori_pos, orn)

    def moveto(self, object_id, noise=0):
        commend = []
        newxy, Orn = p.getBasePositionAndOrientation(object_id)
        newxy = list(newxy)
        #newxy[2] = hand_height
        newxy[2] += 2.0
        rand = np.random.rand(3)
        rand = [rand[i] * noise for i in range(2)] + [0.0]
        rd_sign = np.random.randint(2, size=3) * 2 - 1
        move = [rd_sign[i] * rand[i] for i in range(3)]
        newxy[0] = newxy[0] + move[0]
        newxy[1] = newxy[1] + move[1]

        orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        self.moveKukaEndtoPos(newxy, orn)
        commend.append(self.record_seq(newxy, orn))

        return commend

    """
  def snapshot_kuka(self, objects):
    link_state = dict()
    link_states = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)
    link_state[self.kukaEndEffectorIndex] = dict()
    link_state[self.kukaEndEffectorIndex]['pos'] = np.array(link_states[0])
    link_state[self.kukaEndEffectorIndex]['orn'] = np.array(link_states[1])
    lf_id = 8
    rf_id = 9
    link_states = p.getLinkState(self.kukaId, lf_id)
    link_state[lf_id] = dict()
    link_state[lf_id]['pos'] = np.array(link_states[0])
    link_state[lf_id]['orn'] = np.array(link_states[1])
    link_states = p.getLinkState(self.kukaId, rf_id)
    link_state[rf_id] = dict()
    link_state[rf_id]['pos'] = np.array(link_states[0])
    link_state[rf_id]['orn'] = np.array(link_states[1])
    
    link_state['gripper_close'] = self.is_gripper_open
    print(link_state)

    return link_state


  def load_kuka_snapshot(self, link_state, objects):
    pos = link_state[self.kukaEndEffectorIndex]['pos']
    orn = link_state[self.kukaEndEffectorIndex]['orn']

    self.moveKukaEndtoPos(pos, orn)  
    
    lf_id = 8
    rf_id = 9
    p.setJointMotorControl2(bodyIndex=self.kukaId,jointIndex=lf_id,controlMode=p.POSITION_CONTROL,\
      targetPosition=-pose,targetVelocity=0,force=self.fingerTipForce)
    p.setJointMotorControl2(bodyIndex=self.kukaId,jointIndex=rf_id,controlMode=p.POSITION_CONTROL,\
      targetPosition=pose,targetVelocity=0,force=self.fingerTipForce)
  """

    #""" This is not working
    def snapshot_kuka(self):
        num_link = p.getNumJoints(self.kukaId)
        link_state = dict()
        for link_id in range(num_link):
            jp, jv, _, _ = p.getJointState(self.kukaId, link_id)
            link_state[link_id] = dict()
            link_state[link_id]['jp'] = jp
            link_state[link_id]['jv'] = jv
        link_state['gripper_close'] = self.is_gripper_open
        #objects

        return link_state

    def load_kuka_snapshot(self, link_state):
        num_link = p.getNumJoints(self.kukaId)
        for link_id in range(num_link):
            jp = link_state[link_id]['jp']
            jv = link_state[link_id]['jv']

        for kuka_sec in range(500):
            for link_id in range(num_link):
                p.setJointMotorControl2(bodyIndex=self.kukaId,jointIndex=link_id,controlMode=p.POSITION_CONTROL,\
                  targetPosition=link_state[link_id]['jp'],targetVelocity=link_state[link_id]['jv'],force=self.maxForce,positionGain=0.03,velocityGain=1)
            p.stepSimulation()
            if self.render:
                if kuka_sec < 50:
                    sleep(0.005)
                else:
                    sleep(0.001)

            #p.resetJointState(self.kukaId, link_id, jp, jv)
        self.is_gripper_open = link_state['gripper_close']

    def moveKukaEndtoPos(self, newxy, orn=None):
        if orn is None:
            orn = (-0.707, -0.707, 0, 0)  # so gripper is always pointing down
        kuka_min_height = 0.0  #limit min height
        newxy[2] = max(kuka_min_height, newxy[2])
        for kuka_sec in range(500):
            jointPoses = p.calculateInverseKinematics(
                self.kukaId,
                self.kukaEndEffectorIndex,
                newxy,
                orn,
                lowerLimits=ll,
                upperLimits=ul,
                jointRanges=jr,
                restPoses=rp)
            for i in range(self.numJoints):
                p.setJointMotorControl2(bodyIndex=self.kukaId,jointIndex=i,controlMode=p.POSITION_CONTROL,\
                  targetPosition=jointPoses[i],targetVelocity=0,force=self.maxForce,positionGain=0.03,velocityGain=1)
            self.gripper_stabler()

            p.stepSimulation()
            if self.render:
                if kuka_sec < 50:
                    sleep(0.001)
                else:
                    sleep(0.0001)

    def instantMoveKukaEndtoPos(self, newxy=None, orn=None):
        if orn is None:
            orn = (-0.707, -0.707, 0, 0)  # so gripper is always pointing down
        if newxy is None:
            newxy = self.get_gripper_pose()[:3]
        kuka_min_height = 0  #limit min height
        newxy = list(newxy)
        newxy[2] = max(kuka_min_height, newxy[2])

        jointPoses = p.calculateInverseKinematics(
            self.kukaId,
            self.kukaEndEffectorIndex,
            newxy,
            orn,
            lowerLimits=ll,
            upperLimits=ul,
            jointRanges=jr,
            restPoses=rp)
        for i in range(self.numJoints):
            p.setJointMotorControl2(bodyIndex=self.kukaId,jointIndex=i,controlMode=p.POSITION_CONTROL,\
              targetPosition=jointPoses[i],targetVelocity=0,force=self.maxForce,positionGain=0.03,velocityGain=1)
        self.gripper_stabler()

    def instant_open_gripper(self):
        self.is_gripper_open = True
        self.gripper_stabler()

    def instant_close_gripper(self):
        self.is_gripper_open = False
        self.gripper_stabler()

    def open_gripper(self):
        self.is_gripper_open = True
        for kuka_sec in range(30):
            self.gripper_stabler()
            p.stepSimulation()
            if self.render:
                sleep(0.001)

    def close_gripper(self):
        self.is_gripper_open = False
        for kuka_sec in range(30):
            self.gripper_stabler()
            p.stepSimulation()
            if self.render:
                sleep(0.001)

    def gripper_stabler(self):
        if self.is_gripper_open:
            pose = 0
        else:
            pose = self.gripper_close_pose
        p.setJointMotorControl2(bodyIndex=self.kukaId,jointIndex=self.lf_id,controlMode=p.POSITION_CONTROL,\
          targetPosition=-pose,targetVelocity=0,force=self.fingerTipForce)
        p.setJointMotorControl2(bodyIndex=self.kukaId,jointIndex=self.rf_id,controlMode=p.POSITION_CONTROL,\
          targetPosition=pose,targetVelocity=0,force=self.fingerTipForce)

    def gripper_centered(self):
        for kuka_sec in range(5):
            b = p.getJointState(self.kukaId, self.lf_id)[0]
            p.setJointMotorControl2(self.kukaId, self.rf_id, p.POSITION_CONTROL, targetPosition=-b, force=500)

    def gripper_stabler_keep_commands(self):
        if self.is_gripper_open:
            pose = self.gripper_open_pose
        else:
            pose = self.gripper_close_pose
        for kuka_sec in range(100):
            p.setJointMotorControl2(bodyIndex=self.kukaId,jointIndex=self.lf_id,controlMode=p.POSITION_CONTROL,\
              targetPosition=-pose,targetVelocity=0,force=self.fingerTipForce)
            p.setJointMotorControl2(bodyIndex=self.kukaId,jointIndex=self.rf_id,controlMode=p.POSITION_CONTROL,\
              targetPosition=pose,targetVelocity=0,force=self.fingerTipForce)
            self.gripper_centered()

class kuka_iiwa_prototype(kuka_iiwa):
    def __init__(self,
                 urdfRootPath='/home/msieb/libs/pybullet_data/data',
                 pos=[-0.2, 0.0, -.12],
                 orn=[0, 0, 0, 1.0],
                 render=True,
                 global_t=0):
        self.lf_id = 8
        self.rf_id = 9
        self.gripper_close_pose = 0.06
        self.gripper_open_pose = 0.0
        self.maxForce = 200  #200.
        self.scale = 0.01
        self.object_gripper_offset = 0
        self.fingerTipForce = 8
        self.is_gripper_open = True
        print(urdfRootPath)
        print(os.path.join(urdfRootPath, "kuka_iiwa/kuka_with_gripper5.sdf"))
        self.kukaId = p.loadSDF(
            os.path.join(urdfRootPath, "kuka_iiwa/kuka_with_gripper5.sdf"),
            globalScaling=1)[0]
        self.kukaEndEffectorIndex = 7  #Link index of gripper base
        p.resetBasePositionAndOrientation(self.kukaId, pos, orn)
        self.numJoints = p.getNumJoints(self.kukaId)
        far = 1
        #near = 0.001
        #self.cam_viewMatrix = list(p.computeViewMatrix(cameraEyePosition=[0, 0, 0.5],
        #  cameraTargetPosition=[0.4, 0, 0.2], cameraUpVector = [0.0, 0.0, 1.0]))
        #self.cam_projMatrix = p.computeProjectionMatrixFOV(80, 1, near, far);
        #self.img_size_h = 200#700
        near = 0.1
        self.cam_viewMatrix = list(
            p.computeViewMatrix(
                cameraEyePosition=[-0.2, 0, 0.04],
                cameraTargetPosition=[0.4, 0, -0.1],
                cameraUpVector=[0.6, 0.0, 0.8]))
        self.cam_projMatrix = p.computeProjectionMatrixFOV(100, 1, near, far)
        self.img_size_h = 360  #700
        self.img_size_w = 360  #1280
        self.far = far
        self.near = near
        self.render = render
        self.global_t = global_t  #for debugging
        self.default = {}

        self.maxVelocity = .35
        self.maxForce = 200.
        self.fingerAForce = 2 
        self.fingerBForce = 2.5
        self.fingerTipForce = 2
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 1
        self.useOrientation = 1
        self.kukaEndEffectorIndex = 6
        self.kukaGripperIndex = 7
        #lower limits for null space
        self.ll=[-.967,-2 ,-2.96,0.19,-2.96,-2.09,-3.05]
        #upper limits for null space
        self.ul=[.967,2 ,2.96,2.29,2.96,2.09,3.05]
        #joint ranges for null space
        self.jr=[5.8,4,5.8,4,5.8,4,6]
        #restposes for null space
        self.rp=[0,0,0,0.5*math.pi,0,-math.pi*0.5*0.66,0]
        #joint damping coefficents
        self.jd=[0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001, 0.00001]

    def applyAction(self, motorCommands):
        #print ("self.numJoints")
        #print (self.numJoints)
        xl = 0.8
        xh = 1.1
        yl = -0.700000
        yh = -0.0
        zl = 0.698
        zh =  2.0
        if (self.useInverseKinematics):
          dx = motorCommands[0]
          dy = motorCommands[1]
          dz = motorCommands[2]
          da = motorCommands[3]
          # fingerAngle = motorCommands[4]
          fingerAngle = 0.06
          
          state = p.getLinkState(self.kukaId,self.kukaEndEffectorIndex)
          actualEndEffectorPos = list(state[0])
          actualEndEffectorOrn = list(state[1])
          #print("pos[2] (getLinkState(kukaEndEffectorIndex)")
          #print(actualEndEffectorPos[2])

          
          self.endEffectorPos[0] = self.endEffectorPos[0]+dx
          if (self.endEffectorPos[0]> xh):
            self.endEffectorPos[0]= xh
          if (self.endEffectorPos[0]<xl):
            self.endEffectorPos[0]=xl
          self.endEffectorPos[1] = self.endEffectorPos[1]+dy
          if (self.endEffectorPos[1]<yl):
            self.endEffectorPos[1]=yl
          if (self.endEffectorPos[1]>yh):
            self.endEffectorPos[1]=yh

          #print ("self.endEffectorPos[2]")
          #print (self.endEffectorPos[2])
          #print("actualEndEffectorPos[2]")
          #print(actualEndEffectorPos[2])
          #if (dz<0 or actualEndEffectorPos[2]<0.5):
          self.endEffectorPos[2] = self.endEffectorPos[2]+dz
          if (self.endEffectorPos[1]<zl):
            self.endEffectorPos[1]=zl
          self.cur_joint_pos = [p.getJointState(self.kukaId, i)[0] for i in range(p.getNumJoints(self.kukaId))]

          self.endEffectorAngle = self.cur_joint_pos[7]
          pos = self.endEffectorPos
          # orn = p.getQuaternionFromEuler([0,-math.pi,0]) # -math.pi,yaw])
          orn = actualEndEffectorOrn
          if (self.useNullSpace==1):
            if (self.useOrientation==1):
              jointPoses = p.calculateInverseKinematics(self.kukaId,self.kukaEndEffectorIndex,pos,orn,self.ll,self.ul,self.jr,self.rp)
            else:
              jointPoses = p.calculateInverseKinematics(self.kukaId,self.kukaEndEffectorIndex,pos,lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp)
          else:
            if (self.useOrientation==1):
              jointPoses = p.calculateInverseKinematics(self.kukaId,self.kukaEndEffectorIndex,pos,orn,jointDamping=self.jd)
            else:
              jointPoses = p.calculateInverseKinematics(self.kukaId,self.kukaEndEffectorIndex,pos)

          #print("jointPoses")
          #print(jointPoses)
          #print("self.kukaEndEffectorIndex")
          #print(self.kukaEndEffectorIndex)
          if (self.useSimulation):
            for i in range (self.kukaEndEffectorIndex+1):
              #print(i)
              p.setJointMotorControl2(bodyUniqueId=self.kukaId,jointIndex=i,controlMode=p.POSITION_CONTROL,targetPosition=jointPoses[i],targetVelocity=0,force=self.maxForce,maxVelocity=self.maxVelocity, positionGain=0.3,velocityGain=1)
          else:
            #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
            for i in range (self.numJoints):
              p.resetJointState(self.kukaId,i,jointPoses[i])
          #fingers
          p.setJointMotorControl2(self.kukaId,7,p.POSITION_CONTROL,targetPosition=self.endEffectorAngle,force=self.maxForce)
          p.setJointMotorControl2(self.kukaId,8,p.POSITION_CONTROL,targetPosition=-fingerAngle,force=self.fingerAForce)
          p.setJointMotorControl2(self.kukaId,9,p.POSITION_CONTROL,targetPosition=fingerAngle,force=self.fingerBForce)
          # p.setJointMotorControl2(self.kukaId,10,p.POSITION_CONTROL,targetPosition=0,force=self.fingerTipForce)
          # p.setJointMotorControl2(self.kukaId,13,p.POSITION_CONTROL,targetPosition=0,force=self.fingerTipForce)
          
          
        else:
          for action in range (len(motorCommands)):
            motor = self.motorIndices[action]
            p.setJointMotorControl2(self.kukaId,motor,p.POSITION_CONTROL,targetPosition=motorCommands[action],force=self.maxForce)
        self.gripper_centered()
