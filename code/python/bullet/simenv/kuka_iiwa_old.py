import pybullet as p
import numpy as np
import os
import math
from time import sleep
import scipy
from . import grisp_proposal
from pdb import set_trace
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
                 urdfRootPath='',
                 pos=[-0.2, 0.0, -.12],
                 orn=[0, 0, 0, 1.0],
                 render=True,
                 global_t=0):
        self.lf_id = 8
        self.rf_id = 9
        self.gripper_close_pose = 0.06
        self.gripper_open_pose = 0.0
        self.maxForce = 1000  #200.
        self.scale = 0.01
        self.object_gripper_offset = 0
        self.fingerTipForce = 300
        self.is_gripper_open = True
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
            orn = [0.70603903128,0.708148792076,0,0] #p.getQuaternionFromEuler([0, -math.pi, 0])
            self.moveKukaEndtoPos([0.96, -0.100000, 0.8], orn)

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

    def gripper_stabler_keep_commands(self):
        if self.is_gripper_open:
            pose = self.gripper_open_pose
        else:
            pose = self.gripper_close_pose

        p.setJointMotorControl2(bodyIndex=self.kukaId,jointIndex=self.lf_id,controlMode=p.POSITION_CONTROL,\
          targetPosition=-pose,targetVelocity=0,force=self.fingerTipForce)
        p.setJointMotorControl2(bodyIndex=self.kukaId,jointIndex=self.rf_id,controlMode=p.POSITION_CONTROL,\
          targetPosition=pose,targetVelocity=0,force=self.fingerTipForce)


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
            self.gripper_stabler_keep_commands()
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
        for kuka_sec in range(20):
            self.gripper_stabler()
            p.stepSimulation()
            if self.render:
                sleep(0.001)

    def close_gripper(self):
        self.is_gripper_open = False
        for kuka_sec in range(20):
            self.gripper_stabler()
            p.stepSimulation()
            if self.render:
                sleep(0.001)

    def move_gripper(self, delta):
        for kuka_sec in range(20):
            self.gripper_stabler(delta)
            p.stepSimulation()
            if self.render:
                sleep(0.001)

    def gripper_stabler(self, delta=None):
        if self.is_gripper_open:
            pose = self.gripper_open_pose
        else:
            pose = self.gripper_close_pose

        if delta is not None:
            pose = delta
        joint_vals = [p.getJointState(self.kukaId, i)[0] for i in range(p.getNumJoints(self.kukaId))]

        joint_vals[self.lf_id] = -pose
        joint_vals[self.rf_id] = pose
        for i in range(self.numJoints):
            p.setJointMotorControl2(bodyIndex=self.kukaId,jointIndex=i,controlMode=p.POSITION_CONTROL,\
              targetPosition=joint_vals[i],targetVelocity=0,force=self.fingerTipForce)

    def gripper_centered(self):
        b = p.getJointState(self.kukaId, self.lf_id)[0]
        p.setJointMotorControl2(self.kukaId, self.rf_id, p.POSITION_CONTROL, targetPosition=-b, force=300)