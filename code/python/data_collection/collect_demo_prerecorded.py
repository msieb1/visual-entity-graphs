import pybullet as p
import time
import argparse
import pybullet_data
import os
from os.path import join
import numpy as np
from utils import euclidean_dist, readLogFile
import importlib
import scipy
import cv2
import torch
import imageio
from copy import deepcopy as copy
from pdb import set_trace
import json
import sys
import matplotlib.pyplot as plt
# from env.kuka_iiwa import kuka_iiwa
#p.connect(p.UDP,"192.168.86.100")
plt.ion()
sys.path.append('../')
#from experiments.cube_and_bowl_mrcnn.configs import Inference_Camera_Config
#camconf = Inference_Camera_Config()

TARGET_POSITION = [1.32, -0.33, 0.46]
ATTACH_DIST = 0.12
SAMPLE_RATE = 20
IMG_SIZE = 240
FOV = 30
NEARVAL = 1
FARVAL = 3
CAMERA_DISTANCE = 2.0
YAW = -90
PITCH = -30

DEMOS_DIR = '/home/msieb/projects/lang2sim/demos'
SAVE_ROOT_DIR = '/home/msieb/projects/gps-lfd/demo_data'

# Note depth is in OpenGL format. It is non-linear. [0, 1]
# Nonlinearity helps to preserve more details when the object is close.
# Checkout https://learnopengl.com/Advanced-OpenGL/Depth-testing
def tinyDepth_no_rescale(depthSample, near, far):
    zLinear = far * near / (far - (far - near) * depthSample) 
    return np.asarray(zLinear)

def linDepth(depthSample, zNear, zFar):
    depthSample = 2.0 * depthSample - 1.0
    zLinear = 2.0 * zNear * zFar / (zFar + zNear - depthSample * (zFar - zNear))
    return zLinear

def nonLinearDepth(depthSample, zNear, zFar):
    linearDepth = linDepth(depthSample, zNear, zFar)
    nonLinearDepth = (zFar + zNear - 2.0 * zNear * zFar / linearDepth) / (zFar - zNear)
    nonLinearDepth = (nonLinearDepth + 1.0) / 2.0
    return nonLinearDepth;

def create_data_folders(visual ,save_path):
    rgb_folder = 'rgb'
    depth_folder = 'depth'
    seg_folder = 'masks'
    flow_folder = 'flow'
    info_folder = 'info'
    rgb_folder = join(save_path, rgb_folder)
    depth_folder = join(save_path, depth_folder)
    seg_folder = join(save_path, seg_folder)
    flow_folder = join(save_path, flow_folder)
    info_folder = join(save_path, info_folder)
    base_folder = save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if visual is True:
        if not os.path.isdir(rgb_folder):
            os.makedirs(rgb_folder)
        if not os.path.isdir(depth_folder):
            os.makedirs(depth_folder)
        if not os.path.isdir(seg_folder):
            os.makedirs(seg_folder)
        if not os.path.isdir(flow_folder):
            os.makedirs(flow_folder)
        if not os.path.isdir(info_folder):
            os.makedirs(info_folder)
        return rgb_folder, depth_folder, seg_folder, flow_folder, info_folder, base_folder
    else:
        if not os.path.isdir(rgb_folder):
            os.makedirs(rgb_folder)
        if not os.path.isdir(depth_folder):
            os.makedirs(depth_folder)
        if not os.path.isdir(seg_folder):
            os.makedirs(seg_folder)
        if not os.path.isdir(info_folder):
            os.makedirs(info_folder)
        return rgb_folder, depth_folder, seg_folder, info_folder, base_folder


def compute_optical_flow(prev, cur):
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def set_obj_invisible(objId):
    num_of_joints = p.getNumJoints(objId)
    for i in range(num_of_joints+1):
        p.changeVisualShape(objId, linkIndex=i-1, rgbaColor=(0, 0, 0, 0))


def getPose(objID):
    pos, orn = p.getBasePositionAndOrientation(objID)
    return list(pos + orn)

    
def main(args, no_connect=False):
    print(args.visual)
    save_path = join(SAVE_ROOT_DIR, args.demo)

    demo_path = join(DEMOS_DIR, args.play)
    demo_info = os.path.basename(demo_path).split('_')
    demo_num = int(demo_info[2].split('.')[0])
    print('DEMO NUM', demo_num)
    args.env = demo_info[1]
    # args.env = 'bowlenv1_occ'
    module = importlib.import_module('bullet.simenv.' + args.env)
    envClass = getattr(module, 'UserEnv')
    env = envClass()
    cid = -1
    if demo_path is None:
        cid = p.connect(p.SHARED_MEMORY)

    if cid<0 and not no_connect:
        cid = p.connect(p.GUI)
    p.resetSimulation()
    #disable rendering during loading makes it much faster
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # Env is all loaded up here
    h, o = env.load()
    print('Total Number:', p.getNumBodies())
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    p.setGravity(0.000000,0.000000,0.000000)
    p.setGravity(0,0,-10)

    ##show this for 10 seconds
    #now = time.time()
    #while (time.time() < now+10):
    #	p.stepSimulation()
    p.setRealTimeSimulation(1)

    # Replay and generate object centeric log.
    recordID = 0
    log = readLogFile(demo_path)
    recordNum = len(log)
    itemNum = len(log[0])
    print('record num:'),
    print(recordNum)
    print('item num:'),
    print(itemNum)
    init_setup = True
    startID = log[0][2]
    imgCount = 0
    timeCount = 0
    if args.visual:
        rgb_folder, depth_folder, seg_folder, flow_folder, info_folder, base_folder =  create_data_folders(args.visual, save_path)
    else:
        rgb_folder, depth_folder, seg_folder, info_folder, base_folder = create_data_folders(args.visual, save_path)
    
    image_buffer = []
    rgb_writer = imageio.get_writer('{}/{}.mp4'.format(
                    rgb_folder, demo_num), fps=SAMPLE_RATE)
    end_effector_trajectory = []
    # Save high level plan
    # objects of interest
    objects = env.objects()
    obj_start_pose = None
    current_label = -1

    excluded_ids = ['kuka', 'pr2_gripper','pr2_cid', 'pr2_cid2']
    relevant_ids = {val: key for key, val in h.items() if key not in excluded_ids}
    # relevant_ids = {h.cubeid: 'cube1', h.bowlid: 'bowl'}
    relevant_ids_names = {val: str(key) for key, val in relevant_ids.items()}
    object_trajectories = {key: [] for key in relevant_ids.keys()}
    init_object_poses = {}
    # Set arm invisible
    # set_obj_invisible(objects[0])
    set_obj_invisible(1)

    # Optical flow related
    prev = None
    hsv = np.zeros((IMG_SIZE, IMG_SIZE, 3))
    hsv[...,1] = 255

    # For record
    info_record = {}
    for objseq, obj in enumerate(objects):
        info_record[objseq] = []
    cam_pose = getPose(4)[:3]
    # cam_pose[0] -= 0.4
    for i, record in enumerate(log):
        Id = record[2]

        if i != 0 and Id == startID and init_setup:
            init_setup = False
            obj_start_pose = [getPose(x) for x in objects[1:]]
        if init_setup:
            pos = [record[3],record[4],record[5]]
            orn = [record[6],record[7],record[8],record[9]]
            if Id == 6:
                pos = [0.9032135611658497, -0.3798930514455489, 0.6500000245783583]
            if Id == 7:
                pos = [0.9032135611658497, -0.6830514455489, 0.6500000245783583]

            p.resetBasePositionAndOrientation(Id,pos,orn)
            if Id in relevant_ids.keys():
                # if Id == 5:
                #     set_trace()
                init_object_poses[Id] = pos + orn

            numJoints = p.getNumJoints(Id)
            for i in range (numJoints):
                jointInfo = p.getJointInfo(Id,i)
                qIndex = jointInfo[3]
                if qIndex > -1:
                    p.resetJointState(Id,i,record[qIndex-7+17])
        elif Id == h.kuka:  # can also be objects[0]
            numJoints = p.getNumJoints(Id)
            for i in range(numJoints):
                jointInfo = p.getJointInfo(Id,i)
                qIndex = jointInfo[3]
                if i not in (env.o.kukaobject.lf_id, env.o.kukaobject.rf_id) and qIndex > -1:
                    p.setJointMotorControl2(Id, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                            targetPosition=record[qIndex-7+17], targetVelocity=0,
                                            force=env.o.kukaobject.maxForce,
                                            positionGain=0.12,
                                            velocityGain=1)
            rf = record[p.getJointInfo(Id, env.o.kukaobject.rf_id)[3]-7+17]
            lf = record[p.getJointInfo(Id, env.o.kukaobject.lf_id)[3]-7+17]
            position = max(abs(rf), abs(lf))
            if position > 1e-4:
                env.o.kukaobject.instant_close_gripper()
            else:
                env.o.kukaobject.instant_open_gripper()
            env.o.kukaobject.gripper_centered()

            p.setGravity(0.000000,0.000000,-10.000000)
            p.stepSimulation()
            time.sleep(0.003)


        if Id == startID:
            # Don't take pic from the first frame 0
            if timeCount % SAMPLE_RATE == 1 and timeCount // SAMPLE_RATE > 0:
                info = o.kukaobject.get_gripper_pose()
                info.append(o.kukaobject.is_gripper_open)
                # 0, the glory number reserved for kuka arm
                info_record[0].append(np.array(info))
                for objseq, obj in enumerate(objects[1:]):
                    objseq += 1
                    pose = np.array(getPose(obj))
                    info_record[objseq].append(pose)

                viewMat = p.computeViewMatrixFromYawPitchRoll( cam_pose,   #TARGET_POSITION, #getPose(objects[-1])[:3],
                                                              distance=CAMERA_DISTANCE,
                                                              yaw=YAW,
                                                              pitch=PITCH, 
                                                              roll=0,
                                                              upAxisIndex=2)
                projMat = p.computeProjectionMatrixFOV(FOV, aspect=1, nearVal=NEARVAL, farVal=FARVAL)
                results = p.getCameraImage(width=IMG_SIZE,
                                           height=IMG_SIZE,
                                           viewMatrix=viewMat,
                                           projectionMatrix=projMat)

                ### DEBUG for 3D RECONSTRUCTION#####
                # viewMat2 = p.computeViewMatrixFromYawPitchRoll(getPose(4)[:3],
                #                                               distance=CAMERA_DISTANCE,
                #                                               yaw=-10,
                #                                               pitch=-10, 
                #                                               roll=0,
                #                                               upAxisIndex=2)
                # projMat2 = p.computeProjectionMatrixFOV(FOV, aspect=1, nearVal=NEARVAL, farVal=FARVAL)
                # results2 = p.getCameraImage(width=IMG_SIZE,
                #                            height=IMG_SIZE,
                #                            viewMatrix=viewMat2,
                #                            projectionMatrix=projMat2)

                # vin = np.linalg.pinv(np.reshape(projMat, [4,4]).dot(np.reshape(viewMat, [4,4])))
                # vin2 = np.linalg.pinv(np.reshape(projMat2, [4,4]).dot(np.reshape(viewMat2, [4,4])))
                # pin = np.linalg.pinv(np.reshape(projMat, [4,4]))
                # pin2 = np.linalg.pinv(np.reshape(projMat2, [4,4]))                

                # d1 = results[3]
                # d2 = results2[3]
                # d1 = np.swapaxes(copy(d1), 0, 1)
                # d2 = np.swapaxes(copy(d2), 0, 1)
                # d1_r = tinyDepth_no_rescale(copy(d1), NEARVAL, FARVAL)
                # d2_r = tinyDepth_no_rescale(copy(d2), NEARVAL, FARVAL)
                # # d2_r2 = linDepth(d2, NEARVAL, FARVAL)
                # # d2_r3 = nonLinearDepth(d2, NEARVAL, FARVAL)


                # print("tiny: ",d2_r[120,120])



                # w1, h1 = 193, 133
                # w2, h2 = 123, 143
                # vun1 = vin.dot(np.array([w1, h1, d1[w1, h1], 1]))
                # vun2 = vin2.dot(np.array([w2, h2, d2[w2, h2], 1]))
                # pun1 = pin.dot(np.array([w1, h1, d1[w1, h1], 1]))
                # pun2 = pin2.dot(np.array([w2, h2, d2[w2, h2], 1]))
                # pun1 /= pun1[-1]
                # pun2 /= pun2[-1]
                # vun1 /= vun1[-1]
                # vun2 /= vun2[-1]

                # w1_bowl, h1_bowl = 89, 136
                # w2_bowl, h2_bowl = 34, 117
                # vun1_bowl = vin.dot(np.array([w1_bowl, h1_bowl, d1[w1_bowl, h1_bowl], 1]))
                # vun2_bowl = vin2.dot(np.array([w2_bowl, h2_bowl, d2[w2_bowl, h2_bowl], 1]))
                # pun1_bowl = pin.dot(np.array([w1_bowl, h1_bowl, d1[w1_bowl, h1_bowl], 1]))
                # pun2_bowl = pin2.dot(np.array([w2_bowl, h2_bowl, d2[w2_bowl, h2_bowl], 1]))
                # # pun1_bowl /= pun1_bowl[-1]
                # # pun2_bowl /= pun2_bowl[-1]
                # # vun1_bowl /= vun1_bowl[-1]
                # # vun2_bowl /= vun2_bowl[-1]


                # vun1_bowl[2] = d1_r[w1_bowl, h1_bowl]
                # vun2_bowl[2] = d2_r[w2_bowl, h2_bowl]
                # vun1[2] = d1_r[w1, h1]
                # vun2[2] = d2_r[w2, h2]


                # dist1_z = d1_r[w1_bowl, h1_bowl] - d1_r[w1, h1]
                # dist2_z = d2_r[w2_bowl, h2_bowl] - d2_r[w2, h2]


                # dist1 = pun1_bowl[:2] - pun1[:2]
                # dist2 = pun2_bowl[:2] - pun2[:2]


 

                # w1 = 71
                # h1 = 122
                # x_cube_1 = (w1 - 120) * d2_r[w1,h1] / 3.732/100
                # y_cube_1 = (h1 - 120) * d2_r[w1,h1] / 3.732/100
                # z_cube_1 = d1_r[w1, h1]
                # p3d_cube_1 = np.array([x_cube_1, y_cube_1, z_cube_1])
                # x_red_1 = (120 - 120) * d2_r[120,120] / 3.732/100
                # y_red_1 = (120 - 120) * d2_r[120,120] / 3.732/100
                # z_red_1 = d1_r[120, 120]
                # p3d_red_1 = np.array([x_red_1, y_red_1, z_red_1])

                # w2 = 136
                # h2 = 128
                # x_cube_2 = (120 - 120) * d2_r[120,120] / 3.732/100
                # y_cube_2 = (120 - 120) * d2_r[120,120] / 3.732/100
                # z_cube_2 = d2_r[120, 120]
                # p3d_cube_2 = np.array([x_cube_2, y_cube_2, z_cube_2])
                # x_red_2 = (w2 - 120) * d2_r[w2,h2] / 3.732/100
                # y_red_2 = (h2 - 120) * d2_r[w2,h2] / 3.732/100
                # z_red_2 = d2_r[w2, h2]
                # p3d_red_2 = np.array([x_red_2, y_red_2, z_red_2])

                # # print(np.linalg.norm(p3d_red_1 - p3d_cube_1))
                # # print(np.linalg.norm(p3d_red_2 - p3d_cube_2))


                # persp = np.reshape(projMat, [4,4])
                # persp[0:2, 2] = 120
                # alpha, s, x0, beta, y0 = persp[0, 0], persp[0,1], persp[0,2], persp[1,1], persp[1,2]
                # K = np.array([[alpha, s, x0], [0, beta, y0], [0,0,-1]])
                # I_m = np.eye(3)
                # I_m[1,1] = -1.0
                # I_h = np.eye(3)
                # I_h[1,2] = 120.0
                # Ks = I_h.dot(I_m).dot(K)
                # cx = Ks[0,2]
                # cy = Ks[1,2]
                # fx = Ks[0,0]*100
                # fy = Ks[1,1]*100


                # w_eeright = 131
                # h_eeright = 88
                # w_eeleft= 103
                # h_eeleft = 88
                # # w_eeleft= 120
                # # h_eeleft = 120                

                # x_eeleft_1 = (w_eeleft - cx) * d2_r[w_eeleft,h_eeleft] / fx
                # y_eeleft_1 = (h_eeleft - cy + 120) * d2_r[w_eeleft,h_eeleft] / fy
                # z_eeleft_1 = d1_r[w_eeleft, h1]
                # p3d_eeleft_1 = np.array([x_eeleft_1, y_eeleft_1, z_eeleft_1])

                # x_eeright_1 = (w_eeright - cx) * d2_r[w_eeright,h_eeright] / fx
                # y_eeright_1 = (h_eeright - cy + 120) * d2_r[w_eeright,h_eeright] / fy
                # z_eeright_1 = d1_r[w_eeright, h_eeright]
                # p3d_eeright_1 = np.array([x_eeright_1, y_eeright_1, z_eeright_1])

                # w_eeright = 140
                # h_eeright = 92
                # w_eeleft= 133
                # h_eeleft = 89

                # x_eeleft_2 = (w_eeleft - cx) * d2_r[w_eeleft,h_eeleft] / fx
                # y_eeleft_2 = (h_eeleft - cy + 120) * d2_r[w_eeleft,h_eeleft] / fy
                # z_eeleft_2 = d2_r[w_eeleft, h_eeleft]
                # p3d_eeleft_2 = np.array([x_eeleft_2, y_eeleft_2, z_eeleft_2])

                # x_eeright_2 = (w_eeright - cx) * d2_r[w_eeright,h_eeright] / fx
                # y_eeright_2 = (h_eeright - cy + 120) * d2_r[w_eeright,h_eeright] / fy
                # z_eeright_2 = d2_r[w_eeright, h_eeright]
                # p3d_eeright_2 = np.array([x_eeright_2, y_eeright_2, z_eeright_2])
                
                # print(np.linalg.norm(p3d_eeright_1 - p3d_eeleft_1))
                # print(np.linalg.norm(p3d_eeright_2 - p3d_eeleft_2))

                # plt.figure()
                # plt.imshow(results[2])
                # plt.figure()
                # plt.imshow(results2[2])
                # # plt.figure()
                # # plt.imshow(results[3])
                # # plt.figure()
                # # plt.imshow(results2[3])

                # # plt.show()
                # set_trace()
                ############# END DEBUG #####3
                set_trace()
                rgb_writer.append_data(results[2])
                
                for key in relevant_ids.keys():
                    object_trajectories[key].append(getPose(key))


                link_state = p.getLinkState(o.kukaobject.kukaId, o.kukaobject.kukaEndEffectorIndex)
                end_effector_trajectory.append(np.array(link_state[0] + link_state[1]))

                # Note: we can compute flow online
                imageio.imwrite('{0}/{1}_{2:05d}.png'.format(
                    rgb_folder, demo_num, imgCount), results[2])
                imageio.imwrite('{0}/{1}_{2:05d}.png'.format(
                    depth_folder, demo_num, imgCount), results[3])
                imageio.imwrite('{0}/{1}_{2:05d}.png'.format(
                    seg_folder, demo_num, imgCount), results[4])
                np.save('{0}/{1}_{2:05d}_mask.npy'.format(
                    seg_folder, demo_num, imgCount), results[4])
                # With visual flag, save visualizable imgs
                if args.visual:
                    cur = cv2.cvtColor(results[2], cv2.COLOR_BGR2GRAY)
                    # First frame
                    if timeCount // SAMPLE_RATE == 1:
                        prev = cur
                    flow = compute_optical_flow(prev, cur)

                    # Visualize optical flow
                    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                    hsv[...,0] = ang*180/np.pi/2
                    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                    hsv = hsv.astype(np.uint8)
                    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

                    scipy.misc.imsave('{}/flow_{}_{}.png'.format(
                        flow_folder, demo_num, imgCount), bgr)
                    prev = cur

                imgCount += 1
            timeCount += 1

    print("Saving files to ", base_folder)
    np.save('{}/{}_ee.npy'.format(
        base_folder, demo_num), np.array(end_effector_trajectory))

    with open('{}/{}_objects.json'.format(base_folder, demo_num), 'w') as f:
        json.dump(object_trajectories, f)

    with open('{}/{}_init_object_poses.json'.format(base_folder, demo_num), 'w') as f:
        json.dump(init_object_poses, f)
    with open('{}/{}_relevant_ids_names.json'.format(base_folder, demo_num), 'w') as f:
        json.dump(relevant_ids_names, f)
    # Check if consistent with above paramsd
    camera_params = {"cameraTargetPosition": TARGET_POSITION, "distance": CAMERA_DISTANCE, "yaw": YAW, "pitch":   PITCH, 
    "roll": 0.0, "upAxisIndex": 2, "nearPlane":  NEARVAL, "farPlane":  FARVAL, "fov":  FOV,  "aspect":  1}
    with open('{}/{}_camera_params.json'.format(base_folder, demo_num), 'w') as f:
        json.dump(camera_params, f)


    if args.visual:
        for i in range(len(objects)):
            with open('{}/test_{}_{}.txt'.format(
                    info_folder, demo_num, i), 'w') as f:
                print_list = [' '.join(map(str, x)) + '\n' for x in info_record[i]]
                f.writelines(print_list)
    else:
        torch.save(info_record, '{}/info-{}-.tch'.format(info_folder, demo_num))
    rgb_writer.close()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VR Demo and Play")
    parser.add_argument('--play', type=str, help='The path to the log file to be played. No need to add env flag if trace file name is not modified.')
    parser.add_argument('--demo', type=str, default='tmp', help='demo name to save')
    parser.add_argument('--visual', action='store_true', help='Output visualized data instead of compact dataset')
    args = parser.parse_args()
    main(args)
