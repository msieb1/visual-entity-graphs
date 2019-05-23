import pybullet as p
import time
import argparse
import numpy as np
import pybullet_data
import os
from utils import euclidean_dist, pressed, readLogFile
import importlib
from reward import DecisionTree
from pdb import set_trace
# from env.kuka_iiwa import kuka_iiwa
#p.connect(p.UDP,"192.168.86.100")

ATTACH_DIST = 0.12


def getPose(objID):
    pos, orn = p.getBasePositionAndOrientation(objID)
    return list(pos + orn)


def main(args, no_connect=False):
    if args.play is None and args.env is None:
        print('Either --play or --env is needed.')
        return 1
    if args.play is not None and args.env is None:
        args.env = args.play.split('_')[1]
    module = importlib.import_module('simenv.' + args.env)
    envClass = getattr(module, 'UserEnv')
    env = envClass()

    cid = -1
    if args.play is None:
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

    # Decision tree
    dtree = DecisionTree()
    dtree.load()

    # objects of interest
    objects = env.objects()
    # Replay and generate object centeric log.
    if args.play is not None:
        if args.video:
            recordID = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                args.play + '.mp4')
        else:
            recordID = 0
        log = readLogFile(args.play)
        recordNum = len(log)
        itemNum = len(log[0])
        print('record num:'),
        print(recordNum)
        print('item num:'),
        print(itemNum)

        init_setup = True
        startID = log[0][2]
        obj_start_pose = None
        current_label = -1
        for i, record in enumerate(log):
            Id = record[2]

            if i != 0 and Id == startID and init_setup:
                init_setup = False
                obj_start_pose = [getPose(x) for x in objects[1:]]
                break
            if init_setup:
                pos = [record[3],record[4],record[5]]
                orn = [record[6],record[7],record[8],record[9]]
                p.resetBasePositionAndOrientation(Id,pos,orn)
                numJoints = p.getNumJoints(Id)
                for i in range (numJoints):
                    jointInfo = p.getJointInfo(Id,i)
                    qIndex = jointInfo[3]
                    if qIndex > -1:
                        p.resetJointState(Id,i,record[qIndex-7+17])
                if args.detect:
                    feature_info = [getPose(x) for x in objects[1:]]  # exclude gripper
                    predicted_label = dtree.predict(obj_start_pose, feature_info)
                    if predicted_label != current_label:
                        current_label = predicted_label
                        p.removeAllUserDebugItems()
                        p.addUserDebugText(str(bool(predicted_label[0])), [-2, 0, 1], lifeTime=5000, textSize=10)
                    print(predicted_label)
                    # print(feature_info)


            if Id == startID and i != 0:
                timeCount += 1
        if args.video:
            p.stopStateLogging(recordID)
    else:
        obj_start_pose = [getPose(x) for x in objects[1:]]


    feature_info = [getPose(x) for x in objects[2:]]
    print(feature_info)

    for i in np.arange(-0.1, 0.1, 0.01):
        for j in np.arange(-0.1, 0.1, 0.01):
            for k in np.arange(-0.1, 0.1, 0.01):

                man_feat_info = [i, j, k]
                startp = [0.95, -0.1, 0.7]
                for x in range(len(man_feat_info)):
                    man_feat_info[x] += startp[x]
                comb_feat_info = [man_feat_info + [0, 0, 0, 1]] + feature_info
                predicted_label = dtree.predict(obj_start_pose, comb_feat_info)
                if predicted_label[0]:
                    p.addUserDebugText('.', man_feat_info, lifeTime=0, textSize=3, textColorRGB=(0, 255, 0))
                else:
                    p.addUserDebugText('.', man_feat_info, lifeTime=0, textSize=3, textColorRGB=(255, 0, 0))
                
    # feature_info = [getPose(x) for x in objects[1:]]  # exclude gripper
    # predicted_label = dtree.predict(obj_start_pose, feature_info)
    # if predicted_label != current_label:
        # current_label = predicted_label
        # p.removeAllUserDebugItems()
        # p.addUserDebugText(str(bool(predicted_label[0])), [-2, 0, 1], lifeTime=5000, textSize=10)
    time.sleep(30)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VR Demo and Play")
    parser.add_argument('--play', type=str, help='The path to the log file to be played. No need to add env flag if trace file name is not modified.')
    parser.add_argument('--env', type=str, help='The selected environment')
    parser.add_argument('--video', action='store_true', help='Flag to enable video recording. Only used in replay.')
    args = parser.parse_args()
    main(args)
