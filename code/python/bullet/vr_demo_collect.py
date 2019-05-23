import pybullet as p
import time
import argparse
import pybullet_data
import os
from utils import euclidean_dist, pressed, readLogFile
import importlib
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

    # Replay and generate object centeric log.
    if args.play is not None:
        if args.detect:
            from reward import DecisionTree
            dtree = DecisionTree()
            dtree.load()
        recordID = 0
        if args.video:
            recordID = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,
                                args.play + '.mp4')
        log = readLogFile(args.play)
        recordNum = len(log)
        itemNum = len(log[0])
        print('record num:'),
        print(recordNum)
        print('item num:'),
        print(itemNum)
        init_setup = True
        startID = log[0][2]
        timeCount = 0
        # Save high level plan
        fid = open(args.play + '.txt', 'w')
        # objects of interest
        objects = env.objects()
        fid.write(' '.join(map(str, objects)) + '\n')
        obj_start_pose = None
        current_label = -1
        for i, record in enumerate(log):
            Id = record[2]

            if i != 0 and Id == startID and init_setup:
                init_setup = False
                obj_start_pose = [getPose(x) for x in objects[1:]]
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
                if args.detect:
                    feature_info = [getPose(x) for x in objects[1:]]  # exclude gripper
                    predicted_label = dtree.predict(obj_start_pose, feature_info)
                    if predicted_label != current_label:
                        current_label = predicted_label
                        p.removeAllUserDebugItems()
                        p.addUserDebugText(str(bool(predicted_label[0])), [-2, 0, 1], lifeTime=5000, textSize=10)
                    print(predicted_label)
                    # print(feature_info)

                gripper_pos, gripper_orn = p.getLinkState(o.kukaobject.kukaId, o.kukaobject.kukaEndEffectorIndex)[0:2]
                p.stepSimulation()
                time.sleep(0.003)

            # Write to trace
            posorn = None
            if Id == objects[0]:
                posorn = env.o.kukaobject.get_gripper_pose()
                lq = p.getJointState(Id, o.kukaobject.lf_id)[3]
                rq = p.getJointState(Id, o.kukaobject.rf_id)[3]
                # posorn = [timeCount, Id] + posorn + [record[lqIndex-7+17], record[rqIndex-7+17]]
                posorn = [timeCount, Id] + getPose(Id) + [lq, rq]
            else:
                # posorn = [timeCount, Id] + record[3:9+1]
                posorn = [timeCount, Id] + getPose(Id)

            fid.write(' '.join(map(str, posorn)) + '\n')

            if Id == startID and i != 0:
                timeCount += 1
        if args.video:
            p.stopStateLogging(recordID)
        return 0
    if cid < 0:
        while True:
            p.setGravity(0,0,-10)

    p.setVRCameraState(rootPosition=(0, -0.200000, -0.200000), rootOrientation=(0, 0, -0.423, 0.906))

    CONTROLLER_ID = 0
    POSITION=1
    ORIENTATION=2
    ANALOG=3
    BUTTONS=6


    controllerId1 = -1
    controllerId2 = -1
    kuka_sync_status = False  #whether or not to sync kuka with controllerId2
    logId = -1
    logCount = args.start
    fileName = ''
    start_recording = False
    pose = 0  # 0: normal, 1, 2: for pouring

    print("waiting for VR controller trigger for free gripper")
    while (controllerId1 < 0 or controllerId2 < 0):
        events = p.getVREvents()
        for e in (events):
            if controllerId1 < 0 and pressed(e, 'trigger') == p.VR_BUTTON_IS_DOWN:
                controllerId1 = e[CONTROLLER_ID]
                print("waiting for VR controller trigger for kuka gripper")
            if controllerId1 >= 0 and controllerId2 < 0 and pressed(e, 'trigger') == p.VR_BUTTON_IS_DOWN:
                if e[CONTROLLER_ID] != controllerId1:
                    controllerId2 = e[CONTROLLER_ID]

    # Sometimes strange problems can occur and it is handy to swap controllers.
    controllerId1, controllerId2 = controllerId2, controllerId1


    print("Using controllerId="+str(controllerId1) + ' for free gripper.')
    print("Using controllerId="+str(controllerId2) + ' for kuka gripper.')

    while (1):
        #keep the gripper centered/symmetric
        b = p.getJointState(h.pr2_gripper,2)[0]
        p.setJointMotorControl2(h.pr2_gripper, 0, p.POSITION_CONTROL, targetPosition=b, force=3)
        o.kukaobject.gripper_centered()

        if logId < 0 and start_recording:
            fileName = args.log + '_' + args.env + '_' + str(logCount) + '.bin'
            fileName = os.path.join(os.path.dirname(os.path.abspath(__file__)), fileName)
            logId = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, fileName, maxLogDof=20)
            print('Start logging at ', fileName)

        if logId >= 0 and not start_recording:
            p.stopStateLogging(logId)
            logId = -1
            logCount += 1
            print('End logging at', fileName)


        events = p.getVREvents()
        for e in (events):
            if e[CONTROLLER_ID] == controllerId2:
                #sync the vr pr2 gripper with the vr controller position
                p.changeConstraint(h.pr2_cid, e[POSITION], e[ORIENTATION], maxForce=500)
                relPosTarget = 1 - e[ANALOG]
                #open/close the gripper, based on analogue
                p.changeConstraint(h.pr2_cid2,gearRatio=1, erp=1, relativePositionTarget=relPosTarget, maxForce=3)
                if pressed(e, 'menu'):
                    pose = 1

                if pressed(e, 'pad'):
                    pose = 2

                if pressed(e, 'side'):
                    pose = 0

            if e[CONTROLLER_ID] == controllerId1:
                if pressed(e, 'trigger'):
                    if not kuka_sync_status:
                        gripper_pos, gripper_orn = p.getLinkState(o.kukaobject.kukaId, o.kukaobject.kukaEndEffectorIndex)[0:2]
                        if euclidean_dist(e[POSITION], gripper_pos) < ATTACH_DIST:
                            kuka_sync_status = True


                if pressed(e, 'side'):
                    kuka_sync_status = False
                    env.reset()

                if kuka_sync_status:
                    orn = None
                    if pose == 0:
                        orn = None
                    elif pose == 1:
                        orn = (0.500,-0.500,-0.500,0.500)
                    elif pose == 2:
                        orn = (-0.500,-0.500,0.500,0.500)

                    o.kukaobject.instantMoveKukaEndtoPos(e[POSITION], orn)  # can also include e[ORIENTATION]
                    if e[ANALOG] == 1:
                        o.kukaobject.instant_close_gripper()
                    if e[ANALOG] == 0:
                        o.kukaobject.instant_open_gripper()

                if pressed(e, 'menu'):
                    start_recording = True

                if pressed(e, 'pad'):
                    start_recording = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VR Demo and Play")
    parser.add_argument('--log', type=str, default='demo', help='The path to the log file for demo recording.')
    parser.add_argument('--play', default='testRecordings_cubeenv1_0.bin', type=str, help='The path to the log file to be played. No need to add env flag if trace file name is not modified.')
    parser.add_argument('--env', default='cubeenv1', type=str, help='The selected environment')
    parser.add_argument('--novis', action='store_true', help='Flag to disable visualization')
    parser.add_argument('--video', action='store_true', help='Flag to enable video recording. Only used in replay.')
    parser.add_argument('--detect', action='store_true', help='Enable detector')
    parser.add_argument('--start', type=int, default=0, help='Log starting number')
    args = parser.parse_args()
    main(args)
