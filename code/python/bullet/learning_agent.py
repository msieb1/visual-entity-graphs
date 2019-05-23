import pybullet as p
import time
import argparse
import numpy as np
import pybullet_data
import os
from os.path import join
from utils import euclidean_dist, pressed, readLogFile
import importlib
from algos import FiniteDifferenceLearner
from trace import Trace
import pyquaternion as pq
from transformation import get_H, transform_trajectory, get_rotation_between_vectors
# from env.kuka_iiwa import kuka_iiwa
#p.connect(p.UDP,"192.168.86.100")
from pdb import set_trace


ATTACH_DIST = 0.12

DEMO_PATH = '/home/msieb/projects/lang2sim/bullet/demos'


class BulletEnv(object):
    def __init__(self):
        self._setup_world()
        self._initialize_world()

    def _setup_world(self):    
        """
        Setup bullet robot environment and load all relevant objects
        """
        module = importlib.import_module('simenv.' + args.env)
        envClass = getattr(module, 'UserEnv')
        self._env = envClass()
        self.ids = None

        cid = -1
        if args.init is None:
            cid = p.connect(p.SHARED_MEMORY)

        if (cid<0):
            p.connect(p.GUI)
        p.resetSimulation()
        #disable rendering during loading makes it much faster
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # Env is all loaded up here
        h, o = self._env.load()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        p.setGravity(0.000000,0.000000,0.000000)
        p.setGravity(0,0,-10)

        ##show this for 10 seconds
        #now = time.time()
        #while (time.time() < now+10):
        #	p.stepSimulation()
        # p.setRealTimeSimulation(1)  

    def _initialize_world(self):
        demo_name = 'putAInFrontOfB_cubeenv2_1.bin.txt'

        # Robot ID is 2,  the object IDs are given as 4 and 5... (A and B)

        with open(join(DEMO_PATH, demo_name), 'r') as f:
            lines = f.readlines()
        #read joint names specified in file
        IDs = lines[0].rstrip().split(' ')
        self.ids = IDs
        all_traj = {key: [] for key in IDs}
        for idx, values in enumerate(lines[1:]):
            #clean each line of file
            ID, values = clean_line(values)
            if ID in IDs:
                all_traj[str(int(ID))].append(values)
        t = Trace(join(DEMO_PATH, demo_name))


        self.init_traj_B = np.asarray(all_traj['5'])[:-1:160, 2:5]
        self.init_traj_A = np.asarray(all_traj['4'])[:-1:160, 2:5]
        self.policy = np.asarray(all_traj['2'])[:-1:160, 2:-1]
        self.n_iter = 5
        self.n_runs = 20
        self.eps = 1e-5

        self.object_ids = [4, 5]
        self.object_poses = [self.init_traj_A[0].tolist(), self.init_traj_B[0].tolist()]

    def reset(self):
        
        self._env.reset()
        for i, pos in zip(self.object_ids, self.object_poses):
            self._env.setObjectPose(i, pos, [0,0,0,1])
            self._env.o.kukaobject.close_gripper()


    def sample(self, policy):
        orn = None
        self.reset()
        manipulated_obj_traj = []
        robot_traj = []
        for i in range(len(policy)):
            pos = policy[i, :-1]
            if np.abs(policy[i, -1]) > 0.015:
                self._env.o.kukaobject.close_gripper()
            else:
                self._env.o.kukaobject.open_gripper()
            self._env.o.kukaobject.instantMoveKukaEndtoPos(pos, orn)
            p.stepSimulation()
            # time.sleep(0.001)
            robot_pose  = self._env.getEndEffectorPose()
            robot_traj.append(robot_pose)
            manipulated_obj_pose = self._env.getObjectPose(int(self.ids[1]))
            manipulated_obj_traj.append(manipulated_obj_pose)
        return np.asarray(robot_traj), np.asarray(manipulated_obj_traj)       

def main(args):
    env = BulletEnv()
    print('=' * 20)

    init_traj_A = env.init_traj_A
    init_traj_B = env.init_traj_B
    policy = env.policy[:,[0,1,2,-1]]
    n_iter = env.n_iter
    n_runs = env.n_runs
    eps = env.eps
    # delta_quat =  pq.Quaternion(np.hstack([initial_reference_A_pose[-1], initial_reference_A_pose[3:-1]]))*pq.Quaternion(np.hstack([initial_object_A_pose[-1], initial_object_A_pose[3:-1]])).conjugate
    # R = delta_quat.rotation_matrix
    # t = initial_object_A_pose[:3]- initial_reference_A_pose[:3]
    # H = get_H(R_rel, t)
    # policy = transform_trajectory(policy, H)
    # init_traj_A = transform_trajectory(init_traj_A, H)
    algo = FiniteDifferenceLearner(init_traj_A, env, policy)

    for r in range(n_runs):
        new_rollout = env.sample(policy)
        cost = algo.cost_fn(new_rollout[1])
        print("Start run ", r)
        for i in range(n_iter):
            delta_end_state = np.linalg.norm(init_traj_A[-1] - new_rollout[1][-1][0:3])
            print('end state residual: ', delta_end_state)

            if delta_end_state < 0.05:
                break
            print("iteration: ", i)
            new_policy, new_cost, new_rollout = algo.iteration(policy, n_samples=10)
            print("current cost: ", new_cost)
            print("residual: ", new_cost - cost)
            delta_end_state = np.linalg.norm(init_traj_A[-1] - new_rollout[1][-1][0:3])
            print('=' * 20)

        policy = algo.new_policy
        init_traj_A = new_rollout[1][:, :3]

        algo.ref_object_traj = init_traj_A
        algo.policy = policy
        for i, pos in enumerate(env.object_poses):
            env.object_poses[i] = [a + b for a,b in zip(env.object_poses[i], [np.random.uniform(-0.03, 0.03), np.random.uniform(-0.03,0.03), 0])]    


    # Replay and generate object centeric log.
    if args.init is not None:
        log = readLogFile(args.init)
        recordNum = len(log)
        itemNum = len(log[0])
        print('record num:'),
        print(recordNum)
        print('item num:'),
        print(itemNum)
        init_setup = True
        startID = log[0][2]
        # Save high level plan
        fid = open(args.init + '.txt', 'w')
        # objects of interest
        objects = env._env.objects()
        set_trace()
        for i, record in enumerate(log):
            Id = record[2]
            posorn = record[3:9+1]
            if Id == objects[0]:
                lqIndex = p.getJointInfo(Id, o.kukaobject.lf_id)[3]
                rqIndex = p.getJointInfo(Id, o.kukaobject.rf_id)[3]
                posorn += [record[lqIndex-7+17], record[rqIndex-7+17]]
            fid.write(' '.join(map(str, posorn)) + '\n')
            if init_setup:
                if i != 0 and Id == startID:
                    init_setup = False
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
                    if qIndex > -1:
                        p.setJointMotorControl2(Id, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                                targetPosition=record[qIndex-7+17], targetVelocity=0,
                                                force=o.kukaobject.maxForce,
                                                positionGain=0.03,
                                                velocityGain=1)
                p.setGravity(0.000000,0.000000,-10.000000)
                p.stepSimulation()
                time.sleep(0.005)
        exit()
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
    logCount = 0
    fileName = ''
    start_recording = False

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
            if e[CONTROLLER_ID] == controllerId2:  # To make sure we only get the value for one of the remotes
                #sync the vr pr2 gripper with the vr controller position
                p.changeConstraint(h.pr2_cid, e[POSITION], e[ORIENTATION], maxForce=500)
                relPosTarget = 1 - e[ANALOG]
                #open/close the gripper, based on analogue
                p.changeConstraint(h.pr2_cid2,gearRatio=1, erp=1, relativePositionTarget=relPosTarget, maxForce=3)

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
                    o.kukaobject.instantMoveKukaEndtoPos(e[POSITION])  # can also include e[ORIENTATION]
                    if e[ANALOG] == 1:
                        o.kukaobject.instant_close_gripper()
                    if e[ANALOG] == 0:
                        o.kukaobject.instant_open_gripper()

                if pressed(e, 'menu'):
                    start_recording = True

                if pressed(e, 'pad'):
                    start_recording = False



def clean_line(line):
    """
    Cleans a single line of recorded joint positions
    @param line: the line described in a list to process
    @param joint_names: joint name keys
    @return command: returns dictionary {joint: value} of valid commands
    @return line: returns list of current line values stripped of commas
    """
    def try_float(x):
        try:
            return float(x)
        except ValueError:
            return None
    #convert the line of strings to a float or None
    line = [try_float(x) for x in line.rstrip().split(' ')]
    #zip the values with the joint names
    #take out any tuples that have a none value
    #convert it to a dictionary with only valid commands

    ID = str(int(line[1]))
    return (ID, line,)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VR Demo and Play")
    parser.add_argument('--log', type=str, default='demo', help='The path to the log file for demo recording.')
    parser.add_argument('--init', type=str, default='demos/putABehindB_cubeenv2_4.bin', help='The path to the log file to be initialized with.')
    parser.add_argument('--env', type=str, default='cubeenv2', help='The selected environment')
    args = parser.parse_args()
    main(args)
