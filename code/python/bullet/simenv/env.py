import os
import pybullet as p


def userdatapath(*paths):
    """Note: __file__ has to be stayed in this file"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),  'data', *paths)


class Env:
    """Note class inherited from this class must call themselves as 'UserEnv'"""
    def __init__(self):
        # A flag for indicating loaded or not
        self.prelim = False

    def load(self):
        """This method loads the environment"""
        raise NotImplementedError

    def objects(self):
        """Return the object id that specific action operates.
        The first id will always be the kuka robot.
        """
        raise NotImplementedError

    def reset(self):
        """Note: not necessary to reset everything"""
        raise NotImplementedError

    def getObjectPose(self, object_id):
        # assert object_id != self.h.kukaid, 'object id cannot be kuka itself.'
        pos, pose = p.getBasePositionAndOrientation(object_id)
        return list(pos) + list(pose)

    def getEndEffectorPose(self):
        return self.o.kukaobject.get_gripper_pose()

    def setObjectPose(self, object_id, pos, orn):
        p.resetBasePositionAndOrientation(object_id, pos, orn)
