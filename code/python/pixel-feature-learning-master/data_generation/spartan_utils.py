__author__ = 'manuelli'
import numpy as np
import numpy
import collections
import yaml
import os
import time
import math

# # director
# from director import transformUtils
#
# import spartan.utils.transformations as transformations
_EPS = numpy.finfo(float).eps * 4.0

def getSpartanSourceDir():
    return os.getenv("SPARTAN_SOURCE_DIR")

def getDictFromYamlFilename(filename):
    stream = file(filename)
    return yaml.load(stream)

def saveToYaml(data, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

# def poseFromTransform(transform):
#     pos, quat = transformUtils.poseFromTransform(transform)
#     pos = pos.tolist()
#     quat = quat.tolist()
#     d = dict()
#     d['translation'] = dict()
#     d['translation']['x'] = pos[0]
#     d['translation']['y'] = pos[1]
#     d['translation']['z'] = pos[2]
#
#     d['quaternion'] = dict()
#     d['quaternion']['w'] = quat[0]
#     d['quaternion']['x'] = quat[1]
#     d['quaternion']['y'] = quat[2]
#     d['quaternion']['z'] = quat[3]
#
#     return d

def dictFromPosQuat(pos, quat):
    d = dict()
    d['translation'] = dict()
    d['translation']['x'] = pos[0]
    d['translation']['y'] = pos[1]
    d['translation']['z'] = pos[2]

    d['quaternion'] = dict()
    d['quaternion']['w'] = quat[0]
    d['quaternion']['x'] = quat[1]
    d['quaternion']['y'] = quat[2]
    d['quaternion']['z'] = quat[3]

    return d

# def transformFromPose(d):
#     pos = [0]*3
#     pos[0] = d['translation']['x']
#     pos[1] = d['translation']['y']
#     pos[2] = d['translation']['z']
#
#     quatDict = getQuaternionFromDict(d)
#     quat = [0]*4
#     quat[0] = quatDict['w']
#     quat[1] = quatDict['x']
#     quat[2] = quatDict['y']
#     quat[3] = quatDict['z']
#
#     return transformUtils.transformFromPose(pos, quat)

"""
msg: geometry_msgs/Pose
"""
# def transformFromROSPoseMsg(msg):
#     pos = [msg.position.x, msg.position.y, msg.position.z]
#     quat = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
#
#     return transformUtils.transformFromPose(pos,quat)
#
# def transformFromROSTransformMsg(msg):
#     pos = [msg.translation.x, msg.translation.y, msg.translation.z]
#     quat = [msg.rotation.w, msg.rotation.x, msg.rotation.y, msg.rotation.z]
#
#     return transformUtils.transformFromPose(pos,quat)

def getQuaternionFromDict(d):
    quat = None
    quatNames = ['orientation', 'rotation', 'quaternion']
    for name in quatNames:
        if name in d:
            quat = d[name]


    if quat is None:
        raise ValueError("Error when trying to extract quaternion from dict, your dict doesn't contain a key in ['orientation', 'rotation', 'quaternion']")

    return quat

def get_current_time_unique_name():
    """
    Converts current date to a unique name

    Note: this function will return variable-length strings.

    :return:
    :rtype: str
    """

    unique_name = time.strftime("%Y%m%d-%H%M%S")
    return unique_name

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def matrix_from_trans_quat(trans, quat_wxyz):
    q = numpy.array(quat_wxyz, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], trans[0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], trans[1]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], trans[2]],
        [                0.0,                 0.0,                 0.0, 1.0]])

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q


def homogenous_transform_from_dict(d):
    """
    Returns a transform from a standard encoding in dict format
    :param d:
    :return:
    """
    pos = [0]*3
    pos[0] = d['translation']['x']
    pos[1] = d['translation']['y']
    pos[2] = d['translation']['z']

    quatDict = getQuaternionFromDict(d)
    quat = [0]*4
    quat[0] = quatDict['w']
    quat[1] = quatDict['x']
    quat[2] = quatDict['y']
    quat[3] = quatDict['z']

    transform_matrix = quaternion_matrix(quat)
    transform_matrix[0:3,3] = np.array(pos)

    return transform_matrix


# def get_current_YYYY_MM_DD_hh_mm_ss():
#     """
#     Returns a string identifying the current:
#     - year, month, day, hour, minute, second
#
#     Using this format:
#
#     YYYY-MM-DD-hh-mm-ss
#
#     For example:
#
#     2018-04-07-19-02-50
#
#     Note: this function will always return strings of the same length.
#
#     :return: current time formatted as a string
#     :rtype: string
#
#     """
#
#     now = datetime.datetime.now()
#     string =  "%0.4d-%0.2d-%0.2d-%0.2d-%0.2d-%0.2d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
#     return string

def compute_angle_between_quaternions(q, r):
    """
    Computes the angle between two quaternions.

    theta = arccos(2 * <q1, q2>^2 - 1)

    See https://math.stackexchange.com/questions/90081/quaternion-distance
    :param q: numpy array in form [w,x,y,z]. As long as both q,r are consistent it doesn't matter
    :type q:
    :param r:
    :type r:
    :return: angle between the quaternions, in radians
    :rtype:
    """

    theta = 2*np.arccos(2 * np.dot(q,r)**2 - 1)
    return theta

def compute_translation_distance_between_poses(pose_a, pose_b):
    """
    Computes the linear difference between pose_a and pose_b
    :param pose_a: 4 x 4 homogeneous transform
    :type pose_a:
    :param pose_b:
    :type pose_b:
    :return: Distance between translation component of the poses
    :rtype:
    """

    pos_a = pose_a[0:3,3]
    pos_b = pose_b[0:3,3]

    return np.linalg.norm(pos_a - pos_b)

# def compute_angle_between_poses(pose_a, pose_b):
#     """
#     Computes the angle distance in radians between two homogenous transforms
#     :param pose_a: 4 x 4 homogeneous transform
#     :type pose_a:
#     :param pose_b:
#     :type pose_b:
#     :return: Angle between poses in radians
#     :rtype:
#     """
#
#     quat_a = transformations.quaternion_from_matrix(pose_a)
#     quat_b = transformations.quaternion_from_matrix(pose_b)
#
#     return compute_angle_between_quaternions(quat_a, quat_b)

def get_kuka_joint_names():
    return [
     'iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3',
     'iiwa_joint_4', 'iiwa_joint_5', 'iiwa_joint_6',
     'iiwa_joint_7']