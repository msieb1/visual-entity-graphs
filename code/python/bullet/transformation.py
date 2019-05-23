import numpy as np
import pyquaternion as pq

def get_ortho_basis(v1, v2):
    """Produces an orthonormal basis given 2 vectors

    Parameters
    ----------
    v1 : array (1, 3)
        first vector (e.g, P1 to P2)
    v2 : array (1, 3)
        second vector (e.g., P1 to P3)

    Returns
    -------
    i, j, k : tupel of arrays (1, 3)
        orthonormal basis given 2 input vectors that are not collinear

    """
    
    i = v1/np.linalg.norm(v1)
    j = np.cross(v1, v2)
    j /= np.linalg.norm(j)
    k = np.cross(i, j)
    return i, j, k

def get_H(R, t):
    """Computes the homogeneous transformation given a rotation matrix and translation vector between two frames

    Parameters
    ----------
    R : array (3, 3)
        rotation matrix
    t : array (1, 3)
        translation vector

    Returns
    -------
    H : array (4, 4)
        homogeneous transformation matrix
    """
    H = np.eye(4, 4)
    H[0:-1, 0:-1] = R
    H[0:-1, -1] = t
    return H

def get_rotation_between_vectors(a, b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.abs(v)
    c = a.dot(b)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + skew + skew.dot(skew)*(1/(1+c))
    return R

def get_homogeneous_transform_from_basis(p1, p2, p3):
    """Computes the homogeneous transformation from three linearly independent vectors in world frame

    Parameters
    ----------
    p1 : array (1, 3)
        vector
    p2 : array (1, 3)
        vector
    p3 : array (1, 3)
        vector

    Returns
    -------
    H : array (4, 4)
        homogeneous transformation matrix
    """
    v1 = p2 - p1
    v2 = p3 - p1
    (i, j, k) = get_ortho_basis(v1, v2)
    # Obtain rigid body transformation matrices
    R = np.vstack([i, j, k]).T
    t = p1
    H = get_H(R, t)
    return H

def inv_H(H):
    """Computes inverse homogeneous matrix


    Parameters
    ----------
    H : array (4, 4)
        homogeneous matrix
    Returns
    -------

    H_inv : array (4, 4)
        inverse homogeneous matrix

    """
    R = H[0:-1, 0:-1]
    t = H[0:-1, -1]
    H_inv = np.eye(4, 4)
    H_inv[0:-1, 0:-1] = R.T
    H_inv[0:-1, -1] = -R.T.dot(t[:, None])[:, 0]
    return H_inv

def transform_pose(pose_as_array, H, mode='standard'):
    """Transform 7 dimensional pose (position + quaternion) to another coordinate frame

    Parameters
    ----------
    pose_as_array : array (7, )
        pose - x,y,z and qw, qx, qy, qz, given in frame1 (W is specified FIRST in pyquaternion and LAST in moveit)
    H : array (4, 4)
        homogeneous transformation matrix from frame1 to frame2 (if get_homogeneous_transform_from_basis was used, this mean
            that the basis vectors are expressed as coordinates in frame1, therefore H*p=p', where p is local frame2, and p' is world frame1)

    Returns
    -------

    pose_transformed : array (7, )
        transformed pose

    """
    if mode == 'from_baxter':

        pos_local = np.asarray(pose_as_array[:3])
        qu_buff = [pose_as_array[-1], pose_as_array[3], pose_as_array[4], pose_as_array[5]]
        qu_local = pq.Quaternion(qu_buff)
        pos_transformed = H.dot(np.hstack((pos_local, 1)))
        qu_transformed = pq.Quaternion(matrix=H) * qu_local
        qu_buff = np.asarray(qu_transformed.elements) 
        qu_as_array = np.asarray([qu_buff[1], qu_buff[2], qu_buff[3], qu_buff[0]])
        pose_transformed = np.hstack((pos_transformed[:-1], qu_as_array))
    else:
        pos_local = np.asarray(pose_as_array[:3])
        qu_local = pq.Quaternion(pose_as_array[3:].tolist())
        pos_transformed = H.dot(np.hstack((pos_local, 1)))
        qu_transformed = pq.Quaternion(matrix=H) * qu_local
        qu_as_array = np.asarray(qu_transformed.elements) 
        pose_transformed = np.hstack((pos_transformed[:-1], qu_as_array))
    return pose_transformed

def get_H_from_point_sets(ps1, ps2):

    cent_1 = np.mean(ps1, axis=0)
    cent_2 = np.mean(ps2, axis=0)
    E = np.zeros((3, 3))
    for i in range(ps1.shape[0]):
        if ps1[i] is None or ps2[i] is None:
            continue
        E += (ps1[i, :] - cent_1)[:, None].dot((ps2[i, :] - cent_2)[None, :])
    U, S, V = np.linalg.svd(E)
    R = V.dot(U.T) # make second V to U.T if more than 3 points
    t = -R.dot(cent_1) + cent_2
    H = np.zeros((4, 4))
    H[:-1,:-1] = R
    H[:-1, -1] = t
    H[-1, -1] = 1
    return H

def rigid_transform_3D(A, B):
    """
    Parameters
    ----------
    A, B: array (N, 3)
        
    Returns
    -------

    H : array (4,4)
        homogeneous transform

    """    
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.dot(AA.T, BB)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
       # print "Reflection detected"
       Vt[2,:] *= -1
       R = np.dot(Vt.T, U.T)

    t = np.dot(-R, centroid_A.T) + centroid_B.T
    H = np.zeros((4, 4))
    H[:-1,:-1] = R
    H[:-1, -1] = t
    H[-1, -1] = 1
    return H

def transform_trajectory(trajectory, H):
    """Transform trajectory (coordinates) from one frame to another, where the transformation
    is given by H. Base_points are three points within the new frame that constitute the basis
    of that frame and serve to construct the basis vectors of that frame

    Parameters
    ----------

    trajectory : array (3, time_steps) 
        3D-trajectory (possible 4th coordinate is confidence of that point) over all time steps
    H : array (4, 4)
        Homogeneous transformation matrix
        
    Returns
    -------

    trajectory_transformed : array (3, time_steps)
        trajectory expressed within the new coordinate system

    """


    # Transform trajectory

    trajectory_hom = np.ones((trajectory.shape[0], trajectory.shape[1] + 1))
    trajectory_hom[:, :-1] = trajectory
    trajectory_base_frame = np.dot(H, trajectory_hom.T) # (4, 2998, 60)
    trajectory_base_frame_cart = trajectory_base_frame[:-1, :].T

    return trajectory_base_frame_cart
