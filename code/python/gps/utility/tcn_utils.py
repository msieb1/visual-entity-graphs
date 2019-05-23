import cv2
import numpy as np
import sys
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import tf
import rospy
from geometry_msgs.msg import PointStamped


### Hand Processing ###
def deproject_pixel_to_point_full(p2d, depth_z, intrin):
    ### Width / Height
	p3d = np.zeros(3,)
	p3d[0] = (p2d[0] - intrin[0, 2]) / intrin[0, 0]
	p3d[1] = (p2d[1] - intrin[1, 2]) / intrin[1, 1]
	p3d[2] = 1.0
	p3d *= depth_z
	return p3d

def deproject_pixel_to_point(p2d, depth_z, intrin):
    # dont use depth to scale xy, i.e. this is not the true camera frame but a depth normalized frame in xy dimension
    ### Width / Height
	p3d = np.zeros(3,)
	p3d[0] = (p2d[0] - intrin[0, 2]) / intrin[0, 0]
	p3d[1] = (p2d[1] - intrin[1, 2]) / intrin[1, 1]
	p3d[2] = 1.0
	p3d[2] *= depth_z
	return p3d

def project_point_to_pixel(p3d, intrin):
    # returns (width, height) indexed
	p2d = np.zeros(2,)
	p2d[0] = p3d[0] / p3d[2] * intrin[0, 0] + intrin[0, 2]
	p2d[1] = p3d[1] / p3d[2] * intrin[1, 1] + intrin[1, 2]
	return p2d

def smooth_3d_trajectory(traj, window=61, polyorder=3):
    assert traj.shape[1] == 3
    smoothed_traj = np.copy(traj)
    for j in range(3):
        smoothed_traj[:, j] = savgol_filter(traj[:, j], window, polyorder)
    return smoothed_traj

def preprocess_hand_trajectory(hand_trajectory_path, depth_raw_path):
    # some changeable constants
    indices = [4, 8] #indices of tracked fingers
    depth_scaling = 0.001
    window = 21 # window size for savgol
    polyorder = 3 # polyorder for savgol
    conf_threshold = 0.016 # confidence below which we throw away traj. predictions
    inpaint = True # antiquated, just means we inpaint instead of interpolate	
    show = False # antiquated, just means we save instead of show graphs

    # get smoothed trajectories
    traj_smooth, traj_deproj = get_smoothed_trajectories(hand_trajectory_path, depth_raw_path, \
    indices, conf_threshold, window, polyorder, \
    depth_scaling,  inpaint)
    # assert traj_smooth.shape[1] == 2
    return np.transpose(traj_smooth, (1, 0, 2)), np.transpose(traj_deproj, (1, 0, 2))

def get_2d_depth_finger_trajectories(hand_traj, depth):
    ### Gets 2d finger trajectory with raw depth value as 3rd coordinate
    l_f_traj = hand_traj[:, 4, :]
    r_f_traj = hand_traj[:, 8, :]
    # cv2.namedWindow('depth')

    for i in range(len(hand_traj)):
        mask = np.zeros(depth[i].shape, dtype=np.uint8)
        mask[np.where(depth[i] == 0)] = 1
        depth_inpainted = cv2.inpaint(depth[i],mask,3,cv2.INPAINT_TELEA)
        cur_depth = depth[i]
        # patch_depth_l = depth_inpainted[int(l_f_traj[i, 1])-3:int(l_f_traj[i, 1])+3, int(l_f_traj[i, 0])-3:int(l_f_traj[i, 0])+3]
        # patch_depth_r = depth_inpainted[int(r_f_traj[i, 1])-3:int(r_f_traj[i, 1])+3, int(r_f_traj[i, 0])-3:int(r_f_traj[i, 0])+3]
        # patch_depth_l = patch_depth_l[np.where(masked_depth > 0)]
        # z = np.median(np.sort(masked_depth.flatten()))
        depth_l = depth_inpainted[int(l_f_traj[i, 1]), int(l_f_traj[i, 0])] * 0.001
        depth_r = depth_inpainted[int(r_f_traj[i, 1]), int(r_f_traj[i, 0])] * 0.001
        # depth_l = np.median(np.sort(patch_depth_l.flatten())) * 0.001
        # depth_r = np.median(np.sort(patch_depth_r.flatten())) * 0.001
        l_f_traj[i, 2] = depth_l
        r_f_traj[i, 2] = depth_r  
        # cv2.imshow('depth', depth_inpainted)
        # cv2.waitKey(50)
    return l_f_traj, r_f_traj

def get_unprojected_3d_trajectory(traj, intrin):
    # takes width, height as input
    dep_traj = []
    for i in range(len(traj)):
        p2d = np.array([traj[i, 0], traj[i, 1]]) 
        dep = deproject_pixel_to_point([p2d[0], p2d[1]], traj[i, 2], intrin)  # 0: height, 1: width
        dep_traj.append(dep)
    dep_traj = np.array(dep_traj)
    return dep_traj

def get_unprojected_3d_mean_finger_and_gripper_trajectory(l_f_traj, r_f_traj, l_f_traj_orig, r_f_traj_orig, gripper_threshold, intrin, img_height=1e6, img_width=1e6):
    ### Gets unprojected 3d mean finger trajectory with binarized gripper values for each point
    ### Provide image height & width for proper clipping if trajectories contain invalid 2d values 
    assert len(l_f_traj) == len(r_f_traj)

    # Smooth temporally with savgol filter
    # l_f_traj = smooth_3d_trajectory(l_f_traj)
    # r_f_traj = smooth_3d_trajectory(r_f_traj)

    # Deprojected 3D finger trajectories
    dep_f_mean_traj = []
    f_mean_traj = []
    gripper_binary_traj = []
    for i in range(len(l_f_traj)):

        # l_f_p2d = np.array([l_f_traj[i, 0], l_f_traj[i, 1]]) 
        # r_f_p2d = np.array([r_f_traj[i, 0], r_f_traj[i, 1]])
        l_f_p2d = project_point_to_pixel(l_f_traj[i], intrin)
        r_f_p2d = project_point_to_pixel(r_f_traj[i], intrin)
               
        
        depth_l = l_f_traj[i, 2]
        depth_r = r_f_traj[i, 2]
        dep_l_f = deproject_pixel_to_point([l_f_p2d[0], l_f_p2d[1]], depth_l, intrin)  # 0: width, 1: height
        dep_r_f = deproject_pixel_to_point([r_f_p2d[0], r_f_p2d[1]], depth_r, intrin)  # x indexes column, y indexes row, so order is col - row - z, not row- col -z as usually in numpy

        f_mean_traj.append((l_f_p2d + r_f_p2d) / 2.0)
        dep_f_mean_traj.append((dep_l_f + dep_r_f) / 2.0)
        gripper_binary_traj.append(np.linalg.norm(l_f_traj_orig[i] - r_f_traj_orig[i]) > gripper_threshold)
    f_mean_traj = np.array(f_mean_traj)
    dep_f_mean_traj = np.array(dep_f_mean_traj)
    gripper_binary_traj = np.round(np.array(gripper_binary_traj))[..., None]
    return f_mean_traj, dep_f_mean_traj, gripper_binary_traj

def get_unprojected_3d_mean_finger_and_gripper_trajectory_(l_f_traj, r_f_traj, l_f_traj_orig, r_f_traj_orig, gripper_threshold, intrin, img_height=1e6, img_width=1e6):
    ### Gets unprojected 3d mean finger trajectory with binarized gripper values for each point
    ### Provide image height & width for proper clipping if trajectories contain invalid 2d values 
    assert len(l_f_traj) == len(r_f_traj)

    # Smooth temporally with savgol filter
    # l_f_traj = smooth_3d_trajectory(l_f_traj)
    # r_f_traj = smooth_3d_trajectory(r_f_traj)

    # Deprojected 3D finger trajectories
    dep_f_mean_traj = []
    f_mean_traj = []
    gripper_binary_traj = []
    for i in range(len(l_f_traj)):

        l_f_p2d = np.array([l_f_traj[i, 0], l_f_traj[i, 1]]) 
        r_f_p2d = np.array([r_f_traj[i, 0], r_f_traj[i, 1]])

        depth_l = l_f_traj[i, 2]
        depth_r = r_f_traj[i, 2]
        dep_l_f = deproject_pixel_to_point([l_f_p2d[0], l_f_p2d[1]], depth_l, intrin)  # 0: width, 1: height
        dep_r_f = deproject_pixel_to_point([r_f_p2d[0], r_f_p2d[1]], depth_r, intrin)  # x indexes column, y indexes row, so order is col - row - z, not row- col -z as usually in numpy

        f_mean_traj.append((l_f_p2d + r_f_p2d) / 2.0)
        dep_f_mean_traj.append((dep_l_f + dep_r_f) / 2.0)
        gripper_binary_traj.append(np.linalg.norm(l_f_traj_orig[i] - r_f_traj_orig[i]) > gripper_threshold)
    f_mean_traj = np.array(f_mean_traj)
    dep_f_mean_traj = np.array(dep_f_mean_traj)
    gripper_binary_traj = np.round(np.array(gripper_binary_traj))[..., None]
    return f_mean_traj, dep_f_mean_traj, gripper_binary_traj


class EEFingerListener(object):
    def __init__(self):
        self.listener = tf.TransformListener()

    def get_3d_pose_original(self, gripper='l', finger='r', frame=None):
        assert(gripper == 'r' or gripper == 'l')
        assert(finger == 'l' or finger == 'r')
        # camera frame is usually /camera{camID}_color_optical_frame (camID=2 in ourcase)
        self.listener.waitForTransform("/base", "/{}_gripper_{}_finger_tip".format(gripper, finger), rospy.Time(0), rospy.Duration(4.0))
        (trans,rot) = self.listener.lookupTransform("/base", "/{}_gripper_{}_finger_tip".format(gripper, finger), rospy.Time(0))
        p3d=PointStamped()
        p3d.header.frame_id = "base"
        p3d.header.stamp =rospy.Time(0)
        p3d.point.x=trans[0]
        p3d.point.y=trans[1]
        p3d.point.z=trans[2]
        if frame is not None:
            self.listener.waitForTransform("/base", frame, rospy.Time(0),rospy.Duration(4.0))
            p3d_transformed = self.listener.transformPoint(frame, p3d)
        p3d_transformed = np.array([p3d_transformed.point.x, p3d_transformed.point.y, p3d_transformed.point.z])
        return p3d_transformed   

    def get_3d_pose(self, gripper='l', finger='r', frame=None):
        OFFSET_FINGER_RIGHT = OFFSET_FINGER_LEFT = [-0.00, 0, -0.018]

        # camera frame is usually /camera{camID}_color_optical_frame (camID=2 in ourcase)
        # Intermediate transform from base to gripper base
        # !!! Do not just add translation, perform standard homogeneous transform to multiply both frame transforms !!!
        if gripper == 'l':
            # Base to Gripper Base
            self.listener.waitForTransform("/base", "/{}_gripper_base".format('left'), rospy.Time(0), rospy.Duration(1.0))
            (trans,rot) = self.listener.lookupTransform("/base", "/{}_gripper_base".format('left'), rospy.Time(0))
            # Gripper Base to Finger Tip
            self.listener.waitForTransform("/left_gripper_base", "/{}_gripper_{}_finger_tip".format(gripper, finger), rospy.Time(0), rospy.Duration(1.0))
            (trans_f,rot_f) = self.listener.lookupTransform("/left_gripper_base", "/{}_gripper_{}_finger_tip".format(gripper, finger), rospy.Time(0))

            hom_mat = tf.transformations.quaternion_matrix(rot)
            hom_mat[:-1, -1] = np.array(trans)
            # Compute finger tip position via multiplying current base->gripper_base transform on top of finger tip position in gripper_base frame using homogeneous coordinates (adding a 1 as 4th dimension)
            if finger == 'l':
                offset= OFFSET_FINGER_LEFT
            else:
                offset = OFFSET_FINGER_RIGHT
            new_trans_f = [a + b for a, b in zip(trans_f, offset)]
            new_trans = hom_mat.dot(np.array(new_trans_f + [1])[:, None])
        else:
          # Base to Gripper Base
          self.listener.waitForTransform("/base", "/{}_gripper_base".format('right'), rospy.Time(0), rospy.Duration(1.0))
          (trans,rot) = self.listener.lookupTransform("/base", "/{}_gripper_base".format('right'), rospy.Time(0))
          # Gripper Base to Finger Tip
          self.listener.waitForTransform("/right_gripper_base" "/{}_gripper_{}_finger_tip".format(gripper, finger), rospy.Time(0), rospy.Duration(1.0))
          (trans_f,rot_f) = self.listener.lookupTransform("/right_gripper_base" "/{}_gripper_{}_finger_tip".format(gripper, finger), rospy.Time(0))

          hom_mat = tf.transformations.quaternion_matrix(rot)
          hom_mat[:-1, -1] = np.array(trans)
          # Compute finger tip position via multiplying current base->gripper_base transform on top of finger tip position in gripper_base frame using homogeneous coordinates (adding a 1 as 4th dimension)
          if finger == 'l':
            offset= OFFSET_FINGER_LEFT
          else:
            offset = OFFSET_FINGER_RIGHT
          new_trans_f = [a + b for a, b in zip(trans_f, offset)]
          new_trans = hom_mat.dot(np.array(new_trans_f + [1])[:, None])
        p3d=PointStamped()
        p3d.header.frame_id = "base"
        p3d.header.stamp = rospy.Time(0)
        p3d.point.x = new_trans[0]
        p3d.point.y = new_trans[1]
        p3d.point.z = new_trans[2]
        if frame is not None:
            self.listener.waitForTransform("/base", frame, rospy.Time(0),rospy.Duration(4.0))
            p3d_transformed = self.listener.transformPoint(frame, p3d)
        p3d_transformed = np.array([p3d_transformed.point.x, p3d_transformed.point.y, p3d_transformed.point.z])
        return p3d_transformed[:, 0]
     
                

# linearly interpolates values for stretches of 0-values 
def interpolate_linear(depth): 
	n = len(depth)
	inter_depth = np.zeros(n)
	index = 0
	while depth[index] == 0: 
		index += 1
	inter_depth[:index] = depth[index]
	start = n-1

	while index < n: 
		if depth[index] != 0:
			inter_depth[index] = depth[index]
		else: 
			start = index - 1

			while depth[index] == 0 and index < n-1: 
				index += 1
			if index < n-2:
				inter_depth[start:index+1] = np.linspace(depth[start], depth[index], num=index - start + 1)
			else: 
				inter_depth[start+1:] = depth[start]
		index += 1

	return inter_depth



# extracts the confidences of each position predicted in the trajectory
def get_trajectory_confidences(hand_trajectory, num_indices, frames): 
	confidences = np.zeros([num_indices, frames])
	for i in range(num_indices):  
		confidences[i] = hand_trajectory[:, indices[i], -1]
	return confidences

# use cv2 inpainting to fill in 0-values of the depth
def inpaint_depth(depth_raw, frames): 
    max_value = float(np.max(depth_raw))
    depth = np.zeros(np.shape(depth_raw))

    for i in range(frames): 
		d = depth_raw[i]
		mask = np.zeros(d.shape).astype(np.uint8)
		mask[np.where(d == 0)] = 1
		
		depth[i] = cv2.inpaint(d,mask,3,cv2.INPAINT_TELEA)
    return depth

# find indices that remove any start/end stretches of low-conf frames
def clipping_indices(threshold, confidences, frames): 
    start = 0
    while np.any(confidences[:, start] < threshold): 
        start += 1

    end = frames - 1
    while np.any(confidences[:, end] < threshold): 
        end -= 1
    start = 0
    end = frames - 1
    return start, end, end - start + 1

# extract trajectory, confidences, depths for points along the trajectory
def process_traj(hand_trajectory, depth_, indices, num_frames, num_indices): 
	traj = np.zeros([num_indices, num_frames, 2])
	conf = np.zeros([num_indices, num_frames])
	depth = np.zeros([num_indices, num_frames])

	for i in range(num_indices):
		traj[i] = hand_trajectory[:, indices[i], :2]
		conf[i] = hand_trajectory[:, indices[i], -1]

		for j in range(num_frames): 
			depth[i, j] = depth_[j, traj[i, j, 1].astype(int), traj[i, j, 0].astype(int)]

	return traj, conf, depth

# clip the trajectories according to start and end indices
def clip_traj(traj, confidences, depth, start, end): 
	return traj[:, start:end+1, :], confidences[:, start:end+1], depth[:, start:end+1]

# interpolate trajectories linearly if confidence of pred less than thresh
def interpolate_traj(traj, conf, threshold, num_indices): 
	traj_inter = np.zeros(np.shape(traj))
	traj[np.where(conf < threshold*0)] = 0
	for i in range(num_indices): 
		for j in range(2): 
			traj_inter[i, :, j] = interpolate_linear(traj[i, :, j])
	return traj_inter

# interpolate depth of trajectory linearly if depth missing (=0)
def interpolate_depth(depth, threshold, num_indices): 
	depth_inter = np.zeros(np.shape(depth))
	for i in range(num_indices): 
		depth_inter[i] = interpolate_linear(depth[i])
	return depth_inter

# take trajectories and depths, deproject to (xyz) coordinates
def deproject_points(traj, inter_depth, intrin, num_indices, frames): 
	traj_reprojected = np.zeros([num_indices, frames, 3])
	for i in range(num_indices): 
		for j in range(frames): 
			traj_reprojected[i, j] = deproject_pixel_to_point_full(traj[i, j], inter_depth[i, j], intrin)
			# assert np.allclose(project_point_to_pixel(traj_reprojected[i, j], intrin), traj[i, j])
	return traj_reprojected

# use savgol filtering to smooth trajectories
def savgol_smooth_traj(traj_reprojected, num_indices, frames, window, polyorder): 
	traj_smooth = np.zeros([num_indices, frames, 3])
	for i in range(num_indices):
		for j in range(3):  
			traj_smooth[i, :, j] = savgol_filter(traj_reprojected[i, :, j], window, polyorder)
	return traj_smooth


# reads in video as pixel array
def read_in_video(video_path, num_frames, height, width): 
	vidcap = cv2.VideoCapture(video_path)
	success, image = vidcap.read()
	video_raw = np.zeros([num_frames, height, width])
	for i in range(num_frames): 
		video_raw[i] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		success, image = vidcap.read()
	return video_raw

'''
returns: 
	- array of smoothed trajectory positions of tracked fingers
		with shape [finger index, frame, xyz]

given: 
	- path of raw hand trajectories
	- path of raw depths

	- finger indices to track 
	- confidence threshold for interpolating
	- savgol window size
	- savgol polyorder

	- depth scaling factor 
	- whether to use inpaint (T = inpaint, F = interpolate)
	- whether to show the graphs (T = show, F = save)

	- path to video demonstration
	- path to save smoothed traj .npy to
	- path to save graphs to

'''
def get_smoothed_trajectories(hand_trajectory_path, depth_raw_path, \
	indices, conf_threshold, window, polyorder, \
	depth_scaling,  inpaint): 
	

    # load hand trajectories and depths from file
    hand_trajectory = np.load(hand_trajectory_path)
    depth_raw = np.load(depth_raw_path) # depth has 1 extra start/end frame
    if np.shape(depth_raw)[0] > np.shape(hand_trajectory)[0]: 
        depth_raw = depth_raw[1:-1]

    # dimensional information 
    num_indices = len(indices)
    num_frames_raw = np.shape(hand_trajectory)[0]
    height, width = np.shape(depth_raw)[1:]
    #print('raw dim', np.shape(hand_trajectory), np.shape(depth_raw))
    #print('num frames raw: ', num_frames_raw)

    # inpaint the depth
    if inpaint:
        depth_inpainted = inpaint_depth(depth_raw, num_frames_raw)
    else:
        depth_inpainted = depth_raw


    # get trajectories, confidence, depth of full trajectory
    traj_full, confidences_full, depth_full = process_traj(hand_trajectory, depth_inpainted, indices, num_frames_raw, num_indices)
    #print('full: ', np.shape(traj_full), np.shape(confidences_full), np.shape(depth_full))

    # find clipping indices for beginning/start of trajectory 
    traj_start, traj_end, num_frames = clipping_indices(conf_threshold, confidences_full, num_frames_raw)
    #print('clipping indices: ', traj_start, traj_end, num_frames)

    # clip trajectories
    traj, confidences, depth = clip_traj(traj_full, confidences_full, depth_full, traj_start, traj_end)
    #print('clipped: ', np.shape(traj), np.shape(confidences), np.shape(depth))

    # interpolate trajectory
    traj_inter = interpolate_traj(traj, confidences, conf_threshold, num_indices)

    # interpolate depth if not using inpainting 
    if inpaint: 
        depth_inter = depth * depth_scaling
    else: 
        depth_inter = interpolate_depth(depth, threshold, num_indices) 
        depth_inter = depth_inter * depth_scaling
    #print('interpolated: ', np.shape(traj_inter), np.shape(depth_inter))

    # These are the parameters of the intrinsic matrix K
    intrin = np.array([615.3900146484375, 0.0, 326.35467529296875, 
        0.0, 615.323974609375, 240.33250427246094, 
        0.0, 0.0, 1.0]).reshape(3, 3)

    # deproject points
    traj_reprojected = deproject_points(traj_inter, depth_inter, intrin, num_indices, num_frames)
    #print('reprojected: ', np.shape(traj_reprojected))

    # smooth using savgol filter
    traj_smooth = savgol_smooth_traj(traj_reprojected, num_indices, num_frames, window, polyorder)

    return traj_smooth, traj_reprojected

	