import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import math
T = 25 # get from policy config of experiment
np.set_printoptions(precision=2)
#### HAND PROCESSING
# D435 intrinsics matrix
INTRIN = np.array([615.3900146484375, 0.0, 326.35467529296875, 
		0.0, 615.323974609375, 240.33250427246094, 
		0.0, 0.0, 1.0]).reshape(3, 3)
####################
IMG_HEIGHT = 480
IMG_WIDTH = 640

def _deproject_pixel_to_point(p2d, depth_z, intrin):
	p3d = np.zeros(3,)
	p3d[0] = (p2d[0] - intrin[0, 2]) / intrin[0, 0]
	p3d[1] = (p2d[1] - intrin[1, 2]) / intrin[1, 1]
	p3d[2] = 1.0
	p3d *= depth_z
	return p3d

def _project_point_to_pixel(p3d, intrin):
    # returns (width, height) indexed
	p2d = np.zeros(2,)
	p2d[0] = p3d[0] / p3d[2] * intrin[0, 0] + intrin[0, 2]
	p2d[1] = p3d[1] / p3d[2] * intrin[1, 1] + intrin[1, 2]
	return p2d
agent = {"T": T}
tt = np.arange(0, agent['T'], 1)[..., None]
R = (1.610e-01 - 0.0301e-01) / 2.0
cx_l = (1.610e-01 + 0.00301e-01) / 2.0
cy_l = (-1.063e-01 + 2.169e-02) / 2.0 
cx_r = cx_l + 0.03
cy_r = cy_l

cost_tgt = np.hstack([-R*np.sin(2*math.pi*tt / agent['T']) + cx_l,-R*np.cos(2*math.pi*tt / agent['T']) + cy_l, 0*tt + 0.4, -R*np.sin(2*math.pi*tt / agent['T']) + cx_r, -R*np.cos(2*math.pi*tt / agent['T']) + cy_r, 0*tt + 0.4, 0*tt])
print(cost_tgt)
import ipdb; ipdb.set_trace()
hand_traj = np.load(
'/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/imitation_learning/hand_2/videos/train/poses/circle_view0/right_hand_trajectory.npy'
)
depth = np.load(
'/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/imitation_learning/hand_2/depth/train/circle_view0.npy'
)
l_f_traj = hand_traj[5:-160, 4, :]
r_f_traj = hand_traj[5:-160, 8, :]

dep_l_f_traj = []
dep_r_f_traj = []
gripper_binary_traj = []

imgs = []
reader  = imageio.get_reader(
'/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/imitation_learning/hand_2/videos/train/poses/circle_view0/circle_view0.mp4'
)
for img in reader:
    imgs.append(img)

for i in range(T):
    step_size = int(np.floor(1.0*len(l_f_traj) / T))
    if i == 0 :
        plot_imgs = imgs[i]
    else:
        plot_imgs = np.hstack([plot_imgs, imgs[i*step_size]])

    mask = np.zeros(depth[i].shape, dtype=np.uint8)
    mask[np.where(depth[i*step_size] == 0)] = 1
    depth_inpainted = cv2.inpaint(depth[i*step_size],mask,3,cv2.INPAINT_TELEA)
    # plt.imshow(depth_inpainted)
    # plt.show()
    l_f_p2d = [np.clip(int(l_f_traj[i*step_size, 1]), 0, IMG_HEIGHT-1), np.clip(int(l_f_traj[i*step_size, 0]), 0, IMG_WIDTH-1)]
    r_f_p2d = [np.clip(int(r_f_traj[i*step_size, 1]), 0, IMG_HEIGHT-1), np.clip(int(r_f_traj[i*step_size, 0]), 0, IMG_WIDTH-1)]
    
    depth_l = depth_inpainted[l_f_p2d[0], l_f_p2d[1]] * 0.001
    depth_r = depth_inpainted[r_f_p2d[0], r_f_p2d[1]] * 0.001

    dep_l_f = _deproject_pixel_to_point([l_f_p2d[0], l_f_p2d[1]], depth_l, INTRIN)[[1,0,2]]
    dep_r_f = _deproject_pixel_to_point([r_f_p2d[0], r_f_p2d[1]], depth_r, INTRIN)[[1,0,2]]

    dep_l_f_traj.append(dep_l_f)
    dep_r_f_traj.append(dep_r_f)
    gripper_binary_traj.append(int(np.linalg.norm(dep_l_f - dep_r_f) > 0.075))

    # cv2.imshow('1', imgs[i*step_size][:,:,::-1])
    # cv2.waitKey(5)
    # print(gripper_binary_traj[-1])
    # print(np.linalg.norm(dep_l_f - dep_r_f))
    # p2d_l_proj = _project_point_to_pixel(dep_l_f, INTRIN)
    # p2d_r_proj = _project_point_to_pixel(dep_r_f, INTRIN)
dep_l_f_traj = np.array(dep_l_f_traj)
dep_r_f_traj = np.array(dep_r_f_traj)
gripper_binary_traj = np.array(gripper_binary_traj)[..., None]
cost_tgt = np.hstack([dep_l_f_traj, dep_r_f_traj, gripper_binary_traj])

for tt in range(T):

    if tt == 0 :
        plot_imgs = imgs[tt]
    else:
        plot_imgs = np.hstack([plot_imgs, imgs[tt*step_size]])
print(cost_tgt)
plt.figure(figsize=(20,5))
plt.imshow(plot_imgs)
plt.axis('off')
plt.show()