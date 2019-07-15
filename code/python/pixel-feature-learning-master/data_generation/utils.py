import numpy as np

def depth_to_pc(depth_img, T_camera_ref=np.eye(4), scale=1000.0, K=None):
    if K is None:
        K = np.array([
            [615.3900146484375, 0.0, 326.35467529296875],
            [0.0, 615.323974609375, 240.33250427246094],
            [0.0, 0.0, 1.0]])
    img_H, img_W = depth_img.shape
    depth_vec = depth_img/float(scale)
    zero_depth_ids = np.where(depth_vec == 0)
    depth_vec = depth_vec.ravel()
    u_vec, v_vec = np.meshgrid(np.arange(img_W), np.arange(img_H))
    u_vec = u_vec.ravel() * depth_vec
    v_vec = v_vec.ravel() * depth_vec
    full_vec = np.vstack((u_vec, v_vec, depth_vec))

    pc_camera = np.linalg.inv(K).dot(full_vec)
    pc_camera = np.vstack([pc_camera, np.ones(img_H*img_W)])
    pc_ref = T_camera_ref.dot(pc_camera)[:3].T

    return pc_ref.reshape((img_H, img_W, -1)), zero_depth_ids

def depth_to_mask(depth_img, limit_lower, limit_upper, T_camera_world, scale=1000.0, K=None):
    pc_world, zero_depth_ids = depth_to_pc(depth_img, T_camera_world, scale, K)

    mask = np.logical_and(pc_world<limit_upper, 
                          pc_world>limit_lower).all(axis=2)
    mask[zero_depth_ids[0], zero_depth_ids[1]] = False

    return mask

