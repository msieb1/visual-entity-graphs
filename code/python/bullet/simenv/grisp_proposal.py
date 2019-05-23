import scipy.io as sio
import numpy as np
import scipy.misc
import math
import copy
#data = sio.loadmat("rgb_depth.mat")
#depth = data['depth']
#print(depth.shape)

def unit_vector(vector):
  """ Returns the unit vector of the vector.  """
  return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
  v1_u = unit_vector(v1)
  v2_u = unit_vector(v2)
  if v1_u[1] < 0:
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
  else:
    angle =  -np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
  if angle < -1.57:
    angle += 3.14
  if angle > 1.57:
    angle -= 3.14
  return angle

def mask2prop(mask, obj_id):
  h = mask.shape[0]
  w = mask.shape[1]
  non_mask = copy.deepcopy(mask)
  non_mask[non_mask==obj_id] = 100
  non_mask[non_mask==1] = 100
  non_mask[non_mask!=100] = 1
  non_mask[non_mask==100] = 0

  indices = np.where(mask == obj_id)
  y = np.mean(indices[0])
  x = np.mean(indices[1])
  center = np.array([x - w/2,y - h/2])
  
  best_value = 100
  best_angle = 0

  #scipy.misc.imsave("non_mask.png", non_mask)
  for iter in range(10):
    # sample 10 time and choose the best
    angle = 3.14*np.random.rand(1) - 1.57
    line_map = np.zeros((h,w))
    hit_non_mask = 0
   
    for x_id in range(w):
      y_id = y + (x_id + 0.5 - x)*math.tan(angle)
      #print(x, y, x_id, math.tan(angle), y_id)
      y_id = max(min(int(y_id), h-1), 0)
      if x_id == 0:
        y_id_old = y_id
      else:
        if np.abs(y_id_old - y_id) > h - 10:
          break
      y_id_old = y_id
      line_map[y_id, x_id] = 1
      
      if non_mask[y_id, x_id] == 1:
        hit_non_mask += 1
    if hit_non_mask < best_value:
      best_value = hit_non_mask
      best_angle = -angle
      #scipy.misc.imsave("line_map.png", line_map)
      #print(x,y, center, "best value", best_value)
  
  angle = best_angle
  x1 = x - math.cos(angle)
  y1 = y - math.sin(angle)
  x2 = x + math.cos(angle)
  y2 = y + math.sin(angle)
  return center, angle, 2, [x1, x2, y1, y2]


def depth2prop(depth):
  dzdx = (depth[1:-1, 2:] - depth[1:-1, :-2])/2.0
  dzdy = (depth[2:, 1:-1] - depth[:-2, 1:-1])/2.0

  h = dzdx.shape[0]
  w = dzdx.shape[1]

  sn = 0.000001 * np.ones([h,w,3])
  sn[:,:,0] = dzdx[:,:]
  sn[:,:,1] = dzdy[:,:]
  norm = np.sqrt(np.sum(np.power(sn, 2), 2))
  sn = sn/np.expand_dims(norm, 2)
  
  mask = np.zeros([h,w])
  mask[29:-29, 29:-29] = 1
  #mask[9:-9, 9:-9] = 1
  graspable_point = np.argwhere( np.transpose(np.multiply(mask, np.sum(np.power(sn[:,:,:2], 2), 2))) > 0.98);

  num_graspable_point = graspable_point.shape[0]
  point = graspable_point[np.random.randint(num_graspable_point), :]
  x = point[0] # x is row, y is column
  y = point[1]

  sn_point = (-np.sum(np.tile(sn[y,x,:2], [h, w, 1])*sn[:,:,:2],2) > 0.9).astype(np.float)*mask

  posmap = np.concatenate((np.tile(np.reshape(np.arange(0, w, 1), [1, -1, 1]), [h, 1, 1]),
                         np.tile(np.reshape(np.arange(0, h, 1), [-1, 1, 1]), [1, w, 1])), 2)\
         - np.tile(np.reshape(np.array([x, y]), [1, 1, 2]),[h,w,1]);
  wn = sn[y,x,:2]/np.sqrt(np.sum(np.power(np.reshape(sn[y,x,:2], [2,1]), 2)));
  dist = np.sqrt(np.sum(np.power(posmap, 2), 2)) #- \
               #np.power(np.sum(np.multiply(posmap, np.tile(wn, [h, w, 1])), 2), 2));

  idx = np.argmin(dist[sn_point == 1])
  idxs = np.argwhere(sn_point == 1)
  #scipy.misc.imsave("diff.png", np.tile(np.expand_dims(sn_point, 2), [1,1,3]))
  idx = idxs[idx]
  x2 = idx[1]
  y2 = idx[0]
  center = [(x + x2)/2.0 - int(w/2), (y + y2)/2.0 - int(h/2)]
  angle = angle_between([x2-x, y2-y], [1, 0])
  gripper_dist = np.linalg.norm([x2 - x, y2 - y]) 
  return center, angle, gripper_dist, [x - int(w/2), y - int(h/2), x2 - int(w/2), y2 - int(h/2)]



#
#center, angle = depth2prop(depth) 
#print(center)
#print("angle", angle)
