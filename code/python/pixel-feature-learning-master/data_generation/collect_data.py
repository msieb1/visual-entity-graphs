from __future__ import print_function
import os
import tf
import sys
import math
import cv2
import rospy
import argparse
import numpy as np
import subscribers
import select, time
import spartan_utils, utils

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scene', dest='scene_name')
    parser.add_argument('-d', '--dir', dest='data_dir', default='/home/zhouxian/git/pixel-feature-learning/pdc/logs_proto')
    parser.add_argument('-n', '--name', dest='camera_name', default='/camera')
    parser.add_argument('-t', '--tf_prefix', dest='camera_tf_prefix', default='/camera')
    parser.add_argument('-f', '--fps', dest='fps', type=int, default=10)
    return parser.parse_args()

def check_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_K():
    K = np.array([
        [615.3900146484375, 0.0, 326.35467529296875],
        [0.0, 615.323974609375, 240.33250427246094],
        [0.0, 0.0, 1.0]])
    return K

def main():
    args = parse_arguments()

    img_topic = args.camera_name + '/color/image_raw'
    depth_topic = args.camera_name + '/aligned_depth_to_color/image_raw'
    world_frame = '/base'
    camera_frame = args.camera_tf_prefix + '_color_optical_frame'

    # for toy
    # limit_lower = np.array([ 0.4, -0.35 , -0.02])
    # limit_upper = np.array([ 0.86, 0.35 , 0.25])

    # for destroyer
    # limit_lower = np.array([ 0.55, 0.05 , 0.03])
    # limit_upper = np.array([ 0.80, 0.20 , 0.3])

    limit_lower = np.array([ 0.3, -0.35 , -0.03])
    limit_upper = np.array([ 0.80, 0.35 , 0.25])

    rospy.init_node('data_collector')

    tf_listener = tf.TransformListener()
    img_subscriber = subscribers.img_subscriber(topic=img_topic)
    depth_subscriber = subscribers.depth_subscriber(topic=depth_topic)

    scene_name = args.scene_name
    data_dir = args.data_dir
    mask_dir = "image_masks/"
    depth_dir = "rendered_images/"
    rgb_dir = "images/"
    output_dir = os.path.join(data_dir, scene_name, 'processed/')
    # output_dir = 'temp/'
    check_dir(os.path.join(output_dir, mask_dir))
    check_dir(os.path.join(output_dir, depth_dir))
    check_dir(os.path.join(output_dir, rgb_dir))

    rospy.sleep(0.5)
    a = raw_input('Press ENTER to start collecting data...')
    print('Start recording. Press ENTER to stop.')

    idx = 0
    pose_data = dict()
    rate = rospy.Rate(args.fps)
    while True:
        rate.sleep()

        # stop collection if user inputs something
        rlist, _, _ = select.select([sys.stdin], [], [], 0.0001)
        if rlist:
            break
        # get data
        rgb_img = img_subscriber.img
        depth_img = depth_subscriber.img
        trans, quat_xyzw = tf_listener.lookupTransform(world_frame, camera_frame, rospy.Time(0))
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        transform_dict = spartan_utils.dictFromPosQuat(trans, quat_wxyz)

        # compute mask
        T_camera_world = tf_listener.fromTranslationRotation(trans, quat_xyzw)
        # pc_world, zero_depth_ids = utils.depth_to_pc(depth_img, T_camera_world)
        mask = utils.depth_to_mask(depth_img, limit_lower, limit_upper, T_camera_world)
        # import IPython;IPython.embed()
        # from visualization import Visualizer3D as vis
        # vis.figure(bgcolor=(1,1,1), size=(500,500))
        # vis.points(pc_world.reshape(640*480, 3)[np.random.choice(640*480, 10000)], scale=0.004)
        # vis.points(np.array([ 0.55, 0.05 , 0.03]), scale=0.01, color=(1,0,0))
        # vis.points(np.array([ 0.80, 0.22 , 0.3]), scale=0.01, color=(1,0,0))
        # vis.plot3d(np.array(([0, 0, 0], [0.2, 0, 0])).astype(np.float32), color=(1,0,0), tube_radius=0.01)
        # vis.plot3d(np.array(([0, 0, 0], [0, 0.2, 0])).astype(np.float32), color=(0,1,0), tube_radius=0.01)
        # vis.plot3d(np.array(([0, 0, 0], [0, 0, 0.2])).astype(np.float32), color=(0,0,1), tube_radius=0.01)
        # vis.show()

        # pc_world_masked = pc_world[np.where(mask)[0], np.where(mask)[1], :]
        # vis.points(pc_world_masked[np.random.choice(pc_world_masked.shape[0], 5000)], scale=0.002)
        # vis.points(np.array([ 0.4, -0.35 , 0.05]), scale=0.01, color=(1,0,0))
        # vis.points(np.array([ 0.86, 0.35 , -0.02]), scale=0.01, color=(1,0,0))
        # vis.show()
        # PIL.Image.fromarray(np.array(100*mask).astype(np.uint8)).show()

        # saving imgs
        rgb_filename = "%06i_%s.png" % (idx, "rgb")
        rgb_filename_full = os.path.join(output_dir, rgb_dir, rgb_filename)
        depth_filename = "%06i_%s.png" % (idx, "depth")
        depth_filename_full = os.path.join(output_dir, depth_dir, depth_filename)
        mask_filename = "%06i_%s.png" % (idx, "mask")
        mask_filename_full = os.path.join(output_dir, mask_dir, mask_filename)

        sys.stdout.write('Saving images %06i \r' % idx)
        sys.stdout.flush()
        cv2.imwrite(rgb_filename_full, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(depth_filename_full, depth_img)
        cv2.imwrite(mask_filename_full, mask.astype(np.int32))
        pose_data[idx] = dict()
        pose_data[idx]['camera_to_world'] = transform_dict
        pose_data[idx]['timestamp'] = rospy.get_time() # random timestamp
        pose_data[idx]['rgb_image_filename'] = rgb_filename
        pose_data[idx]['depth_image_filename'] = depth_filename

        idx += 1

    spartan_utils.saveToYaml(pose_data, os.path.join(output_dir, rgb_dir, 'pose_data.yaml'))


if __name__ == '__main__':
    main()