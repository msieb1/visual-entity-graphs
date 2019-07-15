from __future__ import division
import os
import csv
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file')
    parser.add_argument('-s', '--scale', dest='scale', type=int, default=10)
    parser.add_argument('-c', '--cycle_time', dest='cycle_time', type=int, default=5)
    return parser.parse_args()

def main():
    args = parse_arguments()

    data = list(csv.reader(open(args.file, "r")))
    header = data[0]
    wpts = np.array(data[1:]).astype(float)

    # slow trajectory down for adding wrist rotation
    wpts = np.repeat(wpts, args.scale, axis=0)

    # create timestamps
    wpts[:, 0] = np.arange(0, wpts.shape[0]/100, 0.01)

    # add wrist rotation
    UPPER_LIMIT = 3.05
    LOWER_LIMIT = -3.05
    cycle_length = int(args.cycle_time * 100)
    wrist_angles = np.linspace(LOWER_LIMIT, UPPER_LIMIT, cycle_length)
    wrist_angles = np.hstack([wrist_angles, wrist_angles[:-1][::-1]])

    n_cycles = int(np.ceil(wpts.shape[0]/wrist_angles.shape[0]))
    wrist_angles = np.tile(wrist_angles, n_cycles)
    wpts[:, -2] = wrist_angles[:wpts.shape[0]]
    # save traj
    folder, filename = os.path.split(args.file)
    filename = os.path.splitext(filename)[0] + '_processed' + '.csv'
    with open(os.path.join(folder, filename), mode='w') as traj_file:
        writer = csv.writer(traj_file, delimiter=',')
        writer.writerow(header)
        writer.writerows(wpts)
    
if __name__ == '__main__':
    main()