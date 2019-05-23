import os
import argparse
import pybullet as p
from vr_demo_collect import main as gen
from easydict import EasyDict

DEMO_DIR = 'demos'


def batch_trace_gen(files, video=False, detect=False):
    p.connect(p.GUI)
    for fname in files:
        body, ext = os.path.splitext(fname)
        if ext == '.bin':
            print("Generating trace for {}".format(fname))
            args = EasyDict()
            args.play = fname
            args.env = None
            args.video = video
            args.detect = detect
            gen(args, no_connect=True)
    p.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate trace life for a dir of bin files')
    parser.add_argument('--dir', required=True, nargs='+', type=str, help='Directory specified')
    parser.add_argument('--video', action='store_true', help='Flag to enable video recording.')
    args = parser.parse_args()
    rootdirs = set(os.path.abspath(x) for x in args.dir)
    files = []
    for d in rootdirs:
        files += [os.path.join(d, x) for x in os.listdir(d)]
    files = sorted(files)
    batch_trace_gen(files, video=args.video)
