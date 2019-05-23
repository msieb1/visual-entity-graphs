from pdb import set_trace
import struct
import argparse
import collections
import numpy as np
from sklearn import tree
import pydotplus
import io
import os
import random
from sklearn.externals import joblib
from trace import Trace
from batch_trace_gen import batch_trace_gen


def get_demo_path(path, demopath):
    without_ext = os.path.splitext(path)[0]
    basename = os.path.basename(without_ext)
    return os.path.join(demopath, basename)


def generateData(pos_list, neg_list, save_path, shuffle=True):
    fid = open(save_path, 'w')
    data = []
    for trace_name in pos_list:
        trace = Trace(trace_name)
        first_point = trace.getFirstPoint()
        last_point = trace.getLastPoint()
        # Generate two examples for a single demo
        data.append((first_point, first_point, 0))
        data.append((first_point, last_point, 1))

    for trace_name in neg_list:
        trace = Trace(trace_name)
        first_point = trace.getFirstPoint()
        last_point = trace.getLastPoint()
        data.append((first_point, last_point, 0))
    if shuffle:
        random.shuffle(data)
    joblib.dump(data, save_path)
    return data


class DecisionTree:
    def __init__(self):
        """
        Parameters
        ----------
        trace_list : list of str
            Trace file name list
        """
        self.data = None
        self.crf = tree.DecisionTreeClassifier()

    def _get_raw_feature(self, state):
        """state : list of pose tuples of different objs"""
        num_obj = len(state)
        feature = []
        compfeat = []
        for i in range(num_obj):
            feature += state[i][:3]

        for i in range(1, num_obj):
            compfeat += (np.array(state[0][:3]) - np.array(state[i][:3])).tolist()
        compfeat += [np.linalg.norm(np.array(state[0][:3]) - np.array(state[1][:3]))]
        return np.array(feature), np.array(compfeat)

    def load(self, model='data/dtree.model'):
        self.crf = joblib.load(model)
        return self

    def save(self, model='data/dtree.model'):
        joblib.dump(self.crf, model)
        return self

    def compute_feature(self, record):
        first_feat, first_comp = self._get_raw_feature(record[0])
        last_feat, last_comp = self._get_raw_feature(record[1])
        # dist = np.array([np.linalg.norm(last_feat[:3] - first_feat[:3])])
        # set_trace()
        return np.concatenate((last_feat - first_feat, first_comp, last_comp))

    def train(self, fname='data/train.pth'):
        self.data = joblib.load(fname)
        feature = list(map(self.compute_feature, self.data))
        labels = [x[2] for x in self.data]
        # set_trace()
        self.crf.fit(feature, labels)

    def visualize(self, filename='tree'):
        filename += '.png'
        dot_data = io.StringIO()
        feature_names = ['delta_objA_x',
                         'delta_objA_y',
                         'delta_objA_z',
                         'delta_objB_x',
                         'delta_objB_y',
                         'delta_objB_z',
                         'relative_x_atBeginning',
                         'relative_y_atBeginning',
                         'relative_z_atBeginning',
                         'distance_of_A_and_B_atBeginning',
                         'relative_x_atEnd',
                         'relative_y_atEnd',
                         'relative_z_atEnd',
                         'distance_of_A_and_B_atEnd',
                         ]
        tree.export_graphviz(self.crf, dot_data,
                             rounded=True,
                             proportion=True,
                             special_characters=True,
                             class_names=['No', 'Yes'])
                             # feature_names=feature_names,
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(filename)

    def predict_test(self, test_fname='data/test.pth'):
        data = joblib.load(test_fname)
        feature = list(map(self.compute_feature, data))
        labels = [x[2] for x in data]
        prediction = self.crf.predict(feature).tolist()
        # print()
        # print(data[1][1])
        # print(feature[1])
        # print()
        # print(data[3][1])
        # print(feature[3])
        # print()
        # print(data[5][1])
        # print(feature[5])
        print('Labels:     ', labels)
        print('Predictions:', prediction)
        return prediction

    def predict(self, obj_pos_start, obj_pos_end):
        """
        obj_pos_start : list of iterables
            Each object's position at first frame, in the order of A, B, C, etc.
        obj_pos_end : list of iterables
            Each object's position at first frame, in the order of A, B, C, etc.
        """
        data = (obj_pos_start, obj_pos_end, 0)
        feature = [self.compute_feature(data)]
        # print(feature[0].tolist())
        # feature = list(map(self.compute_feature, data))
        prediction = self.crf.predict(feature)
        return prediction.tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pos example keyword')
    parser.add_argument('--pos', type=str, required=True, help='pos specified')
    parser.add_argument('--train', type=str, default='data/train.pth', help='Training data path')
    parser.add_argument('--test', type=str, default='data/test.pth', help='Test data path')
    parser.add_argument('--demo', type=str, default='data/demos/2',
                        help='Path to demo trace folder')
    parser.add_argument('--demobin', type=str, default='data/demos',
                        help='Path to demo bin folder')

    parser.add_argument('--vis', action='store_true', help='Gen detector visualization')
    args = parser.parse_args()
    POS_STR = args.pos
    traces = [fname for fname in os.listdir(args.demo) if os.path.splitext(fname)[1]=='.txt']
    pos = [os.path.join(args.demo, f) for f in traces if f.startswith(POS_STR)]
    neg = [os.path.join(args.demo, f) for f in traces if not f.startswith(POS_STR)]
    random.shuffle(pos)
    random.shuffle(neg)
    generateData(pos[:-3], neg[:-3], args.train)
    # generateData(pos[:], neg[:], args.train)
    generateData(pos[-3:], neg[-3:], args.test, shuffle=False)
    testset = pos[-3:] + neg[-3:]
    testset = [get_demo_path(x, args.demobin) for x in testset]
    print('File tested:')
    for f in testset:
        print(f)

    dtree = DecisionTree()
    dtree.train(args.train)
    dtree.save()
    visual_file_path = os.path.join('data', POS_STR)
    dtree.visualize(visual_file_path)
    dtree.predict_test(args.test)
    if args.vis:
        batch_trace_gen(testset, video=False, detect=True)
