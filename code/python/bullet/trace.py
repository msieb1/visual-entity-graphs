import numpy as np


def getField(record, index, func):
    """Get index field in record.
    
    Parameters
    ----------
    record : str
        Record string, divided by space
    index : int
        Index being queried
    func : function
        
    """
    return func(record.split()[index])


class Trace:
    def __init__(self, fname):
        fid = open(fname, 'r')
        lines = [line.strip() for line in fid]
        fid.close()
        # first id in self.objects is always kuka itself,
        # the second is object A, and the third is object B, etc.
        self.objects = list(map(int, lines[0].split()))
        self.objdict = {v:i for i, v in enumerate(self.objects)}
        self.totalTime = getField(lines[-1], 0, int) + 1

        # Read trace data from given file
        shape = len(self.objects), self.totalTime, 7 + 2
        # self.data's shape = (object, time, pose)
        self.data = np.zeros(shape)
        for line in lines[1:]:
            records = line.split()
            time = int(records[0])
            objectId = int(records[1])
            if objectId not in self.objdict:
                continue
            if objectId != self.objects[0]:
                records += [0, 0]
            objectIndex = self.objdict[objectId]
            kin = np.array(list(map(float, records[2:])))
            self.data[objectIndex, time] = kin

    def getTraj(self, objectId):
        objectIndex = self.objdict[objectId]
        return self.data[objectIndex]

    def getFirstPoint(self):
        """
        Returns
        -------
        list of objects info in the trace
        """
        # These are the objects being operated.
        objs = [self.objdict[x] for x in self.objects[1:]]
        kindata = self.data[objs, 0, :-2].tolist()
        return kindata

    def getLastPoint(self):
        """
        Returns
        -------
        list of objects info in the trace
        """
        # These are the objects being operated.
        objs = [self.objdict[x] for x in self.objects[1:]]
        kindata = self.data[objs, -1, :-2].tolist()
        return kindata


if __name__ == '__main__':
    t = Trace('./demos/pickAup_cubeenv1_0.bin.txt')
    print(t.getFirstPoint(), t.getLastPoint())
