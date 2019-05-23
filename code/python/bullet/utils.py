import pybullet_data
import os
import numpy as np
import pybullet as p
import struct


CONTROLLER_ID = 0
POSITION=1
ORIENTATION=2
ANALOG=3
BUTTONS=6

BUTTON_MAPS = {'trigger':33, 'menu':1, 'side':2, 'pad':32}


def sysdatapath(*filepath):
  return os.path.join(pybullet_data.getDataPath(), *filepath)


def euclidean_dist(point1, point2):
	assert len(point1) == 3
	assert len(point2) == 3
	point1 = np.array(point1)
	point2 = np.array(point2)
	return np.linalg.norm(point1 - point2)


def pressed(vrevent, button_name):
	return vrevent[BUTTONS][BUTTON_MAPS[button_name]] == p.VR_BUTTON_IS_DOWN
	

def readLogFile(filename, verbose = True):
  f = open(filename, 'rb')

  print('Opened'),
  print(filename)

  keys = f.readline().decode('utf8').rstrip('\n').split(',')
  fmt = f.readline().decode('utf8').rstrip('\n')

  # The byte number of one record
  sz = struct.calcsize(fmt)
  # The type number of one record
  ncols = len(fmt)

  if verbose:
    print('Keys:'),
    print(keys)
    print('Format:'),
    print(fmt)
    print('Size:'),
    print(sz)
    print('Columns:'),
    print(ncols)

  # Read data
  wholeFile = f.read()
  # split by alignment word
  chunks = wholeFile.split(b'\xaa\xbb')
  log = list()
  for chunk in chunks:
    if len(chunk) == sz:
      values = struct.unpack(fmt, chunk)
      record = list()
      for i in range(ncols):
        record.append(values[i])
      log.append(record)

  return log
  
def clean_line(line):
    """
    Cleans a single line of recorded joint positions
    @param line: the line described in a list to process
    @param joint_names: joint name keys
    @return command: returns dictionary {joint: value} of valid commands
    @return line: returns list of current line values stripped of commas
    """
    def try_float(x):
        try:
            return float(x)
        except ValueError:
            return None
    #convert the line of strings to a float or None
    line = [try_float(x) for x in line.rstrip().split(' ')]
    #zip the values with the joint names
    #take out any tuples that have a none value
    #convert it to a dictionary with only valid commands

    ID = str(int(line[1]))
    return (ID, line,)
