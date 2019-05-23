import pybullet as p


cid = p.connect(p.SHARED_MEMORY)
if (cid<0):
	p.connect(p.GUI)

objects = [p.loadURDF("/Users/tangyihe/Dev/workspace/imitation_learning/learn_in_simulator/data/bowl.urdf")]
while True:
    p.setGravity(0,0,-10)
p.disconnect()
