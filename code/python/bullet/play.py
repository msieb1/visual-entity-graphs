from simenv.kuka_iiwa import kuka_iiwa
import pybullet as p
import os


print(os.path.abspath(__file__))
exit()
p.connect(p.GUI)
kuka = kuka_iiwa(urdfRootPath="../data/")
print(kuka.numJoints)
while True:
    p.setGravity(0,0,-10)
