#!/bin/bash

# Collecting demos
#python python/collect_demo_w_agent.py --play 2/StackAOnB_cubeenv3_0.bin --demo cube
#python python/collect_demo_w_agent.py --play 2/putAIntoB_bowlenv1_16.bin --demo bowl
python python/collect_demo_w_agent.py --play 2/PushAOffB_cubeenv3_9.bin --demo cube_push

# Running PILQR Training
#python python/gps/gps_main.py -experiment bowl_and_cube
#python python/gps/gps_main.py -experiment bowl_and_cube_mrcnn
#python python/gps/gps_main.py -experiment red_cube_stacking
#python python/gps/gps_main.py -experiment red_cube_stacking_mrcnn

# Compute MRCNN features
python python/compute_mrcnn_output.py -d /home/msieb/projects/gps-lfd/demo_data/duck -s 99
#python python/compute_mrcnn_output.py -d /home/msieb/projects/gps-lfd/demo_data/cube_bowl -s 9

# Collecting demos from gps-stick (allows for handcrafted demos, fix in future!)
# python python/gps/collect_demo_w_agent_v2.py --demo duck

