#!/bin/bash
EXPNAME=$1
# From laptop
#rsync -vaP --exclude '*.pk' /home/max/projects/data/experiments/$1 isaac:/media/hdd/msieb/data/tcn_data/experiments/

#rsync -vaP /home/max/projects/data/experiments/$1 baxter:/media/msieb/data/tcn_data/experiments/

# From Baxter
rsync -vaP --exclude '*.pk' /media/msieb/data/tcn_data/experiments/$1 isaac:/media/hdd/msieb/data/tcn_data/experiments/

