#!/bin/bash
EXPNAME=$1
rsync -vaP baxter:/media/msieb/data/tcn_data/experiments/$1/videos_features /home/max/projects/data/experiments/$1/
#rsync -vaP baxter:/media/msieb/data/tcn_data/experiments/$1/videos_features /home/max/projects/data/experiments/$1/
