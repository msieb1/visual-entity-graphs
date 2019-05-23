#!/bin/bash
EXPNAME="$1"
MODEL="$2"
TIME="$3"
CKPT="$4"
#rsync -vaP baxter:/media/msieb/data/tcn_data/experiments/$1/trained_models /home/max/projects/data/experiments/$1
mkdir -p /media/msieb/data/tcn_data/experiments/$1/trained_models/$2/$3
rsync -vaP isaac:/media/hdd/msieb/data/tcn_data/experiments/$1/trained_models/$2/$3/$CKPT /media/msieb/data/tcn_data/experiments/$1/trained_models/$2/$3

#mkdir -p /home/max/projects/data/experiments/$1/trained_models/$2/$3
#rsync -vaP isaac:/media/hdd/msieb/data/tcn_data/experiments/$1/trained_models/$2/$3/$CKPT /home/max/projects/data/experiments/$1/trained_models/$2/$3
