#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH -t 20:00:00
#SBATCH --mem=1000

module load compiler/intel/19.1

# User specific aliases and functions
#*MCTDH*A***********************************************************************
# Following lines written by install_mctdh.  Wed Jul  8 10:03:26 CEST 2020
export MCTDH_DIR=/home/fr/fr_fr/fr_lr251/mctdh85.13
. $MCTDH_DIR/install/mctdh.profile
if [ -f ~/.mctdhrc ] && [ -t 0 ] ; then . ~/.mctdhrc ; fi
#*MCTDH*B***********************************************************************

mctdh85 -c $input
