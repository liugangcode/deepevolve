#!/bin/bash
#$ -M liugangswu@outlook.com
#$ -m be
#$ -q gpu@ta-a6k-002
#$ -l gpu=1
#$ -N tmp_run_nuclei_image

# gpu@qa-titanx-001.crc.nd.edu
###### gpu@qa-titanx-001.crc.nd.edu
###### gpu@qa-2080ti-006.crc.nd.edu 
###### gpu@ta-a6k-003
###### gpu@qa-h100-001.crc.nd.edu
###### qrsh -q gpu@qa-h100-001.crc.nd.edu -l gpu_card=1

conda activate aplus

fsync $SGE_STDOUT_PATH &


cd /afs/crc.nd.edu/group/dmsquare/vol2/gliu7/a-plus-dev/examples/nuclei_image/initial_code

python main.py
