#!/bin/sh

EXPECTED_ARGS=2
E_BADARGS=65
BASE='./src/frames'
BASE_DATA=${BASE}'/data'
BASE_OUTPUT=${BASE}'/frame_data'

if [ $# -lt ${EXPECTED_ARGS} ]
then
  echo "Usage: `basename $0` video frames/sec [size=256]"
  exit ${E_BADARGS}
fi

ACTION=$1
FILE_NAME=$2
FPS=$3

ffmpeg -i ${BASE_DATA}/${ACTION}/${FILE_NAME}.mp4 -f image2 -vf fps=${FPS} ${BASE_OUTPUT}/${ACTION}/${FILE_NAME}/%04d.jpg