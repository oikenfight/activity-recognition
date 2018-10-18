#!/bin/sh

EXPECTED_ARGS=5
E_BADARGS=65

if [ $# -lt ${EXPECTED_ARGS} ]
then
  echo "Usage: `basename $0` video frames/sec [size=256]"
  exit ${E_BADARGS}
fi

BASE_INPUT=$1
ACTION=$2
FILE_NAME=$3
BASE_OUTPUT=$4
FPS=$5

ffmpeg -i ${BASE_INPUT}/${ACTION}/${FILE_NAME}.mp4 -f image2 -vf fps=${FPS} ${BASE_OUTPUT}/${ACTION}/${FILE_NAME}/%04d.jpg