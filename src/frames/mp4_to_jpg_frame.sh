#!/bin/sh

EXPECTED_ARGS=2
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` video frames/sec [size=256]"
  exit $E_BADARGS
fi

ACTION=$1
FILE_NAME=$2
FPS=$3

ffmpeg -i data/$ACTION/$FILE_NAME.mp4 -f image2 -vf fps=$FPS frame_data/$ACTION/$FILE_NAME/%04d.jpg