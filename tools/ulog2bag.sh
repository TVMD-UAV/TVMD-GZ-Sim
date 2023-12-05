#!/bin/bash

while getopts u:a:f: flag
do
    case "${flag}" in
        f) filename=${OPTARG};;
        a) topics=${OPTARG};;
    esac
done

if [[ ! -e $filename ]]; then
    echo "Filename not given or file do not exist. Exiting...";
    exit 1;
fi

DIR_NAME="$(dirname "$filename")" ; 
FILE_BASENAME="$(basename "$filename")"
EXTENSION="${FILE_BASENAME##*.}"
FILENAME="${FILE_BASENAME%.*}"

# topics to be converted
DEFAULT_TOPIC="control_allocation_meta_data,"\
"actuator_motors,actuator_servos,"\
"vehicle_thrust_setpoint,vehicle_torque_setpoint,"\
"vehicle_rate_setpoint,vehicle_attitude_setpoint,"\
"vehicle_attitude,vehicle_angular_velocity,"\
"vehicle_acceleration,vehicle_local_position,"\
"control_allocator_status" 

if [[ ${topics+x} ]]; then
    echo "Given Topics: $topics";
else
    topics=$DEFAULT_TOPIC
    echo "Using default topics: $topics";
fi

echo "Converting $FILE_BASENAME to $DIR_NAME/$FILENAME.bag"

SOURCE="$(pwd)/$filename"
TARGET="$(pwd)/$DIR_NAME/$FILENAME.bag"

ulog2rosbag -m $topics $SOURCE $TARGET
