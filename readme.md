# TVMD GZ Sim

This project establishes basic simulation environment for TVMD in GZ-Sim.

# Features

- Generation of TVMD sdf model (for GZ-Sim)
- Preview of TVMD model in rviz
- Convert the ulog file to rosbag file, which can be used to replay the flight in rviz
    - The rosbag file can be republished using plotjuggler, and `display.launch` will start the `republisher` node to republish `px4/actuator_motors` and `px4/actuator_servos`.
    - The two topics are republished to `/joint_states`, which is the control input of the model in rviz.
    - The conversion is done by `tools/pyulog/pyulog/ulog2rosbag.py`.
    ```bash
    /usr/local/bin/ulog2rosbag -m <the_topic_name_to_convert> <ulog_filename>.ulg <output_bag_filename>.bag
    # e.g.
    # /usr/local/bin/ulog2rosbag -m control_allocation_meta_data logs/sitl_ebrca/14_57_40.ulg logs/sitl_ebrca/14_57_40.bag
    ```

# Quick Start

To build this project:
```bash
catkin_make
. ./devel/setup.bash
```
Note that if the topics in `src/px4_msgs/msg` are modified, you need to make a clean build.

## Model Preview
Use rviz to inspect the model:
```bash
roslaunch tvmd display.launch
```
- This will launch rviz with TVMD model and start the republisher.

To generate the SDF file for GZ-Sim:
```bash
sh ./tools/gen_sdf.sh
```

## Flight Review

This feature works for SITLs and real flights.
1. Convert the ulog file to rosbag file:
    ```bash
    /usr/local/bin/ulog2rosbag -m <the_topic_name_to_convert> <ulog_filename>.ulg <output_bag_filename>.bag
    # e.g.
    # /usr/local/bin/ulog2rosbag -m control_allocation_meta_data logs/sitl_ebrca/14_57_40.ulg logs/sitl_ebrca/14_57_40.bag
    ```
2. Open `PlorJuggler`, go to the `Publisher` tab, click the `ROS Topic Re-Publisher` button, and select the topics to republish.
3. Launch rviz with TVMD model and start the republisher:
    ```bash
    roslaunch tvmd display.launch
    ```
