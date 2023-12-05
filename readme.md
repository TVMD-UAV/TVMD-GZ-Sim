# TVMD Model Preview and Flight Review

This project establishes basic simulation environment for TVMD in GZ-Sim.

# Features

- Generation of TVMD sdf model (for GZ-Sim)
- Preview of TVMD model in Rviz
- Convert the ulog file to rosbag file, which can be used to replay the flight in Rviz

# Quick Start

To build this project:
```bash
catkin_make
. ./devel/setup.bash
```
Note that if the topics in `src/px4_msgs/msg` are modified, you need to make a clean build.

## Model Preview
Use Rviz to inspect the model:
```bash
roslaunch tvmd display.launch
```
- This will launch Rviz with TVMD model and start the republisher.

## Flight Review

This feature works for SITL and real flight logs.
1. Convert the ulog file to rosbag file using the provided [script](#ulog-file-conversion):
    ```bash
    ./ulog2bag.sh -f <ulog_filename>.ulg
    ```
    This will create a rosbag file in the same directory as the ulog file.
2. Launch rviz with TVMD model and start the republisher:
    ```bash
    roslaunch tvmd display.launch
    ```
    The rosbag file can be published using PlotJuggler, and `display.launch` will start the `republisher` node to initialize Rviz and  subscribe topics in rosbag i.e.,`px4/actuator_motors` and `px4/actuator_servos`. 
    - Note: launch `display.launch` first to start a ROS master for PlotJuggler.
3. Open `PlorJuggler`, go to the `Publisher` tab, click the `ROS Topic Re-Publisher` button, and select the topics to republish. 
    - The two topics are republished to `/joint_states`, which is the control input of the model in Rviz.


If the firmware is compiled in control allocation debug mode, the `ControlAllocationMetaData` topic will be available in the rosbag file. This topic contains the control allocation pseudo-forces and the calculation intermediate variables, which are helpful in debugging. The `republisher` node will publish this topic and convert them to [/visualization_msgs](https://wiki.ros.org/visualization_msgs), such that this data can be displayed in Rviz.

## Additional Tools
### SDF File Generation
To generate the SDF file for GZ-Sim:
```bash
sh ./tools/gen_sdf.sh
```

### ULog File Conversion
The conversion is done by `tools/pyulog/pyulog/ulog2rosbag.py`. Please follow the instruction in [`tools/pyulog`](https://github.com/PX4/pyulog) to install the tool.
A script is provided to facilitate the conversion:
```bash
./ulog2bag.sh -f <ulog_filename>.ulg
```
Args: 
+ `-f`: specify the ulog file to convert
+ `-a`: specify the topic to convert (default: `control_allocation_meta_data,actuator_motors,actuator_servos`, etc.)

Note that this feature requires a ROS environment with `px4_msgs` built and sourced. And the `px4_msgs` package should have `ControlAllocationMetaData.msg` in `px4_msgs/msg`, which can be obtained in [TVMD-UAV/TVMD-PX4](https://github.com/TVMD-UAV/TVMD-PX4). 