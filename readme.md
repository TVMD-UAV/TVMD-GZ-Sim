# TVMD GZ Sim

This project establishes basic simulation environment for TVMD in GZ-Sim.

# Quick Start

To build this project:
```bash
catkin_make
. ./devel/setup.bash
```

Use rviz to inspect the model:
```bash
roslaunch tvmd display.launch
```

To generate the SDF file for GZ-Sim:
```bash
sh ./tools/gen_sdf.sh
```