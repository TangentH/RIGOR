#!/bin/bash
# Script to publish all the floor plans stacked on top of one another
# This is only for demonstration purposes
#
# Use the installed floorplans. The last number is the height offset.
#
# Usage:
#   ./publish_floorplans.sh [windows|no_windows]
#
# Default is "windows".

SHARE_DIR=$(ros2 pkg prefix challenge_tools_ros)/share/challenge_tools_ros
MASK_SET=${1:-windows}
if [ "$MASK_SET" = "no_windows" ]; then
	FLOORPLAN_DIR="$SHARE_DIR/floorplans/masks_no-window"
else
	FLOORPLAN_DIR="$SHARE_DIR/floorplans/masks_with-windows"
fi

ros2 run challenge_tools_ros map_server.py "$FLOORPLAN_DIR/floor_EG.png" 0 &

ros2 run challenge_tools_ros map_server.py "$FLOORPLAN_DIR/floor_1.png" 10 &
ros2 run challenge_tools_ros map_server.py "$FLOORPLAN_DIR/floor_2.png" 20 &
ros2 run challenge_tools_ros map_server.py "$FLOORPLAN_DIR/floor_3.png" 30 &
ros2 run challenge_tools_ros map_server.py "$FLOORPLAN_DIR/floor_4.png" 40 &

ros2 run challenge_tools_ros map_server.py "$FLOORPLAN_DIR/floor_5.png" 50 &
ros2 run challenge_tools_ros map_server.py "$FLOORPLAN_DIR/floor_6.png" 60 &
ros2 run challenge_tools_ros map_server.py "$FLOORPLAN_DIR/floor_7.png" 70 &

# special logic added in code to offset this image to -10 metres and to translate it to the same x,y frame
ros2 run challenge_tools_ros map_server.py "$FLOORPLAN_DIR/floor_UG1.png" 80 &

# Wait for all background processes
wait
