# This is an autonomous robot which finds its way round a maze using QR codes for directions, it is built on the ROS2 Framework (Jazzy) and utilises OpenCV for the QR codes, and the robot runs in Gazebo in a custom world. 

# --------------------------------------

# There are three separate drivers:

# 1. qr_code_maze_driver_v1.py | Very basic driver, gets around the maze clumsily and requires lots of corrections

# 2. qr_code_maze_driver_v2.py | Sligtly more sophisticated, gets around the maze ok, still requires a lot of self correcting

# 3. qr_code_maze_driver_v3.py | A much more intelligent driver, uses line of best fit and cartesian coordinates to perform proper wall following. Drives down the maze gracefully and requires little to no self correction along the way.

# --------------------------------------

# To run and launch
# Use ROS2 Jazzy
# Gazebo Harmonic
# Rviz2
# Source environment
# ros2 launch qrcode_mazebot_bringup mazebot_bringup_v3.py

# --------------------------------------

# For tele-op and testing
# ros2 launch qrcode_mazebot_description gz_custom.py 
# ros2 launch qrcode_mazebot_description gz_empty.py



