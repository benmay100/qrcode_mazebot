import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


#THIS LAUNCH FILE UTILISED THE VERSION 2 MAZE DRIVER 
#===========================================================

def generate_launch_description():
    # Get the path to the 'my_robot_description' package
    description_pkg_dir = get_package_share_directory('qrcode_mazebot_description')
    
    # Get the path to the 'my_robot_nodes' package
    nodes_pkg_dir = get_package_share_directory('qrcode_mazebot')

    # Define the include action for the Gazebo/RViz launch file
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(description_pkg_dir, 'launch', 'gz_custom.launch.py')
            #os.path.join(description_pkg_dir, 'launch', 'gz_empty.launch.py')
        )
    )

    # Define the node action for the QR code maze drive
    maze_driver_node = Node(
        package='qrcode_mazebot',
        executable='qr_code_maze_driver_v3',
        name='qr_code_maze_driver_v3',
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        maze_driver_node
    ])