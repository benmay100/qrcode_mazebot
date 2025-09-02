import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ben-may/ros_workspaces/qrcode_mazebot_ws/install/qrcode_mazebot'
