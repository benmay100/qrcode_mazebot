#!/usr/bin/env python3

# dependencies 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge   #This allows us to convert the ROS image message into OpenCV datatype
import cv2
import math
import numpy as np


# Based on 640 points, +/- 45 degrees from the sides. 160 deg / 640 pts = 0.25 deg/pt.
# 45 degrees is 45 / 0.25 = 180 points.
RIGHT_WALL_INDICES = range(0, 181)       # Rightmost 181 points
LEFT_WALL_INDICES = range(459, 640)      # Leftmost 181 points
RIGHT_WALL_INDICES_SHORT = range(0, 50) #Rightmost 50 points, used for when robot is in a corner!
LEFT_WALL_INDICES_SHORT = range(590, 640) #Leftmost 50 points, used for when robot is in a corner!
DIRECT_FRONT_START = 315 # Center 10 points
DIRECT_FRONT_END = 325 # Center 10 points
DIRECT_RIGHT_START = 0 # Rightmost 10 points
DIRECT_RIGHT_END = 10 # Rightmost 10 points
DIRECT_LEFT_START = 630 # Leftmost 10 points
DIRECT_LEFT_END = 640 # Leftmost 10 points
LEFT_OF_FRONT_START = 400 # 120 point segment left of the front rays (for spotting corridor openings)
LEFT_OF_FRONT_END = 520 # 120 point segment left of the front rays (for spotting corridor openings)
RIGHT_OF_FRONT_START = 120 # 120 point segment right of the front rays (for spotting corridor openings)
RIGHT_OF_FRONT_END = 240 # 120 point segment right of the front rays (for spotting corridor openings)

# Import the helper class from the other file
from .robot_control_logic import RobotControl



class QrCodeMazeDriver(Node):
    def __init__(self):
        super().__init__('qr_code_mazer_driver')
        self.get_logger().info('[qr_code_mazer_driver] Node initialized')

        # Instantiate RobotControl, passing the node's logger
        self.RobotControl = RobotControl(self.get_logger()) # Pass self.get_logger() here

        #Customisable variables (depending on specific environment!)
        self.corridor_width = 0.6 #metres >> important that this is set to suit the maze corridor width you're using in the simulation!
        self.robot_centered_distance_to_wall_lower_bound = (self.corridor_width/2) - ((self.corridor_width/2)*0.15)
        self.robot_centered_distance_to_wall_upper_bound = (self.corridor_width/2) + ((self.corridor_width/2)*0.15)
        self.ideal_qr_measuring_distance = 0.38 #metres >> usually around 2/3 of corridor with but depends on size of QR codes simulation so needs to be tested empirically
        self.ideal_qr_measuring_distance_lower_bound = self.ideal_qr_measuring_distance - (self.ideal_qr_measuring_distance*0.075)
        self.ideal_qr_measuring_distance_upper_bound = self.ideal_qr_measuring_distance + (self.ideal_qr_measuring_distance*0.075)
        self.lateral_error_clamp = self.corridor_width*0.6 #Usually works well!

        #Lidar and position properties
        self.lidar_data = []
        self.lidar_angle_min = 0.0
        self.lidar_angle_increment = 0.0
        self.lateral_error = 0 #Lateral error is dependent on how the robot is angled, so will use the angular error as part of its calculations
        self.angular_error = 0.0 #(rads)
        self.average_direct_front_data = 10 #set to 10 to start to avoid triggering "read_qr" status before lidar readings come through
        self.average_direct_left_data = self.corridor_width /2 #set to  centralised value for start of program
        self.average_direct_right_data = self.corridor_width /2 #set to  centralised value for start of program
        self.average_left_of_front_data = 0.0 #A segment slightly off to the left but still poiting quite forward (used for establishing when a corridor is opening out)
        self.average_right_of_front_data = 0.0 #A segment slightly off to the left but still poiting quite forward (used for establishing when a corridor is opening out)
        self.estimated_corridor_length = 0.0 #Only used for estimated length of corridor the robot is TURNING into!
        
        #States, flags, timings etc
        self.current_state = "drive"
        self.current_position = "corridor" #Options 'corridor' 'corridor_after_left_turn' 'corridor_after_right_turn' 'corridor_opening_left' 'corridor_opening_right' 'corner_left' 'corner_right' 'end_of_maze' 'undetermined'
        self.last_qr_code = "" # Store the last detected QR code to avoid continuous detection
        self.is_lateral_correcting = False #A flag to show when robot is performing lateral correction (i.e. making sure driving centrally down corridor)
        self.is_qr_correcting = False #A flag to show when robot is performing qr code read correction (i.e. making sure it is aligned parallel to corridor)
        self.is_turning = False

        # Create publishers, subscribers, services, etc. here
        self.camera_subscriber_ = self.create_subscription(Image, "camera/image_raw", self.callback_camera_subscriber, 10)
        self.lidar_subscriber_ = self.create_subscription(LaserScan, "scan", self.callback_lidar_subscriber, 10)
        self.cmd_vel_publisher_ = self.create_publisher(Twist, "cmd_vel", 10)
        self.cmd_vel_timer_ = self.create_timer(0.1, self.callback_cmd_vel_publisher)
        self.cv_bridge=CvBridge()
        
        # Create a timer that calls the log_positions function every 0.5 seconds
        self.timer_period = 0.5  # seconds
        self.timer = self.create_timer(self.timer_period, self.log_positions)


    #Add callback functions here
    #---------------------------------------------------------------

    def callback_lidar_subscriber(self, msg):
        #Store important data
        self.lidar_data = msg.ranges
        self.lidar_angle_min = msg.angle_min
        self.lidar_angle_increment = msg.angle_increment
        # 1. Create the necessary lidar slices to perform functions throughout the program
        direct_front_data_raw = self.lidar_data[DIRECT_FRONT_START:DIRECT_FRONT_END]
        direct_right_data_raw = self.lidar_data[DIRECT_RIGHT_START:DIRECT_RIGHT_END]
        direct_left_data_raw = self.lidar_data[DIRECT_LEFT_START:DIRECT_LEFT_END][::-1] #this reverses the order, which is needed for left data arrays!
        right_of_front_data_raw = self.lidar_data[RIGHT_OF_FRONT_START:RIGHT_OF_FRONT_END]
        left_of_front_data_raw = self.lidar_data[LEFT_OF_FRONT_START:LEFT_OF_FRONT_END][::-1] #this reverses the order, which is needed for left data arrays!
        # 2. Filter out 'inf' values and get formatted data AND averages!
        self.average_direct_front_data = self.RobotControl.filter_out_inf_and_calculate_average(direct_front_data_raw)
        self.average_direct_right_data = self.RobotControl.filter_out_inf_and_calculate_average(direct_right_data_raw)
        self.average_direct_left_data = self.RobotControl.filter_out_inf_and_calculate_average(direct_left_data_raw)
        self.average_right_of_front_data = self.RobotControl.filter_out_inf_and_calculate_average(right_of_front_data_raw)
        self.average_left_of_front_data = self.RobotControl.filter_out_inf_and_calculate_average(left_of_front_data_raw)
        # 4. Calculate error for the ANGLE of the robot (see robot_control_logic.py for how this all works). Positive angle error means skewed right, Negative angle error means skewed left.
        self.angular_error, self.right_wall_slope, self.right_wall_angle, self.left_wall_slope, self.left_wall_angle = self.RobotControl._calculate_angle_error(
            RIGHT_WALL_INDICES, 
            LEFT_WALL_INDICES,
            RIGHT_WALL_INDICES_SHORT,
            LEFT_WALL_INDICES_SHORT, 
            self.lidar_data, 
            self.lidar_angle_min, 
            self.lidar_angle_increment, 
            self.angular_error, 
            self.current_position)
        # 5. Calculate the error for LATERAL DISTANCE of robot (see robot_control_logic.py for how this all works). Positive error means too far right, Negative error means too far left.
        self.lateral_error, self.average_direct_right_data, self.average_direct_left_data = self.RobotControl._calculate_lateral_error(
            self.average_direct_right_data, 
            self.average_direct_left_data, 
            self.angular_error, 
            self.corridor_width, 
            self.lateral_error,
            self.current_position) 
                

    def callback_camera_subscriber(self, msg):
        # Only process camera frames if the robot is in the 'read_qr' state
        if self.current_state == "read_qr":
            self.frame = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale for better performance
            # Create a QR Code detector instance
            qr_detector = cv2.QRCodeDetector()
            # Detect and decode the QR code - The detectAndDecode function returns three values: the decoded string, the bounding box of the QR code, and a straight-on version of the code. We only need the data string.
            data, _, _ = qr_detector.detectAndDecode(gray_frame)
            if data:  # A QR code was found and decoded successfully
                self.get_logger().info(f"QR Code detected with message: {data}")
                qr_data = data
                if qr_data and qr_data != self.last_qr_code:
                    self.last_qr_code = qr_data
                    if qr_data == "left":
                        self.get_logger().info("Instruction: Turn left")
                        self.current_state = "turn_left"
                    elif qr_data == "right":
                        self.get_logger().info("Instruction: Turn right")
                        self.current_state = "turn_right"
                    elif qr_data == "stop":
                        self.get_logger().info("Instruction: Stop")
                        self.current_state = "stop"
                    else:
                        self.get_logger().info("Unknown QR code. Sticking with last instruction.")
            else: # No QR code was found
                self.get_logger().warn("No QR code was found")
                self.current_state = "fine_tune" #perform correction function in 'fine_tune' case to try and find it!
                self.is_qr_correcting = True


    def callback_cmd_vel_publisher(self):
        #1. Declare message type we will use
        msg = Twist()
        #2. Before we do ANY publishing, we must always estimate the current_position
        self.current_position = self.RobotControl._establish_position(
            self.average_direct_front_data, 
            self.average_direct_right_data, 
            self.average_direct_left_data,
            self.average_right_of_front_data,
            self.average_left_of_front_data,
            self.lateral_error, 
            self.corridor_width, 
            self.current_position, 
            self.is_turning)
        #3. Now carry out state machine
        match self.current_state:
            case "drive":
                self.last_qr_code = ""
                Kp_lateral = 2.0
                Kp_angular = 3.5
                
                match self.current_position:
                    case "corridor" | "corridor_after_left_turn" | "corridor_after_right_turn" | "corridor_opening_left" | "corridor_opening_right":
                        if abs(self.lateral_error) >= 0.04:
                            self.is_lateral_correcting = True
                            msg.angular.z = self.lateral_error * Kp_lateral
                        elif 0.00 < abs(self.lateral_error) < 0.04 and self.is_lateral_correcting:
                            if abs(self.angular_error) >= 0.01:
                                msg.angular.z = self.angular_error * Kp_angular
                            else:
                                msg.angular.z = 0.00
                                self.is_lateral_correcting = False
                        else:
                            self.is_lateral_correcting = False
                            msg.angular.z = 0.00
                        
                        if self.current_position in ["corridor_opening_left", "corridor_opening_right"]: #Slows down as approaching corner
                            msg.linear.x = 0.45
                        else:
                            msg.linear.x = 0.65
                    
                    case "corner_left" | "corner_right" | "end_of_maze":
                        msg.linear.x = 0.00
                        msg.angular.z = 0.00
                        self.get_logger().warn(f"Forward distance is now {self.average_direct_front_data:.2f} meters, stopping and...")
                        self.current_state = "fine_tune"
                    
                    case _:
                        msg.linear.x = 0.00
                        msg.angular.z = 0.00
                        self.get_logger().warn("We have current_state [drive] but position [undetermined]")

            case "fine_tune":  #Handles fine tuning before reading a QR code, and after a turn
                self.get_logger().warn("fine tuning...")
                #For cases when about to read a QR code
                if not self.is_turning and not self.is_qr_correcting:
                    if self.average_direct_front_data < 0.33:
                        msg.linear.x = -0.2
                    elif self.average_direct_front_data >= 0.38:
                        msg.linear.x = 0.2
                    else:
                        msg.linear.x = 0.0
                        if abs(self.angular_error) > 0.03:
                            msg.angular.z = self.angular_error * 2
                        else:
                            msg.angular.z = 0.00
                            self.current_state = "read_qr"
                            self.get_logger().info("current_state set to 'read_qr'")
                #For cases when going in for second attempt to read a QR code
                elif not self.is_turning and self.is_qr_correcting:
                    #reverse until back to over 0.5 metres away
                    if 0.33 < self.average_direct_front_data <= 0.38:
                        msg.linear.x = -0.2
                        msg.angular.z = 0.0 #Keep steering straight
                    else:
                        msg.linear.x = 0.0
                        if abs(self.angular_error) > 0.03:
                            msg.angular.z = self.angular_error * 2
                        else:
                            msg.angular.z = 0.00
                            self.is_qr_correcting = False
                            self.current_state = "fine_tune" #Set state back to usual fine tune to go in for another attempt at reading
                            self.get_logger().info("Corrective fine-tune done, now reverting to normal fine-tune to attempt another read at QR code")
                #For cases when just completed a turn and making sure aligned nicely before driving off down next corridor
                else:
                    msg.linear.x = 0.0
                    if abs(self.angular_error) > 0.03:
                            msg.angular.z = self.angular_error * 2
                    else:
                        msg.angular.z = 0.00
                        self.is_turning = False
                        self.current_state = "drive"
                        self.get_logger().info("current_state set to 'drive'")


            case "read_qr":
                self.get_logger().info("Robot is stationary, waiting for camera callback to read QR code...")
                msg.linear.x = 0.0
                msg.angular.z = 0.0

            case "turn_left" | "turn_right":
                msg.linear.x = 0.0
                
                if self.estimated_corridor_length == 0.0:
                    if self.current_state == "turn_left":
                        self.estimated_corridor_length = self.average_direct_left_data
                        self.get_logger().info(f"Estimated corridor length updated to {self.estimated_corridor_length} metres")
                    else: # "turn_right"
                        self.estimated_corridor_length = self.average_direct_right_data
                        self.get_logger().info(f"Estimated corridor length updated to {self.estimated_corridor_length} metres")
                
                msg.angular.z, self.is_turning, self.current_state, self.estimated_corridor_length = self.RobotControl._perform_turn(
                    self.current_state,
                    self.current_position, 
                    self.average_direct_front_data, 
                    self.angular_error,
                    self.estimated_corridor_length,
                    self.is_turning)

            case "stop":
                self.is_turning = False
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.get_logger().warn("MAZE COMPLETE")

            case "stuck":
                self.is_turning = False
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.get_logger().warn("ROBOT STUCK - REASSESS CODE AND RESTART PROGRAM")

            case _:
                pass
                # self.get_logger().warn("Unknown state")
                
        self.cmd_vel_publisher_.publish(msg)


    def log_positions(self):
        if self.current_position not in ["end_of_maze"]:
            self.get_logger().info("==============================================")
            self.get_logger().info("CURRENT POSITION: ["+self.current_position+"]")
            self.get_logger().info("CURRENT STATE: ["+self.current_state+"]")
            self.get_logger().info("==============================================")
            self.get_logger().info(f"Average direct front distance: {self.average_direct_front_data:.2f} meters")
            self.get_logger().info(f"Average direct right distance: {self.average_direct_right_data:.2f} meters")
            self.get_logger().info(f"Average direct left distance: {self.average_direct_left_data:.2f} meters")
            if self.lateral_error > 0.01:
                self.get_logger().info(f"Lateral Error: {self.lateral_error:.2f} meters, too far right!")
            elif self.lateral_error < -0.01:
                self.get_logger().info(f"Lateral Error: {self.lateral_error:.2f} meters, too far left!")
            else:   
                self.get_logger().info(f"Lateral Error: {self.lateral_error:.2f} meters, CENTRALISED!")
            self.get_logger().info("__________________________________________")
            # if self.right_wall_slope is not None:
            #     self.get_logger().info(f"Right Wall: Slope={self.right_wall_slope:.3f}, Angle={self.right_wall_angle:.2f} rads")
            # else:
            #     self.get_logger().info("Right Wall: Not detected.")
            # if self.left_wall_slope is not None:
            #     self.get_logger().info(f"Left Wall:  Slope={self.left_wall_slope:.3f}, Angle={self.left_wall_angle:.2f} rads")
            # else:
            #     self.get_logger().info("Left Wall: Not detected.")
            if self.angular_error > 0.01:
                self.get_logger().info(f"Angular Error ={self.angular_error:.2f} rads - skewed Right!")
            elif self.angular_error < -0.01:
                self.get_logger().info(f"Angular Error ={self.angular_error:.2f} rads - skewed Left!")
            else:
                self.get_logger().info(f"Angular Error ={self.angular_error:.2f} rads - Angled CENTRALLY!")

    #---------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = QrCodeMazeDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()












        