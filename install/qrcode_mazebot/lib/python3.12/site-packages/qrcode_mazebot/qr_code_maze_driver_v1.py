#!/usr/bin/env python3

# dependencies 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge   #This allows us to convert the ROS image message into OpenCV datatype
import cv2
import math

class QrCodeMazeDriver(Node):
    def __init__(self):
        super().__init__('qr_code_mazer_driver')
        self.get_logger().info('[qr_code_mazer_driver] Node initialized')

        #Customisable variables (depending on environment!)
        self.corridor_width = 0.6 #metres >> important that this is set to suit the maze corridor width you're using in the simulation!
        self.robot_centered_distance_to_wall_lower_bound = (self.corridor_width/2) - ((self.corridor_width/2)*0.15)
        self.robot_centered_distance_to_wall_upper_bound = (self.corridor_width/2) + ((self.corridor_width/2)*0.15)
        self.ideal_qr_measuring_distance = 0.38 #metres >> usually around 2/3 of corridor with but depends entirely on size of QR codes in the simulation you're running so needs to be tested empirically
        self.ideal_qr_measuring_distance_lower_bound = self.ideal_qr_measuring_distance - (self.ideal_qr_measuring_distance*0.075)
        self.ideal_qr_measuring_distance_upper_bound = self.ideal_qr_measuring_distance + (self.ideal_qr_measuring_distance*0.075)

        #States, flags, timings etc
        self.current_state = "drive"
        self.last_qr_code = "" # Store the last detected QR code to avoid continuous detection
        self.turn_duration = 5.1 # Time in seconds to complete a 90-degree turn. You'll need to tune this value.
        self.move_timer = None # This will hold our timer object for peforming certain time-defined moves.
        self.is_turning = False # A flag to indicate if we're currently in the middle of a turn.
        self.average_forward_distance = 10 #set to 10 to start to avoid triggering "read_qr" status before lidar readings come through
        self.error = 0
        self.error_clamp = self.corridor_width*0.75 #Usually works well!
        self.reverse_turn_correction_phase = 0 #Complex reverse turns compose of two phases, 1 and 2 (and 0 for when not being used)
        self.reverse_turn_correction_duration = 1 # Time in seconds (for each phase) of a complex reverse turn to realign to QR code

        # Create publishers, subscribers, services, etc. here
        self.camera_subscriber_ = self.create_subscription(Image, "camera/image_raw", self.callback_camera_subscriber, 10)
        self.lidar_subscriber_ = self.create_subscription(LaserScan, "scan", self.callback_lidar_subscriber, 10)
        self.cmd_vel_publisher_ = self.create_publisher(Twist, "cmd_vel", 10)
        self.cmd_vel_timer_ = self.create_timer(0.1, self.callback_cmd_vel_publisher)
        self.cv_bridge=CvBridge()


        


    #Add callback functions here
    def callback_lidar_subscriber(self, msg):
        lidar_data = msg.ranges
        # 1. Separate the right, front, and left side data
        # front_data (points 315-326) are indices 314-325
        front_data = lidar_data[314:326]
        # We will now take just the first 30 data points from the lidar_data to get an accurate right-wall reading
        ninety_degree_right_data = lidar_data[0:30]
        # We will now take just the last 30 data points from the lidar_data to get an accurate left-wall reading
        ninety_degree_left_data = lidar_data[610:640]
        #We will also calculate some very specific averages now, as these may be needed for specific corrections when the program runs (see no_qr_code_found_correction_attempt function)
        absolute_right_data = lidar_data[0:5]
        offset_right_data = lidar_data[25:30]
        absolute_left_data = lidar_data[635:640]
        offset_left_data = lidar_data[610:615]

        # 2. Filter out 'inf' values using a list comprehension for each segment
        filtered_front_data = [d for d in front_data if not math.isinf(d)]
        filtered_ninety_degree_right_data = [d for d in ninety_degree_right_data if not math.isinf(d)]
        filtered_ninety_degree_left_data = [d for d in ninety_degree_left_data if not math.isinf(d)]
        filtered_absolute_right_data = [d for d in absolute_right_data if not math.isinf(d)]
        filtered_offset_right_data = [d for d in offset_right_data if not math.isinf(d)]
        filtered_absolute_left_data = [d for d in absolute_left_data if not math.isinf(d)]
        filtered_offset_left_data = [d for d in offset_left_data if not math.isinf(d)]

        # 3. Calculate the average distance for each filtered segment
        def calculate_average(data_list):
            if len(data_list) > 0:
                return sum(data_list) / len(data_list)
            else:
                return 0.0

        self.average_right_wall_distance = calculate_average(filtered_ninety_degree_right_data)
        self.average_left_wall_distance = calculate_average(filtered_ninety_degree_left_data)
        self.average_forward_distance = calculate_average(filtered_front_data)
        self.average_absolute_right_data = calculate_average(filtered_absolute_right_data)
        self.average_offset_right_data = calculate_average(filtered_offset_right_data)
        self.average_absolute_left_data = calculate_average(filtered_absolute_left_data)
        self.average_offset_left_data = calculate_average(filtered_offset_left_data)

        # 4. Calculate the error
        # A positive error means the robot is too close to the right wall, and negative error means robot is too close to left wall
        # However if robot is in a corner, with an open corridor to one side and a wall to the other this logic breaks down and we don't want robot to self-correct
        # So we will set an error_clamp in the attributes above which can be tweaked
        # If error exceeds error clamp robot to continue straight and not perform any turns - this is what the below does
        calculated_error = self.average_left_wall_distance - self.average_right_wall_distance
        if calculated_error > self.error_clamp or calculated_error < (self.error_clamp*-1): #may need to tune these distances
            if self.current_state == "drive":
                self.get_logger().info("Error exceeds the clamp, therefore in a corner and will not seek to correct")
            self.error = 0
        else:
            self.error = calculated_error

        # Print the results (only when driving)
        if self.current_state == "drive":
            self.get_logger().info(f"Average right wall distance: {self.average_right_wall_distance:.2f} meters")
            self.get_logger().info(f"Average left wall distance: {self.average_left_wall_distance:.2f} meters")
            self.get_logger().info(f"Average forward distance: {self.average_forward_distance:.2f} meters")
            self.get_logger().info(f"Calculated steering error: {self.error:.2f} meters")
            self.get_logger().info("__________________________________________")
    

    def callback_camera_subscriber(self, msg):
        # Only process camera frames if the robot is in the 'read_qr' state
        if self.current_state == "read_qr":
            self.frame = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale for better performance
            qr_data = self.detect_qr_code(gray_frame)

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
         

    def detect_qr_code(self, image):
        # Create a QR Code detector instance
        qr_detector = cv2.QRCodeDetector()
        # Detect and decode the QR code - The detectAndDecode function returns three values: the decoded string, the bounding box of the QR code, and a straight-on version of the code. We only need the data string.
        data, _, _ = qr_detector.detectAndDecode(image)
        if data:
            # A QR code was found and decoded successfully
            self.get_logger().info(f"QR Code detected with message: {data}")
            return data
        else:
            # No QR code was found
            self.get_logger().warn("No QR code was found")
            #perform correction function to try and find it!
            self.no_qr_code_found_correction_attempt()
            return None
    

    def no_qr_code_found_correction_attempt(self):
        #1. Work out which wall distance we need to reference here
        if self.average_left_wall_distance < self.average_right_wall_distance: #then robot is in corner with open corridor to right
            wall_to_reference = "left"
            current_distance_to_wall = self.average_left_wall_distance
        else:  #then robot is in corner with open corridor to left
            wall_to_reference = "right"
            current_distance_to_wall = self.average_right_wall_distance
        self.get_logger().info(f"Wall to reference is the {wall_to_reference} wall - current distance {current_distance_to_wall:.2f} meters")
        #2. Check robot is largely centralised
        if (current_distance_to_wall < self.robot_centered_distance_to_wall_upper_bound) and (current_distance_to_wall > self.robot_centered_distance_to_wall_lower_bound): #then we can be confident robot is nicely centralised
            if (self.average_forward_distance < self.ideal_qr_measuring_distance_lower_bound) or (self.average_forward_distance >= self.ideal_qr_measuring_distance_upper_bound):
                self.get_logger().info("Adjusting distance to try and view the QR code")
                self.current_state = "distance_correction"
            else:
                self.get_logger().warn(f"Robot sensible front distance from wall ({self.average_forward_distance:.2f} metres), so must be alignment issue")
                #Here we would perform no reverse, just try a left turn to sort alignment
                #To work out whether we need turn left or right, we need to use certain bits of the lidar data
                if wall_to_reference == "left":
                    if (self.average_absolute_left_data > self.average_offset_left_data):
                        self.get_logger().info(f"Far left lidar scan is {self.average_absolute_left_data:.2f} metres, and 30-offset from far left is {self.average_offset_left_data}, so we are aligned too far left, turning right now")
                        self.current_state = "turn_right_correction"
                    else:
                        self.get_logger().info(f"Far left lidar scan is {self.average_absolute_left_data:.2f} metres, and 30-offset from far left is {self.average_offset_left_data}, so we are aligned too far right, turning left now")
                        self.current_state = "turn_left_correction"
                elif wall_to_reference == "right":
                    if (self.average_absolute_right_data > self.average_offset_right_data):
                        self.get_logger().info(f"Far right lidar scan is {self.average_absolute_right_data:.2f} metres, and 30-offset from far right is {self.average_offset_right_data}, so we are aligned too far right, turning left now")
                        self.current_state = "turn_left_correction"
                    else:
                        self.get_logger().info(f"Far right lidar scan is {self.average_absolute_right_data:.2f} metres, and 30-offset from far right is {self.average_offset_right_data}, so we are aligned too far left, turning right now")
                        self.current_state = "turn_left_correction"                    
        else:
            self.get_logger().warn("ROBOT NOT CENTRAL - performing complex correction")
            #If robot too far from wall we need to reverse and shift robot towards the wall
            if (current_distance_to_wall > self.robot_centered_distance_to_wall_upper_bound and wall_to_reference == "right") or (current_distance_to_wall < self.robot_centered_distance_to_wall_lower_bound and wall_to_reference == "left"):
                self.get_logger().info("Performing reverse and shift to the right")
                self.current_state = "reverse_and_shift_right"
                self.reverse_turn_correction_phase = 1 #phase 1 of reverse turn will be reverse and right turn
            if (current_distance_to_wall > self.robot_centered_distance_to_wall_upper_bound and wall_to_reference == "left") or (current_distance_to_wall < self.robot_centered_distance_to_wall_lower_bound and wall_to_reference == "right"):
                self.get_logger().info("Performing reverse and shift to the left")
                self.current_state = "reverse_and_shift_left"
                self.reverse_turn_correction_phase = 1 #phase 1 of reverse turn will be reverse and right left



    def turn_completed_callback(self):
        self.is_turning = False
        self.move_timer.destroy() # It's good practice to destroy the timer once it's finished to free up resources.
        self.move_timer = None
        if self.current_state == "turn_left" or self.current_state == "turn_right":
            self.get_logger().info("Turn completed. Returning to drive state.")
            self.current_state = "drive"
            self.last_qr_code = "" #necessary to put this back to nil when transitioning from sucessful turn back to drive mode, otherwise consecutive left or right turns cause a loop in the callback_camera_subscriber method
        if self.current_state == "turn_left_correction" or self.current_state == "turn_right_correction":
            self.get_logger().info("Small correction turn completed. Returning to read_qr state.")
            self.current_state = "read_qr"
            self.last_qr_code = "" #necessary to put this back to nil when transitioning from sucessful turn back to drive mode, otherwise consecutive left or right turns cause a loop in the callback_camera_subscriber method
        if (self.current_state == "reverse_and_shift_right") or (self.current_state == "reverse_and_shift_left"):
            if self.reverse_turn_correction_phase == 2:
                self.get_logger().info("Complex reverse correction turn completed. Now entering 'distance_correction' state.")
                self.current_state = "distance_correction"
                self.reverse_turn_correction_phase = 0
            else:
                self.get_logger().info("Complex reverse correction turn phase 1 complete, moving to phase 2...")
                self.reverse_turn_correction_phase = 2


    def callback_cmd_vel_publisher(self):
        msg = Twist()
        if self.current_state == "drive":
            Kp = 1.5  # A proportional constant (Kp) is needed to scale central to corridor errors, you'll need to tune this value
            distance_to_start_slow_down = (self.corridor_width*1.33)
            if self.average_forward_distance >= distance_to_start_slow_down:
                msg.linear.x = 0.65
                msg.angular.z = self.error * Kp
            elif self.average_forward_distance < distance_to_start_slow_down and self.average_forward_distance >= self.ideal_qr_measuring_distance_upper_bound: #slows robot and keeps it on track
                msg.linear.x = 0.55
                msg.angular.z = self.error * Kp
            else:
                self.get_logger().warn(f"Forward distance is now {self.average_forward_distance:.2f} meters, STOPPING")
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.current_state = "read_qr"
                self.get_logger().info("current_state set to 'read_qr'")
        elif self.current_state == "read_qr":
            # Robot is stationary, waiting for camera callback to read QR code
            self.get_logger().info("Robot is stationary, waiting for camera callback to read QR code...")
            msg.linear.x = 0.0
            msg.angular.z = 0.0
        elif self.current_state == "turn_left":
            if not self.is_turning:
                self.get_logger().info("Starting a 90-degree left turn...")
                self.is_turning = True
                self.move_timer = self.create_timer(self.turn_duration, self.turn_completed_callback)
            # Keep publishing the turn command while the timer is active
            msg.linear.x = 0.0
            msg.angular.z = 1.0 # Positive value for left turn. Tune this.
        elif self.current_state == "turn_right":
            if not self.is_turning:
                self.get_logger().info("Starting a 90-degree right turn...")
                self.is_turning = True
                self.move_timer = self.create_timer(self.turn_duration, self.turn_completed_callback)
            # Keep publishing the turn command while the timer is active
            msg.linear.x = 0.0
            msg.angular.z = -1.0 # Negative value for right turn. Tune this.
        elif self.current_state == "turn_left_correction":
            if not self.is_turning:
                self.get_logger().info("Starting a small left turn to sort alignment...")
                self.is_turning = True
                self.move_timer = self.create_timer((self.turn_duration/4), self.turn_completed_callback)
            # Keep publishing the turn command while the timer is active
            msg.linear.x = 0.0
            msg.angular.z = 1.0 # Positive value for left turn. Tune this.
        elif self.current_state == "turn_right_correction":
            if not self.is_turning:
                self.get_logger().info("Starting a small right turn to sort alignment...")
                self.is_turning = True
                self.move_timer = self.create_timer((self.turn_duration/4), self.turn_completed_callback)
            # Keep publishing the turn command while the timer is active
            msg.linear.x = 0.0
            msg.angular.z = -1.0 # Negative value for right turn. Tune this.
        elif self.current_state == "distance_correction":
            if self.average_forward_distance < 0.38:
                msg.linear.x = -0.1
            elif self.average_forward_distance >= 0.4:
                msg.linear.x = 0.1
            else:
                self.get_logger().warn(f"Forward distance is now {self.average_forward_distance:.2f} meters, reverting to 'read_qr' state for another attempt")
                self.current_state = "read_qr"
        elif self.current_state == "reverse_and_shift_right":
            if not self.is_turning:
                self.get_logger().info(f"Phase {self.reverse_turn_correction_phase} of 'reverse and shift to right' procedure beginning.")
                self.is_turning = True
                self.move_timer = self.create_timer(self.reverse_turn_correction_duration, self.turn_completed_callback)
            # Keep publishing the turn command while the timer is active
            if self.reverse_turn_correction_phase == 1:
                msg.linear.x = -0.25
                msg.angular.z = 1.0 # Positive value for reverse right turn. Tune this.
            elif self.reverse_turn_correction_phase == 2:
                msg.linear.x = -0.25
                msg.angular.z = -0.85 # Negative value for reverse left turn. Tune this.
            else:
                self.get_logger().warn("Unknown reverse turn phase!")
        elif self.current_state == "reverse_and_shift_left":
            if not self.is_turning:
                self.get_logger().info(f"Phase {self.reverse_turn_correction_phase} of 'reverse and shift to right' procedure beginning.")
                self.is_turning = True
                self.move_timer = self.create_timer(self.reverse_turn_correction_duration, self.turn_completed_callback)
            # Keep publishing the turn command while the timer is active
            if self.reverse_turn_correction_phase == 1:
                msg.linear.x = -0.25
                msg.angular.z = -1.0 # Negative value for reverse left turn. Tune this.
            elif self.reverse_turn_correction_phase == 2:
                msg.linear.x = -0.25
                msg.angular.z = 0.85 # Positive value for reverse right turn. Tune this.

            else:
                self.get_logger().warn("Unknown reverse turn phase!")
        elif self.current_state == "stop":
            # Maze complete, robot stopped
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.get_logger().warn("MAZE COMPLETE")
        else:
            self.get_logger().warn("Unknown state")
        self.cmd_vel_publisher_.publish(msg)
    


def main(args=None):
    rclpy.init(args=args)
    node = QrCodeMazeDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()