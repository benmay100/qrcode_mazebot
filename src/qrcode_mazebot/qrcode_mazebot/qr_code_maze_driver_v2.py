#!/usr/bin/env python3

# dependencies 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge   #This allows us to convert the ROS image message into OpenCV datatype
import cv2
import math

class QrCodeMazeTester(Node):
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
        self.lidar_data = []

        #States, flags, timings etc
        self.current_state = "drive"
        self.last_qr_code = "" # Store the last detected QR code to avoid continuous detection
        self.move_timer = None # This will hold our timer object for peforming certain time-defined moves.
        self.is_turning = False # A flag to indicate if we're currently in the middle of a turn.
        self.average_direct_front_data = 10 #set to 10 to start to avoid triggering "read_qr" status before lidar readings come through
        self.error = 0
        self.error_clamp = self.corridor_width*0.6 #Usually works well!
        self.reverse_turn_correction_phase = 0 #Complex reverse turns compose of two phases, 1 and 2 (and 0 for when not being used)
        self.turn_duration = 1.5 # Time in seconds to complete each phase of the complex reverse turn (for corrections). You'll need to tune this value.

        # Create publishers, subscribers, services, etc. here
        self.camera_subscriber_ = self.create_subscription(Image, "camera/image_raw", self.callback_camera_subscriber, 10)
        self.lidar_subscriber_ = self.create_subscription(LaserScan, "scan", self.callback_lidar_subscriber, 10)
        self.cmd_vel_publisher_ = self.create_publisher(Twist, "cmd_vel", 10)
        self.cmd_vel_timer_ = self.create_timer(0.1, self.callback_cmd_vel_publisher)
        self.cv_bridge=CvBridge()


    #Helper functions here:
    #---------------------------------------------------------------
    
    def calculate_average(self, data_list): #A function used throughout the program to calculate average values for lidar slice arrays
        if len(data_list) > 0:
            return sum(data_list) / len(data_list)
        else:
            return 0.0

    #---------------------------------------------------------------


    #Add callback functions here
    #---------------------------------------------------------------

    def callback_lidar_subscriber(self, msg):
        self.lidar_data = msg.ranges
        # 1. Create the necessary lidar slices to perform functions throughout the program
        direct_front_data = self.lidar_data[315:325] #central 10 pts
        direct_right_data = self.lidar_data[0:32]
        direct_left_data = self.lidar_data[608:640][::-1] #this reverses the order, which is needed for all left data arrays!
        broad_right_data = self.lidar_data[0:160]
        broad_left_data = self.lidar_data[480:640][::-1] #this reverses the order, which is needed for all left data arrays!
        # 2. Filter out 'inf' values using a list comprehension for each segment
        direct_front_data = [d for d in direct_front_data if not math.isinf(d)]
        direct_right_data = [d for d in direct_right_data if not math.isinf(d)]
        direct_left_data = [d for d in direct_left_data if not math.isinf(d)]
        broad_right_data = [d for d in broad_right_data if not math.isinf(d)]
        broad_left_data = [d for d in broad_left_data if not math.isinf(d)]
        # 3. Calculate the average distance for each filtered segment (uses a helper function below in the code)
        self.average_direct_front_data = self.calculate_average(direct_front_data)
        self.average_direct_right_data = self.calculate_average(direct_right_data)
        self.average_direct_left_data = self.calculate_average(direct_left_data)
        self.average_broad_right_data = self.calculate_average(broad_right_data)
        self.average_broad_left_data = self.calculate_average(broad_left_data)
        # 4. Calculate the error
        # A positive error means the robot is too close to the right wall, and negative error means robot is too close to left wall
        # However if robot is in a corner, with an open corridor to one side and a wall to the other this logic breaks down and we don't want robot to self-correct
        # So we will set an error_clamp in the attributes above which can be tweaked
        # If error exceeds error clamp robot to continue straight and not perform any turns - this is what the below does
        calculated_error = self.average_direct_left_data - self.average_direct_right_data
        if calculated_error > self.error_clamp or calculated_error < (self.error_clamp*-1): #may need to tune these distances
            if self.current_state == "drive":
                self.get_logger().info("Error exceeds the clamp, therefore in a corner and will not seek to correct")
            self.error = 0
        else:
            self.error = calculated_error
        # 5. Print the results (only when driving)
        if self.current_state == "drive":
            self.get_logger().info(f"Average direct front distance: {self.average_direct_front_data:.2f} meters")
            self.get_logger().info(f"Average direct right distance: {self.average_direct_right_data:.2f} meters")
            self.get_logger().info(f"Average direct left distance: {self.average_direct_left_data:.2f} meters")
            self.get_logger().info(f"Average broad right distance: {self.average_broad_right_data:.2f} meters")
            self.get_logger().info(f"Average broad left distance: {self.average_broad_left_data:.2f} meters")
            self.get_logger().info(f"Calculated steering error: {self.error:.2f} meters")
            self.get_logger().info("__________________________________________")


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
                self.no_qr_code_found_correction_attempt() #perform correction function to try and find it!



    def no_qr_code_found_correction_attempt(self):
        #1. Work out which wall distance we need to reference here
        if self.average_direct_left_data < self.average_direct_right_data: #then robot is in corner with wall to left, and open corridor to right
            wall_to_reference = "left"
            current_distance_to_wall = self.average_direct_left_data
        else:  #then robot is in corner with wall to right, and open corridor to left
            wall_to_reference = "right"
            current_distance_to_wall = self.average_direct_right_data
        self.get_logger().info(f"Wall to reference is the {wall_to_reference} wall - current distance {current_distance_to_wall:.2f} meters")
        #2. Check robot is largely centralised
        if (current_distance_to_wall < self.robot_centered_distance_to_wall_upper_bound) and (current_distance_to_wall > self.robot_centered_distance_to_wall_lower_bound): #then we can be confident robot is nicely centralised
            if (self.average_direct_front_data < self.ideal_qr_measuring_distance_lower_bound) or (self.average_direct_front_data >= self.ideal_qr_measuring_distance_upper_bound): #then robot probably too close or too far away, so will adjust distance
                self.get_logger().info("Adjusting distance to try and view the QR code")
                self.current_state = "distance_correction"
            else: #Then robot is centralised, and a sensible distance from wall, so it is likely out of parallel alignment
                self.get_logger().warn(f"Robot sensible front distance from wall ({self.average_direct_front_data:.2f} metres), so must be alignment issue")
                self.alignment_issue_turn_callback()     
        else: #Robot not centralised
            self.get_logger().warn("ROBOT NOT CENTRAL - performing complex reverse shift correction")
            #If robot too far from wall we need to reverse and shift robot towards the wall
            if (current_distance_to_wall > self.robot_centered_distance_to_wall_upper_bound and wall_to_reference == "right") or (current_distance_to_wall < self.robot_centered_distance_to_wall_lower_bound and wall_to_reference == "left"):
                self.get_logger().info("Performing reverse and shift to the right")
                self.current_state = "complex_reverse_turn_right"
                self.reverse_turn_correction_phase = 1 #phase 1 of reverse turn will be reverse and right turn
            if (current_distance_to_wall > self.robot_centered_distance_to_wall_upper_bound and wall_to_reference == "left") or (current_distance_to_wall < self.robot_centered_distance_to_wall_lower_bound and wall_to_reference == "right"):
                self.get_logger().info("Performing reverse and shift to the left")
                self.current_state = "complex_reverse_turn_left"
                self.reverse_turn_correction_phase = 1 #phase 1 of reverse turn will be reverse and right left


    def complex_reverse_turn_callback(self):
        self.is_turning = False
        self.move_timer.destroy() # It's good practice to destroy the timer once it's finished to free up resources.
        self.move_timer = None
        if (self.current_state == "complex_reverse_turn_right") or (self.current_state == "complex_reverse_turn_left"):
            if self.reverse_turn_correction_phase == 2:
                self.get_logger().info("Complex reverse correction turn completed. Now entering 'distance_correction' state.")
                self.current_state = "distance_correction"
                self.reverse_turn_correction_phase = 0
            else:
                self.get_logger().info("Complex reverse correction turn phase 1 complete, moving to phase 2...")
                self.reverse_turn_correction_phase = 2


    def alignment_issue_turn_callback(self):
        #To work out whether we need turn left or right, we need to use certain bits of the lidar data
        #We will take a front scan range of 120 points (so an angle of approx 67 degrees)
        #If the two ends of that scan data are broadly the same AND higher than the centre then the robot is centred
        #If left end is higher, then robot is skewed left, and needs to turn right
        #If right end is higher, then robot is skewed right, and needs to turn left
        front_scan_range = self.lidar_data[260:380]
        right_side_of_front_scan_range = front_scan_range[0:20]
        left_side_of_front_scan_range = front_scan_range[99:120]
        centre_of_front_scan_range = front_scan_range[49:70]
        right_side_of_front_scan_range = [d for d in right_side_of_front_scan_range if not math.isinf(d)]
        left_side_of_front_scan_range = [d for d in left_side_of_front_scan_range if not math.isinf(d)]
        centre_of_front_scan_range = [d for d in centre_of_front_scan_range if not math.isinf(d)]
        average_right_side_of_front_scan_range = self.calculate_average(right_side_of_front_scan_range)
        average_left_side_of_front_scan_range = self.calculate_average(left_side_of_front_scan_range)
        average_centre_of_front_scan_range = self.calculate_average(centre_of_front_scan_range)

        if (average_centre_of_front_scan_range > average_left_side_of_front_scan_range) and (average_centre_of_front_scan_range > average_right_side_of_front_scan_range):
            self.get_logger().info("left and right extremes of the front scan smaller than central region. This indicates robot may be facing a corner and is stuck")
            self.current_state = "stuck"
        elif (abs(average_right_side_of_front_scan_range - average_left_side_of_front_scan_range) <= 0.05 * ((average_right_side_of_front_scan_range + average_left_side_of_front_scan_range) / 2)):
        # Then the left and right values are similar (within 5% of each other) and must also be larger than central scan range... so we will try to do distance correction and re-read the QR code
                self.get_logger().info("robot is centrally aligned and facing the wall, and we will now move to the distance correction phase")
                self.current_state = "distance_correction"                        
        elif average_right_side_of_front_scan_range > average_left_side_of_front_scan_range:
            #Then robot is skewed right, and needs to turn left until left and right scan ranges are similar AND bigger than central range
            self.current_state = "turn_left_correction"
        elif average_left_side_of_front_scan_range > average_right_side_of_front_scan_range:
            #Then robot is skewed left, and needs to turn right until left and right scan ranges are similar AND bigger than central range
            self.current_state = "turn_right_correction"
        else:
            self.get_logger().info("Unknown scan range combination during realignment - robot stuck.")
            self.current_state = "stuck"


    # Calculations for working out if robot approx central to corridor - works best for when robot completing a turn (i.e. when it is at other end of corridor!)
    # Basically here we take front facing range, and then a range offset to the left, and to the right of this range. 
    # The robot is pointing pretty much directly down the corridor when the central range is larger than both the offset ranges.
    # Now, the shorter the corridor, the wider the central ray and the offset rays need to be in order to get an accurate "stop turning" message
    # So we will build in functionality that takes ranges of a size that are inversely proportional to the estimated length of the corridor
    def turning_stop_when_parallel(self): #Only called once we have  QR code reading

        if not self.is_turning:
            if self.current_state == "turn_left":
                self.get_logger().info("Starting a 90-degree left turn...")
            elif self.current_state == "turn_right":
                self.get_logger().info("Starting a 90-degree right turn...")
            else:
                self.get_logger().info("Invalid current state - cannot estimate corridor length")
            #Set is turning flag to true
            self.is_turning = True
        
        #1. Get corridor length and keep it updated as accurately as possible...
        # The easiest way to do this, is simply to look for the largest value in the lidar-array, and to keep checking for it. If we take the value and reduce by 10% this will give us a pretty good indication of the "straight ahead" corridor length - not perfect, but doesn't need to be
        # Note - this model breaks down for corridors shorter than they are wide, so if this is the case clamp at corridor_width + 10%
        filtered_lidar_data = [d for d in self.lidar_data if not math.isinf(d)]
        if max(filtered_lidar_data) < self.corridor_width:
            self.estimated_corridor_length = self.corridor_width*1.1
        else:
            self.estimated_corridor_length = (max(filtered_lidar_data)*0.9)
        
        #Apply multiplier based on corridor length to determine size of rays which help determine how central we are!
        self.corridor_length_in_terms_of_corridor_width = self.estimated_corridor_length / self.corridor_width
        self.inverse_ray_width_multiplier = (1/self.corridor_length_in_terms_of_corridor_width) 
        self.standard_broad_front_ray_width = 120 #can tune this
        self.how_much_bigger_offset_rays_are_than_central_ray = 2 #can tune this
        center_ray = 320
        self.broad_front_range_start = round(center_ray - ((self.standard_broad_front_ray_width*self.inverse_ray_width_multiplier)/2))
        self.broad_front_range_end = round(center_ray + ((self.standard_broad_front_ray_width*self.inverse_ray_width_multiplier)/2))

        #These lines create our ray sections from the main lidar data, and using the proportional operator which is based on the estimated corridor length to adjust ray width
        broad_front_data = self.lidar_data[self.broad_front_range_start:self.broad_front_range_end]
        front_right_offset_data = self.lidar_data[(self.broad_front_range_start - round(((self.standard_broad_front_ray_width*self.inverse_ray_width_multiplier)*self.how_much_bigger_offset_rays_are_than_central_ray))):self.broad_front_range_start] 
        front_left_offset_data = self.lidar_data[self.broad_front_range_end:(self.broad_front_range_end + round(((self.standard_broad_front_ray_width*self.inverse_ray_width_multiplier)*self.how_much_bigger_offset_rays_are_than_central_ray)))][::-1] 

        #Remove 'inf values' and then calculate averages
        broad_front_data = [d for d in broad_front_data if not math.isinf(d)]
        front_right_offset_data = [d for d in front_right_offset_data if not math.isinf(d)]
        front_left_offset_data = [d for d in front_left_offset_data if not math.isinf(d)]
        self.average_broad_front_data = self.calculate_average(broad_front_data)
        self.average_front_right_offset_data = self.calculate_average(front_right_offset_data)
        self.average_front_left_offset_data = self.calculate_average(front_left_offset_data)
    
        self.get_logger().info(f"Estimated corridor length: {self.estimated_corridor_length:.2f} meters")
        self.get_logger().info(f"Average of broad front data: {self.average_broad_front_data:.2f} meters")
        self.get_logger().info(f"Average of front right offset data: {self.average_front_right_offset_data:.2f} meters")
        self.get_logger().info(f"Average of front left offset data: {self.average_front_left_offset_data:.2f} meters")
        
        #3. Once robot parallel, we return to drive mode, and set self.is_turning flag back to False
        multiplier_to_cope_with_long_corridors = 2.4 * (1/self.corridor_length_in_terms_of_corridor_width)
        if (self.average_broad_front_data > self.estimated_corridor_length * multiplier_to_cope_with_long_corridors) and (self.average_broad_front_data > self.average_front_left_offset_data) and (self.average_broad_front_data > self.average_front_right_offset_data):
            self.get_logger().info("Robot parallel to corridor walls, turn complete, returning to 'drive' mode")
            self.current_state = "drive"
            self.is_turning = False
            self.last_qr_code = "" #necessary to put this back to nil when transitioning from sucessful turn back to drive mode, otherwise consecutive left or right turns cause a loop in the callback_camera_subscriber method
        else:
            self.get_logger().info("turning...")


    def callback_cmd_vel_publisher(self):
        msg = Twist()
        if self.current_state == "drive":
            Kp = 1.5  # A proportional constant (Kp) is needed to scale central to corridor errors, you'll need to tune this value
            distance_to_start_slow_down = self.corridor_width*1.33
            if self.average_direct_front_data >= distance_to_start_slow_down:
                msg.linear.x = 0.65
                msg.angular.z = self.error * Kp
            elif self.average_direct_front_data < distance_to_start_slow_down and self.average_direct_front_data >= self.ideal_qr_measuring_distance_upper_bound: #slows robot and keeps it on track
                msg.linear.x = 0.55
                msg.angular.z = self.error * Kp
            else:
                self.get_logger().warn(f"Forward distance is now {self.average_direct_front_data:.2f} meters, STOPPING")
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
            # Keep publishing the turn command and running the turning_stop_when_parallel function...
            msg.linear.x = 0.0
            if self.is_turning and self.estimated_corridor_length > 4.00:
                msg.angular.z = 0.4 # Positive value for left turn. Slowed down for long corridors
            else:
                msg.angular.z = 0.7
            self.turning_stop_when_parallel()
        elif self.current_state == "turn_right":
            # Keep publishing the turn command and running the turning_stop_when_parallel function...
            msg.linear.x = 0.0
            if self.is_turning and self.estimated_corridor_length > 4.00:
                msg.angular.z = -0.4 # Positive value for left turn. Slowed down for long corridors
            else:
                msg.angular.z = -0.7
            self.turning_stop_when_parallel()
        elif self.current_state == "turn_left_correction":
            # Keep publishing the turn command and running the alignment_issue_turn_callback until the robot gets properly aligned
            msg.linear.x = 0.0
            msg.angular.z = 0.5 # Positive value for left turn. Tune this.
            self.alignment_issue_turn_callback()
        elif self.current_state == "turn_right_correction":
            # Keep publishing the turn command and running the alignment_issue_turn_callback until the robot gets properly aligned
            msg.linear.x = 0.0
            msg.angular.z = -0.5 # Negative value for right turn. Tune this.
            self.alignment_issue_turn_callback()
        elif self.current_state == "distance_correction":
            self.is_turning = False
            if self.average_direct_front_data < 0.38:
                msg.linear.x = -0.1
            elif self.average_direct_front_data >= 0.4:
                msg.linear.x = 0.1
            else:
                self.get_logger().warn(f"Forward distance is now {self.average_direct_front_data:.2f} meters, reverting to 'read_qr' state for another attempt")
                self.current_state = "read_qr"
        elif self.current_state == "complex_reverse_turn_right":
            if not self.is_turning:
                self.get_logger().info(f"Phase {self.turn_duration} of 'complex_reverse_turn_right' procedure beginning.")
                self.is_turning = True
                self.move_timer = self.create_timer(self.turn_duration, self.complex_reverse_turn_callback)
            # Keep publishing the turn command while the timer is active
            if self.reverse_turn_correction_phase == 1:
                msg.linear.x = -0.25
                msg.angular.z = 1.0 # Positive value for reverse right turn. Tune this.
            elif self.reverse_turn_correction_phase == 2:
                msg.linear.x = -0.25
                msg.angular.z = -0.85 # Negative value for reverse left turn. Tune this.
            else:
                self.get_logger().warn("Unknown reverse turn phase!")
        elif self.current_state == "complex_reverse_turn_left":
            if not self.is_turning:
                self.get_logger().info(f"Phase {self.turn_duration} of 'complex_reverse_turn_left' procedure beginning.")
                self.is_turning = True
                self.move_timer = self.create_timer(self.turn_duration, self.complex_reverse_turn_callback)
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
            self.is_turning = False
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.get_logger().warn("MAZE COMPLETE")
        elif self.current_state == "stuck":
            # Maze complete, robot stopped
            self.is_turning = False
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.get_logger().warn("ROBOT STUCK - REASSESS CODE AND RESTART PROGRAM")
        else:
            self.get_logger().warn("Unknown state")
        self.cmd_vel_publisher_.publish(msg)

    #---------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = QrCodeMazeTester()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()