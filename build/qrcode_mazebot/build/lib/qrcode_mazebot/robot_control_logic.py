#!/usr/bin/env python3

# dependencies 
import math
import numpy as np


"""
A class which handles the logic behind ensuring the robot drives accurately and gracefully down the maze corridors and completes turns correctly. 
The logic here also handles to corrective moves where needed.

"""

class RobotControl:
    def __init__(self, logger): # Accept the logger as an argument
        """Initializes the RobotControl class with a logger."""
        self.logger = logger # Store the logger object

    def _calculate_angle_error(self, right_indices, left_indices, right_indices_short, left_indices_short, lidar_data, 
                               lidar_angle_min, lidar_angle_increment, previous_angular_error, current_position):
        
        if current_position in ["corner_left", "corner_right", "end_of_maze"]: #Then we'll use a MUCH shorter section of the wall to avoid corner confusion!
            right_wall_points = self._get_cartesian_points(right_indices_short, lidar_data, lidar_angle_min, lidar_angle_increment) #Identify wall points & convert to Cartesian, this is done in a robust way that preserves original indices.
            left_wall_points = self._get_cartesian_points(left_indices_short, lidar_data, lidar_angle_min, lidar_angle_increment) #Identify wall points & convert to Cartesian, this is done in a robust way that preserves original indices.
        else: #We will use full length wall section for current_positions: 'corridor' 'corridor_after_left_turn' 'corridor_after_right_turn' 'corridor_opening_left' 'corridor_opening_right' and 'undetermined'
            right_wall_points = self._get_cartesian_points(right_indices, lidar_data, lidar_angle_min, lidar_angle_increment) #Identify wall points & convert to Cartesian, this is done in a robust way that preserves original indices.
            left_wall_points = self._get_cartesian_points(left_indices, lidar_data, lidar_angle_min, lidar_angle_increment) #Identify wall points & convert to Cartesian, this is done in a robust way that preserves original indices.

        if right_wall_points:
            right_wall_slope, right_wall_intercept = self._calculate_line_of_best_fit(right_wall_points)
        else:
            right_wall_slope, right_wall_intercept = None, None
        if left_wall_points:
            left_wall_slope, left_wall_intercept = self._calculate_line_of_best_fit(left_wall_points)
        else:
            left_wall_slope, left_wall_intercept = None, None
        
        right_wall_angle = self._get_angle_from_slope(right_wall_slope) # slope 'm' is tan(theta) so theta = atan(m). Gives angle relative to robot's forward direction (the x-axis).
        left_wall_angle = self._get_angle_from_slope(left_wall_slope) # slope 'm' is tan(theta) so theta = atan(m). Gives angle relative to robot's forward direction (the x-axis).
        
        if current_position == "corner_right":
            self.logger.info("In a right hand corner, so using a shortened left wall for angle error!")
            angular_error = left_wall_angle
        elif current_position == "corner_left":
            self.logger.info("In a left hand corner, so using a shortened right wall for angle error!")
            angular_error = right_wall_angle
        elif current_position == "corridor":
            self.logger.info("Travelling down corridor so using average of both full length walls for angle error!")
            angular_error = (right_wall_angle + left_wall_angle) / 2 #Average of the two angle errors!
        elif current_position == "end_of_maze":
            self.logger.info("Near end of maze so using average of both 'shortened' walls for angle error!")
            angular_error = (right_wall_angle + left_wall_angle) / 2 #Average of the two angle errors!
        elif current_position == "corridor_opening_left":
            self.logger.info("Approaching a left hand corner, so using full length right wall for angle error!")
            angular_error = right_wall_angle
        elif current_position == "corridor_opening_right":
            self.logger.info("Approaching a right hand corner, so using full length left wall for angle error!")
            angular_error = left_wall_angle  
        elif current_position == "corridor_after_left_turn":
            self.logger.info("Looking down corridor after a left turn, so using full length right wall for angle error!")
            angular_error = right_wall_angle 
        elif current_position == "corridor_after_right_turn":
            self.logger.info("Looking down corridor after a right turn, so using full length left wall for angle error!")
            angular_error = left_wall_angle       
        elif current_position == "undetermined":
            self.logger.info("Current position undetermined, so we will not updated previous angular error...")
            angular_error = previous_angular_error
        else:
            self.logger.warn("Unknown position state (check code)")
            angular_error = previous_angular_error
        return angular_error, right_wall_slope, right_wall_angle, left_wall_slope, left_wall_angle 


    def _calculate_lateral_error(self, average_direct_right_data, average_direct_left_data, angular_error, corridor_width, previous_lateral_error, current_position):
        # A positive error means the robot is too close to the right wall, and negative error means robot is too close to left wall
        # We also need to make sure we are using the 'corrected' left and right wall distance which account for the angle of the robot!!
        # We also need to account for corners, with an open corridor to one side!
        corrected_dist_right = average_direct_right_data * math.cos(angular_error)
        corrected_dist_left = average_direct_left_data * math.cos(angular_error)
        desired_dist_from_wall = corridor_width / 2.0

        #If in corridor, we will use standard calculation (which uses right wall)
        if current_position == "corridor":
            lateral_error = desired_dist_from_wall - corrected_dist_right
        #If at end of maze, we will just use the previous error so we don't get any confusion as we approach the corners of the wall at the end
        elif current_position == "end_of_maze":
            lateral_error = previous_lateral_error
        #So we need to use the left wall for our lateral measurement, but we MUST make sure we swap the signs to ensure that +'ve still means too far right and -'ve still means too far left!
        elif current_position == "corridor_opening_right" or current_position == "corridor_after_right_turn":
            lateral_error = (desired_dist_from_wall - corrected_dist_left)*-1
        #Corridor opening to our left so we need to use the right wall!
        elif current_position == "corridor_opening_left" or current_position == "corridor_after_left_turn":
            lateral_error = desired_dist_from_wall - corrected_dist_right
        #If in a corner, we will stop measuring any new lateral errors as we're too close to the front wall, it confuses the readings 
        elif current_position == "corner_left" or current_position == "corner_right":
            lateral_error = previous_lateral_error
        else:
            lateral_error = previous_lateral_error
        
        return lateral_error, corrected_dist_right, corrected_dist_left 
    

    def _establish_position(self, average_front_data, average_right_data, average_left_data, right_of_front_data, left_of_front_data, lateral_error, corridor_width, previous_position, is_turning):
        #Firstly, we need to understand that the robot is using left and right distances to try and estimate where it is
        #If the robot is not perfectly central in the corridor, it may make the wrong decisions
        #So we need to apply a fix to the left and right data to effectively make it as if the robot is perfectly centralised
        #We can use the last known lateral error to do this
        half_corridor = corridor_width / 2
        if abs(lateral_error) > 0.01:
            percentage_fix_to_distances = lateral_error / half_corridor
            average_left_data_corrected = average_left_data - (half_corridor * percentage_fix_to_distances)
            average_right_data_corrected = average_right_data + (half_corridor * percentage_fix_to_distances)
        else:
            average_left_data_corrected = average_left_data
            average_right_data_corrected = average_right_data

        #If robot is in process of turning, then readings will be all over the place, but we know robot is in a corner, so we will just keep it that way until is_turning becomes FALSE!
        if is_turning:
            current_position = previous_position
        else:
            #If we're in a corner or corridor end...
            if average_front_data <= corridor_width:
                #...and the left wall and right distances are pretty close to eachother, then we're likely at the end of the maze
                if ((average_right_data_corrected*0.85) < average_left_data_corrected < (average_right_data_corrected*1.15)):
                    current_position = "end_of_maze"
                # ... otherwise if left wall distances are greater than right wall then we're in a left corner
                elif (average_left_data_corrected > average_right_data_corrected):
                    current_position = "corner_left"
                # ...otherwise if right wall distances are greater than left wall distances then we're in a right corner
                elif (average_right_data_corrected > average_left_data_corrected):
                    current_position = "corner_right"
                else:
                    current_position = "undetermined"
            #If we're NEARING the end of a corridor... then we need to try and spot a corridor opening up so we need to use the left_of_front and right_of_front data segments...
            elif (average_front_data < (corridor_width * 2.5)):
                #...and the left of front, right of front distances are pretty close to eachother, then we'll consider ourselves still in the corridor...
                if ((right_of_front_data*0.85) < left_of_front_data < (right_of_front_data*1.15)):
                    current_position = "corridor"
                # ... otherwise if left of front distances are greater than right of front then we're seeing the corridor open out to the left...
                elif (left_of_front_data > right_of_front_data):
                    current_position = "corridor_opening_left"
                # ...otherwise if right of front distances are greater than left of front we're seeing the corridor open out to the right...
                elif (right_of_front_data > left_of_front_data):
                    current_position = "corridor_opening_right"
                else:
                    current_position = "undetermined"             
            #If LOTS of space out front... 
            elif (average_front_data >= (corridor_width * 2.5)):
                #...and the left wall and right distances are pretty close to eachother, then we're likely in a normal corridor still
                if ((average_right_data_corrected*0.8) < average_left_data_corrected < (average_right_data_corrected*1.2)):
                    current_position = "corridor"
                # ... otherwise if left wall distances are greater than right wall then we're seeing the corridor open out to the left...
                elif (average_left_data_corrected > average_right_data_corrected):
                    current_position = "corridor_after_left_turn"
                # ...otherwise if right wall distances are greater than left wall we're seeing the corridor open out to the right...
                elif (average_right_data_corrected > average_left_data_corrected):
                    current_position = "corridor_after_right_turn"
                else:
                    current_position = "undetermined"    
            #Default position
            else:
                current_position = "undetermined"
        return current_position


    def _perform_turn(self, current_state, current_position, average_direct_front_data, angular_error, estimated_corridor_length, previous_turn_state):
        if current_state == "turn_left":
            turn_value_sign_swap = 1
        elif current_state == "turn_right":
            turn_value_sign_swap = -1 #negative value needed for right turn
        
        if not previous_turn_state:
            self.logger.info(f"Performing a turn. Estimated corridor length is {estimated_corridor_length} metres")
            is_turning = True
        else:
            is_turning = previous_turn_state

        if (average_direct_front_data < (estimated_corridor_length*0.95)):
            turn_value = 0.7 * turn_value_sign_swap
            new_state = current_state
            new_estimated_corridor_length = estimated_corridor_length
        else: #now robot is mostly turned through 90 degrees - so we will perform fine-tune which once done, will then revert us to drive mode
            self.logger.info(f"Turn completed. Status changed to [fine_tune] and estimated position is: [{current_position}]")
            turn_value = 0.0
            new_estimated_corridor_length = 0.0 #Critical this is done when turn completed!!
            new_state = "fine_tune"
            #We MUST leave 'is_turning' as TRUE as this will ensure the fine-tune method behaves correctly!
        return turn_value, is_turning, new_state, new_estimated_corridor_length



# --- THESE ARE ALL USED BY THE CALCULATE_ANGLE_ERROR FUNCTION --- #
    #Filters out 'inf' values and calculates average, returns both formatted array, and average value
    def filter_out_inf_and_calculate_average(self, data_list): 
        formatted_data = [d for d in data_list if not math.isinf(d)]
        if len(formatted_data) > 0:
            formatted_data_average = sum(formatted_data) / len(formatted_data)
            return formatted_data_average
        else:
            return 0.0
        
    #Converts LiDAR points for given indices to Cartesian coordinates.
    def _get_cartesian_points(self, indices, lidar_data, lidar_angle_min, lidar_angle_increment):
        points = [] 
        if not lidar_data or lidar_angle_increment == 0.0: # Ensure we have the necessary lidar properties before proceeding
            return points
        for i in indices:
            distance = lidar_data[i]
            if math.isinf(distance) or math.isnan(distance) or distance < 0.01: # Filter out invalid readings
                continue
            angle = lidar_angle_min + (i * lidar_angle_increment) # Calculate angle using the point's ORIGINAL index `i` - ensures angle is correct relative to robot's frame.
            # Convert polar (distance, angle) to Cartesian (x, y) - In the standard ROS robot frame (REP 103): +x is forward, +y is left, +z is up
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            points.append((x, y))
        return points

    #Calculates the slope and intercept of the line of best fit using numpy.
    def _calculate_line_of_best_fit(self, points):
        if len(points) < 2:
            return None, None # Not enough points to define a line
        x_coords = np.array([p[0] for p in points]) # Unpack points into separate x and y numpy arrays
        y_coords = np.array([p[1] for p in points]) # Unpack points into separate x and y numpy arrays
        # Use numpy's polyfit to find the slope (m) and intercept (b) of a 1st-degree polynomial - This is a robust way to perform linear regression.
        try:
            m, b = np.polyfit(x_coords, y_coords, 1)
            return m, b
        except np.linalg.LinAlgError:
            return None, None # This can happen in rare cases, e.g., a perfectly vertical line
            
    #Calculates the angle of the wall in degrees from its slope.
    def _get_angle_from_slope(self, slope):
        if slope is None:
            return 0.0
        # The angle is the arctangent of the slope
        angle_rad = math.atan(slope)
        #angle_deg = math.degrees(angle_rad) - uncomment if needed!
        return angle_rad
