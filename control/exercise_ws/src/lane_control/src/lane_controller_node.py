#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, Segment, SegmentList

from lane_controller.controller import PurePursuitLaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.pp_controller = PurePursuitLaneController(self.params)

        # Initialize variables
        self.car_control_msg = Twist2DStamped()
        self.segment_list = SegmentList().segments

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)

        self.sub_segment_list = rospy.Subscriber("/agent/ground_projection_node/lineseglist_out",
                                                 SegmentList,
                                                 self.cbSegmentList,
                                                 queue_size=1)

        print("d,phi,target_x,target_y,seg_min_dist,seg_max_dist,cluster_x,cluster_y,cluster_c,cluster_pos_mean_x,cluster_pos_mean_y,cluster_n_mean_x,cluster_n_mean_y,lane_mid_x,lane_mid_y,v,L,sin_alpha,w,seg_list_size,cluster_size")
        self.log("Initialized!")

    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        csv_string = ""
        
        self.pose_msg = input_pose_msg
        csv_string += str(round(self.pose_msg.d, 3)) + ','
        csv_string += str(round(self.pose_msg.phi, 3)) + ','
        
        segment_list = self.segment_list
        
        # Initialize default values
            # TODO: Set a default non-zero value?
            v = 0
            w = 0
        
        if len(segment_list) > 0:
            # Initialize target point and list of distances to it
            # TODO: make K, min L_0 configurable
            K = 0.6
            L_0 = max(0.15, K * self.car_control_msg.v)
            target = np.dot(self.rotation2D(-self.pose_msg.phi), np.array([L_0, 0]))
            
            csv_string += str(round(target[0], 3)) + ','
            csv_string += str(round(target[1], 3)) + ','
            #print("target:" + str(target))
            
            # TODO: check if the target is out of the camera scope
            segments_dist = np.zeros(len(segment_list))
            
            # List distances from segments to target
            for i, seg in enumerate(segment_list):
                # Set segment that are not white or yellow very far to exclude them
                if seg.color != seg.WHITE:# and seg.color != seg.YELLOW:
                    segments_dist[i] = np.inf
                else:
                    v = self.mat2x2(seg.points)
                    v -= target
                    segments_dist[i] = np.linalg.norm(v.mean(axis=0))
            
            csv_string += str(round(segments_dist.min(), 3)) + ','
            csv_string += str(round(segments_dist.max(initial=0.0, where=~np.isnan(segments_dist)), 3)) + ','
            
            if len(segments_dist) > 0:
            # Initialize the cluster with the segment closest to the target 
                cluster_center = segment_list[segments_dist.argmin()].points
            cluster_center = self.mat2x2(cluster_center).mean(axis=0)
            
            csv_string += str(round(cluster_center[0], 3)) + ','
            csv_string += str(round(cluster_center[1], 3)) + ','
                colors = { segment_list[0].WHITE : 'WHITE', 
                        segment_list[0].YELLOW : 'YELLOW' }
                csv_string += colors[segment_list[segments_dist.argmin()].color] + ','
            
            cluster_segs = []

            # Find segments in the cluster
                for i, seg in enumerate(segment_list):
                # Only keep white and yellow segments
                    if seg.color == seg.WHITE:# or seg.color == seg.YELLOW:
                    v = self.mat2x2(seg.points)
                    v -= cluster_center
                    # TODO: make radius configurable
                        # TODO: increase it if cluster_center color is WHITE?
                        if np.linalg.norm(v.mean(axis=0)) < 0.02:
                        cluster_segs.append(seg)
            
            # Initialize list of position and normal of segments in the cluster
            cluster_segs_pos = np.zeros((len(cluster_segs), 2))
            cluster_segs_n = np.zeros((len(cluster_segs), 2))

            # Iterate segments in the cluster to extract their position and normal
            #print("cluster_segs_pos[0]=" + str(self.mat2x2(cluster_segs[0].points).mean(axis=0)))
            for i, seg in enumerate(cluster_segs):
                seg_points = self.mat2x2(seg.points)
                cluster_segs_pos[i] = seg_points.mean(axis=0)
                
                t = seg_points[1] - seg_points[0]
                t = t / np.linalg.norm(t)

                cluster_segs_n[i] = np.array([-t[1], t[0]])
                
                # Correct the normal given the color of the segments (and the phi?)
                if seg.color == seg.YELLOW:
                    unit_vector = np.array([0, -1])
                else:
                    unit_vector = np.array([0, 1])

                if np.dot(unit_vector, cluster_segs_n[i]) < 0:
                    cluster_segs_n[i] *= -1

            # Compute the mean of both the position and the normal of the cluster
            cluster_pos_mean = cluster_segs_pos.mean(axis=0)
            cluster_n_mean = cluster_segs_n.mean(axis=0)

            # Find the middle of the lane with the mean normal and position
            #print("lane_mid = cluster_pos_mean:" + str(cluster_pos_mean) 
            #    + " + 0.105 * cluster_n_mean:" + str(cluster_n_mean))
            lane_mid = cluster_pos_mean + 0.105 * cluster_n_mean
            csv_string += str(round(cluster_pos_mean[0], 3)) + ','
            csv_string += str(round(cluster_pos_mean[1], 3)) + ','
            csv_string += str(round(cluster_n_mean[0], 3)) + ','
            csv_string += str(round(cluster_n_mean[1], 3)) + ','
            #print("lane_mid:" + str(lane_mid))
            csv_string += str(round(lane_mid[0], 3)) + ','
            csv_string += str(round(lane_mid[1], 3)) + ','

            # TODO: If target point gets too near of ourself, gradually increase look ahead distance?
            
            # Use the point in the PPC algorithm
            L = np.linalg.norm(lane_mid)
            #print("sin_alpha = lane_mid[1]:" + str(lane_mid[1]) + " / L:" + str(L))
            sin_alpha = lane_mid[1] / L
                w = (2 * self.car_control_msg.v * sin_alpha) / L
                
                # Adapt linear speed to angular speed
                v = min(1.0, 0.5 / abs(w))

            if v == np.nan:
                v = 0
            if w == np.nan:
                w = 0
            
            csv_string += str(round(v, 3)) + ','
            csv_string += str(round(L, 3)) + ','
            csv_string += str(round(sin_alpha, 3)) + ','
            csv_string += str(round(w, 3)) + ','
            else:
                csv_string += str(np.nan) + ','
                csv_string += str(np.nan) + ','
                csv_string += str(np.nan) + ','
                csv_string += str(np.nan) + ','
                csv_string += str(np.nan) + ','
                csv_string += str(np.nan) + ','
                csv_string += str(np.nan) + ','
                csv_string += str(np.nan) + ','
                csv_string += str(np.nan) + ','
                csv_string += str(round(v, 3)) + ','
                csv_string += str(np.nan) + ','
                csv_string += str(np.nan) + ','
                csv_string += str(round(w, 3)) + ','
        else:
            csv_string += str(np.nan) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(round(v, 3)) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(np.nan) + ','
            csv_string += str(round(w, 3)) + ','
            
            
        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        # TODO This needs to get changed
        car_control_msg.v = v
        car_control_msg.omega = w

        #print('v=' + str(v) + ", w=" + str(w))
        #print('seg_list length: ' + str(len(segment_list)) + ', cluster length: ' + str(len(cluster_segs)))
        csv_string += str(len(segment_list)) + ','
        csv_string += str(len(cluster_segs))
        print(csv_string)

        # Save the message to use at next iteration
        self.car_control_msg = car_control_msg

        self.publishCmd(car_control_msg)

    def cbSegmentList(self, input_seg_list_msg):
        """Callback receiving line segment list messages.

        Args:
            input_seg_list_msg (:obj:`SegmentList`): Message containing a list of line segments projected on the ground plane.
        """
        self.segment_list = input_seg_list_msg.segments


    def publishCmd(self, car_cmd_msg):
        """Publishes a car command message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)


    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)

    def mat2x2(self, matrix):
        return np.array([[matrix[0].x, matrix[0].y],
                         [matrix[1].x, matrix[1].y]])

    def rotation2D(self, angle):
        """Returns the 2D rotation matrix from a radian angle.
        Reference: https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/

        Args:
            angle (float): The angle in radian of the rotation in the 2D plane.
        """
        
        c, s = np.cos(angle), np.sin(angle)
        return np.array(((c, -s), (s, c)))


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
