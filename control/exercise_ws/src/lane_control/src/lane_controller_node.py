#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, Segment, SegmentList
from geometry_msgs.msg import Point

from lane_controller.controller import PurePursuitLaneController

#import debugpy
#debugpy.listen(("localhost", 5678))

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
        #self.sub_segment_list = rospy.Subscriber("/agent/lane_filter_node/seglist_filtered",
                                                 SegmentList,
                                                 self.cbSegmentList,
                                                 queue_size=1)

        self.count = 0
        print("d,phi,target_x,target_y,seg_min_dist,seg_max_dist,cluster_x,cluster_y,cluster_c,cluster_pos_mean_x,cluster_pos_mean_y,cluster_n_mean_x,cluster_n_mean_y,lane_mid_x,lane_mid_y,v,L,sin_alpha,w,seg_list_size,cluster_size")
        self.log("Initialized!")

    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        MODES = {
            'naive' : 0,
            'path' : 1,
            'cluster' : 2,
            'lwlr' : 3
        }
        self.count = 1#(self.count + 1) % 5
        csv_string = ""
        
        self.pose_msg = input_pose_msg
        csv_string += str(round(self.pose_msg.d, 3)) + ','
        csv_string += str(round(self.pose_msg.phi, 3)) + ','
        
        segment_list = self.segment_list
        
        # Initialize default values
        
            # TODO: Set a default non-zero value?
        #v = 0
        #w = 0
        
        mode = MODES['lwlr']
            # Initialize target point and list of distances to it
            # TODO: make K, min L_0 configurable
            K = 0.6
            L_0 = max(0.15, K * self.car_control_msg.v)
        offset = np.dot(self.rotation2D(-self.pose_msg.phi), np.array([0, -self.pose_msg.d]))
        target = offset + np.dot(self.rotation2D(-self.pose_msg.phi), np.array([L_0, 0]))
            
        v, w = self.getControlValues(target)
        
        if len(segment_list) > 0 and np.abs(self.pose_msg.d) < 0.15 and mode > MODES['naive']:
            
            csv_string += str(round(target[0], 3)) + ','
            csv_string += str(round(target[1], 3)) + ','
            #print("target:" + str(target))
            
            if mode == MODES["lwlr"]:
                x = L_0
                x, y = self.getLaneMid(x, segment_list, target[1])
                v, w = self.getControlValues(np.array([x, y]))
            elif mode == MODES['path']:
                lane_mid_candidats = self.getDirectCandidats(segment_list)
                #lane_mid_candidats = self.getClusterCandidats(segment_list)
                
                if self.count == 0:
                    print('x,y')
                    for mid in lane_mid_candidats:
                        print(str(mid[0]) + "," + str(mid[1]))

                path = [offset]

                # TODO: make initial dist configurable
                next_target = offset + np.dot(self.rotation2D(-self.pose_msg.phi), np.array([0.1, 0]))
                
                while np.linalg.norm(path[-1]) < L_0 and len(lane_mid_candidats) > 0:
                    chosen, lane_mid_candidats = self.nextPoint(next_target, path[-1], lane_mid_candidats)
                    path.append(chosen)
                    # TODO: find make dist to next point configurable
                    diff_points = path[-1] - path[-2]
                    if np.linalg.norm(diff_points) == 0:
                        path.pop()
                    else:
                        next_target = path[-1] + diff_points / np.linalg.norm(diff_points) * 0.05

                if self.count == 0:
                    print('path_x,path_y')
                    for point in path:
                        print(str(point[0]) + "," + str(point[1]))
                
                if (len(path) > 1):
                    v, w = self.getControlValues(path[-1])

            # TODO: check if the target is out of the camera scope
            segments_dist = np.zeros(len(segment_list))
            
            # List distances from segments to target
            for i, seg in enumerate(segment_list):
                # Set segment that are not white or yellow very far to exclude them
                if seg.color != seg.WHITE:# and seg.color != seg.YELLOW:
                    segments_dist[i] = np.inf
                else:
                    points = self.mat2x2(seg.points)
                    points -= target
                    segments_dist[i] = np.linalg.norm(points.mean(axis=0))
            
            csv_string += str(round(segments_dist.min(), 3)) + ','
            csv_string += str(round(segments_dist.max(initial=0.0, where=~np.isnan(segments_dist)), 3)) + ','
            
            if len(segments_dist) > 0 and mode == MODES['cluster']:
            # Initialize the cluster with the segment closest to the target 
                cluster_center = segment_list[segments_dist.argmin()].points
            cluster_center = self.mat2x2(cluster_center).mean(axis=0)
                cluster_color = segment_list[segments_dist.argmin()].color
            
            csv_string += str(round(cluster_center[0], 3)) + ','
            csv_string += str(round(cluster_center[1], 3)) + ','
                colors = { segment_list[0].WHITE : 'WHITE', 
                        segment_list[0].YELLOW : 'YELLOW' }
                csv_string += colors[cluster_color] + ','
            
            cluster_segs = []

            # Find segments in the cluster
                for i, seg in enumerate(segment_list):
                    # Only keep segments of the cluster's color
                    if seg.color == cluster_color:
                        points = self.mat2x2(seg.points)
                        points -= cluster_center
                    # TODO: make radius configurable
                        # TODO: increase it if cluster_center color is WHITE?
                        if np.linalg.norm(points.mean(axis=0)) < 0.02:
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
            csv_string += str(round(lane_mid[0], 3)) + ','
            csv_string += str(round(lane_mid[1], 3)) + ','

            # TODO: If target point gets too near of ourself, gradually increase look ahead distance?
            
                v, w = self.getControlValues(lane_mid)
            
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
            
        # Create control message
        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        # Set the control values
        car_control_msg.v = v
        car_control_msg.omega = w

        #print('v=' + str(v) + ", w=" + str(w))
        #print('seg_list length: ' + str(len(segment_list)) + ', cluster length: ' + str(len(cluster_segs)))
        #csv_string += str(len(segment_list)) + ','
        #csv_string += str(len(cluster_segs))
        if self.count == 0:
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

    def segPos2D(self, points):
        seg_points = self.mat2x2(points)
        return seg_points.mean(axis=0)
    
    def segNormal2D(self, points, color=np.nan):
        seg_points = self.mat2x2(points)
        
        t = seg_points[1] - seg_points[0]
        t = t / np.linalg.norm(t)
        t = np.reshape(t, (2,1))

        normal = np.array([-t[1], t[0]])
        normal = np.reshape(normal, (2))
        
        if ~np.isnan(color):
            segment = Segment()
            
            if color == segment.YELLOW:
                unit_vector = np.array([0, -1])
            elif color == segment.WHITE:
                unit_vector = np.array([0, 1])
            else:
                unit_vector = normal / np.linalg.norm(normal)

            if np.dot(unit_vector, normal) < 0:
                normal *= -1
        
        return normal

    def nextPoint(self, point, previous_point, point_list):
        # Compute distances to points in point_list
        distance_list = self.distance(point, point_list)
        prev_distance_list = self.distance(previous_point, point_list)
        
        # TODO: remove that check since it is now useless
        if np.array_equal(point, np.zeros(2)):
            index = distance_list.argmin()
            return point_list[index], np.delete(point_list, index, axis=0)
        else:
            # Filter out the point closer to previous_point
            distance_filter = (distance_list < prev_distance_list) & (prev_distance_list < 0.06)
            new_point_list = point_list[distance_filter]
            filtered_distance_list = distance_list[distance_filter]
            
            if len(filtered_distance_list) == 0:
                return previous_point, new_point_list
            else:
                index = filtered_distance_list.argmin()

                # Creating return values
                point = new_point_list[index]
                new_list = np.delete(new_point_list, index, axis=0)
                point_list_complement = point_list[~distance_filter]
                new_list = np.vstack((new_list, point_list_complement))
                return point, new_list

    def distance(self, point, point_list):
        # Compute distance between point and points in point_list
        rel_pos_list = point_list - point
        return np.linalg.norm(rel_pos_list, axis=1)

    def getControlValues(self, target):
        # TODO: make min, max speed configurable
        v_min, v_max = 0.5, 1.0
        v_range = v_max - v_min
        
        # Use the point in the PPC algorithm
        L = np.linalg.norm(target)
        #print("sin_alpha = target[1]:" + str(target[1]) + " / L:" + str(L))
        
        # Adapt linear speed to anticipated angular speed
        cos_alpha = target[0] / L
        v = 0.5#v_min + np.power(cos_alpha, 2) * v_range
        
        # Compute omega as per PPC algorithm
        sin_alpha = target[1] / L
        w = (2 * v * sin_alpha) / L

        if np.isnan(v):
            v = 0.0
        if np.isnan(w):
            w = 0.0

        return v, w

    def getClusters(self, segment_list):
        segment = Segment()
        lists_by_color = self.splitByColor(segment_list)

        cluster_pos = []
        cluster_normal = []
        for c in { segment.WHITE, segment.YELLOW }:
            point_list = self.getPointList(lists_by_color[c])
            normal_list = self.getNormalList(lists_by_color[c], c)
            
            for i in range(len(point_list)):
                rel_pos = point_list - point_list[i]
                mask = np.linalg.norm(rel_pos, axis=1) < 0.02
                cluster_pos.append(point_list[mask].mean(axis=0))
                cluster_normal.append(normal_list[mask].mean(axis=0))
        
        return np.array(cluster_pos), np.array(cluster_normal)

    def splitByColor(self, segment_list):
        dummy_segment = Segment()
        lists_by_color = {
            dummy_segment.WHITE : [],
            dummy_segment.YELLOW : [],
            dummy_segment.RED : []
        }
        
        for seg in segment_list:
            lists_by_color[seg.color].append(seg)

        return lists_by_color
    
    def getPointList(self, segment_list):
        return np.array([self.segPos2D(seg.points) for seg in segment_list])

    def getNormalList(self, segment_list, color=np.nan):
        return np.array([self.segNormal2D(seg.points, color) for seg in segment_list])

    def getDirectCandidats(self, segment_list):
        lane_mid_candidats = np.zeros((len(segment_list), 2))
        to_remove_indexes = []

        # List distances from segments to target
        for i, seg in enumerate(segment_list):
            if seg.color != seg.WHITE and seg.color != seg.YELLOW:
                to_remove_indexes.append(i)
            else:
                position = self.segPos2D(seg.points)
                normal = self.segNormal2D(seg.points, seg.color)
                lane_mid_candidats[i][:] = position + 0.105 * normal
        
        return np.delete(lane_mid_candidats, to_remove_indexes, axis=0)

    def getClusterCandidats(self, segment_list):
        positions, normals = self.getClusters(segment_list)
        return positions + 0.105 * normals

    # Source of LWLR code: https://www.geeksforgeeks.org/implementation-of-locally-weighted-linear-regression/
    # kernel smoothing function
    
    # function to calculate W weight diagnal Matric used in calculation of predictions 
    def get_WeightMatrix_for_LOWES(self, query_point, training_examples, Bandwidth): 
        # M is the No of training examples 
        M = training_examples.shape[0] 
        # Initialising W with identity matrix 
        W = np.mat(np.eye(M)) 
        # calculating weights for query points 
        for i in range(M): 
            xi = training_examples[i] 
            denominator = (-2 * Bandwidth * Bandwidth) 
            W[i, i] = np.exp(np.dot((xi-query_point), (xi-query_point).T)/denominator) 
        return W 

    # function to make predictions 
    def predict(self, training_examples, Y, query_x, Bandwidth): 
        M = training_examples.shape[0] 
        if training_examples.ndim == 1:
            training_examples = np.reshape(training_examples, (M, 1))
        if Y.ndim == 1:
            Y = np.reshape(Y, (M, 1))
        all_ones = np.ones((M, 1)) 
        X_ = np.hstack((training_examples, all_ones)) 
        qx = np.mat([query_x, 1]) 
        W = self.get_WeightMatrix_for_LOWES(qx, X_, Bandwidth) 
        # calculating parameter theta 
        theta = np.linalg.pinv(X_.T*(W * X_))*(X_.T*(W * Y)) 
        # calculating predictions 
        pred = np.dot(qx, theta) 
        return theta, pred[0, 0]

    def getLaneMid(self, x, segment_list, default):
        # Initialize a dummy segment to access Segment's constants
        segment = Segment()
        
        # Split the segments according to there color
        lists_by_color = self.splitByColor(segment_list)

        y = np.array([np.nan, np.nan])
        
        # Estimate the y position corresponding to the requested x position
        # given the segments near the DB
        for c in { segment.WHITE, segment.YELLOW }:
            if len(lists_by_color[c]) > 10:
                y[c] = self.getLaneMidHelper(x, lists_by_color[c])
        
        if np.isnan(y).all():
            # Not enough information was available to approximate the y value
            return x, default
        elif np.isnan(y).any():
            # Only the line of one side was detected 
            color = int(np.argwhere(~np.isnan(y)))
            points = [Point(), Point()]
            # Finding y values around the x value of interest to estimate the 
            # normal at that point
            points[0].x = np.maximum(0.0, x - 0.02)
            points[1].x = x + 0.02
            points[0].y = self.getLaneMidHelper(points[0].x, lists_by_color[color])
            points[1].y = self.getLaneMidHelper(points[1].x, lists_by_color[color])
            # Get normal from point around the point of interest
            normal = self.segNormal2D(points, color)
            # Find the estimated middle of the lane given y estimated at x and 
            # the normal estimated at the same point.
            lane_mid = np.array([x, y[color]]) + 0.105 * normal
            return lane_mid[0], lane_mid[1]
        else:
            # Both lanes on both sides where detected
            return x, y.mean()

    def getLaneMidHelper(self, x, segment_list):
        # Convert the segment list to a numpy array of positions
        point_list = self.getPointList(segment_list)
        # Filter out the points that are not near enough
        mask = np.linalg.norm(point_list, axis=1) < 0.60
        point_list = point_list[mask]
        # Predict the value of y at x given the segment position on the list
        sorted_point_list = point_list[point_list[:,0].argsort()]
        _, value = self.predict(sorted_point_list[:, 0], sorted_point_list[:, 1], x, 0.2)
        
        """For debugging
        string = ""
        for i in np.linspace(0.1, 0.55, 10):
            _, value = self.predict(sorted_point_list[:, 0], sorted_point_list[:, 1], i, 0.2)
            string += str("{:0.4f}".format(value[0,0])) + ","
        print(string[:-1])"""
        return value

if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
