
from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt

import debugpy
debugpy.listen(("localhost", 5678))

class LaneFilterHistogramKF():
    """ Generates an estimate of the lane pose.

    TODO: Fill in the details

    Args:
        configuration (:obj:`List`): A list of the parameters for the filter

    """

    def __init__(self, **kwargs):
        param_names = [
            # TODO all the parameters in the default.yaml should be listed here.
            'mean_d_0',
            'mean_phi_0',
            'sigma_d_0',
            'sigma_phi_0',
            'delta_d',
            'delta_phi',
            'd_max',
            'd_min',
            'phi_max',
            'phi_min',
            'cov_v',
            'linewidth_white',
            'linewidth_yellow',
            'lanewidth',
            'min_max',
            'sigma_d_mask',
            'sigma_phi_mask',
            'range_min',
            'range_est',
            'range_max',
        ]

        for p_name in param_names:
            assert p_name in kwargs
            setattr(self, p_name, kwargs[p_name])

        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]

        self.belief = {'mean': self.mean_0, 'covariance': self.cov_0}

        self.encoder_resolution = 0
        self.wheel_radius = 0.0
        self.baseline = 0
        self.initialized = False
        
        print("categ,d_mean,phi_mean,mean_var,phi_var,custom1,custom2")

    def predict(self, dt, left_encoder_delta, right_encoder_delta):
        if not self.initialized:
            return

        # TODO: set them only once
        DELTA_TICKS_TO_DIST = 2 * np.pi * self.wheel_radius / self.encoder_resolution
        DELTA_TICKS_DIFF_TO_OMEGA = np.pi / (self.baseline * self.encoder_resolution)
        
        #delta_position = 0.5 * DELTA_TICKS_TO_DIST * (left_encoder_delta + right_encoder_delta)
        delta_ticks_sum = right_encoder_delta + left_encoder_delta
        delta_ticks_diff = right_encoder_delta - left_encoder_delta

        A = np.eye(2)
        B = np.array([[np.sin(self.belief['mean'][1]), 0], [0, dt]])

        dist = delta_ticks_sum / 2.0 * DELTA_TICKS_TO_DIST
        omega = delta_ticks_diff * DELTA_TICKS_DIFF_TO_OMEGA
        
        x_t_last = self.belief['mean']
        u_t = np.array([dist, omega])

        self.belief['mean'] = A @ x_t_last + B @ u_t
        
        dist_error = 0.5 / self.encoder_resolution * DELTA_TICKS_TO_DIST
        omega_error = 1.0 / self.encoder_resolution * DELTA_TICKS_DIFF_TO_OMEGA

        P_last = self.belief['covariance']
        Q = np.diag([dist_error, omega_error]) + np.diag([0.0001, 0.0025])
        self.belief['covariance'] = A @ P_last @ A.T + Q
        print("predict," 
        + str(round(self.belief['mean'][0], 4)) + "," 
        + str(round(self.belief['mean'][1], 4)) + "," 
        + str(round(Q[0, 0], 4)) + "," 
        + str(round(Q[1, 1], 4)) + "," 
        + str(round(dist, 4)) + "," 
        + str(round(omega, 4)))

    def update(self, segments):
        # prepare the segments for each belief array
        segmentsArray = self.prepareSegments(segments)
        
        # generate all belief arrays
        measurement_likelihood = self.generate_measurement_likelihood(
            segmentsArray)

        if measurement_likelihood is not None:
            # TODO: Parameterize the measurement likelihood as a Gaussian
            # TODO: Set them only once
            D_GRID = np.mgrid[self.d_min:self.d_max:self.delta_d]
            PHI_GRID = np.mgrid[self.phi_min:self.phi_max:self.delta_phi]

            d_weights = measurement_likelihood.sum(axis=1)
            d_mean, d_var = self.weighted_avg_and_var(D_GRID, weights=d_weights)

            phi_weights = measurement_likelihood.sum(axis=0)
            phi_mean, phi_var = self.weighted_avg_and_var(PHI_GRID, weights=phi_weights)

            H = np.eye(2)
            z = np.array([d_mean, phi_mean])
            R = np.diag([d_var, phi_var])
            
            # TODO: Apply the update equations for the Kalman Filter to self.belief
            predicted_mu = self.belief['mean']
            predicted_Sigma = self.belief['covariance']
            
            try:
                residual_mean = z - H @ predicted_mu
                residual_covariance = H @ predicted_Sigma @ H.T + R
                kalman_gain = predicted_Sigma @ H.T @ np.linalg.inv(residual_covariance)
                self.belief['mean'] = predicted_mu + kalman_gain @ residual_mean
                self.belief['covariance'] = predicted_Sigma - kalman_gain @ H @ predicted_Sigma
                print("update," 
                + str(round(self.belief['mean'][0], 4)) + "," 
                + str(round(self.belief['mean'][1], 4)) + "," 
                + str(round(R[0, 0], 4)) + "," 
                + str(round(R[1, 1], 4)) + "," 
                + str(round(z[0], 4)) + "," 
                + str(round(z[1], 4)))
            except np.linalg.LinAlgError:
                print("Singular Matrix encountered.")

    def getEstimate(self):
        return self.belief

    def generate_measurement_likelihood(self, segments):

        if len(segments) == 0:
            return None

        grid = np.mgrid[self.d_min:self.d_max:self.delta_d,
                                    self.phi_min:self.phi_max:self.delta_phi]

        # initialize measurement likelihood to all zeros
        measurement_likelihood = np.zeros(grid[0].shape)

        for segment in segments:
            d_i, phi_i, l_i, _ = self.generateVote(segment)

            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                continue

            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] = measurement_likelihood[i, j] + 1

        if np.linalg.norm(measurement_likelihood) == 0:
            return None

        # lastly normalize so that we have a valid probability density function

        measurement_likelihood = measurement_likelihood / \
            np.sum(measurement_likelihood)
        return measurement_likelihood

    # generate a vote for one segment
    def generateVote(self, segment):
        p1 = np.array([segment.points[0].x, segment.points[0].y])
        p2 = np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if (l1 < 0):
            l1 = -l1
        if (l2 < 0):
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])
        if segment.color == segment.WHITE:  # right lane is white
            if(p1[0] > p2[0]):  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane

                d_i = - d_i

                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2

        elif segment.color == segment.YELLOW:  # left lane is yellow
            if (p2[0] > p1[0]):  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i

        # weight = distance
        weight = 1
        return d_i, phi_i, l_i, weight

    def get_inlier_segments(self, segments, d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l, w = self.generateVote(segment)
            if abs(d_s - d_max) < 3 * self.delta_d and abs(phi_s - phi_max) < 3 * self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    # get the distance from the center of the Duckiebot to the center point of a segment
    def getSegmentDistance(self, segment):
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c**2 + y_c**2)

    # prepare the segments for the creation of the belief arrays
    def prepareSegments(self, segments):
        segmentsArray = []
        self.filtered_segments = []
        for segment in segments:

            # we don't care about RED ones for now
            if segment.color != segment.WHITE and segment.color != segment.YELLOW:
                continue
            # filter out any segments that are behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                continue

            self.filtered_segments.append(segment)
            # only consider points in a certain range from the Duckiebot for the position estimation
            point_range = self.getSegmentDistance(segment)
            if point_range < self.range_est:
                segmentsArray.append(segment)

        return segmentsArray

    def get_next_pose(self, icc_pos, d, cur_theta, theta_displacement):
        """
        Compute the new next position in global frame
        Input:
            - icc_pos: numpy array of ICC position [x,y] in global frame
            - d: distance from robot to the center of curvature
            - cur_theta: current yaw angle in radian (float)
            - theta_displacement: the amount of angular displacement if we apply w for 1 time step
        Return:
            - next_position:
            - next_orientation:
        """
        
        # First, let's define the ICC frame as the frame centered at the location of ICC
        # and oriented such that its x-axis points towards the robot
        
        # Compute location of the point where the robot should be at (i.e., q)
        # in the frame of ICC.
        x_new_icc_frame = d * np.cos(theta_displacement)
        y_new_icc_frame = d * np.sin(theta_displacement)
        
        # Build transformation matrix from origin to ICC
        T_oc_angle = -(np.deg2rad(90) - cur_theta) # 
        icc_x, icc_y = icc_pos[0], icc_pos[1]
        T_oc = np.array([
            [np.cos(T_oc_angle), -np.sin(T_oc_angle), icc_x],
            [np.sin(T_oc_angle), np.cos(T_oc_angle), icc_y],
            [0, 0, 1]
        ]) # Transformation matrix from origin to the ICC
        
        # Build transformation matrix from ICC to the point where the robot should be at (i.e., q)
        T_cq = np.array([
            [1, 0, x_new_icc_frame],
            [0, 1, y_new_icc_frame],
            [0, 0, 1]
        ]) # Transformation matrix from ICC to the point where the robot should be at (i.e., q)
        
        # Convert the local point q to the global frame
        T_oq = np.dot(T_oc, T_cq) # Transformation matrix from origin to q
        
        next_position = np.array([T_oq[0,2], T_oq[1,2]])
        next_orientation = cur_theta + theta_displacement
        return next_position, next_orientation
    
    # Adapted from: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    def weighted_avg_and_var(self, values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values-average)**2, weights=weights)
        return (average, variance)