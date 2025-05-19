# Pratik Chaudhari (pratikac@seas.upenn.edu)
# Minku Kim (minkukim@seas.upenn.edu)

import os, sys, pickle, math
from scipy import io
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from load_data import load_kitti_lidar_data, load_kitti_poses, load_kitti_calib
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    def __init__(s, resolution=0.5):
        s.resolution = resolution
        s.xmin, s.xmax = -700, 700
        s.zmin, s.zmax = -500, 900
        #s.xmin, s.xmax = -400, 1100
        #s.zmin, s.zmax = -300, 1200

        s.szx = int(np.ceil((s.xmax - s.xmin) / s.resolution + 1))
        s.szz = int(np.ceil((s.zmax - s.zmin) / s.resolution + 1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szz), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds,
        # and similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh / (1 - s.occupied_prob_thresh))

    def grid_cell_from_xz(s, x, z):
        """
        x and z can be 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/z go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        ##### TODO: XXXXXXXXXX

        x_idx = np.clip(((x - s.xmin) / s.resolution).astype(int), 0, s.szx - 1)
        z_idx = np.clip(((z - s.zmin) / s.resolution).astype(int), 0, s.szz - 1)

        return np.vstack([x_idx,z_idx])

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.5, Q=1e-3*np.eye(3), resampling_threshold=0.3):
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

        # dynamics noise for the state (x, z, yaw)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

        s.prev_cells = s.map.cells.copy()



    def read_data(s, src_dir, idx):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar_dir = src_dir + f'odometry/{s.idx}/velodyne/'
        s.poses = load_kitti_poses(src_dir + f'poses/{s.idx}.txt')
        s.lidar_files = sorted(os.listdir(src_dir + f'odometry/{s.idx}/velodyne/'))
        s.calib = load_kitti_calib(src_dir + f'calib/{s.idx}/calib.txt')

    def init_particles(s, n=100, p=None, w=None):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3, s.n))
        s.w = deepcopy(w) if w is not None else np.ones(n) / n

    @staticmethod
    def stratified_resampling(p, w):
        """
        Resampling step of the particle filter.
        """
        ##### TODO: XXXXXXXXXXX
        new_p = p.copy()
        new_w = np.ones(p.shape[1])/p.shape[1]
        n = p.shape[1]
        c = w[0]
        i = 0
        r = np.random.uniform(0,1/n)
        for m in range(n):
            u = r+ (m-1)/n
            while u > c:
                i +=1
                c +=w[i]
            new_p[:,m] = p[:,i]

        return new_p ,new_w 

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def lidar2world(s, p, points):
        """
        Transforms LiDAR points to world coordinates.

        The particle state p is now interpreted as [x, z, theta], where:
        - p[0]: x translation
        - p[1]: z translation
        - p[2]: rotation in the x-z plane

        The input 'points' is an (N, 3) array of LiDAR points in xyz.
        """
        #### TODO: XXXXXXXXX
        calib = np.vstack([s.calib,[0,0,0,1]])
        x,z,yaw = p[0],p[1],p[2]
        T_world = get_se2(yaw,[x,z])
        points = points.T
        #plt.scatter(points[0,:], points[1,:], s=1, c='r', alpha=0.5)

        # 1. Convert LiDAR points to homogeneous coordinates
        homogenous_points = make_homogeneous_coords_3d(points[:3,:])
        
        # 2. Transform Velodyne Frame -> Camera Frame
        camera_coords = calib @ homogenous_points 

        camera_xz = np.vstack([camera_coords[0,:],camera_coords[2,:],camera_coords[3,:]])

        # 3. from camera frame to world frame
        world_coords = (T_world @ camera_xz).T
        #plt.scatter(world_coords[:,0], world_coords[:,1], s=1, c='b', alpha=0.5)

        return world_coords

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d
        function to get the difference of the two poses and we will simply
        set this to be the control.
        Extracts control in the state space [x, z, rotation] from consecutive poses.
        [x, z, theta]
        theta is the rotation around the Y-axis
              | cos  0  -sin |
        R_y = |  0   1    0  |
              |+sin  0   cos |
        R31 = +sin
        R11 =  cos
        yaw = atan2(R_31, R_11)
        """
        if t == 0:
            return np.zeros(3)

        #### TODO: XXXXXXXXXXX
        pose_t = pose_to_xztheta(s.poses[t])
        pose_tprev = pose_to_xztheta(s.poses[t-1])
        
        return smart_minus_2d(pose_t,pose_tprev)

    def dynamics_step(s, t):
        """
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter
        """
        #### TODO: XXXXXXXXXXX
        u = s.get_control(t)
        for i in range(s.n):
            noise = np.random.multivariate_normal(mean=np.zeros(3), cov=s.Q)
            u_noisy = u + noise
            s.p[:,i] = smart_plus_2d(s.p[:,i],u_noisy)
        
    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        new_w = w * np.exp(obs_logp - slam_t.log_sum_exp(obs_logp))
        new_w /= np.sum(new_w)
        return new_w

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data
        you can also store a thresholded version of the map here for plotting later
        """
        obs_logp =[]
        lidar_points = np.array(load_kitti_lidar_data(os.path.join(s.lidar_dir, f"{t:06d}.bin")))
        lidar_points = clean_point_cloud(lidar_points)
        for i in range(s.n):
            world_points = s.lidar2world(s.p[:, i], lidar_points)
            occ_cells = s.map.grid_cell_from_xz(world_points[:, 0], world_points[:, 1])
            logp = np.sum(s.prev_cells[occ_cells[0,:], occ_cells[1,:]])
            obs_logp.append(logp)

        s.prev_cells = np.zeros(s.map.cells.shape)
        s.w = s.update_weights(s.w, np.array(obs_logp))

        best_particle_idx = np.argmax(s.w)
        best_points = s.lidar2world(s.p[:, best_particle_idx], lidar_points)

        occ_cells = s.map.grid_cell_from_xz(best_points[:, 0], best_points[:, 1])

        s.map.log_odds[occ_cells[0], occ_cells[1]] += 2*s.lidar_log_odds_occ 
        s.map.log_odds[:,:] += s.lidar_log_odds_free

        s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)

        s.map.cells[occ_cells[0], occ_cells[1]] += 1
        s.prev_cells[occ_cells[0], occ_cells[1]] += 1

        s.map.cells = np.clip(s.map.cells, 0, 1)



    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
