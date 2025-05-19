# Pratik Chaudhari (pratikac@seas.upenn.edu)
# Minku Kim (minkukim@seas.upenn.edu)

import click, tqdm, random

from slam import *
import matplotlib.pyplot as plt

def run_dynamics_step(src_dir, log_dir, idx, t0=0, draw_fig=False):
    """
    This function is for you to test your dynamics update step. It will create
    two figures after you run it. The first one is the robot location trajectory
    using odometry information obtained form the lidar. The second is the trajectory
    using the PF with a very small dynamics noise. The two figures should look similar.
    """
    slam = slam_t(Q=1e-8*np.eye(3))
    slam.read_data(src_dir, idx)

    # Trajectory using odometry (xz and yaw) in the lidar data
    d = slam.poses
    pose = np.column_stack([d[:,0,3], d[:,1,3], d[:,2,3]]) # X Y Z
    plt.figure(1)
    plt.clf()
    plt.title('Trajectory using onboard odometry')
    plt.plot(pose[:,0], pose[:,2])
    logging.info('> Saving odometry plot in '+os.path.join(log_dir, 'odometry_%s.jpg'%(idx)))
    plt.savefig(os.path.join(log_dir, 'odometry_%s.jpg'%(idx)))

    # dynamics propagation using particle filter
    # n: number of particles, w: weights, p: particles (3 dimensions, n particles)
    # S covariance of the xyth location
    # particles are initialized at the first xyth given by the lidar
    # for checking in this function
    n = 3
    w = np.ones(n)/float(n)
    p = np.zeros((3,n), dtype=np.float64)
    slam.init_particles(n,p,w)
    slam.p[:,0] = deepcopy(pose[0])

    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)
    plt.figure(2)
    plt.clf()
    ax = plt.subplot(111)
    for t in tqdm.tqdm(range(t0+1,T)):
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))

        if draw_fig:
            ax.clear()
            ax.plot(slam.p[0], slam.p[0], '*r')
            plt.title('Particles %03d'%t)
            plt.draw()
            plt.pause(0.01)

    plt.plot(ps[0], ps[1], '*c')
    plt.title('Trajectory using PF')
    logging.info('> Saving plot in '+os.path.join(log_dir, 'dynamics_only_%s.jpg'%(idx)))
    plt.savefig(os.path.join(log_dir, 'dynamics_only_%s.jpg'%(idx)))

def run_observation_step(src_dir, log_dir, idx, is_online=False):
    """
    This function is for you to debug your observation update step
    It will create three particles np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]])
    * Note that the particle array has the shape 3 x num_particles so
    the first particle is at [x=0.2, y=0.4, z=0.1]
    This function will build the first map and update the 3 particles for one time step.
    After running this function, you should get that the weight of the second particle is the largest since it is the closest to the origin [0, 0, 0]
    """
    slam = slam_t(resolution=0.5)
    slam.read_data(src_dir, idx)

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for
    # other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step
    t0 = 0
    d = slam.poses
    pose = np.column_stack([d[t0,0,3], d[t0,1,3], np.arctan2(-d[t0,2,0], d[t0,0,0])])
    logging.debug('> Initializing 1 particle at: {}'.format(pose))
    slam.init_particles(n=1,p=pose.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # reinitialize particles, this is the real test
    logging.info('\n')
    n = 3
    w = np.ones(n)/float(n)
    p = np.array([[2, 0.2, 3],[2, 0.4, 5],[2.7, 0.1, 4]])
    slam.init_particles(n, p, w)

    slam.observation_step(t=1)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

def run_slam(src_dir, log_dir, idx):
    """
    This function runs SLAM. It initializes the SLAM system with 50 particles
    and performs dynamics and observation updates iteratively.
    """
    # Initialize SLAM instance with 50 particles and larger dynamics noise
    slam = slam_t(resolution=0.5, Q=np.diag([1e-6, 1e-6, 1e-6]))
    slam.read_data(src_dir, idx)

    T = len(slam.lidar_files)

    d = slam.poses
    pose = np.column_stack([d[0,0,3], d[0,1,3], np.arctan2(-d[0,2,0], d[0,0,0])])
    slam.init_particles(n=1,p=pose.reshape((3,1)),w=np.array([1]))
    slam.observation_step(0)

    slam.init_particles(n=50)

    trajectory = []
    particle_trajectories = []
    # Plot odometry trajectory (ground truth)
    odometry_x = np.array([pose[0, 3] for pose in slam.poses])
    odometry_z = np.array([pose[2, 3] for pose in slam.poses])

    for t in range(1, T):
        # plt.plot(odometry_x, odometry_z, label="Odometry Trajectory", color="blue")
        # plt.plot(slam.p[:, 0],slam.p[:, 1], label="Estimated Trajectory", color="red")
        # plt.show()
        slam.dynamics_step(t)

        # plt.plot(odometry_x, odometry_z, label="Odometry Trajectory", color="blue")
        # plt.plot(slam.p[:, 0],slam.p[:, 1], label="Estimated Trajectory", color="red")
        # plt.show()
        slam.observation_step(t)

        
        slam.resample_particles()


        best_particle_idx = np.argmax(slam.w)
        trajectory.append(slam.p[:, best_particle_idx])
        """# Plot the map and overlay particles
        plt.figure(figsize=(10, 10))
        plt.imshow(slam.map.cells, cmap='gray')
        plt.scatter(p_in_map[0, :], p_in_map[1, :], color='red', s=15, label="Particles")
        plt.title(f"SLAM Map and Particles at Time {t}")
        plt.xlabel("X [m]")
        plt.ylabel("Z [m]")
        plt.legend()
        plt.grid(True)
        plt.show()"""


        particle_trajectories.append(slam.p.copy())
        logging.info(f"Processed frame {t}/{T}")

    trajectory = np.array(trajectory)
    particle_cells = slam.map.grid_cell_from_xz(trajectory[:,0], trajectory[:,1])
    particle_odometry = slam.map.grid_cell_from_xz(odometry_x, odometry_z)
    plt.imshow(slam.map.cells.T, cmap='gray',origin='lower')
    #plt.ylim(400,2700)
    #plt.xlim(400,2200)
    plt.show()

    plt.imshow(slam.map.cells.T, cmap='gray',origin='lower')
    plt.scatter(particle_cells[0, :], particle_cells[1, :], color='red', s=1, label="Particles")
    plt.scatter(particle_odometry[0], particle_odometry[1], color='blue', s=1, label="Odometry")
    plt.legend()
    #plt.ylim(400,2700)
    #plt.xlim(400,2200)
    plt.show()

    plt.plot(odometry_x, odometry_z, label="Odometry Trajectory", color="blue")
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="Estimated Trajectory", color="red")
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.legend()
    plt.grid(True)
    plt.show()

    



@click.command()
@click.option('--src_dir', default='./KITTI/', help='data directory', type=str)
@click.option('--log_dir', default='logs', help='directory to save logs', type=str)
@click.option('--idx', default='02', help='dataset number', type=str)
@click.option('--mode', default='observation',
              help='choices: dynamics OR observation OR slam', type=str)
def main(src_dir, log_dir, idx, mode):
    # Run python main.py --help to see how to provide command line arguments

    if not mode in ['slam', 'dynamics', 'observation']:
        raise ValueError('Unknown argument --mode %s'%mode)
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_dir, idx)
        sys.exit(0)
    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx)
        sys.exit(0)
    else:
        p = run_slam(src_dir, log_dir, idx)
        return p

if __name__=='__main__':
    main()
