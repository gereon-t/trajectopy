from trajectopy.core.trajectory import Trajectory

open_loop_trajectory = Trajectory.from_file("./test/data/open_loop_trajectory.traj")
open_loop_trajectory.positions.to_epsg(0)


generated_trajectory = Trajectory.from_file("./test/data/generated_trajectory.traj")
noisy_trajectory = Trajectory.from_file("./test/data/noisy_trajectory.traj")
