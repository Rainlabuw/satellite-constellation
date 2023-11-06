from astropy import units as u
from astropy import constants as c
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from collections import defaultdict

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import StaticOrbitPlotter
from poliastro.spheroid_location import SpheroidLocation
from poliastro.core.events import line_of_sight
import time

from constellation_sim.Satellite import Satellite
from constellation_sim.Task import Task

class ConstellationSim(object):
    def __init__(self, dt=5*u.min) -> None:
        self.sats = []
        self.tasks = []

        self.n = 0
        self.m = 0

        self.dt = dt

        self.orbits_over_time = None
        self.graph_over_time = None

        self.assign_over_time = None

    def add_sat(self, sat):
        sat.id = len(self.sats)
        self.sats.append(sat)
        self.n = len(self.sats)

    def add_task(self, task):
        task.id = len(self.tasks)
        self.tasks.append(task)
        self.m = len(self.tasks)

    def add_sats(self, sats):
        for i, sat in enumerate(sats):
            sat.id = len(self.sats) + i
        self.sats.extend(sats)
        self.n = len(self.sats)

    def add_tasks(self, tasks):
        for i, task in enumerate(tasks):
            task.id = len(self.tasks) + i
        self.tasks.extend(tasks)
        self.m = len(self.tasks)

    def update(self):
        """
        Updates the constellation state by propagating things forward in time
        """
        for sat in const.sats:
            sat.propagate_orbit(self.dt)

    def update_plot(self, frame):
        """
        Updates the constellation state and updates the plot accordingly
        """
        print(f"Writing frame {frame} of {len(self.orbits_over_time[0])}")
        self.plotter._ax.clear()
        self.plotter._ax.set_xlabel("y (km)")
        self.plotter._ax.set_ylabel("z (km)")

        for sat in self.sats:
            sat_orbit = self.orbits_over_time[sat.id][frame]
            self.plotter.plot(sat_orbit, label=f"Sat {sat.id}, Plane {sat.plane_id}",color=self.plane_colors[sat.plane_id])

        for task in self.tasks:
            task_xyz = task.loc.cartesian_cords.to_value(u.km)

            #Only plot tasks on the right side of the earth
            if task.loc.cartesian_cords.to_value(u.km)[0] > 0:
                self.plotter._ax.scatter(task_xyz[1], task_xyz[2], label=f"Task {task.id}",color='k', alpha=1)

            # if task.loc.is_visible(*self.orbits_over_time[0][frame].r):
            #     self.plotter._ax.scatter(task_xyz[1], task_xyz[2], label=f"Task {task.id}",color='g', alpha=alpha)
            # else:
            #     self.plotter._ax.scatter(task_xyz[1], task_xyz[2], label=f"Task {task.id}",color='r', alpha=alpha)

        if self.assign_over_time != None:
            assign = self.assign_over_time[frame]

            for sat in self.sats:
                sat_orbit = self.orbits_over_time[sat.id][frame]

                curr_ben = self.benefits_over_time[sat.id, :, frame]
                #find tasks for which curr_ben is greater than zero:
                valid_task_ids = np.where(curr_ben > 0)[0]
                
                for valid_task_id in valid_task_ids:
                    task_pos = self.tasks[valid_task_id].loc.cartesian_cords.to_value(u.km)
                    sat_pos = sat_orbit.r.to_value(u.km)
                    if sat_pos[0] > 0 and task_pos[0] > 0:
                        ys = [task_pos[1], sat_pos[1]]
                        zs = [task_pos[2], sat_pos[2]]

                        if assign[sat.id, valid_task_id] == 1:
                            self.plotter._ax.plot(ys, zs, 'g--')
                        else:
                            self.plotter._ax.plot(ys, zs, 'k--', alpha=0.5)

        plt.show(block=False)

    def run_animation(self, frames=10):
        self.plane_colors = {}
        for sat in self.sats:
            if sat.plane_id not in self.plane_colors.keys():
                self.plane_colors[sat.plane_id] = np.random.rand(3,)

        fig, ax = plt.subplots()
        fig.set_size_inches(12,6)
        self.plotter = StaticOrbitPlotter(ax)
        self.plotter._frame = [0, 1, 0]*u.one, [0, 0, 1]*u.one, [1, 0, 0]*u.one
        ani  = FuncAnimation(fig, self.update_plot, frames=frames, interval=1000, blit=False)

        ani.save('constellation.gif', writer='imagemagick', fps=1, dpi=100)

    def propagate_orbits(self,T):
        """
        Propagate the orbits of all satellites forward in time by T timesteps,
        storing satellite orbits over time a dictionary of the form:
        {sat_id: [orbit_0, orbit_1, ..., orbit_T]}.

        Also compute the benefits over time.
        """
        self.orbits_over_time = defaultdict(list)
        self.benefits_over_time = np.zeros((self.n, self.m, T))
        self.graphs_over_time = []
        for k in range(T):
            print(f"Propagating orbits and computing benefits + neighbors, T={k}/{T}...",end='\r')
            self.graphs_over_time.append(self.determine_connectivity_graph())
            for sat in self.sats:

                sat.propagate_orbit(self.dt)
                self.orbits_over_time[sat.id].append(sat.orbit)
                for task in self.tasks:
                    #Compute the distance 
                    self.benefits_over_time[sat.id, task.id, k] = calc_distance_based_benefits(sat, task)

    def determine_connectivity_graph(self):
        #Build adjacency matrix
        adj = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i+1, self.n):
                sat1_r = self.sats[i].orbit.r.to_value(u.km)
                sat2_r = self.sats[j].orbit.r.to_value(u.km)
                R = self.sats[i].orbit._state.attractor.R.to_value(u.km)

                if line_of_sight(sat1_r, sat2_r, R) >=0 and np.linalg.norm(sat1_r-sat2_r) < 2500:
                    adj[i,j] = 1
                    adj[j,i] = 1

        return nx.from_numpy_array(adj)

def calc_distance_based_benefits(sat, task):
    """
    Given a satellite and a task, computes the benefit of the satellite.

    Benefit here is zero if the task is not visible from the satellite,
    and is a gaussian centered at the minimum distance away from the task,
    and dropping to 5% of the max value at the furthest distance away from the task.
    """
    if task.loc.is_visible(*sat.orbit.r):
        body_rad = np.linalg.norm(sat.orbit._state.attractor.R.to_value(u.km))
        max_distance = np.sqrt(np.linalg.norm(sat.orbit.r.to_value(u.km))**2 - body_rad**2)

        gaussian_height = task.benefit
        height_at_max_dist = 0.05*gaussian_height
        gaussian_sigma = np.sqrt(-max_distance**2/(2*np.log(height_at_max_dist/gaussian_height)))

        sat_height = np.linalg.norm(sat.orbit.r.to_value(u.km)) - body_rad
        task_dist = np.linalg.norm(task.loc.cartesian_cords.to_value(u.km) - sat.orbit.r.to_value(u.km)) - sat_height
        task_benefit = gaussian_height*np.exp(-task_dist**2/(2*gaussian_sigma**2))
    else:
        task_benefit = 0

    return task_benefit

def get_benefit_matrix_from_constellation(n,m,T):
    """
    Generate benefit matrix of size n x m x T
    from a constellation of satellites.

    NOTE: n must be a multiple of 10.
    """
    if n%10 != 0:
        if m == n:
            print(f"WARNING: n={n} is not a multiple of 10. Setting n and m to {n + 10 - n%10}.")
            m = n + 10 - n%10
        else: print(f"WARNING: n={n} is not a multiple of 10. Setting n to {n + 10 - n%10}.")
        n = n + 10 - n%10
        
    const = ConstellationSim(dt=1*u.min)
    earth = Earth

    #~~~~~~~~~Generate a constellation of satellites at 400 km.~~~~~~~~~~~~~
    #10 evenly spaced planes of satellites, each with n/10 satellites per plane
    a = earth.R.to(u.km) + 550*u.km
    ecc = 0.01*u.one
    inc = 58*u.deg
    argp = 0*u.deg

    num_planes = 10
    num_sats_per_plane = n//num_planes
    for plane_num in range(num_planes):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num)
            const.add_sat(sat)

    #~~~~~~~~~Generate m random tasks on the surface of earth~~~~~~~~~~~~~
    num_tasks = m
    for _ in range(num_tasks):
        lon = np.random.uniform(-180, 180)
        lat = np.random.uniform(-50, 50)
        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, earth)
        
        task_benefit = np.random.uniform(1, 2)
        task = Task(task_loc, task_benefit)
        const.add_task(task)

    const.propagate_orbits(T)
    return const.benefits_over_time

if __name__ == "__main__":
    const = ConstellationSim(dt=1*u.min)
    T = int(95 // const.dt.to_value(u.min)) #simulate enough timesteps for ~1 orbit
    T = 10
    earth = Earth

    #~~~~~~~~~Generate a constellation of satellites at 400 km.~~~~~~~~~~~~~
    #5 evenly spaced planes of satellites, each with 10 satellites per plane
    a = earth.R.to(u.km) + 400*u.km
    ecc = 0.01*u.one
    inc = 58*u.deg
    argp = 0*u.deg

    num_planes = 36
    num_sats_per_plane = 15
    for plane_num in range(num_planes):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num)
            const.add_sat(sat)

    #~~~~~~~~~Generate n random tasks on the surface of earth~~~~~~~~~~~~~
    num_tasks = 50

    for i in range(num_tasks):
        lon = np.random.uniform(-180, 180)
        lat = np.random.uniform(-60, 60)
        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, earth)
        
        task_benefit = np.random.uniform(1,2)
        task = Task(task_loc, task_benefit)
        const.add_task(task)

    const.propagate_orbits(T)

    print(sum([nx.is_connected(g) for g in const.graphs_over_time]))

    for graph in const.graphs_over_time:
        nx.draw(graph)
        plt.show()

    const.assign_over_time = [np.eye(const.n, const.m) for i in range(T)]

    # const.run_animation(frames=T)

    # for sat in const.sats:
    #     sat.propagate_orbit(10*u.min)
    #     plotter.plot(sat.orbit)

    #     plotter.show()