from astropy import units as u
from astropy import constants as c
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon
import time

from collections import defaultdict

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import StaticOrbitPlotter
from poliastro.spheroid_location import SpheroidLocation
from poliastro.core.events import line_of_sight
import h3

from constellation_sim.Satellite import Satellite
from constellation_sim.Task import Task

from methods import *

class ConstellationSim(object):
    def __init__(self, dt=1*u.min, isl_dist=None) -> None:
        self.sats = []
        self.tasks = []

        self.n = 0
        self.m = 0

        self.dt = dt

        if isl_dist is None: self.isl_dist = np.inf
        else: self.isl_dist = isl_dist

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

    def propagate_orbits(self,T,benefit_func):
        """
        Propagate the orbits of all satellites forward in time by T timesteps,
        storing satellite orbits over time a dictionary of the form:
        {sat_id: [orbit_0, orbit_1, ..., orbit_T]}.

        Also compute the benefits and connectivity graphs over time and returns them,
        given a benefit function which computes a benefit from a sat and a task.
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
                    self.benefits_over_time[sat.id, task.id, k] = benefit_func(sat, task, k)

        return self.benefits_over_time, self.graphs_over_time

    def determine_connectivity_graph(self):
        #Build adjacency matrix
        adj = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i+1, self.n):
                sat1_r = self.sats[i].orbit.r.to_value(u.km)
                sat2_r = self.sats[j].orbit.r.to_value(u.km)
                R = self.sats[i].orbit._state.attractor.R.to_value(u.km)

                if line_of_sight(sat1_r, sat2_r, R) >=0 and np.linalg.norm(sat1_r-sat2_r) < self.isl_dist:
                    adj[i,j] = 1
                    adj[j,i] = 1

        return nx.from_numpy_array(adj)

def get_constellation_bens_and_graphs_random_tasks(num_planes, num_sats_per_plane, m,T,benefit_func=calc_fov_benefits, altitude=550, fov=60, dt=1*u.min, isl_dist=None):
    """
    Generate benefit matrix of size (num_planes*sats_per_plane) x m x T
    from a constellation of satellites, as well as
    a list of the T connectivity graphs for each timestep.
    """
    const = ConstellationSim(dt=dt, isl_dist=isl_dist)
    earth = Earth

    #~~~~~~~~~Generate a constellation of satellites at 400 km.~~~~~~~~~~~~~
    #10 evenly spaced planes of satellites, each with n/10 satellites per plane
    a = earth.R.to(u.km) + altitude*u.km
    ecc = 0*u.one
    inc = 58*u.deg
    argp = 0*u.deg

    for plane_num in range(num_planes):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num, fov=fov)
            const.add_sat(sat)

    #~~~~~~~~~Generate m random tasks on the surface of earth~~~~~~~~~~~~~
    num_tasks = m
    for _ in range(num_tasks):
        lon = np.random.uniform(-180, 180)
        lat = np.random.uniform(-55, 55)
        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, earth)
        
        task_benefit = np.random.uniform(1, 2, size=T)
        task = Task(task_loc, task_benefit)
        const.add_task(task)

    benefits, graphs = const.propagate_orbits(T, benefit_func)
    return benefits, graphs

def generate_smooth_coverage(lat_range, lon_range):
    # Initialize an empty set to store unique H3 indexes
    hexagons = set()

    # Step through the defined ranges and discretize the globe
    lat_steps, lon_steps = 0.5, 0.5
    lat = lat_range[0]
    while lat <= lat_range[1]:
        lon = lon_range[0]
        while lon <= lon_range[1]:
            # Find the hexagon containing this lat/lon
            hexagon = h3.geo_to_h3(lat, lon, 1)
            hexagons.add(hexagon)
            lon += lon_steps
        lat += lat_steps

    #Add tasks at centroid of all hexagons
    task_lats = []
    task_lons = []
    for hexagon in hexagons:
        boundary = h3.h3_to_geo_boundary(hexagon, geo_json=True)
        polygon = Polygon(boundary)

        task_lats.append(polygon.centroid.y)
        task_lons.append(polygon.centroid.x)
        
    return task_lats, task_lons

def get_constellation_bens_and_graphs_coverage(num_planes, num_sats_per_plane,T,inc,benefit_func=calc_fov_benefits, altitude=550, fov=60, dt=1*u.min, isl_dist=None):
    """
    Generate benefit matrix of with (num_planes*sats_per_plane)
    satellites covering the entire surface of the earth, with tasks
    evenly covering the globe at the lowest H3 reslution possible (~10 deg lat/lon).

    Input an inclination for the satellites and the tasks.
    """
    const = ConstellationSim(dt=dt, isl_dist=isl_dist)
    earth = Earth

    #~~~~~~~~~Generate a constellation of satellites at 400 km.~~~~~~~~~~~~~
    #10 evenly spaced planes of satellites, each with n/10 satellites per plane
    a = earth.R.to(u.km) + altitude*u.km
    ecc = 0*u.one
    inc = inc*u.deg
    argp = 0*u.deg

    for plane_num in range(num_planes):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num, fov=fov)
            const.add_sat(sat)

    #~~~~~~~~~Generate m random tasks on the surface of earth~~~~~~~~~~~~~
    lats, lons = generate_smooth_coverage((-inc.to_value(u.deg), -inc.to_value(u.deg)), (-180, 180))
    for lat, lon in zip(lats, lons):
        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, earth)
        
        task_benefit = np.random.uniform(1, 2, size=T)
        task = Task(task_loc, task_benefit)
        const.add_task(task)

    benefits, graphs = const.propagate_orbits(T, benefit_func)
    return benefits, graphs

if __name__ == "__main__":
    lat_range = (20, 50)
    lon_range = (73, 135)
