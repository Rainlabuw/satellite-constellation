from astropy import units as u
from astropy import constants as c
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import StaticOrbitPlotter
import time

from Satellite import Satellite

class ConstellationSim(object):
    def __init__(self, dt=5*u.min) -> None:
        self.sats = []
        self.tasks = []

        self.dt = dt

    def add_sat(self, sat):
        sat.id = len(self.sats)
        self.sats.append(sat)

    def add_task(self, task):
        task.id = len(self.tasks)
        self.tasks.append(task)

    def add_sats(self, sats):
        for i, sat in enumerate(sats):
            sat.id = len(self.sats) + i
        self.sats.extend(sats)

    def add_tasks(self, tasks):
        for i, task in enumerate(tasks):
            task.id = len(self.tasks) + i
        self.tasks.extend(tasks)

    def gen_random_sats(self, n_sats, orbital_elements):
        """
        Generates n_sats satellites with random orbital elements.
        """
        pass

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
        self.plotter._ax.clear()

        self.update()

        for sat in self.sats:
            self.plotter.plot(sat.orbit, label=f"Sat {sat.id}, Plane {sat.plane_id}",color=self.plane_colors[sat.plane_id])

        plt.show(block=False)

    def run_animation(self, frames=10):
        self.plane_colors = {}
        for sat in self.sats:
            if sat.plane_id not in self.plane_colors.keys():
                self.plane_colors[sat.plane_id] = np.random.rand(3,)

        fig, ax = plt.subplots()
        fig.set_size_inches(12,6)
        self.plotter = StaticOrbitPlotter(ax)
        ani  = FuncAnimation(fig, self.update_plot, frames=frames, interval=1000, blit=False)

        ani.save('constellation.gif', writer='imagemagick', fps=1)

if __name__ == "__main__":
    const = ConstellationSim()

    earth = Earth

    #Generate a constellation of satellites at 400 km.
    #5 evenly spaced planes of satellites, each with 10 satellites per plane
    a = earth.R.to(u.km) + 400*u.km
    ecc = 0.01*u.one
    inc = 70*u.deg
    argp = 0*u.deg

    plotter = StaticOrbitPlotter()

    num_planes = 5
    num_sats_per_plane = 2
    for plane_num, plane_color in zip(range(num_planes), ['r','b','k','g','y']):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num)
            const.add_sat(sat)

    const.run_animation()
    # for sat in const.sats:
    #     sat.propagate_orbit(10*u.min)
    #     plotter.plot(sat.orbit)

    #     plotter.show()
        