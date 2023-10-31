from astropy import units as u
from astropy import constants as c
import numpy as np
import matplotlib.pyplot as plt

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import StaticOrbitPlotter
import time

from Satellite import Satellite

class ConstellationSim(object):
    def __init__(self) -> None:
        self.sats = []
        self.tasks = []

    def add_sat(self, sat):
        self.sats.append(sat)

    def add_task(self, task):
        self.tasks.append(task)

    def add_sats(self, sats):
        self.sats.extend(sats)

    def add_tasks(self, tasks):
        self.tasks.extend(tasks)

    def gen_random_sats(self, n_sats, orbital_elements):
        """
        Generates n_sats satellites with random orbital elements.
        """
        self.tasks.append

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
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [])
            const.add_sat(sat)

            plotter.plot(sat.orbit, label=f"sat {plane_num},{sat_num}", color=plane_color)

    #Animate each satellite in the constellation over a period of 3 huors, every 10 minutes
    for sat in const.sats:
        sat.propagate_orbit(10*u.min)
        plotter.plot(sat.orbit)

        plotter.show()
        