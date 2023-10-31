from astropy import units as u
from astropy import constants as c
import numpy as np
import matplotlib.pyplot as plt

from poliastro.bodies import Earth
from poliastro.twobody import Orbit

class Satellite(object):
    def __init__(self, orbit, neighbors, benefits):
        self.orbit = orbit

        self.neighbors = neighbors
        self.benefits = benefits

    def propagate_orbit(self, time):
        """
        Given a time interval (a astropy quantity object),
        propagates the orbit of the satellite.
        """
        self.orbit = self.orbit.propagate(time)
        return self.orbit