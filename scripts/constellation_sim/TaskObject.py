import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
import h3

class TaskObject(object):
    def __init__(self, start_lat, start_lon, lat_range, lon_range, dir, appear_time, dt, speed=6437*u.km/u.hr) -> None:
        self.lat = start_lat
        self.lon = start_lon
        
        self.lat_range = lat_range
        self.lon_range = lon_range

        self.dir = dir
        self.speed = speed
        self.dt = dt

        #Calculate degree change per timestep (TODO: better movement model)
        #If going east or west
        if dir == (1,0) or dir == (-1,0):
            rad_at_lat = Earth.R.to(u.km)*np.cos(start_lat*np.pi/180)
            ang_vel = self.speed/rad_at_lat #rad/hr
        else: 
            ang_vel = self.speed/Earth.R.to(u.km)
        self.deg_change_per_ts = (ang_vel*self.dt*180/np.pi).to(u.one)

        self.appear_time = appear_time

        self.task_idx = None

        self.in_area = False

    def propagate(self, hex_to_task_mapping, t):
        """
        Propagate object movement over time
        """
        if t < self.appear_time:
            pass
        else:
            self.lat += self.deg_change_per_ts * self.dir[1]
            self.lon += self.deg_change_per_ts * self.dir[0]

            self.in_area = self.lat > self.lat_range[0] and self.lat < self.lat_range[1] and self.lon > self.lon_range[0] and self.lon < self.lon_range[1]
            if self.in_area:
                self.in_area = True

            # Find the hexagon containing this lat/lon, increment target count
            hexagon = h3.geo_to_h3(self.lat, self.lon, 1)

            self.task_idx = hex_to_task_mapping[hexagon]