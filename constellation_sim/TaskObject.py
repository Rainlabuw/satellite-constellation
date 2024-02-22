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
        self.deg_change_per_ts = (ang_vel*self.dt*180/np.pi).to_value(u.one)

        self.appear_time = appear_time

        self.task_idxs = None
        self.lats = []
        self.lons = []

    def propagate(self, hex_to_task_mapping, T):
        """
        Propagate object movement over time, populating the task_idxs list 
        with the task index that the object is associated with at each timestep.
        """
        self.task_idxs = []
        k = 0
        in_region = True
        while k < T:
            if k < self.appear_time or not in_region:
                self.task_idxs.append(None)
                self.lats.append(None)
                self.lons.append(None)
            else:
                # Find the hexagon containing this lat/lon, increment target count
                hexagon = h3.geo_to_h3(self.lat, self.lon, 2)

                self.task_idxs.append(hex_to_task_mapping[hexagon])
                self.lats.append(self.lat)
                self.lons.append(self.lon)

                self.lat += self.deg_change_per_ts * self.dir[1]
                self.lon += self.deg_change_per_ts * self.dir[0]

                if self.lat < self.lat_range[0] or self.lat > self.lat_range[1] or self.lon < self.lon_range[0] or self.lon > self.lon_range[1]:
                    in_region = False
            k += 1