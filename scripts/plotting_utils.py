import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colormaps
import matplotlib.patches as patches
import numpy as np

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import StaticOrbitPlotter
from poliastro.spheroid_location import SpheroidLocation
from astropy import units as u

from constellation_sim.ConstellationSim import ConstellationSim
from constellation_sim.Satellite import Satellite

import h3
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.image as mpimg
from math import radians, cos, sin, asin, sqrt, atan2, degrees

from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def generate_global_hexagons(resolution, lat_max):
    # Initialize an empty set to store unique H3 indexes
    hexagons = set()

    # Latitude and Longitude ranges
    lat_range = (-lat_max, lat_max)
    lon_range = (-180, 180)

    # Step through the defined ranges and discretize the globe
    lat_steps, lon_steps = 0.5, 0.5
    lat = lat_range[0]
    while lat <= lat_range[1]:
        lon = lon_range[0]
        while lon <= lon_range[1]:
            # Find the hexagon containing this lat/lon
            hexagon = h3.geo_to_h3(lat, lon, resolution)
            hexagons.add(hexagon)
            lon += lon_steps
        lat += lat_steps

    return hexagons

def crosses_dateline(polygon):
    """ Check if any edge of the polygon crosses the dateline. """
    for i in range(len(polygon.exterior.coords)-1):
        lon1, lon2 = polygon.exterior.coords[i][0], polygon.exterior.coords[i+1][0]
        if abs(lon1 - lon2) > 180:  # Crosses the dateline
            return True
    return False

def hexagons_to_geometries(hexagons):
    polygons = []
    centroids = []
    for hexagon in hexagons:
        boundary = h3.h3_to_geo_boundary(hexagon, geo_json=True)
        polygon = Polygon(boundary)
        if crosses_dateline(polygon):
            # Create positive polygon
            pos_coords = []
            for x, y in polygon.exterior.coords:
                if x < 0:
                    pos_coords.append((x+360,y))
                else:
                    pos_coords.append((x,y))
            pos_poly = Polygon(pos_coords)

            # Create negative polygon
            neg_coords = []
            for x, y in polygon.exterior.coords:
                if x > 0:
                    neg_coords.append((x-360,y))
                else:
                    neg_coords.append((x,y))
            neg_poly = Polygon(neg_coords)

            # Add both polygons and centroids to the list
            polygons.append(neg_poly)
            polygons.append(pos_poly)
            centroids.append(neg_poly.centroid)
            centroids.append(pos_poly.centroid)
        else:
            polygons.append(polygon)
            centroids.append(polygon.centroid)
    return polygons, centroids

def generate_circle_points(center_lon, center_lat, radius_km, num_points=100):
    """
    Generate points defining a circle on the Earth's surface.
    """
    points = []
    for angle in np.linspace(0, 2*np.pi, num_points):
        d = radius_km
        R = 6371 # Earth radius in km

        lat1 = radians(center_lat)
        lon1 = radians(center_lon)

        lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(angle))
        lon2 = lon1 + atan2(sin(angle) * sin(d/R) * cos(lat1), cos(d/R) - sin(lat1) * sin(lat2))

        lat2 = degrees(lat2)
        lon2 = degrees(lon2)

        points.append((lon2, lat2))

    return points

def get_radius_of_fov(sat):
    earth = sat.orbit.attractor

    earth_r = earth.R.to_value(u.km)
    sat_r = np.linalg.norm(sat.orbit.r.to_value(u.km))

    #Max FOV is when the angle is a tangent to the surface of the earth
    max_fov = np.arcsin(earth_r/sat_r)*180/np.pi
    if max_fov < sat.fov: print(f"Lowering FOV to {max_fov}")
    sat.fov = min(sat.fov, max_fov)

    third_angle = (180 - np.arcsin(sat_r/earth_r*np.sin(sat.fov*np.pi/180))*180/np.pi)
    delta_angle = ((180 - sat.fov - third_angle)) * np.pi/180

    return delta_angle * np.linalg.norm(earth_r)

def get_sat_lat_lon(sat):
    lon = np.arctan2(sat.orbit.r[1].to_value(u.km), sat.orbit.r[0].to_value(u.km))*180/np.pi
    lat = np.arctan2(sat.orbit.r[2].to_value(u.km),np.linalg.norm(sat.orbit.r.to_value(u.km)[:2]))*180/np.pi
    return lat, lon

def haal_experiment_plots():
    earth_image = mpimg.imread('earth.jpg')

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Display the Earth image
    ax.imshow(earth_image, extent=[-180, 180, -90, 90], aspect='auto')

    earth = Earth

    altitude=550
    fov=60

    a = earth.R.to(u.km) + altitude*u.km
    ecc = 0*u.one
    argp = 0*u.deg
    #~~~~~~~~~~~~~~~~~ EXPERIMENT 1~~~~~~~~~~~~~~~~~~~~~~
    m = 450
    for j in range(m):
        lon = np.random.uniform(-180, 180)
        lat = np.random.uniform(-58, 58)

        if j == m-1:
            plt.scatter(lon,lat,color='k',s=3, label='Midsize constellation tasks')
        else:
            plt.scatter(lon,lat,color='k',s=3)
    
    const = ConstellationSim(dt=1*u.min)
    num_sats_per_plane = 18
    inc = 58*u.deg
    raan = 0*u.deg
    for sat_num in range(num_sats_per_plane):
        ta = sat_num*360/num_sats_per_plane*u.deg
        sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], fov=fov)
        const.add_sat(sat)
    raan = 360/18*u.deg
    for sat_num in range(num_sats_per_plane):
        ta = sat_num*360/num_sats_per_plane*u.deg
        sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], fov=fov)
        const.add_sat(sat)

    for sat in const.sats:
        center_lat, center_lon = get_sat_lat_lon(sat)
        rad = get_radius_of_fov(sat)

        # Generate circle points
        circle_points = generate_circle_points(center_lon, center_lat, rad)

        # Create a circle polygon
        circle = Polygon(circle_points)

        # Create a GeoDataFrame for the circle
        circle_gdf = gpd.GeoDataFrame(geometry=[circle])

        # Add the circle to the plot
        circle_gdf.plot(ax=ax, edgecolor='black', facecolor="none", linewidth=2)

    plt.scatter(200, 200, color='black', facecolors='none', label="Midsize constellation sat FOVs, planes 0 and 1")
    #~~~~~~~~~~~~~~~~~ EXPERIMENT 2~~~~~~~~~~~~~~~~~~~~~~
    hexagons = generate_global_hexagons(1, 70)
    print(len(hexagons))
    hexagon_polygons, centroids = hexagons_to_geometries(hexagons)
    gdf = gpd.GeoDataFrame(geometry=hexagon_polygons)

    centroid_xs = [c.x for c in centroids]
    centroid_ys = [c.y for c in centroids]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=hexagon_polygons)

    const = ConstellationSim(dt=1*u.min)
    num_sats_per_plane = 25
    inc = 70*u.deg
    raan = (20/40)*360*u.deg
    for sat_num in range(num_sats_per_plane):
        ta = sat_num*360/num_sats_per_plane*u.deg
        sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], fov=fov)
        const.add_sat(sat)
    raan = (21/40)*360*u.deg
    for sat_num in range(num_sats_per_plane):
        ta = sat_num*360/num_sats_per_plane*u.deg
        sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], fov=fov)
        const.add_sat(sat)

    for sat in const.sats:
        center_lat, center_lon = get_sat_lat_lon(sat)
        rad = get_radius_of_fov(sat)

        # Generate circle points
        circle_points = generate_circle_points(center_lon, center_lat, rad)

        # Create a circle polygon
        circle = Polygon(circle_points)

        # Create a GeoDataFrame for the circle
        circle_gdf = gpd.GeoDataFrame(geometry=[circle])

        # Add the circle to the plot
        circle_gdf.plot(ax=ax, edgecolor='red', facecolor="none", linewidth=2)

    # Overlay the hexagon grid
    # gdf.boundary.plot(ax=ax, color='red', alpha=0.25)
    plt.scatter(centroid_xs, centroid_ys, color='red', s=3, label="Large constellation tasks")
    #phantom dot for legend entry
    plt.scatter(200, 200, color='red', facecolors='none', label="Large constellation sat FOVs, planes 20 and 21")
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    plt.legend(loc='lower left')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()

def update_object_track(k, ax, earth_image, task_to_hex_map, sat_cover_matrix, task_objects, assignments, task_trans_state_dep_scaling_mat):
    ax.clear()

    m = sat_cover_matrix.shape[1]
    T = sat_cover_matrix.shape[2]

    # Display the Earth image background
    ax.imshow(earth_image, extent=[-180, 180, -90, 90], aspect='auto', alpha=0.6)
    lat_range = (20, 50)
    lon_range = (73, 135)
    plt.xlim(lon_range)
    plt.ylim(lat_range)

    #Store the coverage level and existence of handover for each task at this time
    #for use later in plotting task objects
    coverage_by_task = {}
    uncaptured_by_task = {} #by strict unassignment
    handover_by_task = {} #by handover
    #Display the region hexagons
    for task, hex in task_to_hex_map.items():
        coverage = 0
        #If the object is tracked by any satellite in the task
        if np.max(assignments[k][:,task]) == 1:
            assigned_sat = np.argmax(assignments[k][:,task])
        else: assigned_sat = None
        #Determine coverage contribution from primary satellite:
        if assigned_sat is not None:
            coverage += sat_cover_matrix[assigned_sat, task, k]
        
        #if secondary task is assigned, find the satellite that is assigned to it
        if np.max(assignments[k][:,task+m//2] == 1):
            sec_assigned_sat = np.argmax(assignments[k][:,task+m//2])
        else: sec_assigned_sat = None
        #Determine coverage contribution from secondary satellite:
        if sec_assigned_sat is not None:
            coverage += sat_cover_matrix[sec_assigned_sat, task+m//2, k]

        #determine if the task is uncaptured this step bc of handover or unassignment
        uncaptured = False
        handover = False
        if coverage > 0:
            if k == 0: uncaptured = False
            else:
                prim_sat_prev_task = assignments[k-1][assigned_sat,:].nonzero()[0]
                uncaptured = bool(task_trans_state_dep_scaling_mat[prim_sat_prev_task,task])
                handover = bool(task_trans_state_dep_scaling_mat[prim_sat_prev_task,task])
        else: 
            uncaptured = True
            handover = False
        coverage_by_task[task] = coverage
        uncaptured_by_task[task] = uncaptured
        handover_by_task[task] = handover

        hexagon_polygons, _ = hexagons_to_geometries([hex])
        gdf = gpd.GeoDataFrame(geometry=hexagon_polygons)
        gdf.boundary.plot(ax=ax, color='black', alpha=0.2)

        if handover:
            gdf.plot(ax=ax, color='yellow', alpha=0.2)
        elif uncaptured:
            gdf.plot(ax=ax, color='red', alpha=0.2)
        else:
            gdf.plot(ax=ax, color=plt.cm.Greens(min(1,coverage)), alpha=0.2)

    for task_object in task_objects:
        if task_object.lats[k] is not None and task_object.lons[k] is not None:
            task_object_idx = task_object.task_idxs[k]

            coverage = coverage_by_task[task_object_idx]
            uncaptured = uncaptured_by_task[task_object_idx]
            handover = handover_by_task[task_object_idx]
            
            if uncaptured or handover:
                plt.scatter(task_object.lons[k], task_object.lats[k], color='red', s=7)
            else:
                plt.scatter(task_object.lons[k], task_object.lats[k], color='green', s=7)

    #Create legend with phantom dots
    plt.scatter(0, 0, color='green', label='Tracked Objects')
    plt.scatter(0, 0, color='red', label='Untracked Objects')
    fake_verts = [(0, 0), (0, 1), (1, 1)]
    ylw_ptch = patches.Polygon(fake_verts, color='green', label='Tracked Regions (intensity is quality of coverage)')
    ax.add_patch(ylw_ptch)
    ylw_ptch = patches.Polygon(fake_verts, color='yellow', label='Regions undergoing handover transition')
    ax.add_patch(ylw_ptch)
    ylw_ptch = patches.Polygon(fake_verts, color='red', label='Untracked Regions')
    ax.add_patch(ylw_ptch)
    plt.legend(loc='lower right')
    plt.title(f"Time: +{k*30}/{T*30} sec.")
            

def plot_object_track_scenario(hexagon_to_task_mapping, sat_cover_matrix, task_objects, assignments, task_trans_state_dep_scaling_mat,
                               save_loc, show=True):
    n = sat_cover_matrix.shape[0]
    m = sat_cover_matrix.shape[1]
    T = sat_cover_matrix.shape[2]

    earth_image = mpimg.imread('scaled_down_highres_earth.jpg')

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Reverse hex to task mapping:
    task_to_hex_mapping = {}
    for hex, task in hexagon_to_task_mapping.items():
        task_to_hex_mapping[task] = hex

    ani  = FuncAnimation(fig, update_object_track, fargs=(ax, earth_image, task_to_hex_mapping, sat_cover_matrix, task_objects, assignments, task_trans_state_dep_scaling_mat), 
                         frames=T, interval=1000, blit=False)

    if show:
        plt.show()

    ani.save(save_loc, writer='pillow', fps=2, dpi=100)

if __name__ == "__main__":
    plot_object_track_scenario(None, None, None, None)