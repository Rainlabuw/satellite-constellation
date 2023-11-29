import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import split
from shapely.geometry import box
import matplotlib.image as mpimg
from math import radians, cos, sin, asin, sqrt, atan2, degrees

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

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

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

if __name__ == "__main__":
    earth_image = mpimg.imread('/home/josh/code/satellite-constellation/scripts/earth.jpg')

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
    m = 400
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
    gdf.boundary.plot(ax=ax, color='red', alpha=0.25)
    plt.scatter(centroid_xs, centroid_ys, color='red', s=1, label="Large constellation tasks")
    #phantom dot for legend entry
    plt.scatter(200, 200, color='red', facecolors='none', label="Large constellation sat FOVs, planes 20 and 21")
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    plt.legend(loc='lower left')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)

    plt.show()