import numpy as np
from shapely.geometry import Polygon
from copy import deepcopy
import pickle
import matplotlib.image as mpimg

from constellation_sim.ConstellationSim import ConstellationSim
from constellation_sim.Satellite import Satellite
from constellation_sim.Task import Task

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.spheroid_location import SpheroidLocation
import h3

from common.methods import *
from common.plotting_utils import generate_global_hexagons, hexagons_to_geometries

def get_prox_mat_and_graphs_random_tasks(num_planes, num_sats_per_plane, m,T,proximity_func=calc_fov_based_proximities, altitude=550, fov=60, dt=1*u.min, isl_dist=None):
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

    benefits, graphs = const.propagate_orbits(T, proximity_func)
    return benefits, graphs

def generate_smooth_coverage_hexagons(lat_range, lon_range, res=1):
    # Initialize an empty set to store unique H3 indexes
    hexagons = set()

    # Step through the defined ranges and discretize the globe
    lat_steps, lon_steps = 0.2/res, 0.2/res
    lat = lat_range[0]
    while lat <= lat_range[1]:
        lon = lon_range[0]
        while lon <= lon_range[1]:
            # Find the hexagon containing this lat/lon
            hexagon = h3.geo_to_h3(lat, lon, res)
            hexagons.add(hexagon)
            lon += lon_steps
        lat += lat_steps
        
    return list(hexagons) #turn into a list so that you can easily index it later

def get_prox_mat_and_graphs_coverage(num_planes, num_sats_per_plane,T,inc,proximity_func=calc_fov_based_proximities, altitude=550, fov=60, dt=1*u.min, isl_dist=None):
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
    hexagons = generate_smooth_coverage_hexagons((-inc.to_value(u.deg), inc.to_value(u.deg)), (-180, 180))
    
    #Add tasks at centroid of all hexagons
    for hexagon in hexagons:
        boundary = h3.h3_to_geo_boundary(hexagon, geo_json=True)
        polygon = Polygon(boundary)

        lat = polygon.centroid.y
        lon = polygon.centroid.x

        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, earth)
        
        task_benefit = np.random.uniform(1, 2, size=T)
        task = Task(task_loc, task_benefit)
        const.add_task(task)

    benefits, graphs = const.propagate_orbits(T, proximity_func)
    return benefits, graphs

def get_prox_mat_and_graphs_area(num_planes, num_sats_per_plane, T, lat_range, lon_range, inc=70*u.deg, fov=60, isl_dist=2500, dt=1*u.min):
    """
    Generate tasks only in locations that are green in an image
    """
    const = ConstellationSim(dt=dt, isl_dist=isl_dist)
    earth = Earth

    hexagons = generate_global_hexagons(2, 70)
    hexagon_polygons, centroids, hexagons = hexagons_to_geometries(hexagons)

    centroid_xs = [c.x for c in centroids]
    centroid_ys = [c.y for c in centroids]

    #Add tasks at centroid of all hexagons
    hex_to_task_mapping = {}
    for lon, lat, hex in zip(centroid_xs, centroid_ys, hexagons):
        if lon < lon_range[1] and lon > lon_range[0] and lat < lat_range[1] and lat > lat_range[0]:
            task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, earth)
            
            task_benefit = np.random.uniform(1, 2, size=T)
            task = Task(task_loc, task_benefit)
            const.add_task(task)

            hex_to_task_mapping[hex] = len(const.tasks)-1

    altitude=550
    a = earth.R.to(u.km) + altitude*u.km
    ecc = 0*u.one
    argp = 0*u.deg

    #BUILD CONSTELLATION OF SATELLITES
    for plane_num in range(num_planes):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num, fov=fov)
            const.add_sat(sat)

    #generate satellite coverage matrix with all satellites, even those far away from the area
    full_sat_prox_matrix, graphs = const.propagate_orbits(T, calc_fov_based_proximities)

    #Remove satellites which never cover any tasks in the entire T window
    truncated_sat_prox_matrix = np.zeros_like(full_sat_prox_matrix)
    old_to_new_sat_mapping = {}
    active_sats = []
    for i in range(const.n):
        total_sat_scaling = np.sum(full_sat_prox_matrix[i,:,:])
        if total_sat_scaling > 0: #it has nonzero scaling on at least one task at one timestep
            curr_sat = const.sats[i]
            curr_sat.id = len(active_sats)

            truncated_sat_prox_matrix[curr_sat.id,:,:] = full_sat_prox_matrix[i,:,:]

            old_to_new_sat_mapping[i] = curr_sat.id
            active_sats.append(curr_sat)
    
    const.sats = active_sats
    truncated_sat_prox_matrix = truncated_sat_prox_matrix[:len(const.sats),:,:] #truncate unused satellites
    
    #update graphs to reflect new satellite numbering after removing useless sats
    for k in range(T):
        nodes_to_remove = [n for n in graphs[k].nodes() if n not in old_to_new_sat_mapping.keys()]
        graphs[k].remove_nodes_from(nodes_to_remove)
        graphs[k] = nx.relabel_nodes(graphs[k], old_to_new_sat_mapping)

    print(f"Num tasks: {len(const.tasks)}")
    print("\nNum active sats", len(const.sats))
    return truncated_sat_prox_matrix, graphs, hex_to_task_mapping, const

def get_prox_mat_and_graphs_soil_moisture(num_planes, num_sats_per_plane, T, lat_range, lon_range, inc=70*u.deg, fov=60, isl_dist=2500, dt=1*u.min):
    """
    Generate tasks only in locations that are green in an image
    """
    const = ConstellationSim(dt=dt, isl_dist=isl_dist)
    earth = Earth

    #~~~~~~~~~~~~~~~Get tasks only over land:~~~~~~~~~~~~~~~~~~~~~~
    earth_image = mpimg.imread('common/scaled_down_highres_earth.jpg')

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Display the Earth image
    ax.imshow(earth_image, extent=[-180, 180, -90, 90], aspect='auto')

    hexagons = generate_global_hexagons(2, 70)
    hexagon_polygons, centroids, hexagons = hexagons_to_geometries(hexagons)

    centroid_xs = [c.x for c in centroids]
    centroid_ys = [c.y for c in centroids]
    ground_centroid_xs = []
    ground_centroid_ys = []
    ground_hexagons = []

    for (centroid_x, centroid_y, hexagon) in zip(centroid_xs, centroid_ys, hexagons):
        #if pixel in earth_image at centroid_x, centroid_y is not blue, add to the list of hexagons corresponding to ground
        if centroid_x < lon_range[1] and centroid_x > lon_range[0] and centroid_y < lat_range[1] and centroid_y > lat_range[0]:
            classic_blue = np.array([11, 10, 50])
            curr_color = np.array(earth_image[int((-centroid_y+90)/180*earth_image.shape[0]), int((centroid_x+180)/360*earth_image.shape[1]), :])
            if np.linalg.norm(classic_blue-curr_color) > 5:
                ground_centroid_xs.append(centroid_x)
                ground_centroid_ys.append(centroid_y)
                ground_hexagons.append(hexagon)

    #Add tasks at centroid of all hexagons
    hex_to_task_mapping = {}
    for lon, lat, hex in zip(ground_centroid_xs, ground_centroid_ys, ground_hexagons):
        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, earth)
        
        task_benefit = np.random.uniform(1, 2, size=T)
        task = Task(task_loc, task_benefit)
        const.add_task(task)

        hex_to_task_mapping[hex] = len(const.tasks)-1

    altitude=550
    a = earth.R.to(u.km) + altitude*u.km
    ecc = 0*u.one
    argp = 0*u.deg

    #BUILD CONSTELLATION OF SATELLITES
    for plane_num in range(num_planes):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num, fov=fov)
            const.add_sat(sat)

    #generate satellite coverage matrix with all satellites, even those far away from the area
    full_sat_prox_matrix, graphs = const.propagate_orbits(T, calc_fov_based_proximities)

    #Remove satellites which never cover any tasks in the entire T window
    truncated_sat_prox_matrix = np.zeros_like(full_sat_prox_matrix)
    old_to_new_sat_mapping = {}
    active_sats = []
    for i in range(const.n):
        total_sat_scaling = np.sum(full_sat_prox_matrix[i,:,:])
        if total_sat_scaling > 0: #it has nonzero scaling on at least one task at one timestep
            curr_sat = const.sats[i]
            curr_sat.id = len(active_sats)

            truncated_sat_prox_matrix[curr_sat.id,:,:] = full_sat_prox_matrix[i,:,:]

            old_to_new_sat_mapping[i] = curr_sat.id
            active_sats.append(curr_sat)
    
    const.sats = active_sats
    truncated_sat_prox_matrix = truncated_sat_prox_matrix[:len(const.sats),:,:] #truncate unused satellites
    
    #update graphs to reflect new satellite numbering after removing useless sats
    for k in range(T):
        nodes_to_remove = [n for n in graphs[k].nodes() if n not in old_to_new_sat_mapping.keys()]
        graphs[k].remove_nodes_from(nodes_to_remove)
        graphs[k] = nx.relabel_nodes(graphs[k], old_to_new_sat_mapping)

    print(f"Num tasks: {len(const.tasks)}")
    print("\nNum active sats", len(const.sats))
    return truncated_sat_prox_matrix, graphs, hex_to_task_mapping, const

def get_benefit_matrix_and_graphs_multitask_area(lat_range, lon_range, T, fov=60, isl_dist=2500, dt=30*u.second):
    """
    Generate sat coverage area and graphs for all satellites which can
    see a given area over the course of some .

    Also add multiple synthetic agents for each real satellite to approximate beams.
    """
    const = ConstellationSim(dt=dt, isl_dist=isl_dist)
    earth = Earth

    hex_to_task_mapping = {}
    #Generate tasks at the centroid of each hexagon in the area.
    #Resolution 2 is more appropriate to cover America, but is too small to cover the entire globe.
    hexagons = generate_smooth_coverage_hexagons(lat_range, lon_range, 2)
    for j, hexagon in enumerate(hexagons):
        boundary = h3.h3_to_geo_boundary(hexagon, geo_json=True)
        polygon = Polygon(boundary)

        lat = polygon.centroid.y
        lon = polygon.centroid.x

        task_loc = SpheroidLocation(lat*u.deg, lon*u.deg, 0*u.m, Earth)
        task_ben = np.random.uniform(1,2)
        const.add_task(Task(task_loc, task_ben*np.ones(T))) #use benefits which are uniformly 1 to get scaling matrix

        hex_to_task_mapping[hexagon] = j
    print("Num tasks", len(hexagons))

    #add lat and lon range to constellation so we can recover it later
    const.task_lat_range = lat_range
    const.task_lon_range = lon_range

    #~~~~~~~~~Generate a constellation of satellites at 400 km.~~~~~~~~~~~~~
    a = earth.R.to(u.km) + 550*u.km
    ecc = 0*u.one
    inc = 70*u.deg
    argp = 0*u.deg
    ta_offset = 0*u.deg

    num_planes = 18
    num_sats_per_plane = 18

    for plane_num in range(num_planes):
        raan = plane_num*360/num_planes*u.deg
        for sat_num in range(num_sats_per_plane):
            ta = sat_num*360/num_sats_per_plane*u.deg + ta_offset
            ta_offset += 1*u.deg
            sat = Satellite(Orbit.from_classical(earth, a, ecc, inc, raan, argp, ta), [], [], plane_id=plane_num, fov=fov)
            const.add_sat(sat)

    #generate satellite coverage matrix with all satellites, even those far away from the area
    full_sat_prox_matrix, graphs = const.propagate_orbits(T, calc_fov_based_proximities)

    #Remove satellites which never cover any tasks in the entire T window
    truncated_sat_prox_matrix = np.zeros_like(full_sat_prox_matrix)
    old_to_new_sat_mapping = {}
    active_sats = []
    for i in range(const.n):
        total_sat_scaling = np.sum(full_sat_prox_matrix[i,:,:])
        if total_sat_scaling > 0: #it has nonzero scaling on at least one task at one timestep
            curr_sat = const.sats[i]
            curr_sat.id = len(active_sats)

            truncated_sat_prox_matrix[curr_sat.id,:,:] = full_sat_prox_matrix[i,:,:]

            old_to_new_sat_mapping[i] = curr_sat.id
            active_sats.append(curr_sat)
    
    const.sats = active_sats
    sats_to_track = [deepcopy(sat) for sat in active_sats]
    truncated_sat_prox_matrix = truncated_sat_prox_matrix[:len(const.sats),:,:] #truncate unused satellites
    
    #update graphs to reflect new satellite numbering after removing useless sats
    for k in range(T):
        nodes_to_remove = [n for n in graphs[k].nodes() if n not in old_to_new_sat_mapping.keys()]
        graphs[k].remove_nodes_from(nodes_to_remove)
        graphs[k] = nx.relabel_nodes(graphs[k], old_to_new_sat_mapping)

    print("\nNum active sats", len(const.sats))

    #Create synthetic satellites to represent each satellite being able to complete multiple tasks.
    #The nth synthetic satellite will recieve (0.9**(n-1))*100% of the benefit for a given task, to incentivize
    #spreading tasks evenly amongst satellites.
    num_tasks_per_sat = 10
    num_real_sats = len(const.sats)
    num_synthetic_sats = num_real_sats*num_tasks_per_sat
    num_original_tasks = len(hexagons)
    
    print(f"Num synthetic sats: {num_synthetic_sats}")
    full_sat_prox_matrix_w_synthetic_sats = np.zeros((num_synthetic_sats, num_original_tasks, T))
    print(f"Full sat cover matrix shape: {full_sat_prox_matrix.shape}, truncated shape: {truncated_sat_prox_matrix.shape}, synthetic shape: {full_sat_prox_matrix_w_synthetic_sats.shape}")

    # #add dummy tasks to sat cover matrix, if necessary
    # base_synthetic_benefit_matrix = np.zeros((num_real_sats, num_tasks_after_synthetic_sats, T))
    # base_synthetic_benefit_matrix[:,:len(hexagons),:] = truncated_sat_prox_matrix
    # print(f"Base synthetic sat cover matrix shape: {base_synthetic_benefit_matrix.shape}")

    for task_num in range(num_tasks_per_sat):
        #Adjust sat cover matrix to reflect the synthetic satellites and tasks
        full_sat_prox_matrix_w_synthetic_sats[task_num*num_real_sats:(task_num+1)*num_real_sats,:num_original_tasks,:] = truncated_sat_prox_matrix*(0.9**task_num)

        #Add appropriate graph connections for the synthetic satellites
        if task_num > 0: #only add for non-original tasks
            for k in range(T):
                grph = graphs[k]
                for real_sat_num in range(num_real_sats):
                    synthetic_sat_num = grph.number_of_nodes()
                    grph.add_node(synthetic_sat_num)
                    grph.add_edge(real_sat_num, synthetic_sat_num)
                    for neigh in grph.neighbors(real_sat_num):
                        grph.add_edge(neigh, synthetic_sat_num)

    n = full_sat_prox_matrix_w_synthetic_sats.shape[0]
    m = full_sat_prox_matrix_w_synthetic_sats.shape[1]
    
    #Create matrix which indicates that synthetic agents representing the same real agent
    A_eqiv = np.zeros((n,n))
    for agent1 in range(n):
        for agent2 in range(n):
            if agent1 % num_real_sats == agent2 % num_real_sats:
                A_eqiv[agent1,agent2] = 1
            else:
                A_eqiv[agent1,agent2] = 0

    #Create scaling matrix for task transitions
    T_trans = np.ones((m,m))
    #no penalty when transitioning between the same task
    for j in range(num_original_tasks):
        T_trans[j,j] = 0

    # #no penalty when transitioning between tasks which are in adjacent hexagons
    # for j in range(num_original_tasks):
    #     task_hex = hexagons[j]
    #     neighbor_hexes = h3.k_ring(task_hex, 1)
    #     for neighbor_hex in neighbor_hexes:
    #         if neighbor_hex in hex_to_task_mapping.keys():
    #             T_trans[j,hex_to_task_mapping[neighbor_hex]] = 0
    #             T_trans[hex_to_task_mapping[neighbor_hex],j] = 0

    with open('multitask_experiment/sat_prox_matrix.pkl','wb') as f:
        pickle.dump(full_sat_prox_matrix_w_synthetic_sats, f)
    with open('multitask_experiment/graphs.pkl','wb') as f:
        pickle.dump(graphs, f)
    with open('multitask_experiment/T_trans.pkl','wb') as f:
        pickle.dump(T_trans, f)
    with open('multitask_experiment/A_eqiv.pkl','wb') as f:
        pickle.dump(A_eqiv, f)
    with open('multitask_experiment/hex_task_map.pkl','wb') as f:
        pickle.dump(hex_to_task_mapping, f)
    with open('multitask_experiment/const_object.pkl','wb') as f:
        pickle.dump(const, f)
    with open('multitask_experiment/sats_to_track.pkl','wb') as f:
        pickle.dump(sats_to_track, f)
    
    return full_sat_prox_matrix_w_synthetic_sats, graphs, T_trans, A_eqiv, \
        hex_to_task_mapping, const, sats_to_track