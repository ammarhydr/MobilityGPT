import geopandas as gpd
import pandas as pd
from scipy.spatial import distance
import pickle
from tqdm import tqdm
import numpy as np
from shapely.geometry import LineString, Point
import osmnx as ox
import geopy.distance
from collections import Counter
import torch

crs = {'init': 'epsg:4326'} 


def od_pair_to_adjacency_matrix(od_pair_list):
    """
    Converts an od pair to an adjacency matrix.

    Args:
        od_pair (torch.Tensor): A tensor of od pairs, where each pair is a two-dimensional tensor of the form [source, destination].

    Returns:
        torch.Tensor: An adjacency matrix
    """

    od_pair= torch.tensor(od_pair_list)
    # Create a sparse adjacency matrix.
    adjacency_matrix = torch.sparse_coo_tensor(od_pair.t(), torch.ones(od_pair.size(0)))

    # Convert the sparse adjacency matrix to a dense adjacency matrix.
    adjacency_matrix = adjacency_matrix.to_dense().numpy()

    # Return the adjacency matrix.
    return adjacency_matrix

def get_radius(traj):
    """
    get the std of the distances of all points away from center as `gyration radius`
    :param trajs:
    :return:
    """

    xs = []
    ys = []
    for ind, geo in traj.iterrows():
        coordinate = list(map(float, geo['coordinates'].replace('[', '').replace(']', '').split(',')))
        ys.append(coordinate[0])
        xs.append(coordinate[1])
    xcenter, ycenter = np.mean(xs), np.mean(ys)
    dxs = xs - xcenter
    dys = ys - ycenter
    rad = [dxs[i]**2 + dys[i]**2 for i in range(len(traj))]
    
    return np.mean(np.array(rad, dtype=float))
    

def arr_to_distribution(arr, min, max, bins):
    """
    convert an array to a probability distribution
    :param arr: np.array, input array
    :param min: float, minimum of converted value
    :param max: float, maximum of converted value
    :param bins: int, number of bins between min and max
    :return: np.array, output distribution array
    """
    distribution, base = np.histogram(
        arr, np.arange(
            min, max, float(
                max - min) / bins))
    return distribution, base[:-1]

def save_to_gdp(links_all, path, name):
    
    data = list(set(links_all))

    link_counts = Counter(links_all)
    
    df_in=geo.loc[geo['geo_id'].isin(data)].reset_index(drop=True)
    counts_df = pd.DataFrame(list(link_counts.items()), columns=['geo_id', 'counts'])
    df = pd.merge(df_in, counts_df, on='geo_id', how='left')

    coordinates = [list(map(float, c.replace('[', '').replace(']', '').split(','))) for c in df['coordinates'].tolist()]
    
    geometry=[]
    for x in coordinates:
        geometry.append(LineString([Point(pair) for pair in zip(x[::2], x[1::2])]))
        # geometry.append(Point([x[-2], x[-1]]))
    df['geometry'] = geometry
    
    df=df[['geo_id', 'length', 'counts', 'geometry']]
    gdf = gpd.GeoDataFrame(df, geometry="geometry",  crs = crs)
    
    fdf = ox.project_gdf(gdf,to_latlong=True)
    fdf.to_file(path+name+'.geojson',driver='GeoJSON')



def calculate_gravity(trajs, dataset):
    
    file = open(dataset+'-Taxi/regions','rb')
    regions, region_links, links_region = pickle.load(file)

    regions_count=np.zeros((2, len(regions)))
    for links in trajs:
        for r in range(len(regions)):
            if links[0] in region_links[r]:
                regions_count[0][r]+=1
            if links[-1] in region_links[r]:
                regions_count[1][r]+=1
                
    regions_count=regions_count.sum(axis=0, keepdims=True)
    
    gravity=np.zeros((len(regions), len(regions)))
    for r1 in range(len(regions)):
        for r2 in range(len(regions)):
            dist=geopy.distance.geodesic(regions.center.iloc[r1], regions.center.iloc[r2]).m
            if dist!=0:
                gravity[r1, r2]=regions_count[0, r1]*regions_count[0, r2]/(dist**2)
    
    gravity_traj = [] 
    for links in trajs:
        r_o = links_region[links[0]]
        r_d = links_region[links[-1]]
        gravity_traj.append(gravity[r_o, r_d])
        
    return gravity_traj


def plot_gravity_map(trajs, path, name, dataset):
    
    file = open(dataset+'-Taxi/regions','rb')
    regions, region_links, links_region = pickle.load(file)

    regions_count=np.zeros((2, len(regions)))
    for links in trajs:
        for r in range(len(regions)):
            if links[0] in region_links[r]:
                regions_count[0][r]+=1
            if links[-1] in region_links[r]:
                regions_count[1][r]+=1
                
    regions_count=regions_count.sum(axis=0, keepdims=True)
    
    gravity=np.zeros((len(regions), len(regions)))
    for r1 in range(len(regions)):
        for r2 in range(len(regions)):
            dist=geopy.distance.geodesic(regions.center.iloc[r1], regions.center.iloc[r2]).m
            if dist!=0:
                gravity[r1, r2]=regions_count[0, r1]*regions_count[0, r2]/(dist**2)

    regions1=regions[['region_id','geometry']]
    regions1['gravity']=gravity.sum(axis=1)
    
    center=regions['center'].tolist()
    
    lat=[]
    lon=[]
    for x in center:
        lat.append(((x[0])))
        lon.append(((x[1])))
        
    regions1['lat']=lat
    regions1['lon']=lon
    gdf = gpd.GeoDataFrame(regions1, geometry="geometry",  crs = crs)
    gdf.to_file(path+name+'_gravity.geojson',driver='GeoJSON')
    
def query_error(links_test, links_synth, edges):

    sample_edges = edges.sample(500).geo_id.values
    links_test_commom = list(set(links_test).intersection(sample_edges))
    links_synth_commom = list(set(links_synth).intersection(sample_edges))

    links_sample=edges.loc[edges.geo_id.isin(links_test_commom)]        
    links_sample_synth=edges.loc[edges.geo_id.isin(links_synth_commom)]       
    all_osmids=list(set(links_test+links_synth))        
    
    s_b=0.01*(5000)

    link_counts_test = Counter(links_test)
    link_counts_synth = Counter(links_synth)
   
    # Average Query Error
    qe_all=[]
    for l_id in all_osmids:
        link=links_sample[links_sample.geo_id.isin([l_id])]
        link_synth=links_sample_synth[links_sample_synth.geo_id.isin([l_id])]
        if len(link)==1 and len(link_synth)==1:
            qe=abs(link_counts_test[link.geo_id.iloc[0]]-link_counts_synth[link_synth.geo_id.iloc[0]])/max(link_counts_test[link.geo_id.iloc[0]], s_b)
            qe_all.append(qe)
        elif len(link)==1 and len(link_synth)==0:
            qe=abs(link_counts_test[link.geo_id.iloc[0]])/max(link_counts_test[link.geo_id.iloc[0]], s_b)
            qe_all.append(qe)
        elif len(link)==0 and len(link_synth)==1:
            qe=(link_counts_synth[link_synth.geo_id.iloc[0]])/s_b
            qe_all.append(qe)  
    
    return np.mean(qe_all)

def get_popular_origin(trajs):
    origins = [t[0] for t in trajs]
    origin_counts = Counter(origins)
    most_pop = max(origin_counts.values())
    key = next((key for key, value in origin_counts.items() if value == most_pop), None)
    return key


def remove_edges(path):
    
    removed_path = []
    
    for i in range(len(path)):
        if path[i] not in removed_path:
            removed_path.append(path[i])
        else:
            while removed_path[-1]!=path[i]:
                removed_path.pop()
                
    return removed_path

def connectivity_check(links, link_pairs):
    
    conn=0
    for o, d in zip(links, links[1:]):
        if d in link_pairs[o]:
            conn+=1
            
    av_connectivity = conn/(len(links)-1)
    return av_connectivity

dataset = 'Porto'
    
work_dir = './Trajs_'+dataset+'_synthetic/'
geo=pd.read_csv(dataset+'-Taxi/roadmap.geo')


rel=pd.read_csv(dataset+'-Taxi/roadmap.rel')    
rel['combined'] = rel.apply(lambda x: list([x['origin_id'], x['destination_id']]),axis=1)
od_list=rel['combined'].tolist()
adj_matrix=od_pair_to_adjacency_matrix(od_list)

connectivity={}
for indx, row in tqdm(enumerate(adj_matrix)):
    ones_indices = np.where(row == 1)[0]
    connectivity[indx] = list(ones_indices)


df_data=pd.read_csv(dataset+'-Taxi/'+dataset+'_Taxi_trajectory_test.csv')
samples=df_data.sample(n=5000, random_state=1)

samples.to_csv(dataset+'-Taxi/'+dataset+'_Taxi_trajectory_sample.csv')

links_str=samples.rid_list.values.tolist()
links_test = [list(map(int, traj.split(','))) for traj in links_str]
links_test_4count = [e for traj in links_str for e in list(map(int, traj.split(',')))]
links_test_all = list(set(links_test_4count))


file = open(work_dir+'test_PPO_trajectories_1e-6_T0.99.txt', 'rb')
links_synth = pickle.load(file)
# links_synth = [remove_edges(l) for l in links_synth]
links_synth_4count = [ e for traj in links_synth for e in traj]
links_synth_all = list(set(links_synth_4count))

save_to_gdp(links_test_4count, work_dir,'test_map')
save_to_gdp(links_synth_4count, work_dir, 'MobilityGPT_map')
#%%
plot_gravity_map(links_test, work_dir, 'test', dataset)
plot_gravity_map(links_synth, work_dir, 'synth', dataset)

gravity_test = calculate_gravity(links_test, dataset)
gravity_synth = calculate_gravity(links_synth, dataset)



# # origin_id = get_popular_origin(links_test)
# origin_id = 20

# links_most_test = [l for l in links_test if l[0]==origin_id]
# samples_test = random.sample(links_most_test, 10)
# for i, sample in enumerate(samples_test):
#     save_to_gdp(sample, 'samples/test_'+str(i))

# links_most_synth = [l for l in links_synth if l[0]==origin_id]
# samples_synth = random.sample(links_most_synth, 10)
# for i, sample in enumerate(samples_synth):
#     save_to_gdp(sample, 'samples/synth_'+str(i))




av_qe = query_error(links_test_4count, links_synth_4count, geo)

OD_synth=[]
length_synth=[]

OD_test=[]
length_test=[]

rad_test = []
rad_synth = []

per_test = []
per_synth = []

# conn_test = []
conn_synth = []
for i in tqdm(range(len(links_test))):
    link_ids = links_synth[i]
    if len(link_ids)>1:
        link_synth = geo[geo['geo_id'].isin(link_ids)]
        OD_synth.append(link_ids[0])
        OD_synth.append(link_ids[-1])
        length = link_synth.length.sum()
        length_synth.append(length)
        rad_synth.append(get_radius(link_synth))
        per_synth.append(float(len(set(link_ids)))/len(link_ids))
        conn_synth.append(connectivity_check(link_ids, connectivity))
        

        link_ids=links_test[i] 
        link_test = geo[geo['geo_id'].isin(link_ids)]
        OD_test.append(link_ids[0])
        OD_test.append(link_ids[-1])
        length = link_test.length.sum()
        length_test.append(length)
        rad_test.append(get_radius(link_test))
        per_test.append(float(len(set(link_ids)))/len(link_ids))
        # conn_test.append(connectivity_check(link_ids, connectivity))



OD_synth_dist, _ = arr_to_distribution(OD_synth, min(OD_synth+OD_test), max(OD_synth+OD_test), 300)
OD_test_dist, _ = arr_to_distribution(OD_test, min(OD_synth+OD_test), max(OD_synth+OD_test), 300)

length_synth_dist, _ = arr_to_distribution(length_synth, min(length_synth+length_test), max(length_synth+length_test), 300)
length_test_dist, _ = arr_to_distribution(length_test, min(length_synth+length_test), max(length_synth+length_test), 300)

rad_synth_dist, _ = arr_to_distribution(rad_synth, min(rad_synth+rad_test), max(rad_synth+rad_test), 300)
rad_test_dist, _ = arr_to_distribution(rad_test, min(rad_synth+rad_test), max(rad_synth+rad_test), 300)

link_synth_dist, _ = arr_to_distribution(links_synth_all, min(links_synth_all+links_test_all), max(links_synth_all+links_test_all), 300)
link_test_dist, _ = arr_to_distribution(links_test_all, min(links_synth_all+links_test_all), max(links_synth_all+links_test_all), 300)

gravity_synth_dist, _ = arr_to_distribution(gravity_synth, min(gravity_synth+gravity_test), max(gravity_synth+gravity_test), 100)
gravity_test_dist, _ = arr_to_distribution(gravity_test, min(gravity_synth+gravity_test), max(gravity_synth+gravity_test), 100)

# conn_synth_dist, _ = arr_to_distribution(conn_synth, min(conn_synth+conn_test), max(conn_synth+conn_test), 300)
# conn_test_dist, _ = arr_to_distribution(conn_test, min(conn_synth+conn_test), max(conn_synth+conn_test), 300)

js_OD=distance.jensenshannon(OD_synth_dist, OD_test_dist)
print('JS value for OD: ',js_OD)

js_length=distance.jensenshannon(length_synth_dist, length_test_dist)
print('JS value for length: ',js_length)

js_rad=distance.jensenshannon(rad_synth_dist, rad_test_dist)
print('JS value for radius: ',js_rad)

js_link=distance.jensenshannon(link_synth_dist, link_test_dist)
print('JS value for link distribution: ',js_link)

js_gravity=distance.jensenshannon(gravity_synth_dist, gravity_test_dist)
print('JS value for gravity: ',js_gravity)

av_conn = len([c for c in conn_synth if c==1])/len(conn_synth)
print('Average connectivity: ', av_conn)


# links_commom = list(set(links_test_all).intersection(links_synth_all))
percentage_of_coverage = len(set(links_synth_all))/len(geo)
print('Normalized coverage area with respect to test set: ', percentage_of_coverage)


print('Average query error: ',av_qe)
# js_rad=distance.jensenshannon(per_synth_dist, per_test_dist)
# print('JS value for repeatitions: ',js_rad)
    
# OD_synth_count=[OD_synth.count(x) for x in set(OD_synth)]
# OD_test_count=[OD_test.count(x) for x in set(OD_test)]

# fig, ax = plt.subplots(figsize=(10,8))
# ax.plot(OD_test_count, label='Raw data', linewidth=6, color='red')
# ax.plot(OD_synth_count,'-', label='Synthetic data', linewidth=2, color='blue')
# plt.grid(axis='y', alpha=0.75)
# # legend = ax.legend(loc='upper right', fontsize='x-large')
# legend = ax.legend(fontsize='15')
# plt.xticks(fontsize=14)
# plt.yticks( fontsize=14)
# plt.xlabel('Links', fontsize=16)
# plt.ylabel('Frequency', fontsize=16)
# plt.title('OD link distribution with adjacency matrix')
# plt.show()
