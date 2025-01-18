import pandas as pd
import numpy as np
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from scipy.spatial import distance
import geopy.distance
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrajectoryEvaluator:
    """Evaluate generated trajectories against ground truth."""
    
    def __init__(self, dataset: str = "SF"):
        """Initialize evaluator."""
        self.dataset = dataset
        self.data_dir = Path(f'./{dataset}-Taxi')
        self.work_dir = Path(f'./Trajs_{dataset}_synthetic/gpt-mobility')
        self.crs = {'init': 'epsg:4326'}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict]:
        """Load required data files."""
        # Load road network data
        geo = pd.read_csv(self.data_dir / 'roadmap.geo')
        rel = pd.read_csv(self.data_dir / 'roadmap.rel')
        
        # Create graph and get connectivity
        graph = nx.from_pandas_edgelist(rel, source='origin_id', target='destination_id')
        adj_matrix = nx.adjacency_matrix(graph).todense()
        
        # Create connectivity dictionary
        connectivity = {}
        for i, row in enumerate(adj_matrix):
            connectivity[i] = np.where(row == 1)[0].tolist()
            
        return geo, rel, adj_matrix, connectivity
        
    def _get_radius(self, traj: pd.DataFrame) -> float:
        """Calculate gyration radius of trajectory."""
        xs, ys = [], []
        for _, geo in traj.iterrows():
            coordinate = list(map(float, geo['coordinates'].replace('[', '').replace(']', '').replace('(', '').replace(')', '').split(',')))
            ys.append(coordinate[0])
            xs.append(coordinate[1])
            
        if not xs or not ys:
            return 0.0
            
        xcenter, ycenter = np.mean(xs), np.mean(ys)
        rad = [(xs[i] - xcenter)**2 + (ys[i] - ycenter)**2 
               for i in range(len(traj))]
        return np.mean(rad)
    
    def _calculate_gravity(self, trajs: List[List[int]]) -> List[float]:
        """Calculate gravity values for trajectories."""
        # Load region data
        with open(self.data_dir / 'regions', 'rb') as f:
            regions, region_links, links_region = pickle.load(f)
            
        # Calculate region counts
        regions_count = np.zeros((2, len(regions)))
        for links in trajs:
            for r in range(len(regions)):
                if links[0] in region_links[r]:
                    regions_count[0][r] += 1
                if links[-1] in region_links[r]:
                    regions_count[1][r] += 1
                    
        regions_count = regions_count.sum(axis=0, keepdims=True)
        
        # Calculate gravity matrix
        gravity = np.zeros((len(regions), len(regions)))
        for r1 in range(len(regions)):
            for r2 in range(len(regions)):
                dist = geopy.distance.geodesic(
                    regions.center.iloc[r1], 
                    regions.center.iloc[r2]
                ).m
                if dist != 0:
                    gravity[r1, r2] = (regions_count[0, r1] * 
                                     regions_count[0, r2] / (dist**2))
        
        # Calculate trajectory gravity values
        gravity_values = []
        for links in trajs:
            r_o = links_region[links[0]]
            r_d = links_region[links[-1]]
            gravity_values.append(gravity[r_o, r_d])
            
        return gravity_values
    
    def _remove_cycles(self, path: List[int]) -> List[int]:
        """Remove cycles from trajectory."""
        removed_path = []
        for i in range(len(path)):
            if path[i] not in removed_path:
                removed_path.append(path[i])
            else:
                while removed_path[-1] != path[i]:
                    removed_path.pop()
        return removed_path
    
    def _check_connectivity(self, links: List[int], 
                          connectivity: Dict[int, List[int]]) -> float:
        """Check connectivity of trajectory."""
        conn = 0
        for o, d in zip(links, links[1:]):
            if d in connectivity[o]:
                conn += 1
        return conn / (len(links) - 1) if len(links) > 1 else 0
    
    def calculate_metrics(self, test_trajs: List[List[int]], 
                        synth_trajs: List[List[int]],
                        geo: pd.DataFrame,
                        connectivity: Dict[int, List[int]]) -> Dict:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Flatten trajectories
        test_links = [link for traj in test_trajs for link in traj]
        synth_links = [link for traj in synth_trajs for link in traj]
        
        # Calculate basic distributions
        metrics.update(self._calculate_distribution_metrics(
            test_trajs, synth_trajs, test_links, synth_links))
        
        # Calculate trajectory-specific metrics
        metrics.update(self._calculate_trajectory_metrics(
            test_trajs, synth_trajs, geo, connectivity))
        
        # Calculate gravity metrics
        metrics.update(self._calculate_gravity_metrics(test_trajs, synth_trajs))
        
        # Calculate query error
        metrics['query_error'] = self._calculate_query_error(
            test_trajs, synth_trajs, geo)
        
        return metrics
    
    def _calculate_distribution_metrics(self, test_trajs, synth_trajs, 
                                     test_links, synth_links) -> Dict:
        """Calculate distribution-based metrics."""
        metrics = {}
        
        # OD distribution
        od_test = [traj[0] for traj in test_trajs] + [traj[-1] for traj in test_trajs]
        od_synth = [traj[0] for traj in synth_trajs] + [traj[-1] for traj in synth_trajs]
        od_dist_test, _ = self._arr_to_distribution(od_test, 
                                                  min(od_test + od_synth),
                                                  max(od_test + od_synth), 300)
        od_dist_synth, _ = self._arr_to_distribution(od_synth,
                                                   min(od_test + od_synth),
                                                   max(od_test + od_synth), 300)
        metrics['js_od'] = distance.jensenshannon(od_dist_test, od_dist_synth)
        
        # Link distribution
        link_dist_test, _ = self._arr_to_distribution(test_links,
                                                    min(test_links + synth_links),
                                                    max(test_links + synth_links), 300)
        link_dist_synth, _ = self._arr_to_distribution(synth_links,
                                                     min(test_links + synth_links),
                                                     max(test_links + synth_links), 300)
        metrics['js_link'] = distance.jensenshannon(link_dist_test, link_dist_synth)
        
        return metrics
    
    def _calculate_trajectory_metrics(self, test_trajs, synth_trajs, 
                                   geo, connectivity) -> Dict:
        """Calculate trajectory-specific metrics."""
        metrics = {}
        
        # Length distribution
        test_lengths = []
        synth_lengths = []
        test_radii = []
        synth_radii = []
        synth_connectivity = []
        
        for traj in test_trajs:
            traj_df = geo[geo['geo_id'].isin(traj)]
            test_lengths.append(traj_df['length'].sum())
            test_radii.append(self._get_radius(traj_df))
            
        for traj in synth_trajs:
            traj_df = geo[geo['geo_id'].isin(traj)]
            synth_lengths.append(traj_df['length'].sum())
            synth_radii.append(self._get_radius(traj_df))
            synth_connectivity.append(self._check_connectivity(traj, connectivity))
        
        # Calculate distributions and metrics
        length_dist_test, _ = self._arr_to_distribution(test_lengths,
                                                      min(test_lengths + synth_lengths),
                                                      max(test_lengths + synth_lengths), 300)
        length_dist_synth, _ = self._arr_to_distribution(synth_lengths,
                                                       min(test_lengths + synth_lengths),
                                                       max(test_lengths + synth_lengths), 300)
        metrics['js_length'] = distance.jensenshannon(length_dist_test, length_dist_synth)
        
        rad_dist_test, _ = self._arr_to_distribution(test_radii,
                                                   min(test_radii + synth_radii),
                                                   max(test_radii + synth_radii), 300)
        rad_dist_synth, _ = self._arr_to_distribution(synth_radii,
                                                    min(test_radii + synth_radii),
                                                    max(test_radii + synth_radii), 300)
        metrics['js_radius'] = distance.jensenshannon(rad_dist_test, rad_dist_synth)
        
        # Connectivity score
        metrics['connectivity'] = np.mean([c for c in synth_connectivity if c == 1])
        
        return metrics
    
    def _calculate_gravity_metrics(self, test_trajs, synth_trajs) -> Dict:
        """Calculate gravity-based metrics."""
        metrics = {}
        
        gravity_test = self._calculate_gravity(test_trajs)
        gravity_synth = self._calculate_gravity(synth_trajs)
        
        gravity_dist_test, _ = self._arr_to_distribution(gravity_test,
                                                       min(gravity_test + gravity_synth),
                                                       max(gravity_test + gravity_synth), 100)
        gravity_dist_synth, _ = self._arr_to_distribution(gravity_synth,
                                                        min(gravity_test + gravity_synth),
                                                        max(gravity_test + gravity_synth), 100)
        metrics['js_gravity'] = distance.jensenshannon(gravity_dist_test, gravity_dist_synth)
        
        return metrics

    def _arr_to_distribution(self, arr, min, max, bins):
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

    def evaluate(self, test_file: str, synth_file: str):
        """Main evaluation function."""
        logger.info("Loading data...")
        geo, rel, adj_matrix, connectivity = self.load_data()
        
        # Load trajectories
        test_df = pd.read_csv(self.data_dir / test_file).sample(n=5000, random_state=1)
        test_trajs = [list(map(int, traj.split(','))) 
                     for traj in test_df.rid_list.values]
        
        with open(self.work_dir / synth_file, 'rb') as f:
            synth_trajs = pickle.load(f)
            
        # Remove cycles from synthetic trajectories
        synth_trajs = [self._remove_cycles(traj) for traj in synth_trajs]
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = self.calculate_metrics(test_trajs, synth_trajs, geo, connectivity)    
        
        # Log results
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    def _calculate_query_error(self, test_trajs: List[List[int]], 
                             synth_trajs: List[List[int]], 
                             geo: pd.DataFrame) -> float:
        """Calculate average query error between test and synthetic trajectories.
        
        Args:
            test_trajs: List of test trajectories
            synth_trajs: List of synthetic trajectories
            geo: DataFrame containing road network data
            
        Returns:
            Average query error
        """
        # Flatten trajectories
        links_test = [link for traj in test_trajs for link in traj]
        links_synth = [link for traj in synth_trajs for link in traj]
        
        # Sample edges and find common links
        sample_edges = geo.sample(5000).geo_id.values
        links_test_common = list(set(links_test).intersection(sample_edges))
        links_synth_common = list(set(links_synth).intersection(sample_edges))
        
        # Filter links
        links_sample = geo[geo.geo_id.isin(links_test_common)]
        links_sample_synth = geo[geo.geo_id.isin(links_synth_common)]
        
        # Get all unique links
        all_osmids = list(set(links_test + links_synth))
        
        # Set sensitivity bound
        s_b = 0.01 * 5000
        
        # Calculate link counts
        link_counts_test = Counter(links_test)
        link_counts_synth = Counter(links_synth)
        
        # Calculate query errors
        qe_all = []
        for l_id in all_osmids:
            link = links_sample[links_sample.geo_id.isin([l_id])]
            link_synth = links_sample_synth[links_sample_synth.geo_id.isin([l_id])]
            
            if len(link) == 1 and len(link_synth) == 1:
                # Both exist
                qe = abs(link_counts_test[link.geo_id.iloc[0]] - 
                        link_counts_synth[link_synth.geo_id.iloc[0]]) / \
                    max(link_counts_test[link.geo_id.iloc[0]], s_b)
                qe_all.append(qe)
            elif len(link) == 1 and len(link_synth) == 0:
                # Only in test
                qe = abs(link_counts_test[link.geo_id.iloc[0]]) / \
                    max(link_counts_test[link.geo_id.iloc[0]], s_b)
                qe_all.append(qe)
            elif len(link) == 0 and len(link_synth) == 1:
                # Only in synthetic
                qe = link_counts_synth[link_synth.geo_id.iloc[0]] / s_b
                qe_all.append(qe)
        
        return np.mean(qe_all)

def main():
    """Main entry point."""
    evaluator = TrajectoryEvaluator(dataset="SF")
    evaluator.evaluate(
        test_file="SF_Taxi_trajectory_test.csv",
        synth_file="test_trajectories.txt"
    )

if __name__ == "__main__":
    main()
