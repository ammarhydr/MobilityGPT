import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString
import geopy.distance
import pickle
from math import radians, sin, cos, sqrt, atan2
import os
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GravityBasedSampler:
    def __init__(self, dataset: str, grid_distance: float = 1000):
        """Initialize GravityBasedSampler.
        
        Args:
            dataset: Name of the dataset (e.g., 'SF')
            grid_distance: Distance in meters for grid cell size
        """
        self.dataset = dataset
        self.grid_distance = grid_distance
        self.data_dir = f'./{dataset}-Taxi'
        
    def _haversine_distance(self, coord1: Tuple[float, float], 
                           coord2: Tuple[float, float]) -> float:
        """Calculate haversine distance between two coordinates.
        
        Args:
            coord1: First coordinate (lat, lon)
            coord2: Second coordinate (lat, lon)
            
        Returns:
            Distance in meters
        """
        earth_radius = 6371000.0
        lat1, lon1 = map(radians, coord1)
        lat2, lon2 = map(radians, coord2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return earth_radius * c

    def _convert_gps_to_2d_grid(self, gps_locations: List, 
                               lines: List[LineString]) -> Tuple[gpd.GeoDataFrame, Dict, Dict]:
        """Convert GPS locations to 2D grid system.
        
        Args:
            gps_locations: List of [latitudes, longitudes]
            lines: List of LineString geometries
            
        Returns:
            Tuple of (regions GeoDataFrame, region_links dict, links_region dict)
        """
        min_lat, max_lat = min(gps_locations[1]), max(gps_locations[1])
        min_lon, max_lon = min(gps_locations[0]), max(gps_locations[0])
        
        length1 = self._haversine_distance((min_lat, min_lon), (max_lat, min_lon))
        length2 = self._haversine_distance((min_lat, min_lon), (min_lat, max_lon))
        
        grid_lat = int(length1 // self.grid_distance)
        grid_lon = int(length2 // self.grid_distance)
        
        grid_coordinates = [
            np.linspace(min_lat, max_lat, grid_lat),
            np.linspace(min_lon, max_lon, grid_lon)
        ]
        
        grid_polies, center_coords, region_links, links_region = [], [], {}, {}
        
        for i in range(grid_lat):
            for j in range(grid_lon):
                region_id = i * grid_lon + j
                poly = self._create_square_poly(
                    grid_coordinates[0][i], 
                    grid_coordinates[1][j]
                )
                
                grid_polies.append(poly)
                center_coords.append([grid_coordinates[0][i], grid_coordinates[1][j]])
                
                # Find intersecting links
                links = [k for k, line in enumerate(lines) if line.intersects(poly)]
                region_links[region_id] = links
                for k in links:
                    links_region[k] = region_id
                
        regions = gpd.GeoDataFrame({
            'region_id': range(len(grid_polies)),
            'center': center_coords,
            'geometry': grid_polies
        }, crs="EPSG:4326")
        
        return regions, region_links, links_region
    
    def _create_square_poly(self, lat: float, lon: float) -> gpd.GeoSeries:
        """Create square polygon centered at given coordinates."""
        gs = gpd.GeoSeries(wkt.loads(f'POINT ({lon} {lat})'))
        gdf = gpd.GeoDataFrame(geometry=gs, crs='EPSG:4326')
        gdf = gdf.to_crs('EPSG:3857')
        res = gdf.buffer(distance=self.grid_distance, cap_style=3)
        return res.to_crs('EPSG:4326').iloc[0]
    
    def process(self):
        """Main processing function to generate gravity-based sampling."""
        logger.info(f"Processing {self.dataset} dataset...")
        
        # Load road network data
        geo_data = pd.read_csv(os.path.join(self.data_dir, 'roadmap.geo'))
        
        # Extract coordinates and create LineStrings
        coords = []
        lats, lons = [], []
        for _, geo in geo_data.iterrows():
            coordinate = list(map(float, geo['coordinates'].replace('[', '').replace(']', '').replace('(', '').replace(')', '').split(',')))
            coords.append(LineString([[coordinate[i], coordinate[i+1]] 
                                   for i in range(0, len(coordinate), 2)]))
            lats.append(coordinate[0])
            lons.append(coordinate[1])
        
        # Generate regions
        regions, region_links, links_region = self._convert_gps_to_2d_grid(
            [lats, lons], coords
        )
        
        # Save region data
        with open(os.path.join(self.data_dir, 'regions'), 'wb') as f:
            pickle.dump([regions, region_links, links_region], f)
        
        # Process trajectory data
        train_file = os.path.join(self.data_dir, f'{self.dataset}_Taxi_trajectory_train.csv')
        df_data = pd.read_csv(train_file)
        
        # Calculate region counts
        regions_count = self._calculate_region_counts(df_data, region_links, len(regions))
        
        # Calculate gravity matrix
        gravity = self._calculate_gravity_matrix(regions, regions_count)
        
        # Add gravity values to trajectories
        df_data['gravity'] = self._calculate_trajectory_gravity(
            df_data, links_region, gravity
        )
        
        # Save enhanced dataset
        output_file = os.path.join(self.data_dir, 
                                 f'{self.dataset}_Taxi_trajectory_train_w_gravity.csv')
        df_data.to_csv(output_file, index=False)
        logger.info(f"Saved gravity-enhanced dataset to {output_file}")
    
    def _calculate_region_counts(self, df: pd.DataFrame, 
                               region_links: Dict, 
                               num_regions: int) -> np.ndarray:
        """Calculate region counts from trajectory data."""
        regions_count = np.zeros((2, num_regions))
        for _, traj in df.iterrows():
            links = list(map(int, traj.rid_list.split(',')))
            for r in range(num_regions):
                if links[0] in region_links[r]:
                    regions_count[0][r] += 1
                if links[-1] in region_links[r]:
                    regions_count[1][r] += 1
        return regions_count.sum(axis=0, keepdims=True)
    
    def _calculate_gravity_matrix(self, regions: gpd.GeoDataFrame, 
                                regions_count: np.ndarray) -> np.ndarray:
        """Calculate gravity matrix between regions."""
        num_regions = len(regions)
        gravity = np.zeros((num_regions, num_regions))
        
        for r1 in range(num_regions):
            for r2 in range(num_regions):
                dist = geopy.distance.geodesic(
                    regions.center.iloc[r1], 
                    regions.center.iloc[r2]
                ).m
                if dist != 0:
                    gravity[r1, r2] = (regions_count[0, r1] * 
                                     regions_count[0, r2] / (dist**2))
        return gravity
    
    def _calculate_trajectory_gravity(self, df: pd.DataFrame, 
                                    links_region: Dict,
                                    gravity: np.ndarray) -> List[float]:
        """Calculate gravity values for trajectories."""
        gravity_values = []
        for _, traj in df.iterrows():
            links = list(map(int, traj.rid_list.split(',')))
            r_o = links_region[links[0]]
            r_d = links_region[links[-1]]
            gravity_values.append(gravity[r_o, r_d])
        return gravity_values

def main():
    """Main entry point."""
    sampler = GravityBasedSampler(dataset="SF")
    sampler.process()

if __name__ == "__main__":
    main()


    