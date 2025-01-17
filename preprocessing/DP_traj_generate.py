import pickle
import networkx as nx
import pandas as pd
import random
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DPTrajectoryGenerator:
    """Generate trajectories using differential privacy approach."""
    
    def __init__(self, dataset: str = "SF"):
        """Initialize generator.
        
        Args:
            dataset: Name of the dataset (e.g., 'SF')
        """
        self.dataset = dataset
        self.data_dir = Path(f'./{dataset}-Taxi')
        
    def load_data(self) -> Tuple[nx.Graph, Dict, Dict, Dict]:
        """Load required data files.
        
        Returns:
            Tuple of (road network graph, regions, region_links, links_region)
        """
        # Load road network
        rel = pd.read_csv(self.data_dir / 'roadmap.rel')
        graph = nx.from_pandas_edgelist(rel, source='origin_id', 
                                      target='destination_id')
        
        # Load region data
        with open(self.data_dir / 'regions', 'rb') as f:
            regions, region_links, links_region = pickle.load(f)
            
        return graph, regions, region_links, links_region
    
    def calculate_sampling_probabilities(self, region_links: Dict) -> Dict:
        """Calculate sampling probabilities for regions.
        
        Args:
            region_links: Dictionary mapping regions to their road links
            
        Returns:
            Dictionary of region sampling probabilities
        """
        total_elements = sum(len(values) for values in region_links.values())
        return {key: len(values) / total_elements 
                for key, values in region_links.items()}
    
    def generate_trajectory(self, graph: nx.Graph, 
                          region_links: Dict,
                          sampling_probs: Dict) -> str:
        """Generate a single trajectory.
        
        Args:
            graph: Road network graph
            region_links: Dictionary mapping regions to their road links
            sampling_probs: Region sampling probabilities
            
        Returns:
            Comma-separated trajectory string
        """
        # Sample origin and destination regions
        sampled_keys = random.choices(
            list(region_links.keys()),
            weights=list(sampling_probs.values()),
            k=2
        )
        
        # Sample road segments from regions
        sampled_elements = [random.choice(region_links[key]) 
                          for key in sampled_keys]
        
        # Find shortest path
        path = nx.shortest_path(
            G=graph,
            source=sampled_elements[0],
            target=sampled_elements[1]
        )
        
        return ','.join(map(str, path))
    
    def generate_trajectories(self, num_trajectories: int = int(1e6)):
        """Generate multiple trajectories.
        
        Args:
            num_trajectories: Number of trajectories to generate
        """
        logger.info(f"Loading data for {self.dataset} dataset...")
        graph, regions, region_links, links_region = self.load_data()
        
        # Calculate sampling probabilities
        sampling_probs = self.calculate_sampling_probabilities(region_links)
        
        # Generate trajectories
        logger.info(f"Generating {num_trajectories} trajectories...")
        output_file = Path(f"Trajs_{self.dataset}_random.txt")
        
        with open(output_file, "w") as f:
            for _ in tqdm(range(num_trajectories)):
                try:
                    traj = self.generate_trajectory(
                        graph, region_links, sampling_probs
                    )
                    f.write(f"{traj}\n\n")
                except Exception as e:
                    logger.warning(f"Failed to generate trajectory: {e}")
                    continue
        
        logger.info(f"Generated trajectories saved to {output_file}")

def main():
    """Main entry point."""
    generator = DPTrajectoryGenerator()
    generator.generate_trajectories()

if __name__ == "__main__":
    main()
    