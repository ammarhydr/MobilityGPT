import numpy as np
import pandas as pd
import os
import logging
from typing import Tuple, List
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrajectoryProcessor:
    def __init__(self, dataset: str = "SF"):
        """Initialize TrajectoryProcessor.
        
        Args:
            dataset: Name of the dataset (e.g., 'SF')
        """
        self.dataset = dataset
        self.data_dir = Path(f'./{dataset}-Taxi')
        
    def create_train_test_split(self, df: pd.DataFrame, 
                              validation_split: float = 0.2,
                              shuffle_dataset: bool = True,
                              random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/test split of the data.
        
        Args:
            df: Input DataFrame
            validation_split: Fraction of data to use for validation
            shuffle_dataset: Whether to shuffle before splitting
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        dataset_size = len(df)
        indices = list(range(dataset_size))
        
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        train_indices, val_indices = indices[split:], indices[:split]
        return df.iloc[train_indices], df.iloc[val_indices]
    
    def generate_trajectory_file(self, df: pd.DataFrame, output_file: Path):
        """Generate trajectory text file from DataFrame.
        
        Args:
            df: Input DataFrame with trajectory data
            output_file: Path to output file
        """
        rid_list_list = df.rid_list.values.tolist()
        
        trajectories = []
        for traj in rid_list_list:
            traj_str = ''.join(traj)
            trajectories.append(f"{traj_str}\n")
            
        with open(output_file, "w") as fo:
            for trajectory in trajectories:
                fo.write(f"{trajectory}\n")
        
        logger.info(f"Generated trajectory file: {output_file}")
    
    def process(self, validation_split: float = 0.2,
               shuffle_dataset: bool = True,
               random_seed: int = 42):
        """Main processing function.
        
        Args:
            validation_split: Fraction of data to use for validation
            shuffle_dataset: Whether to shuffle before splitting
            random_seed: Random seed for reproducibility
        """
        train_file = self.data_dir / f'{self.dataset}_Taxi_trajectory_train.csv'
        
        if not train_file.exists():
            logger.info(f"Train file not found. Creating train/test split...")
            
            # Read original data
            input_file = self.data_dir / f'{self.dataset}_Taxi_trajectory.csv'
            if not input_file.exists():
                raise FileNotFoundError(f"Dataset file not found: {input_file}")
                
            df = pd.read_csv(input_file)
            
            # Create split
            train_data, test_data = self.create_train_test_split(
                df,
                validation_split=validation_split,
                shuffle_dataset=shuffle_dataset,
                random_seed=random_seed
            )
            
            # Save splits
            train_data.to_csv(train_file, index=False)
            test_file = self.data_dir / f'{self.dataset}_Taxi_trajectory_test.csv'
            test_data.to_csv(test_file, index=False)
            logger.info("Train/test split created and saved.")
        else:
            logger.info("Using existing train file.")
            train_data = pd.read_csv(train_file)
        
        # Generate trajectory file
        output_file = Path(f"Trajs_{self.dataset}.txt")
        self.generate_trajectory_file(train_data, output_file)

def main():
    """Main entry point."""
    processor = TrajectoryProcessor()
    processor.process()

if __name__ == "__main__":
    main()

    
    
