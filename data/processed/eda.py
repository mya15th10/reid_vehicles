# eda.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import seaborn as sns
from collections import Counter
import random
from matplotlib_venn import venn3


class VeRiDatasetAnalyzer:
    """
    A class to analyze the VeRi-776 vehicle re-identification dataset
    """
    
    def __init__(self, dataset_path='./data/raw/VeRi'):
        """
        Initialize the analyzer with the path to the dataset
        
        Args:
            dataset_path: Base path to the VeRi dataset
        """
        self.dataset_path = dataset_path
        self.train_path = os.path.join(dataset_path, 'image_train')
        self.query_path = os.path.join(dataset_path, 'image_query')
        self.test_path = os.path.join(dataset_path, 'image_test')
        
        # Check if paths exist
        for path_name, path in [("Train", self.train_path), 
                               ("Query", self.query_path), 
                               ("Test", self.test_path)]:
            if not os.path.exists(path):
                print(f"Warning: {path_name} directory does not exist at {path}")
    
    def analyze_directory(self, directory_path):
        """
        Analyze the structure of a directory and return basic information
        
        Args:
            directory_path: Path to the directory to analyze
            
        Returns:
            Dictionary containing statistics about the images
        """
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist")
            return None
            
        # Read file list
        files = os.listdir(directory_path)
        
        # Parse filenames to extract vehicle ID and camera ID
        # Format: 0001_c001_00016240_0.jpg (vehicle_id_camera_frame_num.jpg)
        vehicle_ids = []
        camera_ids = []
        
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                parts = file.split('_')
                if len(parts) >= 2:
                    vehicle_ids.append(parts[0])
                    camera_ids.append(parts[1])
        
        # Count unique vehicles and cameras
        unique_vehicles = set(vehicle_ids)
        unique_cameras = set(camera_ids)
        
        # Count images per vehicle
        vehicle_count = Counter(vehicle_ids)
        
        # Calculate distribution of images per vehicle
        images_per_vehicle = list(vehicle_count.values())
        
        return {
            'total_images': len(files),
            'unique_vehicles': len(unique_vehicles),
            'unique_cameras': len(unique_cameras),
            'images_per_vehicle': images_per_vehicle,
            'vehicle_ids': list(unique_vehicles),
            'camera_ids': list(unique_cameras)
        }
    
    def visualize_image_samples(self, directory_path, num_samples=5):
        """
        Display sample images from the dataset
        
        Args:
            directory_path: Path to the directory containing images
            num_samples: Number of samples to display
        """
        files = [f for f in os.listdir(directory_path) 
                if f.endswith('.jpg') or f.endswith('.png')]
        
        if not files:
            print("No image files found in the directory")
            return
        
        # Randomly select samples
        sample_files = random.sample(files, min(num_samples, len(files)))
        
        plt.figure(figsize=(15, 3*num_samples))
        for i, file in enumerate(sample_files):
            img_path = os.path.join(directory_path, file)
            img = Image.open(img_path)
            
            # Extract metadata from filename
            parts = file.split('_')
            vehicle_id = parts[0] if len(parts) > 0 else "Unknown"
            camera_id = parts[1] if len(parts) > 1 else "Unknown"
            
            plt.subplot(num_samples, 1, i+1)
            plt.imshow(img)
            plt.title(f"Vehicle ID: {vehicle_id}, Camera: {camera_id}\nFile: {file}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_image_properties(self, directory_path, sample_size=100):
        """
        Analyze image properties such as dimensions and color channels
        
        Args:
            directory_path: Path to the directory containing images
            sample_size: Number of images to sample for analysis
            
        Returns:
            Dictionary containing image property statistics
        """
        files = [f for f in os.listdir(directory_path) 
                if f.endswith('.jpg') or f.endswith('.png')]
        
        if not files:
            print("No image files found in the directory")
            return
        
        # Randomly select samples
        sample_files = random.sample(files, min(sample_size, len(files)))
        
        # Collect image properties
        widths = []
        heights = []
        channels = []
        aspect_ratios = []
        
        for file in sample_files:
            img_path = os.path.join(directory_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                heights.append(height)
                widths.append(width)
                channels.append(img.shape[2] if len(img.shape) > 2 else 1)
                aspect_ratios.append(width / height)
        
        return {
            'height': {
                'min': min(heights), 
                'max': max(heights), 
                'mean': np.mean(heights),
                'std': np.std(heights)
            },
            'width': {
                'min': min(widths), 
                'max': max(widths), 
                'mean': np.mean(widths),
                'std': np.std(widths)
            },
            'aspect_ratio': {
                'min': min(aspect_ratios),
                'max': max(aspect_ratios),
                'mean': np.mean(aspect_ratios),
                'std': np.std(aspect_ratios)
            },
            'channels': list(set(channels))
        }
    
    def visualize_vehicle_distribution(self, dataset_info):
        """
        Visualize the distribution of images per vehicle ID
        
        Args:
            dataset_info: Dictionary with dataset statistics from analyze_directory
        """
        plt.figure(figsize=(12, 6))
        
        # Histogram of images per vehicle
        plt.hist(dataset_info['images_per_vehicle'], bins=30, alpha=0.7)
        plt.title('Distribution of Images per Vehicle ID')
        plt.xlabel('Number of Images')
        plt.ylabel('Number of Vehicle IDs')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional statistics
        min_images = min(dataset_info['images_per_vehicle'])
        max_images = max(dataset_info['images_per_vehicle'])
        avg_images = np.mean(dataset_info['images_per_vehicle'])
        
        print(f"Minimum images per vehicle: {min_images}")
        print(f"Maximum images per vehicle: {max_images}")
        print(f"Average images per vehicle: {avg_images:.2f}")
        
        # Count vehicles with few images
        few_images = sum(1 for count in dataset_info['images_per_vehicle'] if count < 5)
        print(f"Vehicles with fewer than 5 images: {few_images} ({few_images/len(dataset_info['images_per_vehicle'])*100:.2f}%)")
    
    def analyze_camera_distribution(self, directory_path):
        """
        Analyze distribution of images across different cameras
        
        Args:
            directory_path: Path to the directory containing images
            
        Returns:
            Dictionary with camera distribution statistics
        """
        files = [f for f in os.listdir(directory_path) 
                if f.endswith('.jpg') or f.endswith('.png')]
        
        camera_counts = Counter()
        
        for file in files:
            parts = file.split('_')
            if len(parts) >= 2:
                camera_id = parts[1]
                camera_counts[camera_id] += 1
        
        # Plot camera distribution
        plt.figure(figsize=(12, 6))
        cameras = list(camera_counts.keys())
        counts = [camera_counts[cam] for cam in cameras]
        
        plt.bar(cameras, counts)
        plt.title('Distribution of Images Across Cameras')
        plt.xlabel('Camera ID')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return dict(camera_counts)
    
    def analyze_vehicle_examples(self, directory_path, num_vehicles=3, images_per_vehicle=5):
        """
        Display multiple images of the same vehicles to show variations
        
        Args:
            directory_path: Path to the directory containing images
            num_vehicles: Number of different vehicles to display
            images_per_vehicle: Number of images to show per vehicle
        """
        files = os.listdir(directory_path)
        
        # Create a mapping from vehicle ID to list of images
        vehicle_to_images = {}
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                parts = file.split('_')
                if len(parts) >= 2:
                    vehicle_id = parts[0]
                    if vehicle_id not in vehicle_to_images:
                        vehicle_to_images[vehicle_id] = []
                    vehicle_to_images[vehicle_id].append(file)
        
        # Select vehicles with sufficient images
        eligible_vehicles = [v_id for v_id, imgs in vehicle_to_images.items() 
                            if len(imgs) >= images_per_vehicle]
        
        if len(eligible_vehicles) == 0:
            print("No vehicles with sufficient images found")
            return
        
        selected_vehicles = random.sample(eligible_vehicles, 
                                        min(num_vehicles, len(eligible_vehicles)))
        
        # Display images for each vehicle
        for vehicle_id in selected_vehicles:
            selected_images = random.sample(vehicle_to_images[vehicle_id], 
                                          min(images_per_vehicle, len(vehicle_to_images[vehicle_id])))
            
            plt.figure(figsize=(15, 3))
            for i, img_file in enumerate(selected_images):
                img_path = os.path.join(directory_path, img_file)
                img = Image.open(img_path)
                
                # Extract camera ID
                camera_id = img_file.split('_')[1] if len(img_file.split('_')) > 1 else "Unknown"
                
                plt.subplot(1, images_per_vehicle, i+1)
                plt.imshow(img)
                plt.title(f"Camera: {camera_id}")
                plt.axis('off')
            
            plt.suptitle(f"Vehicle ID: {vehicle_id}")
            plt.tight_layout()
            plt.show()
    
    def compare_datasets(self):
        """
        Compare train, query, and test datasets
        """
        # Analyze each dataset
        train_info = self.analyze_directory(self.train_path)
        query_info = self.analyze_directory(self.query_path)
        test_info = self.analyze_directory(self.test_path)
        
        if not all([train_info, query_info, test_info]):
            print("Cannot compare datasets due to missing information")
            return
        
        # Create comparison table
        data = {
            'Metric': ['Total Images', 'Unique Vehicles', 'Unique Cameras', 
                      'Min Images/Vehicle', 'Max Images/Vehicle', 'Avg Images/Vehicle'],
            'Train': [train_info['total_images'], 
                     train_info['unique_vehicles'], 
                     train_info['unique_cameras'],
                     min(train_info['images_per_vehicle']),
                     max(train_info['images_per_vehicle']),
                     f"{np.mean(train_info['images_per_vehicle']):.2f}"],
            'Query': [query_info['total_images'], 
                     query_info['unique_vehicles'], 
                     query_info['unique_cameras'],
                     min(query_info['images_per_vehicle']),
                     max(query_info['images_per_vehicle']),
                     f"{np.mean(query_info['images_per_vehicle']):.2f}"],
            'Test': [test_info['total_images'], 
                    test_info['unique_vehicles'], 
                    test_info['unique_cameras'],
                    min(test_info['images_per_vehicle']),
                    max(test_info['images_per_vehicle']),
                    f"{np.mean(test_info['images_per_vehicle']):.2f}"]
        }
        
        # Convert to DataFrame for nice display
        df = pd.DataFrame(data)
        print("\nDataset Comparison:")
        print(df.to_string(index=False))
        
        # Check vehicle ID overlap
        train_vehicles = set(train_info['vehicle_ids'])
        query_vehicles = set(query_info['vehicle_ids'])
        test_vehicles = set(test_info['vehicle_ids'])
        
        train_query_overlap = len(train_vehicles.intersection(query_vehicles))
        train_test_overlap = len(train_vehicles.intersection(test_vehicles))
        query_test_overlap = len(query_vehicles.intersection(test_vehicles))
        
        print("\nVehicle ID Overlap:")
        print(f"Train-Query overlap: {train_query_overlap} vehicles")
        print(f"Train-Test overlap: {train_test_overlap} vehicles")
        print(f"Query-Test overlap: {query_test_overlap} vehicles")
        
        # Create Venn diagram for vehicle overlap
        plt.figure(figsize=(10, 6))
        venn3([train_vehicles, query_vehicles, test_vehicles], 
              ('Train', 'Query', 'Test'))
        plt.title('Vehicle ID Overlap Between Datasets')
        plt.show()
    
    def run_complete_analysis(self):
        """
        Run a complete analysis of the dataset
        """
        print("=" * 50)
        print("VeRi-776 Dataset Analysis")
        print("=" * 50)
        
        # Analyze train dataset
        print("\n[1] Train Dataset Analysis:")
        train_info = self.analyze_directory(self.train_path)
        if train_info:
            print(f"Total images: {train_info['total_images']}")
            print(f"Unique vehicles: {train_info['unique_vehicles']}")
            print(f"Unique cameras: {train_info['unique_cameras']}")
            
            print("\n[1.1] Vehicle Distribution:")
            self.visualize_vehicle_distribution(train_info)
            
            print("\n[1.2] Camera Distribution:")
            self.analyze_camera_distribution(self.train_path)
            
            print("\n[1.3] Image Properties:")
            img_props = self.analyze_image_properties(self.train_path)
            print(f"Resolution: {img_props['width']['mean']:.0f}x{img_props['height']['mean']:.0f} (average)")
            print(f"Aspect ratio: {img_props['aspect_ratio']['mean']:.2f} (average)")
            print(f"Color channels: {img_props['channels']}")
            
            print("\n[1.4] Sample Images:")
            self.visualize_image_samples(self.train_path)
            
            print("\n[1.5] Vehicle Examples:")
            self.analyze_vehicle_examples(self.train_path)
        
        # Analyze query dataset
        print("\n" + "=" * 50)
        print("[2] Query Dataset Analysis:")
        query_info = self.analyze_directory(self.query_path)
        if query_info:
            print(f"Total images: {query_info['total_images']}")
            print(f"Unique vehicles: {query_info['unique_vehicles']}")
            print(f"Unique cameras: {query_info['unique_cameras']}")
            
            print("\n[2.1] Vehicle Distribution:")
            self.visualize_vehicle_distribution(query_info)
            
            print("\n[2.2] Camera Distribution:")
            self.analyze_camera_distribution(self.query_path)
            
            print("\n[2.3] Sample Images:")
            self.visualize_image_samples(self.query_path)
        
        # Analyze test dataset
        print("\n" + "=" * 50)
        print("[3] Test Dataset Analysis:")
        test_info = self.analyze_directory(self.test_path)
        if test_info:
            print(f"Total images: {test_info['total_images']}")
            print(f"Unique vehicles: {test_info['unique_vehicles']}")
            print(f"Unique cameras: {test_info['unique_cameras']}")
            
            print("\n[3.1] Vehicle Distribution:")
            self.visualize_vehicle_distribution(test_info)
            
            print("\n[3.2] Camera Distribution:")
            self.analyze_camera_distribution(self.test_path)
            
            print("\n[3.3] Sample Images:")
            self.visualize_image_samples(self.test_path)
        
        # Compare datasets
        print("\n" + "=" * 50)
        print("[4] Dataset Comparison:")
        self.compare_datasets()
        
        print("\n" + "=" * 50)
        print("Analysis Complete!")
        print("=" * 50)


if __name__ == "__main__":
    # Create analyzer and run analysis
    analyzer = VeRiDatasetAnalyzer()
    analyzer.run_complete_analysis()