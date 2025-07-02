import os
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import numpy as np
from .bases import BaseImageDataset

class CustomVehicleDataset(BaseImageDataset):
    """
    Custom Vehicle Dataset for TransReID
    Using only video11 and video21 for both training and testing
    Goal: Get best_model.pth for later streaming with Kafka + Spark
    
    Strategy:
    - Train: 80% samples from video11 + video21 (cross-camera learning)
    - Query: 20% samples from video11  
    - Gallery: 20% samples from video21
    - Same domain train/test for optimal model performance
    """
    
    def __init__(self, root='./data/raw/CustomVehicleDataset', verbose=True, **kwargs):
        super(CustomVehicleDataset, self).__init__()
        self.dataset_dir = root
        
        # Only use video11 and video21
        self.videos = ['video11', 'video21']
        self.annotations = ['annotations_11.xml', 'annotations_21.xml']
        self.camera_ids = [0, 1]  # video11=cam0, video21=cam1
        
        # Parse annotations and create dataset splits
        self.train = self._process_dataset('train')
        self.query = self._process_dataset('query') 
        self.gallery = self._process_dataset('gallery')
        
        if verbose:
            print("=> CustomVehicleDataset loaded (video11 + video21 only)")
            try:
                self.print_dataset_statistics(self.train, self.query, self.gallery)
                self._print_detailed_stats()
            except Exception as e:
                print(f"Warning: Could not print statistics: {e}")
                print(f"Train: {len(self.train)}, Query: {len(self.query)}, Gallery: {len(self.gallery)}")

    def _parse_xml_annotations(self, xml_file):
        """Parse XML annotations to extract vehicle information"""
        if not os.path.exists(xml_file):
            print(f"Warning: Annotation file {xml_file} not found")
            return {}
            
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        annotations = {}
        for image in root.findall('image'):
            image_name = image.get('name')
            width = int(image.get('width'))
            height = int(image.get('height'))
            
            vehicles = []
            for box in image.findall('box'):
                label = box.get('label')
                if label in ['car', 'bicycle']:  # Focus on vehicles
                    # Get vehicle ID from attribute
                    vehicle_id_elem = box.find('attribute[@name="id"]')
                    if vehicle_id_elem is not None and vehicle_id_elem.text:
                        vehicle_id = vehicle_id_elem.text.strip()
                        if vehicle_id:  # Only process vehicles with valid IDs (not empty)
                            vehicles.append({
                                'vehicle_id': vehicle_id,
                                'label': label,
                                'bbox': [
                                    float(box.get('xtl')),
                                    float(box.get('ytl')),
                                    float(box.get('xbr')),
                                    float(box.get('ybr'))
                                ]
                            })
            
            if vehicles:
                annotations[image_name] = {
                    'width': width,
                    'height': height,
                    'vehicles': vehicles
                }
        
        return annotations

    def _extract_all_crops(self):
        """Extract all vehicle crops from video11 and video21"""
        all_crops = []
        
        for i, (video_name, annotation_file) in enumerate(zip(self.videos, self.annotations)):
            annotation_path = os.path.join(self.dataset_dir, annotation_file)
            video_dir = os.path.join(self.dataset_dir, video_name)
            camera_id = self.camera_ids[i]
            
            if not os.path.exists(video_dir):
                print(f"Warning: Video directory {video_dir} not found")
                continue
            
            # Parse annotations
            annotations = self._parse_xml_annotations(annotation_path)
            
            for image_name, annotation in annotations.items():
                image_path = os.path.join(video_dir, image_name)
                
                if os.path.exists(image_path):
                    for vehicle in annotation['vehicles']:
                        all_crops.append({
                            'image_path': image_path,
                            'vehicle_id': vehicle['vehicle_id'],
                            'bbox': vehicle['bbox'],
                            'label': vehicle['label'],
                            'video': video_name,
                            'camera_id': camera_id
                        })
        
        return all_crops

    def _find_cross_camera_vehicles(self):
        """Find vehicles that appear in BOTH video11 AND video21"""
        
        # Get vehicle IDs from each video
        video11_ids = set()
        video21_ids = set()
        
        # Extract from video11
        ann11_path = os.path.join(self.dataset_dir, 'annotations_11.xml')
        ann11 = self._parse_xml_annotations(ann11_path)
        for image_name, annotation in ann11.items():
            for vehicle in annotation['vehicles']:
                video11_ids.add(vehicle['vehicle_id'])
        
        # Extract from video21
        ann21_path = os.path.join(self.dataset_dir, 'annotations_21.xml')
        ann21 = self._parse_xml_annotations(ann21_path)
        for image_name, annotation in ann21.items():
            for vehicle in annotation['vehicles']:
                video21_ids.add(vehicle['vehicle_id'])
        
        # Find intersection (vehicles appearing in BOTH videos)
        cross_camera_vehicles = video11_ids & video21_ids
        
        print(f"\n=== CROSS-CAMERA ANALYSIS ===")
        print(f"Video11 unique vehicles: {len(video11_ids)}")
        print(f"Video21 unique vehicles: {len(video21_ids)}")
        print(f"Cross-camera vehicles: {len(cross_camera_vehicles)}")
        print(f"Cross-camera ratio: {len(cross_camera_vehicles)/max(len(video11_ids), len(video21_ids)) * 100:.1f}%")
        
        return cross_camera_vehicles

    def _create_dataset_splits(self):
        """Create train/query/gallery splits using video11 and video21"""
        
        print("Creating dataset splits using video11 + video21...")
        
        # Find cross-camera vehicles
        cross_camera_vehicles = self._find_cross_camera_vehicles()
        
        # Extract all crops
        all_crops = self._extract_all_crops()
        
        # Filter crops to only include cross-camera vehicles
        cross_camera_crops = [crop for crop in all_crops if crop['vehicle_id'] in cross_camera_vehicles]
        
        print(f"Total crops extracted: {len(all_crops)}")
        print(f"Cross-camera crops used: {len(cross_camera_crops)}")
        
        # Create vehicle ID mapping
        vehicle_id_mapping = {}
        current_pid = 0
        for vehicle_id in cross_camera_vehicles:
            vehicle_id_mapping[vehicle_id] = current_pid
            current_pid += 1
        
        # Split crops by camera
        video11_crops = [crop for crop in cross_camera_crops if crop['video'] == 'video11']
        video21_crops = [crop for crop in cross_camera_crops if crop['video'] == 'video21']
        
        # Strategy: Use 80% for training, 20% for testing
        train_data = []
        query_data = []
        gallery_data = []
        
        # Training: Use 80% of each camera's data
        train_ratio = 0.8
        
        # Group crops by vehicle_id for proper splitting
        from collections import defaultdict
        video11_by_vehicle = defaultdict(list)
        video21_by_vehicle = defaultdict(list)
        
        for crop in video11_crops:
            video11_by_vehicle[crop['vehicle_id']].append(crop)
            
        for crop in video21_crops:
            video21_by_vehicle[crop['vehicle_id']].append(crop)
        
        # Split each vehicle's data
        for vehicle_id in cross_camera_vehicles:
            pid = vehicle_id_mapping[vehicle_id]
            
            # Get crops for this vehicle from both cameras
            v11_crops = video11_by_vehicle[vehicle_id]
            v21_crops = video21_by_vehicle[vehicle_id]
            
            # Skip vehicles that don't appear in both cameras
            if len(v11_crops) == 0 or len(v21_crops) == 0:
                continue
            
            # Use percentage split: 80% train, 20% test
            # Only create test samples if vehicle has enough data
            min_samples_for_test = 3  # Need at least 3 samples to split
            
            # Split video11 crops
            if len(v11_crops) >= min_samples_for_test:
                v11_train_count = int(len(v11_crops) * train_ratio)
                v11_train = v11_crops[:v11_train_count]
                v11_test = v11_crops[v11_train_count:]
            else:
                # Too few samples, use all for training
                v11_train = v11_crops
                v11_test = []
            
            # Split video21 crops  
            if len(v21_crops) >= min_samples_for_test:
                v21_train_count = int(len(v21_crops) * train_ratio)
                v21_train = v21_crops[:v21_train_count]
                v21_test = v21_crops[v21_train_count:]
            else:
                # Too few samples, use all for training
                v21_train = v21_crops
                v21_test = []
            
            # Add to training (both cameras)
            for crop in v11_train:
                train_data.append([
                    crop['image_path'],
                    pid,
                    0,  # camera_id for video11
                    0,  # trackid
                    crop['bbox']
                ])
                
            for crop in v21_train:
                train_data.append([
                    crop['image_path'],
                    pid,
                    1,  # camera_id for video21
                    0,  # trackid
                    crop['bbox']
                ])
            
            # Add to query (video11 test data)
            for crop in v11_test:
                query_data.append([
                    crop['image_path'],
                    pid,
                    0,  # camera_id
                    0,  # trackid
                    crop['bbox']
                ])
            
            # Add to gallery (video21 test data)
            for crop in v21_test:
                gallery_data.append([
                    crop['image_path'],
                    pid,
                    1,  # camera_id
                    0,  # trackid
                    crop['bbox']
                ])
        
        print(f"\n=== DATASET SPLITS SUMMARY ===")
        print(f"Cross-camera vehicles used: {len(cross_camera_vehicles)}")
        print(f"Training samples: {len(train_data)} (both cameras)")
        print(f"Query samples: {len(query_data)} (video11)")
        print(f"Gallery samples: {len(gallery_data)} (video21)")
        print(f"Total samples: {len(train_data) + len(query_data) + len(gallery_data)}")
        
        return train_data, query_data, gallery_data

    def _print_detailed_stats(self):
        """Print detailed statistics"""
        if not hasattr(self, '_dataset_cache'):
            return
            
        train_data, query_data, gallery_data = self._dataset_cache
        
        # Analyze PIDs and cameras
        train_pids = set([item[1] for item in train_data])
        query_pids = set([item[1] for item in query_data])
        gallery_pids = set([item[1] for item in gallery_data])
        
        train_cams = set([item[2] for item in train_data])
        query_cams = set([item[2] for item in query_data])
        gallery_cams = set([item[2] for item in gallery_data])
        
        print(f"\n=== DETAILED STATISTICS ===")
        print(f"Train: {len(train_pids)} vehicles, cameras: {train_cams}")
        print(f"Query: {len(query_pids)} vehicles, cameras: {query_cams}")
        print(f"Gallery: {len(gallery_pids)} vehicles, cameras: {gallery_cams}")
        
        # Check overlap (should be perfect for same vehicles)
        if len(query_pids) > 0:
            query_gallery_overlap = len(query_pids & gallery_pids)
            print(f"Query-Gallery vehicle overlap: {query_gallery_overlap}/{len(query_pids)} = {query_gallery_overlap/len(query_pids)*100:.1f}%")

    def _process_dataset(self, mode):
        """Process dataset based on mode (train/query/gallery)"""
        if not hasattr(self, '_dataset_cache'):
            self._dataset_cache = self._create_dataset_splits()
        
        train_data, query_data, gallery_data = self._dataset_cache
        
        if mode == 'train':
            return train_data
        elif mode == 'query':
            return query_data
        elif mode == 'gallery':
            return gallery_data
        else:
            raise ValueError(f"Invalid mode: {mode}")