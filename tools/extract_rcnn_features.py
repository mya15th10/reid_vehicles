#!/usr/bin/env python3
"""
R-CNN Feature Extraction Pipeline
Extract features from video frames using CVAT annotations
"""

import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import logging
from tqdm import tqdm
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RCNNFeatureExtractor:
    """Extract 256-dim features from vehicle crops using R-CNN + projection"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained Faster R-CNN
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        
        # Extract backbone for feature extraction
        self.backbone = self.model.backbone
        
        # FIXED: Add projection layer to get 256-dim features
        self.feature_projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        ).to(self.device)
        
        # Image preprocessing
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("R-CNN feature extractor initialized with 256-dim output")
    
    def extract_features(self, image_crop):
        """Extract 256-dim features from image crop"""
        
        if len(image_crop.shape) == 3:
            image_tensor = self.transform(image_crop).unsqueeze(0).to(self.device)
        else:
            logger.warning(f"Unexpected image shape: {image_crop.shape}")
            return None
        
        with torch.no_grad():
            # Extract 2048-dim features using backbone
            features = self.backbone(image_tensor)
            
            if isinstance(features, dict):
                feature_key = max(features.keys())
                feature_map = features[feature_key]
            else:
                feature_map = features
            
            # Global average pooling
            pooled_features = F.adaptive_avg_pool2d(feature_map, (1, 1))
            feature_vector_2048 = pooled_features.flatten(1)  # [1, 2048]
            
            # FIXED: Project to 256 dimensions
            feature_vector_256 = self.feature_projector(feature_vector_2048)  # [1, 256]
            
            return feature_vector_256.cpu().numpy()[0]  # Return as numpy array
class VideoProcessor:
    """Process videos and extract frames"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self._open_video()
    
    def _open_video(self):
        """Open video file"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video: {self.video_path}, FPS: {self.fps}, Frames: {self.total_frames}")
    
    def get_frame(self, frame_number: int):
        """Get specific frame from video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            logger.warning(f"Cannot read frame {frame_number} from {self.video_path}")
            return None
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    def __del__(self):
        if self.cap:
            self.cap.release()

class CVATAnnotationParser:
    """Parse CVAT XML annotations"""
    
    def __init__(self):
        self.annotations = []
    
    def parse_xml(self, xml_path: str, camera_id: int, video_processor: VideoProcessor):
        """Parse XML file and extract vehicle annotations"""
        
        logger.info(f"Parsing {xml_path} for camera {camera_id}")
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        detections = []
        
        for image in tqdm(root.findall('image'), desc=f"Processing Camera {camera_id}"):
            frame_id = int(image.get('id'))
            frame_name = image.get('name')
            
            for box in image.findall('box'):
                label = box.get('label')
                
                # Skip non-vehicle labels
                if label not in ['car', 'bicycle', 'truck', 'bus', 'motorcycle']:
                    continue
                
                # Get vehicle ID
                id_attr = box.find('attribute[@name="id"]')
                if id_attr is None or not id_attr.text or not id_attr.text.strip():
                    continue
                
                vehicle_id = id_attr.text.strip()
                
                # Extract bounding box coordinates
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                
                detection = {
                    'vehicle_id': vehicle_id,
                    'camera_id': camera_id,
                    'frame_id': frame_id,
                    'frame_name': frame_name,
                    'label': label,
                    'bbox': (xtl, ytl, xbr, ybr),
                }
                
                detections.append(detection)
        
        logger.info(f"Parsed {len(detections)} detections from camera {camera_id}")
        return detections

class VehicleReIDDatasetBuilder:
    """FIXED: Build ReID dataset with proper splits and consistent labels"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = RCNNFeatureExtractor()
        self.video_processors = {}
        
        # FIXED: Global vehicle ID mapping for consistency
        self.global_vehicle_mapping = {}
        self.next_pid = 0
        
    def get_consistent_pid(self, vehicle_id: str) -> int:
        """Get consistent PID for vehicle across all splits"""
        if vehicle_id not in self.global_vehicle_mapping:
            self.global_vehicle_mapping[vehicle_id] = self.next_pid
            self.next_pid += 1
        return self.global_vehicle_mapping[vehicle_id]
    
    def create_reid_splits(self, features_data):
        """FIXED: Create proper ReID splits with cross-camera validation"""
        
        # Group by vehicle ID
        by_vehicle = defaultdict(list)
        for item in features_data:
            by_vehicle[item['vehicle_id']].append(item)
        
        train_data = []
        query_data = []
        gallery_data = []
        
        # Only use vehicles that appear in BOTH cameras
        cross_camera_vehicles = []
        for vehicle_id, detections in by_vehicle.items():
            cameras = set(d['camera_id'] for d in detections)
            if len(cameras) >= 2:  # Appears in multiple cameras
                cross_camera_vehicles.append(vehicle_id)
        
        logger.info(f"Found {len(cross_camera_vehicles)} cross-camera vehicles")
        
        # Split cross-camera vehicles
        for vehicle_id in cross_camera_vehicles:
            detections = by_vehicle[vehicle_id]
            
            # Group by camera
            by_camera = defaultdict(list)
            for d in detections:
                by_camera[d['camera_id']].append(d)
            
            # For each camera, use 70% for training, 30% for test
            for camera_id, cam_detections in by_camera.items():
                np.random.shuffle(cam_detections)
                train_split = int(0.7 * len(cam_detections))
                
                train_data.extend(cam_detections[:train_split])
                test_detections = cam_detections[train_split:]
                
                # Split test data: even cameras -> query, odd cameras -> gallery
                if camera_id % 2 == 0:
                    query_data.extend(test_detections)
                else:
                    gallery_data.extend(test_detections)
        
        # FIXED: Apply consistent PID mapping
        for split_data in [train_data, query_data, gallery_data]:
            for item in split_data:
                item['pid'] = self.get_consistent_pid(item['vehicle_id'])
        
        # Validate splits
        train_pids = set(item['pid'] for item in train_data)
        query_pids = set(item['pid'] for item in query_data)
        gallery_pids = set(item['pid'] for item in gallery_data)
        
        overlap = query_pids.intersection(gallery_pids)
        logger.info(f"Query-Gallery PID overlap: {len(overlap)} (should be > 0 for ReID)")
        
        # Save splits
        splits = {
            'train': train_data,
            'query': query_data,
            'gallery': gallery_data
        }
        
        for split_name, data in splits.items():
            split_file = self.output_dir / f'{split_name}_features.pkl'
            with open(split_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved {len(data)} {split_name} features to {split_file}")
        
        # Save PID mapping
        mapping_file = self.output_dir / 'pid_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump(self.global_vehicle_mapping, f, indent=2)
        
        return splits
    
    def save_features(self, features_data):
        """Save extracted features to files"""
        
        # Save complete dataset
        features_file = self.output_dir / 'vehicle_features.pkl'
        with open(features_file, 'wb') as f:
            pickle.dump(features_data, f)
        logger.info(f"Saved {len(features_data)} features to {features_file}")
        
        # Create train/query/gallery splits
        self.create_reid_splits(features_data)
        
        # Save metadata
        metadata = {
            'total_features': len(features_data),
            'feature_dimension': features_data[0]['feature_dim'] if features_data else 0,
            'statistics': dict(self.stats),
            'cameras': list(self.video_processors.keys()),
            'vehicle_types': list(set(item['label'] for item in features_data))
        }
        
        with open(self.output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def create_reid_splits(self, features_data):
        """Create train/query/gallery splits for ReID"""
        
        # Group by vehicle ID (not camera!)
        by_vehicle = defaultdict(list)
        for item in features_data:
            by_vehicle[item['vehicle_id']].append(item)
        
        train_data = []
        query_data = []
        gallery_data = []
        
        # For each vehicle, split its detections across train/query/gallery
        for vehicle_id, detections in by_vehicle.items():
            # Group by camera
            cam1_detections = [d for d in detections if d['camera_id'] == 1]
            cam2_detections = [d for d in detections if d['camera_id'] == 2]
            
            # Only use vehicles that appear in BOTH cameras
            if len(cam1_detections) > 0 and len(cam2_detections) > 0:
                # 80% of detections for training
                train_count_cam1 = max(1, int(0.8 * len(cam1_detections)))
                train_count_cam2 = max(1, int(0.8 * len(cam2_detections)))
                
                # Add to train set
                train_data.extend(cam1_detections[:train_count_cam1])
                train_data.extend(cam2_detections[:train_count_cam2])
                
                # Remaining for query/gallery (ensure cross-camera testing)
                remaining_cam1 = cam1_detections[train_count_cam1:]
                remaining_cam2 = cam2_detections[train_count_cam2:]
                
                if remaining_cam1 and remaining_cam2:
                    query_data.extend(remaining_cam1)    # Cam1 as queries
                    gallery_data.extend(remaining_cam2)  # Cam2 as gallery
        
        # ADD THIS MISSING PART:
        # Save splits
        splits = {
            'train': train_data,
            'query': query_data,
            'gallery': gallery_data
        }
        
        for split_name, data in splits.items():
            split_file = self.output_dir / f'{split_name}_features.pkl'
            with open(split_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved {len(data)} {split_name} features to {split_file}")

def main():
    """Main extraction pipeline"""
    
    # Configuration
    config = {
        'xml_files': {
            'data/raw/CustomVehicleDataset/annotations_11.xml': 1,  # Camera 1
            'data/raw/CustomVehicleDataset/annotations_21.xml': 2,  # Camera 2
        },
        'video_files': {
            1: 'data/raw/CustomVehicleDataset/video11.MOV',  # Camera 1
            2: 'data/raw/CustomVehicleDataset/video21.MOV',  # Camera 2
        },
        'output_dir': 'data/processed/CustomVehicleDataset/features'
    }
    
    # Create dataset builder
    builder = VehicleReIDDatasetBuilder(config['output_dir'])
    
    # Add video processors
    for camera_id, video_path in config['video_files'].items():
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return
        builder.add_video(video_path, camera_id)
    
    # Process annotations and extract features
    logger.info("Starting R-CNN feature extraction...")
    features_data = builder.process_annotations(config['xml_files'])
    
    logger.info("Feature extraction completed!")
    logger.info(f"Output directory: {config['output_dir']}")
    logger.info("Files created:")
    logger.info("  - vehicle_features.pkl (all features)")
    logger.info("  - train_features.pkl (training set)")
    logger.info("  - query_features.pkl (query set)")  
    logger.info("  - gallery_features.pkl (gallery set)")
    logger.info("  - dataset_metadata.json (statistics)")

if __name__ == "__main__":
    main()