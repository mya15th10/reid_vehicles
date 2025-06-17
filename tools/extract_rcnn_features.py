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
    """Extract R-CNN features from vehicle crops"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained Faster R-CNN
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        
        # Extract backbone for feature extraction
        self.backbone = self.model.backbone
        
        # Image preprocessing
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),  # Standard input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("R-CNN feature extractor initialized")
    
    def extract_features(self, image_crop):
        """Extract 2048-dim features from image crop"""
        
        # Preprocess image
        if len(image_crop.shape) == 3:
            image_tensor = self.transform(image_crop).unsqueeze(0).to(self.device)
        else:
            logger.warning(f"Unexpected image shape: {image_crop.shape}")
            return None
        
        with torch.no_grad():
            # Extract features using backbone
            features = self.backbone(image_tensor)
            
            # Get features from the last layer (typically 'pool' or '3')
            if isinstance(features, dict):
                # Use the highest resolution feature map
                feature_key = max(features.keys())
                feature_map = features[feature_key]
            else:
                feature_map = features
            
            # Global average pooling to get fixed-size features
            pooled_features = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
            feature_vector = pooled_features.flatten(1)  # [1, feature_dim]
            
            return feature_vector.cpu().numpy()[0]  # Return as numpy array

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
    """Build ReID dataset from R-CNN features"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = RCNNFeatureExtractor()
        self.video_processors = {}
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'vehicles_by_camera': defaultdict(set),
            'cross_camera_vehicles': set()
        }
    
    def add_video(self, video_path: str, camera_id: int):
        """Add video processor for camera"""
        self.video_processors[camera_id] = VideoProcessor(video_path)
        logger.info(f"Added video processor for camera {camera_id}: {video_path}")
    
    def process_annotations(self, xml_files: Dict[str, int]):
        """Process all XML annotations and extract features"""
        
        all_detections = []
        parser = CVATAnnotationParser()
        
        # Parse all XML files
        for xml_path, camera_id in xml_files.items():
            if camera_id not in self.video_processors:
                logger.error(f"No video processor for camera {camera_id}")
                continue
            
            detections = parser.parse_xml(xml_path, camera_id, self.video_processors[camera_id])
            all_detections.extend(detections)
        
        logger.info(f"Total detections to process: {len(all_detections)}")
        self.stats['total_detections'] = len(all_detections)
        
        # Extract features for each detection
        features_data = []
        
        for detection in tqdm(all_detections, desc="Extracting R-CNN features"):
            feature_data = self.extract_single_feature(detection)
            if feature_data is not None:
                features_data.append(feature_data)
                self.stats['successful_extractions'] += 1
                
                # Update statistics
                vehicle_id = detection['vehicle_id']
                camera_id = detection['camera_id']
                self.stats['vehicles_by_camera'][camera_id].add(vehicle_id)
            else:
                self.stats['failed_extractions'] += 1
        
        # Find cross-camera vehicles
        camera_1_vehicles = self.stats['vehicles_by_camera'][1]
        camera_2_vehicles = self.stats['vehicles_by_camera'][2]
        self.stats['cross_camera_vehicles'] = camera_1_vehicles.intersection(camera_2_vehicles)
        
        logger.info(f"Successfully extracted {len(features_data)} feature vectors")
        
        # Save features
        self.save_features(features_data)
        self.print_statistics()
        
        return features_data
    
    def extract_single_feature(self, detection):
        """Extract R-CNN features for single detection"""
        
        camera_id = detection['camera_id']
        frame_id = detection['frame_id']
        bbox = detection['bbox']
        
        # Get video processor
        video_processor = self.video_processors[camera_id]
        
        # Get frame
        frame = video_processor.get_frame(frame_id)
        if frame is None:
            return None
        
        # Crop vehicle region
        xtl, ytl, xbr, ybr = bbox
        xtl, ytl, xbr, ybr = int(xtl), int(ytl), int(xbr), int(ybr)
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        xtl = max(0, min(xtl, w-1))
        ytl = max(0, min(ytl, h-1))
        xbr = max(xtl+1, min(xbr, w))
        ybr = max(ytl+1, min(ybr, h))
        
        vehicle_crop = frame[ytl:ybr, xtl:xbr]
        
        if vehicle_crop.size == 0:
            logger.warning(f"Empty crop for detection: {detection}")
            return None
        
        # Extract R-CNN features
        features = self.feature_extractor.extract_features(vehicle_crop)
        if features is None:
            return None
        
        # Create feature data entry
        feature_data = {
            'vehicle_id': detection['vehicle_id'],
            'camera_id': detection['camera_id'],
            'frame_id': detection['frame_id'],
            'frame_name': detection['frame_name'],
            'label': detection['label'],
            'bbox': detection['bbox'],
            'features': features,
            'feature_dim': len(features)
        }
        
        return feature_data
    
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
        
        # Group by vehicle ID
        by_vehicle = defaultdict(list)
        for item in features_data:
            by_vehicle[item['vehicle_id']].append(item)
        
        # Separate cross-camera and single-camera vehicles
        cross_camera_vehicles = []
        single_camera_vehicles = []
        
        for vehicle_id, detections in by_vehicle.items():
            cameras = set(item['camera_id'] for item in detections)
            if len(cameras) > 1:
                cross_camera_vehicles.append((vehicle_id, detections))
            else:
                single_camera_vehicles.append((vehicle_id, detections))
        
        logger.info(f"Cross-camera vehicles: {len(cross_camera_vehicles)}")
        logger.info(f"Single-camera vehicles: {len(single_camera_vehicles)}")
        
        # Create splits
        train_data = []
        query_data = []
        gallery_data = []
        
        # Use 70% of cross-camera vehicles for training
        train_cross_camera = cross_camera_vehicles[:int(0.7 * len(cross_camera_vehicles))]
        test_cross_camera = cross_camera_vehicles[int(0.7 * len(cross_camera_vehicles)):]
        
        # Training set: 70% cross-camera + all single-camera
        for vehicle_id, detections in train_cross_camera:
            train_data.extend(detections)
        
        for vehicle_id, detections in single_camera_vehicles:
            train_data.extend(detections)
        
        # Test set: remaining cross-camera vehicles
        for vehicle_id, detections in test_cross_camera:
            # Split by camera for query/gallery
            cam1_detections = [d for d in detections if d['camera_id'] == 1]
            cam2_detections = [d for d in detections if d['camera_id'] == 2]
            
            if cam1_detections and cam2_detections:
                query_data.extend(cam1_detections)  # Camera 1 as query
                gallery_data.extend(cam2_detections)  # Camera 2 as gallery
        
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
    
    def print_statistics(self):
        """Print extraction statistics"""
        
        logger.info("="*60)
        logger.info("FEATURE EXTRACTION STATISTICS")
        logger.info("="*60)
        logger.info(f"Total detections: {self.stats['total_detections']}")
        logger.info(f"Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"Failed extractions: {self.stats['failed_extractions']}")
        
        for camera_id, vehicles in self.stats['vehicles_by_camera'].items():
            logger.info(f"Camera {camera_id}: {len(vehicles)} unique vehicles")
        
        logger.info(f"Cross-camera vehicles: {len(self.stats['cross_camera_vehicles'])}")
        logger.info("="*60)

def main():
    """Main extraction pipeline"""
    
    # Configuration
    config = {
        'xml_files': {
            'data/raw/CustomVehicleDataset/annotations_11.xml': 1,  # Camera 1
            'data/raw/CustomVehicleDataset/annotations_21.xml': 2,  # Camera 2
        },
        'video_files': {
            1: 'data/raw/CustomVehicleDataset/video1(1).MOV',  # Camera 1
            2: 'data/raw/CustomVehicleDataset/video2(1).MOV',  # Camera 2
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