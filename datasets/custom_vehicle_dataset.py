import os
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import numpy as np
from .bases import BaseImageDataset

class CustomVehicleDataset(BaseImageDataset):
    """
    Custom Vehicle Dataset for TransReID
    Dataset structure:
    - video11/ (frames)
    - video12/ (frames) 
    - video21/ (frames)
    - video22/ (frames)
    - annotations_11.xml
    - annotations_12.xml
    - annotations_21.xml
    - annotations_22.xml
    """
    
    def __init__(self, root='./data/raw/CustomVehicleDataset', verbose=True, **kwargs):
        super(CustomVehicleDataset, self).__init__()
        self.dataset_dir = root
        self.train_dir = root
        self.query_dir = root
        self.gallery_dir = root
        
        # Session mapping
        self.session_config = {
            'session1': {
                'videos': ['video11', 'video12'],
                'annotations': ['annotations_11.xml', 'annotations_12.xml']
            },
            'session2': {
                'videos': ['video21', 'video22'], 
                'annotations': ['annotations_21.xml', 'annotations_22.xml']
            }
        }
        
        # Parse all annotations and extract vehicle crops
        self.train = self._process_dataset('train')
        self.query = self._process_dataset('query') 
        self.gallery = self._process_dataset('gallery')
        
        if verbose:
            print("=> CustomVehicleDataset loaded")
            try:
                self.print_dataset_statistics(self.train, self.query, self.gallery)
            except Exception as e:
                print(f"Warning: Could not print statistics: {e}")
                print(f"Train: {len(self.train)}, Query: {len(self.query)}, Gallery: {len(self.gallery)}")

    def _parse_xml_annotations(self, xml_file):
        """Parse XML annotations to extract vehicle information"""
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
                        if vehicle_id:  # Only process vehicles with valid IDs
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

    def _extract_vehicle_crops(self, session_name):
        """Extract vehicle crops from frames using annotations"""
        crops = []
        session = self.session_config[session_name]
        
        for video_name, annotation_file in zip(session['videos'], session['annotations']):
            annotation_path = os.path.join(self.dataset_dir, annotation_file)
            video_dir = os.path.join(self.dataset_dir, video_name)
            
            if not os.path.exists(annotation_path):
                print(f"Warning: Annotation file {annotation_path} not found")
                continue
                
            if not os.path.exists(video_dir):
                print(f"Warning: Video directory {video_dir} not found")
                continue
            
            # Parse annotations
            annotations = self._parse_xml_annotations(annotation_path)
            
            for image_name, annotation in annotations.items():
                image_path = os.path.join(video_dir, image_name)
                
                if os.path.exists(image_path):
                    for vehicle in annotation['vehicles']:
                        # Create unique identifier
                        vehicle_id = f"{session_name}_{video_name}_{vehicle['vehicle_id']}"
                        
                        crops.append({
                            'image_path': image_path,
                            'vehicle_id': vehicle_id,
                            'bbox': vehicle['bbox'],
                            'label': vehicle['label'],
                            'session': session_name,
                            'video': video_name,
                            'original_id': vehicle['vehicle_id']
                        })
        
        return crops

    def _create_dataset_splits(self):
        """Create train/query/gallery splits for multi-camera vehicle re-ID"""
        # Extract crops from both sessions
        session1_crops = self._extract_vehicle_crops('session1')  # video11, video12
        session2_crops = self._extract_vehicle_crops('session2')  # video21, video22
        
        train_data = []
        query_data = []
        gallery_data = []
        
        # Vehicle ID mapping for consistent IDs across cameras
        vehicle_id_mapping = {}
        current_pid = 0
        
        # Strategy for multi-camera re-ID:
        # Train: Camera 1 (video11 + video12) 
        # Query: Camera 2, Time 1 (video21)
        # Gallery: Camera 2, Time 2 (video22)
        
        # Process Camera 1 data for training (video11 + video12)
        camera1_crops = [crop for crop in session1_crops]
        
        for crop in camera1_crops:
            original_id = crop['original_id']
            if original_id not in vehicle_id_mapping:
                vehicle_id_mapping[original_id] = current_pid
                current_pid += 1
            
            pid = vehicle_id_mapping[original_id]
            camid = 0 if crop['video'] == 'video11' else 1  # Different times, same camera
            
            train_data.append([
                crop['image_path'],
                pid,
                camid,
                0,  # trackid/viewid
                crop['bbox']
            ])
        
        # Process Camera 2 data for query/gallery (video21, video22)
        camera2_crops = [crop for crop in session2_crops]
        
        video21_crops = [crop for crop in camera2_crops if crop['video'] == 'video21']
        video22_crops = [crop for crop in camera2_crops if crop['video'] == 'video22']
        
        # Query: video21 (Camera 2, Time 1)
        for crop in video21_crops:
            original_id = crop['original_id']
            if original_id in vehicle_id_mapping:  # Only use vehicles seen in training
                pid = vehicle_id_mapping[original_id]
                camid = 2  # Camera 2
                
                query_data.append([
                    crop['image_path'],
                    pid,
                    camid,
                    0,
                    crop['bbox']
                ])
        
        # Gallery: video22 (Camera 2, Time 2)  
        for crop in video22_crops:
            original_id = crop['original_id']
            if original_id in vehicle_id_mapping:  # Only use vehicles seen in training
                pid = vehicle_id_mapping[original_id]
                camid = 3  # Camera 2, different time
                
                gallery_data.append([
                    crop['image_path'],
                    pid,
                    camid,
                    0,
                    crop['bbox']
                ])
        
        print(f"Multi-camera split created:")
        print(f"- Training (Camera 1): {len(train_data)} samples")
        print(f"- Query (Camera 2, Time 1): {len(query_data)} samples") 
        print(f"- Gallery (Camera 2, Time 2): {len(gallery_data)} samples")
        
        return train_data, query_data, gallery_data

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

    def _crop_vehicle(self, image_path, bbox):
        """Crop vehicle from image using bounding box"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure bbox is within image bounds
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            cropped = image[y1:y2, x1:x2]
            return cropped
            
        except Exception as e:
            print(f"Error cropping vehicle from {image_path}: {e}")
            return None