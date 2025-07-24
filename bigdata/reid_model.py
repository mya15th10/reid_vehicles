"""
Vehicle Re-Identification Model using pretrained models
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import pickle
import os
from collections import defaultdict
import logging

# Use manual cosine similarity to avoid sklearn issues
def cosine_similarity_manual(a, b):
    """Calculate cosine similarity between two vectors"""
    a = a.flatten()
    b = b.flatten()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

logger = logging.getLogger(__name__)

class VehicleReIDModel:
    """Vehicle Re-Identification model using pretrained ResNet"""
    
    def __init__(self, model_name: str = 'resnet50', pretrained: bool = True, 
                 feature_dim: int = 2048, device: str = 'auto'):
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.device = self._get_device(device)
        
        # Initialize model
        self.model = self._build_model(pretrained)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize transforms
        self.transform = self._get_transforms()
        
        # Feature gallery for known vehicles
        self.feature_gallery = {}  # vehicle_id -> [features_list]
        self.vehicle_metadata = {}  # vehicle_id -> metadata
        
        logger.info(f"VehicleReIDModel initialized with {model_name} on {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for computation"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')  # Apple Silicon
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _build_model(self, pretrained: bool) -> nn.Module:
        """Build the feature extraction model"""
        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == 'resnet101':
            model = models.resnet101(pretrained=pretrained)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == 'mobilenet_v3':
            model = models.mobilenet_v3_large(pretrained=pretrained)
            # Remove classifier and get features
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, vehicle_crop: np.ndarray) -> np.ndarray:
        """Extract features from a vehicle crop"""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return np.zeros(self.feature_dim)
        
        try:
            # Preprocess image
            if len(vehicle_crop.shape) == 3 and vehicle_crop.shape[2] == 3:
                # Convert BGR to RGB for torchvision
                vehicle_crop = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            input_tensor = self.transform(vehicle_crop).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.view(features.size(0), -1)  # Flatten
                features = features.cpu().numpy().flatten()
                
                # L2 normalize features
                features = features / (np.linalg.norm(features) + 1e-8)
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros(self.feature_dim)
    
    def extract_features_batch(self, vehicle_crops: List[np.ndarray]) -> List[np.ndarray]:
        """Extract features from multiple vehicle crops in batch for better efficiency"""
        if not vehicle_crops:
            return []
        
        try:
            # Preprocess all images
            input_tensors = []
            valid_indices = []
            
            for i, vehicle_crop in enumerate(vehicle_crops):
                if vehicle_crop is None or vehicle_crop.size == 0:
                    continue
                
                try:
                    # Preprocess image
                    if len(vehicle_crop.shape) == 3 and vehicle_crop.shape[2] == 3:
                        # Convert BGR to RGB for torchvision
                        vehicle_crop = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
                    
                    # Apply transforms
                    input_tensor = self.transform(vehicle_crop)
                    input_tensors.append(input_tensor)
                    valid_indices.append(i)
                    
                except Exception as e:
                    logger.error(f"Error preprocessing crop {i}: {e}")
                    continue
            
            if not input_tensors:
                return [np.zeros(self.feature_dim) for _ in vehicle_crops]
            
            # Stack tensors and move to device
            batch_tensor = torch.stack(input_tensors).to(self.device)
            
            # Extract features in batch
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
                batch_features = batch_features.view(batch_features.size(0), -1)  # Flatten
                batch_features = batch_features.cpu().numpy()
                
                # L2 normalize features
                batch_features = batch_features / (np.linalg.norm(batch_features, axis=1, keepdims=True) + 1e-8)
            
            # Create result list with correct ordering
            result_features = []
            batch_idx = 0
            
            for i in range(len(vehicle_crops)):
                if i in valid_indices:
                    result_features.append(batch_features[batch_idx])
                    batch_idx += 1
                else:
                    result_features.append(np.zeros(self.feature_dim))
            
            return result_features
            
        except Exception as e:
            logger.error(f"Error extracting batch features: {e}")
            return [np.zeros(self.feature_dim) for _ in vehicle_crops]
    
    def add_to_gallery(self, vehicle_id: str, features: np.ndarray, 
                      metadata: Dict = None):
        """Add vehicle features to the gallery"""
        if vehicle_id not in self.feature_gallery:
            self.feature_gallery[vehicle_id] = []
            self.vehicle_metadata[vehicle_id] = metadata or {}
        
        self.feature_gallery[vehicle_id].append(features)
        
        # Keep only the last N features to avoid memory issues
        max_features = 10
        if len(self.feature_gallery[vehicle_id]) > max_features:
            self.feature_gallery[vehicle_id] = self.feature_gallery[vehicle_id][-max_features:]
    
    def find_matches(self, query_features: np.ndarray, threshold: float = 0.7,
                    top_k: int = 5) -> List[Tuple[str, float]]:
        """Find matching vehicles in the gallery"""
        if len(self.feature_gallery) == 0:
            return []
        
        matches = []
        
        for vehicle_id, feature_list in self.feature_gallery.items():
            if len(feature_list) == 0:
                continue
            
            # Calculate similarity with all stored features for this vehicle
            similarities = []
            for stored_features in feature_list:
                sim = cosine_similarity_manual(
                    query_features,
                    stored_features
                )
                similarities.append(sim)
            
            # Use the maximum similarity
            max_similarity = max(similarities)
            
            if max_similarity >= threshold:
                matches.append((vehicle_id, max_similarity))
        
        # Sort by similarity and return top_k
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]
    
    def save_gallery(self, filepath: str):
        """Save the feature gallery to disk"""
        gallery_data = {
            'feature_gallery': self.feature_gallery,
            'vehicle_metadata': self.vehicle_metadata,
            'model_name': self.model_name,
            'feature_dim': self.feature_dim
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(gallery_data, f)
        
        logger.info(f"Gallery saved to {filepath}")
    
    def load_gallery(self, filepath: str):
        """Load the feature gallery from disk"""
        if not os.path.exists(filepath):
            logger.warning(f"Gallery file {filepath} not found")
            return
        
        with open(filepath, 'rb') as f:
            gallery_data = pickle.load(f)
        
        self.feature_gallery = gallery_data.get('feature_gallery', {})
        self.vehicle_metadata = gallery_data.get('vehicle_metadata', {})
        
        logger.info(f"Gallery loaded from {filepath}")
    
    def get_gallery_stats(self) -> Dict:
        """Get statistics about the current gallery"""
        total_vehicles = len(self.feature_gallery)
        total_features = sum(len(features) for features in self.feature_gallery.values())
        
        camera_counts = defaultdict(int)
        vehicle_type_counts = defaultdict(int)
        
        for vehicle_id, metadata in self.vehicle_metadata.items():
            if 'camera_id' in metadata:
                camera_counts[metadata['camera_id']] += 1
            if 'vehicle_type' in metadata:
                vehicle_type_counts[metadata['vehicle_type']] += 1
        
        return {
            'total_vehicles': total_vehicles,
            'total_features': total_features,
            'cameras': dict(camera_counts),
            'vehicle_types': dict(vehicle_type_counts)
        }

class CrossCameraReID:
    """Cross-camera vehicle re-identification system"""
    
    def __init__(self, reid_model: VehicleReIDModel, similarity_threshold: float = 0.7):
        self.reid_model = reid_model
        self.similarity_threshold = similarity_threshold
        
        # Track cross-camera matches
        self.cross_camera_matches = {}  # (camera1_id, camera2_id) -> [(vehicle1, vehicle2, confidence)]
        self.global_vehicle_ids = {}    # local_vehicle_id -> global_vehicle_id
        self.next_global_id = 1
        
        logger.info("CrossCameraReID system initialized")
    
    def process_vehicle(self, vehicle_crop: np.ndarray, vehicle_id: str, 
                       camera_id: int, vehicle_type: str, frame_id: int) -> str:
        """Process a vehicle and return global ID"""
        # Extract features
        features = self.reid_model.extract_features(vehicle_crop)
        
        # Create metadata
        metadata = {
            'camera_id': camera_id,
            'vehicle_type': vehicle_type,
            'first_seen_frame': frame_id,
            'local_id': vehicle_id
        }
        
        # Check if this vehicle already has a global ID assigned
        if vehicle_id in self.global_vehicle_ids:
            global_id = self.global_vehicle_ids[vehicle_id]
            # Update gallery with new observation
            self.reid_model.add_to_gallery(vehicle_id, features, metadata)
            return global_id
        
        # New vehicle - look for cross-camera matches first
        matches = self.reid_model.find_matches(features, self.similarity_threshold)
        
        if matches:
            # Found potential match - get the best match
            best_match_id, confidence = matches[0]
            
            # Get metadata of the matched vehicle to check if it's from a different camera
            matched_metadata = self.reid_model.vehicle_metadata.get(best_match_id, {})
            matched_camera_id = matched_metadata.get('camera_id', -1)
            
            # Only consider as cross-camera match if from different camera
            if matched_camera_id != camera_id and matched_camera_id != -1:
                # Get the global ID of the matched vehicle
                if best_match_id in self.global_vehicle_ids:
                    global_id = self.global_vehicle_ids[best_match_id]
                else:
                    # Matched vehicle doesn't have global ID yet - assign one
                    global_id = f"global_{self.next_global_id}"
                    self.next_global_id += 1
                    self.global_vehicle_ids[best_match_id] = global_id
                
                # Assign same global ID to current vehicle
                self.global_vehicle_ids[vehicle_id] = global_id
                
                # Record the cross-camera match
                match_key = tuple(sorted([matched_camera_id, camera_id]))
                
                if match_key not in self.cross_camera_matches:
                    self.cross_camera_matches[match_key] = []
                
                self.cross_camera_matches[match_key].append({
                    'vehicle1': best_match_id,
                    'vehicle2': vehicle_id,
                    'confidence': confidence,
                    'frame_id': frame_id,
                    'global_id': global_id
                })
                
                logger.info(f"Cross-camera match found: {vehicle_id} -> {best_match_id} "
                           f"(confidence: {confidence:.3f}, global_id: {global_id})")
                
            else:
                # Match is from same camera or invalid - assign new global ID
                global_id = f"global_{self.next_global_id}"
                self.next_global_id += 1
                self.global_vehicle_ids[vehicle_id] = global_id
                
                logger.debug(f"Same-camera match ignored: {vehicle_id} -> {best_match_id}, "
                           f"assigning new global_id: {global_id}")
        else:
            # No match found - assign new global ID
            global_id = f"global_{self.next_global_id}"
            self.next_global_id += 1
            self.global_vehicle_ids[vehicle_id] = global_id
            
            logger.debug(f"New vehicle detected: {vehicle_id} -> {global_id}")
        
        # Add to gallery
        self.reid_model.add_to_gallery(vehicle_id, features, metadata)
        
        return global_id
    
    def get_cross_camera_matches(self) -> Dict:
        """Get all cross-camera matches"""
        return self.cross_camera_matches
    
    def get_global_id_mapping(self) -> Dict[str, str]:
        """Get mapping from local to global vehicle IDs"""
        return self.global_vehicle_ids.copy()
    
    def save_state(self, directory: str):
        """Save the current state to disk"""
        os.makedirs(directory, exist_ok=True)
        
        # Save ReID model gallery
        self.reid_model.save_gallery(os.path.join(directory, 'reid_gallery.pkl'))
        
        # Save cross-camera tracking state
        state_data = {
            'cross_camera_matches': self.cross_camera_matches,
            'global_vehicle_ids': self.global_vehicle_ids,
            'next_global_id': self.next_global_id
        }
        
        with open(os.path.join(directory, 'cross_camera_state.pkl'), 'wb') as f:
            pickle.dump(state_data, f)
        
        logger.info(f"State saved to {directory}")
    
    def load_state(self, directory: str):
        """Load state from disk"""
        # Load ReID model gallery
        gallery_path = os.path.join(directory, 'reid_gallery.pkl')
        self.reid_model.load_gallery(gallery_path)
        
        # Load cross-camera tracking state
        state_path = os.path.join(directory, 'cross_camera_state.pkl')
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                state_data = pickle.load(f)
            
            self.cross_camera_matches = state_data.get('cross_camera_matches', {})
            self.global_vehicle_ids = state_data.get('global_vehicle_ids', {})
            self.next_global_id = state_data.get('next_global_id', 1)
            
            logger.info(f"State loaded from {directory}")
        else:
            logger.warning(f"State file {state_path} not found")
