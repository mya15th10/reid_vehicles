import pickle
import numpy as np
import os.path as osp
import json
from .bases import BaseImageDataset

class CustomVehicleDataset(BaseImageDataset):
    """FIXED: Custom Vehicle Re-identification Dataset with consistent PID mapping"""

    dataset_dir = 'CustomVehicleDataset'

    def __init__(self, root='', verbose=True, **kwargs):
        super(CustomVehicleDataset, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        
        # Feature files
        self.feature_dir = osp.join(self.dataset_dir, 'features')
        self.train_file = osp.join(self.feature_dir, 'train_features.pkl')
        self.query_file = osp.join(self.feature_dir, 'query_features.pkl')
        self.gallery_file = osp.join(self.feature_dir, 'gallery_features.pkl')
        self.mapping_file = osp.join(self.feature_dir, 'pid_mapping.json')

        self._check_before_run()

        # FIXED: Load consistent PID mapping
        with open(self.mapping_file, 'r') as f:
            self.pid_mapping = json.load(f)
        
        # Load feature data with consistent mapping
        train = self._load_features(self.train_file, is_train=True)
        query = self._load_features(self.query_file, is_train=False)  
        gallery = self._load_features(self.gallery_file, is_train=False)

        if verbose:
            print("Custom Vehicle Dataset (Feature-based) loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all feature files are available"""
        if not osp.exists(self.feature_dir):
            raise RuntimeError("'{}' is not available".format(self.feature_dir))
        if not osp.exists(self.train_file):
            raise RuntimeError("'{}' is not available".format(self.train_file))
        if not osp.exists(self.query_file):
            raise RuntimeError("'{}' is not available".format(self.query_file))
        if not osp.exists(self.gallery_file):
            raise RuntimeError("'{}' is not available".format(self.gallery_file))
        if not osp.exists(self.mapping_file):
            raise RuntimeError("'{}' is not available".format(self.mapping_file))

    def _load_features(self, feature_file, is_train=False):
        """FIXED: Load feature data with consistent PID mapping"""
        print(f"Loading features from {feature_file}")
        
        with open(feature_file, 'rb') as f:
            feature_data = pickle.load(f)
        
        print(f"Loaded {len(feature_data)} feature vectors")
        
        dataset = []
        for item in feature_data:
            # Use pre-computed consistent PID
            pid = item['pid']  # Already mapped during feature extraction
            
            camera_id = item['camera_id']
            features = item['features']
            
            # Convert camera ID to 0-indexed
            camid = max(0, camera_id - 1)
            viewid = 0
            
            # FIXED: Validate feature dimensions
            if len(features) != 256:
                print(f"Warning: Expected 256-dim features, got {len(features)}")
                continue
            
            dataset.append((features, pid, camid, viewid))
        
        print(f"Processed {len(dataset)} valid feature vectors")
        return dataset