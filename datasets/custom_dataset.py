import pickle
import numpy as np
import os.path as osp
from .bases import BaseImageDataset

class CustomVehicleDataset(BaseImageDataset):
    """
    Custom Vehicle Re-identification Dataset (Feature-based)
    Uses R-CNN extracted features instead of raw images
    """

    dataset_dir = 'CustomVehicleDataset'

    def __init__(self, root='', verbose=True, **kwargs):
        super(CustomVehicleDataset, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        
        # Feature files instead of image directories
        self.feature_dir = osp.join(self.dataset_dir, 'features')
        self.train_file = osp.join(self.feature_dir, 'train_features.pkl')
        self.query_file = osp.join(self.feature_dir, 'query_features.pkl')
        self.gallery_file = osp.join(self.feature_dir, 'gallery_features.pkl')

        self._check_before_run()

        # Load feature data
        train = self._load_features(self.train_file, relabel=True)
        query = self._load_features(self.query_file, relabel=False)  
        gallery = self._load_features(self.gallery_file, relabel=False)

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

    def _load_features(self, feature_file, relabel=False):
        """Load feature data from pickle file"""
        print(f"Loading features from {feature_file}")
        
        with open(feature_file, 'rb') as f:
            feature_data = pickle.load(f)
        
        print(f"Loaded {len(feature_data)} feature vectors")
        
        # FIXED: Create consistent mapping across ALL splits
        if relabel:
            # Training: create mapping and store it
            pid_container = sorted(set(item['vehicle_id'] for item in feature_data))
            self.pid2label = {pid: label for label, pid in enumerate(pid_container)}
            pid2label = self.pid2label
        else:
            # Query/Gallery: use SAME mapping as training
            if hasattr(self, 'pid2label'):
                pid2label = self.pid2label
            else:
                # Fallback: create consistent mapping
                pid_container = sorted(set(item['vehicle_id'] for item in feature_data))
                pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for item in feature_data:
            vehicle_id = item['vehicle_id']
            camera_id = item['camera_id']
            features = item['features']
            
            # Convert camera ID to 0-indexed
            camid = max(0, camera_id - 1)
            
            # FIXED: Use consistent mapping
            if vehicle_id in pid2label:
                pid = pid2label[vehicle_id]
            else:
                # For unseen vehicles, use a consistent fallback
                pid = len(pid2label) + abs(hash(vehicle_id)) % 1000
            
            viewid = 0
            dataset.append((features, pid, camid, viewid))
        
        return dataset