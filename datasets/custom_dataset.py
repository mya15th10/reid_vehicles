import glob
import re
import os.path as osp
import numpy as np

from .bases import BaseImageDataset


class CustomVehicleDataset(BaseImageDataset):
    """
    Custom Vehicle Re-identification Dataset
    Converted from CVAT annotations
    
    Dataset statistics:
    # identities: 377
    # images: 2807 total (1819 train + 663 query + 325 test)
    # cameras: 2 (front and back views)
    """

    dataset_dir = 'CustomVehicleDataset'

    def __init__(self, root='', verbose=True, **kwargs):
        super(CustomVehicleDataset, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        # Read image lists from text files
        train_file = osp.join(self.dataset_dir, 'name_train.txt')
        query_file = osp.join(self.dataset_dir, 'name_query.txt')
        test_file = osp.join(self.dataset_dir, 'name_test.txt')

        # Read and process image lists
        self.train_images = self._read_image_list(train_file, self.train_dir)
        self.query_images = self._read_image_list(query_file, self.query_dir)
        self.gallery_images = self._read_image_list(test_file, self.gallery_dir)

        # Create dataset from image lists
        train = self._process_images(self.train_images, relabel=True)
        query = self._process_images(self.query_images, relabel=False)
        gallery = self._process_images(self.gallery_images, relabel=False)

        if verbose:
            print("Custom Vehicle Dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _read_image_list(self, file_path, dir_path):
        """Read list of image files from text file"""
        if not osp.exists(file_path):
            print(f"Warning: {file_path} not found, using all images in directory")
            return glob.glob(osp.join(dir_path, '*.jpg'))
        
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
        
        # Create full paths
        image_paths = [osp.join(dir_path, line.strip()) for line in lines if line.strip()]
        return image_paths

    def _process_images(self, img_paths, relabel=False):
        """Process image list to create dataset"""
        # Pattern to extract vehicle_id and camera_id from filename
        # Expected format: vehicleID_cameraID_cropID.jpg (e.g., 0001_c001_001.jpg)
        pattern = re.compile(r'(\d+)_c(\d+)_(\d+)\.jpg')

        pid_container = set()
        for img_path in img_paths:
            try:
                img_name = osp.basename(img_path)
                match = pattern.search(img_name)
                if match:
                    pid = int(match.group(1))  # Vehicle ID
                    pid_container.add(pid)
                else:
                    print(f"Warning: Couldn't parse {img_name}")
            except Exception as e:
                print(f"Error parsing {img_name}: {e}")
                continue
        
        # Create pid to label mapping for training (relabeling)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        
        dataset = []
        for img_path in img_paths:
            try:
                img_name = osp.basename(img_path)
                match = pattern.search(img_name)
                if match:
                    pid = int(match.group(1))     # Vehicle ID
                    camid = int(match.group(2))   # Camera ID
                    
                    # Convert camera ID to 0-indexed (1,2 -> 0,1)
                    camid = camid - 1
                    
                    # Relabel vehicle IDs for training if needed
                    if relabel:
                        pid = pid2label[pid]
                    
                    # Use 0 as default viewid (not used in this dataset)
                    viewid = 0
                    
                    dataset.append((img_path, pid, camid, viewid))
                else:
                    print(f"Warning: Couldn't add {img_name}")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
        
        return dataset