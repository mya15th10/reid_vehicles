import glob
import re
import os.path as osp
import numpy as np

from .bases import BaseImageDataset


class VeRi(BaseImageDataset):
    """
       VeRi-776
       Reference:
       Liu, Xinchen, et al. "Large-scale vehicle re-identification in urban surveillance videos." ICME 2016.

       URL:https://vehiclereid.github.io/VeRi/

       Dataset statistics:
       # identities: 776
       # images: 37778 (train) + 1678 (query) + 11579 (gallery)
       # cameras: 20
       """

    dataset_dir = 'VeRi'

    def __init__(self, root='', verbose=True, **kwargs):
        super(VeRi, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        # Sử dụng file danh sách ảnh có sẵn
        train_file = osp.join(self.dataset_dir, 'name_train.txt')
        query_file = osp.join(self.dataset_dir, 'name_query.txt')
        test_file = osp.join(self.dataset_dir, 'name_test.txt')

        # Đọc và phân tích danh sách ảnh để có danh sách đầy đủ đường dẫn
        self.train_images = self._read_image_list(train_file, self.train_dir)
        self.query_images = self._read_image_list(query_file, self.query_dir)
        self.gallery_images = self._read_image_list(test_file, self.gallery_dir)

        # Tạo dataset từ danh sách file
        train = self._process_images(self.train_images, relabel=True)
        query = self._process_images(self.query_images, relabel=False)
        gallery = self._process_images(self.gallery_images, relabel=False)

        if verbose:
            print("VeRi-776 loaded")
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
        """Đọc danh sách file từ text file"""
        if not osp.exists(file_path):
            print(f"Warning: {file_path} not found, using all images in directory")
            return glob.glob(osp.join(dir_path, '*.jpg'))
        
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
        
        # Nếu file chỉ có một dòng và có nhiều tên file được phân tách bằng dấu cách
        if len(lines) == 1:
            lines = lines[0].split()
        
        # Tạo đường dẫn đầy đủ
        image_paths = [osp.join(dir_path, line.strip()) for line in lines]
        return image_paths

    def _process_images(self, img_paths, relabel=False):
        """Xử lý danh sách ảnh để tạo dataset"""
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            try:
                img_name = osp.basename(img_path)
                match = pattern.search(img_name)
                if match:
                    pid, _ = map(int, match.groups())
                    if pid == -1: continue  # junk images are just ignored
                    pid_container.add(pid)
                else:
                    print(f"Warning: Couldn't parse {img_name}")
            except Exception as e:
                print(f"Error parsing {img_name}: {e}")
                continue
        
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for img_path in img_paths:
            try:
                img_name = osp.basename(img_path)
                match = pattern.search(img_name)
                if match:
                    pid, camid = map(int, match.groups())
                    if pid == -1: continue  # junk images are just ignored
                    assert 0 <= pid <= 776  # pid == 0 means background
                    assert 1 <= camid <= 20
                    camid -= 1  # index starts from 0
                    if relabel: pid = pid2label[pid]
                    
                    # Sử dụng 0 làm viewid mặc định
                    viewid = 0
                    
                    dataset.append((img_path, pid, camid, viewid))
                else:
                    print(f"Warning: Couldn't add {img_name}")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
        
        return dataset