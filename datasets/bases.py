import numpy as np

class BaseDataset(object):
    """
    Base class of reid dataset
    """

class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_query_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

    def get_imagedata_info(self, data):
        pids, cams, views = [], [], []
        for _, pid, camid, viewid in data:
            pids.append(pid)
            cams.append(camid)
            views.append(viewid)
        pids = set(pids)
        cams = set(cams)
        views = set(views)
        num_pids = len(pids)
        num_cams = len(cams)
        num_views = len(views)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams, num_views