import mmcv
import numpy as np
from .builder import DATASETS
from .base_dataset import BaseDataset
from random import shuffle

@DATASETS.register_module()
class Filelist_gbc(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            shuffle(samples)

            for cu_sample in samples:
            # for filename, gt_label in samples:

                filename = cu_sample[0]
                gt_label = cu_sample[1]
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos


