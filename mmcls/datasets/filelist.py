import mmcv
import numpy as np
from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class Filelist(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]

            for cu_sample in samples:
                filename = cu_sample[0]
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}

                cu_gt_label_list = []
                gt_label = cu_sample[1:]
                for cu_gt_label in gt_label: # 也可以用map
                    if cu_gt_label=='-1':
                        cu_gt_label_list.append(0) #可能存在-1, 表示没有标签的，这里暂时归为0类
                    elif cu_gt_label=='0':
                        cu_gt_label_list.append(0)
                    elif cu_gt_label=='1':
                        cu_gt_label_list.append(1)
                    elif cu_gt_label=='2':
                        cu_gt_label_list.append(2)
                    elif cu_gt_label=='3':
                        cu_gt_label_list.append(3)
                    elif cu_gt_label=='4':
                        cu_gt_label_list.append(4)
                    elif cu_gt_label=='5':
                        cu_gt_label_list.append(5)
                    elif cu_gt_label=='6':
                        cu_gt_label_list.append(6)
                    elif cu_gt_label=='7':
                        cu_gt_label_list.append(7)

                info['gt_label']= cu_gt_label_list
                data_infos.append(info)
            return data_infos


            #
            # for filename, gt_label in samples:
            #     info = {'img_prefix': self.data_prefix}
            #     info['img_info'] = {'filename': filename}
            #     info['gt_label'] = np.array(gt_label, dtype=np.int64)
            #     data_infos.append(info)
            # return data_infos


