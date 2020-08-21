from __future__ import print_function
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import utils
import numpy as np


SPLIT_FILE = "mimic-cxr-2.0.0-split.csv"
LABEL_FILE = "mimic-cxr-2.0.0-chexpert.csv"
REPORT_FOLDER = "mimic-crx-reports/files/"
REPORT_IMAGES = "mimic-crx-images/files/"

class MimicDataset(Dataset):
    def __init__(self, name, args, dataroot='./data'):
        super(MimicDataset, self).__init__()
        assert name in ['train', 'validate', 'test']
        self.name = name
        self.args = args
        self.dataroot = dataroot

        df_samples = pd.read_csv(os.path.join(self.dataroot, SPLIT_FILE))
        df_samples = df_samples.loc[df_samples['split'] == name]

        df_labels = pd.read_csv(os.path.join(self.dataroot, LABEL_FILE))

        self.keys = list()
        self.labels = {}
        print("Building " + name + " dataset")
        for index, row in tqdm(df_samples.iterrows(), total=df_samples.shape[0]):
            # Keys are a  triple (subject_id, study_id, dicom_id)
            key = (row['subject_id'], row['study_id'], row['dicom_id'])

            # Fetch label
            self.labels[key] = df_labels.loc[(df_labels['subject_id'] == row['subject_id']) &
                                             (df_labels['study_id'] == row['study_id'])]
            if index == 5:
                break
        self.keys = list(self.labels.keys())
        self.transform = utils.get_transforms(name)
        # self.__getitem__(0)

    def __getitem__(self, idx):
        idx = 0
        key = self.keys[idx]
        img_path = os.path.join(self.dataroot,
                                REPORT_IMAGES,
                                'p'+str(key[0])[:2], # 10000032 -> p10
                                'p'+str(key[0]),
                                's'+str(key[1]),
                                str(key[2]) + '.jpg'
                                )

        report_path = os.path.join(self.dataroot,
                                REPORT_FOLDER,
                                'p'+str(key[0])[:2], # 10000032 -> p10
                                'p'+str(key[0]),
                                's'+str(key[1]) +'.txt'
                                )

        img = self.transform(Image.open(img_path).convert('RGB'))
        txt = open(report_path, mode='r').read()
        label = self.labels[key]

        return {'key': key, 'txt': txt, 'img': img, 'label': np.array(label)}

    def __len__(self):
        return len(self.keys)
