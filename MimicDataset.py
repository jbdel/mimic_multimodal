from __future__ import print_function
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import utils
import numpy as np
import re
import gensim



class MimicDataset(Dataset):
    def __init__(self, name, args):
        super(MimicDataset, self).__init__()
        assert name in ['train', 'validate', 'test']
        self.name = name
        self.args = args

        df_samples = pd.read_csv(os.path.join(self.args.data_root, self.args.split_file))
        df_samples = df_samples.loc[df_samples['split'] == name]

        df_labels = pd.read_csv(os.path.join(self.args.data_root, self.args.label_file))

        self.keys = list()
        self.labels = {}
        print("Building " + name + " dataset")
        for index, row in tqdm(df_samples.iterrows(), total=df_samples.shape[0]):
            # Keys are a  triple (subject_id, study_id, dicom_id)
            key = (row['subject_id'], row['study_id'], row['dicom_id'])

            # Fetch label
            self.labels[key] = df_labels.loc[(df_labels['subject_id'] == row['subject_id']) &
                                             (df_labels['study_id'] == row['study_id'])]
            if index == 100:
                break
        self.keys = list(self.labels.keys())
        self.transform = utils.get_transforms(name)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img_path = os.path.join(self.args.data_root,
                                self.args.image_folder,
                                'p'+str(key[0])[:2], # 10000032 -> p10
                                'p'+str(key[0]),
                                's'+str(key[1]),
                                str(key[2]) + '.jpg'
                                )

        report_path = os.path.join(self.args.data_root,
                                   self.args.report_folder,
                                   'p'+str(key[0])[:2], # 10000032 -> p10
                                   'p'+str(key[0]),
                                   's'+str(key[1]) + '.txt'
                                   )
        # image
        img = self.transform(Image.open(img_path).convert('RGB'))

        # report
        txt = open(report_path, mode='r').read()
        txt = txt.replace('FINAL REPORT', '')

        # findings = re.search('FINDINGS:(.*?)(IMPRESSION|$)', txt, flags=re.DOTALL) # Between findings and impression or EOF
        # impression = re.search('IMPRESSION:(.*?)$', txt, flags=re.DOTALL) # Between impression and EOF
        # view = re.search('VIEWS OF THE CHEST:(.*?)$', txt, flags=re.DOTALL)
        # # indication = re.search('INDICATION:(.*?)\n', txt)
        #
        # assert \
        #     'FINDINGS' in txt or \
        #     'IMPRESSION' in txt or \
        #     'VIEWS OF THE CHEST' in txt, 'no findings or impression or impression for ' + str(key) + '\n' + txt
        #
        # assert \
        #     findings is not None or \
        #     view is not None or \
        #     impression is not None, 'no regex match for ' + str(key) + '\n' + txt
        #
        # if findings is not None:
        #     report = findings.group(1)
        # elif impression is not None:
        #     report = impression.group(1)
        # else:
        #     report = view.group(1)

        # label
        label = self.labels[key]

        return {'idx': idx,
                'key': key,
                'report': txt,
                'img': img,
                'label': np.array(label)}


    def __len__(self):
        return len(self.keys)
