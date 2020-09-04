from __future__ import print_function
import os
import utils
import numpy as np
import torch
import sys
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import pickle


class MimicDataset(Dataset):
    def __init__(self, name, args):
        super(MimicDataset, self).__init__()
        assert name in ['train', 'validate', 'test']
        self.name = name
        self.args = args
        self.set_file = os.path.join(self.args.data_root, name + '_set.pkl')

        df_samples = pd.read_csv(os.path.join(self.args.data_root, self.args.split_file))
        df_samples = df_samples.loc[df_samples['split'] == name]
        df_labels = pd.read_csv(os.path.join(self.args.data_root, self.args.label_file))

        self.keys = list()
        self.labels = {}
        print("Building " + name + " dataset")

        # Building set
        if os.path.exists(self.set_file):
            self.labels = pickle.load(open(self.set_file, 'rb'))
        else:
            for index, row in tqdm(df_samples.iterrows(), total=df_samples.shape[0]):
                # Keys are a  triple (subject_id, study_id, dicom_id)
                key = (row['subject_id'], row['study_id'], row['dicom_id'])

                # Fetch label
                label = df_labels.loc[(df_labels['subject_id'] == row['subject_id']) &
                                      (df_labels['study_id'] == row['study_id'])].fillna(0)
                self.labels[key] = (np.array(label.values.tolist()).flatten()[2:] > 0)  # get positive diagnostics

            # save constructed set
            pickle.dump(self.labels, open(self.set_file, 'wb'))

        self.keys = list(self.labels.keys())
        self.transform = utils.get_transforms(name)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img_path = os.path.join(self.args.data_root,
                                self.args.image_folder,
                                'p' + str(key[0])[:2],  # 10000032 -> p10
                                'p' + str(key[0]),
                                's' + str(key[1]),
                                str(key[2]) + '_256.npy'
                                )

        report_path = os.path.join(self.args.data_root,
                                   self.args.report_folder,
                                   'p' + str(key[0])[:2],  # 10000032 -> p10
                                   'p' + str(key[0]),
                                   's' + str(key[1]) + '.txt'
                                   )

        vector_path = os.path.join(self.args.data_root,
                                   self.args.vector_folder,
                                   str(key[0]) + '-' + str(key[1]) + '.npy'
                                   )

        vector = torch.tensor([])
        img = torch.tensor([])
        txt = torch.tensor([])
        # report
        if self.args.model == 'Doc2vec':
            txt = open(report_path, mode='r').read()
            txt = txt.replace('FINAL REPORT', '')

        # vectors and image
        if self.args.model == 'Visual' or self.args.model == 'Finetune':
            try:
                img = self.transform(Image.fromarray(np.load(img_path)).convert('RGB'))
                vector = np.load(vector_path)
            except FileNotFoundError:
                print('Vector or image not found for key', key, vector_path, img_path)
                raise

        # label
        label = self.labels[key].astype(np.float)
        assert True in label, "No Label for key " + str(key)

        return {'idx': idx,
                'key': key,
                'report': txt,
                'img': img,
                'vector': vector,
                'label': label}

    def __len__(self):
        return len(self.keys)
