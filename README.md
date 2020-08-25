### Doc2Vec Mimic-crx

```
.
├── data
│   ├── mimic-crx-images
│   │   └── files
│   │       └── p10
│   │           ├── p10000032
│   │           │   ├── s50414267
│   │           │   │   ├── 02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
                ...
│   ├── mimic-crx-reports
│   │   └── files
│   │       ├── p10
│   │       │   ├── p10000032
│   │       │   │   ├── s50414267.txt
                ...
│   ├── mimic-cxr-2.0.0-chexpert.csv
│   ├── mimic-cxr-2.0.0-metadata.csv
│   └── mimic-cxr-2.0.0-split.csv
├── main.py
├── MimicDataset.py
├── model.py
├── train.py
└── utils.py
```

Get the MIMIC-CXR-JPG data from [physionet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and place it in the data folder. The structure should respect the abovementioned tree.<br/>

To pretrain the doc2vec model on the reports, use
```
python main.py --model Doc2Vec
```
To train a visual model on the doc2vec embeddings, use
```
python main.py --model Visual
```

To finetune the visual model on the mimic labels, use
```
python main.py --model visual --ckpt my_ckpt --use_label True
```

