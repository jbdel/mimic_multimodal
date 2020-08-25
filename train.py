import torch
import torch.nn as nn
import time
import numpy as np
import os
import torch.optim as optim
import gensim
from tqdm import tqdm

def train(net, train_loader, eval_loader, args):
    if args.model == 'Doc2Vec':
        doc2vec = net.model

        # Build corpus
        train_corpus = []
        print("Formating data")
        for i, sample in enumerate(tqdm(train_loader)):
            report = gensim.utils.simple_preprocess(sample['report'][0])
            tag = [int(sample['idx'])]
            train_corpus.append(gensim.models.doc2vec.TaggedDocument(report, tag))

        # Build vocab
        doc2vec.build_vocab(train_corpus)
        print("Vocabulary built, " + str(doc2vec.corpus_total_words) + ' words')

        # Train the model
        doc2vec.train(train_corpus, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)

        # Save the model
        doc2vec.save(os.path.join(args.output, args.name, 'DBOW_vector' +
                     str(doc2vec.vector_size) +
                     '_window' +
                     str(doc2vec.window) +
                     '_count' +
                     str(doc2vec.min_count) +
                     '_epoch' +
                     str(doc2vec.epochs) +
                     '_mimic.doc2vec'))
        print("Model saved")

        # inference
        print('Saving doc2vec vectors')
        os.makedirs(os.path.join(args.data_root, 'mimic_docs_vectors'), exist_ok=True)
        for i, sample in enumerate(tqdm(train_loader)):
            report = gensim.utils.simple_preprocess(sample['report'][0])
            vector = doc2vec.infer_vector(report)
            np.save(os.path.join(
                    args.data_root,
                    'mimic_docs_vectors',
                    str(sample['key'][0]) + '-' + str(sample['key'][1]) + ".npz"
                    ), np.array(vector))
        return

    else:
        net.cuda().train()
        print("Total number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")

        loss_fn = nn.CosineEmbeddingLoss()
        optimizer = optim.Adam(net.parameters(), lr=args.lr_base)

        for epoch in range(0, args.max_epoch):
            time_start = time.time()
            loss_sum = 0
            for step, (sample) in enumerate(train_loader):
                loss_tmp = 0
                optimizer.zero_grad()
                pred = net(sample['img'].cuda())
                print(pred.shape)
                sys.exit()
                # loss = loss_fn(pred, ans)
                # loss.backward()
                # optimizer.step()
