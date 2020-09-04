import torch
import torch.nn as nn
import time
import numpy as np
import os
import torch.optim as optim
import gensim
from tqdm import tqdm
import pickle

def train(net, train_loader, eval_loader, args):
    if args.model == 'Doc2Vec':
        doc2vec = net.model

        # Build corpus
        train_corpus = []
        print("Building corpus")
        doc2vec_corps = os.path.join(args.data_root, "doc2vec_corps.pkl")
        if os.path.exists(doc2vec_corps):
            train_corpus = pickle.load(open(doc2vec_corps, 'rb'))
        else:
            for i, sample in enumerate(tqdm(train_loader)):
                report = gensim.utils.simple_preprocess(sample['report'][0])
                tag = [int(sample['idx'])]
                train_corpus.append(gensim.models.doc2vec.TaggedDocument(report, tag))
            pickle.dump(train_corpus, open(doc2vec_corps, 'wb'))

        # Build vocab
        doc2vec.build_vocab(train_corpus)
        print("Vocabulary built, " + str(doc2vec.corpus_total_words) + ' words')

        # Train the model
        print("Training model")
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
        print('Saving Doc2Vec vectors')
        os.makedirs(os.path.join(args.data_root, args.vector_folder), exist_ok=True)
        for i, sample in enumerate(tqdm(train_loader)):
            report = gensim.utils.simple_preprocess(sample['report'][0])
            vector = doc2vec.infer_vector(report)
            np.save(os.path.join(
                    args.data_root,
                    args.vector_folder,
                    str(sample['key'][0].item()) + '-' + str(sample['key'][1].item())
                    ), np.array(vector))
        return

    elif args.model == 'Visual' or args.model == 'Finetune':
        net.cuda()
        print("Total number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")
        if args.model == 'Visual':
            loss_fn = nn.CosineEmbeddingLoss()
        elif args.model == 'Finetune':
            loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            raise NotImplementedError()

        optimizer = optim.Adam(net.parameters(), lr=args.lr_base)
        loss_sum = 0

        for epoch in range(0, args.max_epoch):
            net.train()
            time_start = time.time()
            for step, (sample) in enumerate(train_loader):
                optimizer.zero_grad()
                img = sample['img'].cuda()
                pred = net(img)

                if args.model == 'Visual':
                    loss = loss_fn(pred, sample['vector'].cuda(), torch.ones((pred.size(0))).cuda())
                elif args.model == 'Finetune':
                    print(sample['label'].cuda().long().dtype)
                    loss = loss_fn(pred, sample['label'].cuda())
                    print(loss)
                    sys.exit()
                else:
                    raise NotImplementedError()

                loss.backward()
                # Gradient norm clipping
                if args.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        args.grad_norm_clip
                    )

                optimizer.step()

                print("\r[Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e, %4d m "
                      "remaining" % (
                          epoch + 1,
                          step,
                          int(len(train_loader.dataset) / args.batch_size),
                          loss.cpu().data.numpy() / args.batch_size,
                          *[group['lr'] for group in optimizer.param_groups],
                          ((time.time() - time_start) / (step + 1)) * (
                                      (len(train_loader.dataset) / args.batch_size) - step) / 60,
                      ), end='          ')
