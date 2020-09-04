import gensim
import multiprocessing
from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):
     '''Callback to log information about training'''
     def __init__(self):
         self.epoch = 0

     def on_epoch_end(self, model):
         print("\rEpoch #{} end".format(self.epoch),
               end='          ')
         self.epoch += 1

     def on_training_end(self, model):
         print('\n')

class Doc2Vec:
    def __init__(self, args):
        self.args = args
        self.model = gensim.models.doc2vec.Doc2Vec(dm=0,
                                                   dbow_words=0,
                                                   vector_size=300,
                                                   window=8,
                                                   min_count=15,
                                                   epochs=args.epochs,
                                                   workers=multiprocessing.cpu_count(),
                                                   callbacks=[EpochLogger()])
        print('Using model: ', self.model)
