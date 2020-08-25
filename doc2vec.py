import gensim
import multiprocessing


class Doc2Vec:
    def __init__(self, args):
        self.args = args
        self.model = gensim.models.doc2vec.Doc2Vec(dm=0,
                                                   dbow_words=1,
                                                   vector_size=300,
                                                   window=8,
                                                   min_count=15,
                                                   epochs=args.epochs,
                                                   workers=multiprocessing.cpu_count())
        print('Using model: ', self.model)
