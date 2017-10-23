import data_utils
from tqdm import tqdm
#import ipdb
vocab , rev_vocab = data_utils.initialize_vocabulary("./data/e.vocab.one")
emb = data_utils.gen_embeddings(vocab, rev_vocab, 300, 100,
                     "/users1/ybsun/glove.840B.300d.txt",
                     "/users1/ybsun/seq2sql/charNgram.txt", save="./data/emb_for_duyu")
