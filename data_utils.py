# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pkl
import collections
import gzip
import os
import re
import tarfile
import numpy as np
from tqdm import tqdm
#import ipdb

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"

def count_lines(fname):
  with open(fname) as f:
    return sum(1 for line in f)

def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Successfully downloaded", filename, statinfo.st_size, "bytes")
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "wb") as new_file:
      for line in gz_file:
        new_file.write(line)


def get_wmt_enfr_train_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  train_path = os.path.join(directory, "newstest2013")
#giga-fren.release2.fixed")
#  if not (gfile.Exists(train_path +".fr") and gfile.Exists(train_path +".en")):
#    corpus_file = maybe_download(directory, "training-giga-fren.tar",
#                                 _WMT_ENFR_TRAIN_URL)
#    print("Extracting tar file %s" % corpus_file)
#    with tarfile.open(corpus_file, "r") as corpus_tar:
#      corpus_tar.extractall(directory)
#    gunzip_file(train_path + ".fr.gz", train_path + ".fr")
#    gunzip_file(train_path + ".en.gz", train_path + ".en")
  return train_path


def get_wmt_enfr_dev_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  dev_name = "newstest2013"
  dev_path = os.path.join(directory, dev_name)
#  if not (gfile.Exists(dev_path + ".fr") and gfile.Exists(dev_path + ".en")):
#    dev_file = maybe_download(directory, "dev-v2.tgz", _WMT_ENFR_DEV_URL)
#    print("Extracting tgz file %s" % dev_file)
#    with tarfile.open(dev_file, "r:gz") as dev_tar:
#      fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
#      en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
#      fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
#      en_dev_file.name = dev_name + ".en"
#      dev_tar.extract(fr_dev_file, directory)
#      dev_tar.extract(en_dev_file, directory)
  return dev_path


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_path)))
    vocab = {}
    counter = 0
    for df in data_path:
      with gfile.GFile(df, mode="rb") as f:
        for line in f:
          counter += 1
          if counter % 100000 == 0:
            print("  processing line %d" % counter)
          line = tf.compat.as_bytes(line)
          tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
          for w in tokens:
            word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
            if word in vocab:
              vocab[word] += 1
            else:
              vocab[word] = 1

    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
      vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
      for w in vocab_list:
        vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                            tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_data(data_dir, en_vocabulary_size, fr_vocabulary_size, tokenizer=None):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """
  # Get wmt data to the specified directory.
  train_path = get_wmt_enfr_train_set(data_dir)
  dev_path = get_wmt_enfr_dev_set(data_dir)

  from_train_path = train_path + ".en"
  to_train_path = train_path + ".fr"
  from_dev_path = dev_path + ".en"
  to_dev_path = dev_path + ".fr"
  return prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, en_vocabulary_size,
                      fr_vocabulary_size, tokenizer)


def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, from_vocabulary_size,
                 to_vocabulary_size, tokenizer=None):
  """Preapre all necessary files that are required for the training.

    Args:
      data_dir: directory in which the data sets will be stored.
      from_train_path: path to the file that includes "from" training samples.
      to_train_path: path to the file that includes "to" training samples.
      from_dev_path: path to the file that includes "from" dev samples.
      to_dev_path: path to the file that includes "to" dev samples.
      from_vocabulary_size: size of the "from language" vocabulary to create and use.
      to_vocabulary_size: size of the "to language" vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for "from language" training data-set,
        (2) path to the token-ids for "to language" training data-set,
        (3) path to the token-ids for "from language" development data-set,
        (4) path to the token-ids for "to language" development data-set,
        (5) path to the "from language" vocabulary file,
        (6) path to the "to language" vocabulary file.
    """
  # Create vocabularies of the appropriate sizes.
  to_vocab_path = os.path.join(data_dir, "vocab%d.to" % to_vocabulary_size)
  from_vocab_path = os.path.join(data_dir, "vocab%d.from" % from_vocabulary_size)
  #create_vocabulary(to_vocab_path, to_train_path , to_vocabulary_size, tokenizer)
  create_vocabulary(from_vocab_path, [from_train_path, from_dev_path] , from_vocabulary_size, tokenizer)

  # Create token ids for the training data.
  to_train_ids_path = to_train_path + (".ids%d" % to_vocabulary_size)
  from_train_ids_path = from_train_path + (".ids%d" % from_vocabulary_size)
  data_to_token_ids(to_train_path, to_train_ids_path, from_vocab_path, tokenizer)
  data_to_token_ids(from_train_path, from_train_ids_path, from_vocab_path, tokenizer)

  # Create token ids for the development data.
  to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
  from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
  data_to_token_ids(to_dev_path, to_dev_ids_path, from_vocab_path, tokenizer)
  data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path, tokenizer)

  return (from_train_ids_path, to_train_ids_path,
          from_dev_ids_path, to_dev_ids_path,
          from_vocab_path, to_vocab_path)

def ngrams(sentence, n):
  """
  Returns:
       list: a list of lists of words corresponding to the ngrams in the sentence.
  """
  return [sentence[i:i+n] for i in range(len(sentence)-n+1)]


def emb(w, char_emb, chars, dim):
  #chars = ['#BEGIN#'] + list(w) + ['#END#']
  embs = np.zeros(dim, dtype=np.float32)
  match = {}
  try:
    for i in [2, 3, 4]:
      grams = ngrams(w, i)
      for g in grams:
        #g = g.decode('utf-8')
        g = u'{}gram-{}'.format(i, ''.join(g))
        if g in chars:
          match[g] = char_emb[chars.index(g)]
  except Exception as e:
    print(e)
    print(grams)
    print(match)
    #ipdb.set_trace()
    pass

  if match:
    embs = sum(match.values()) / len(match)
  return embs 

def load_word2emb(fin_name, dim, show_progress=True):
  char_emd = []
  chars = []
  char_embeddings = np.zeros([874474, 100])
  with open(fin_name) as fin:
    content = fin.read()
    lines = content.splitlines()
    if show_progress:
      lines = tqdm(lines)
    cnt = 0
    for line in lines:
      elems = line.decode('utf-8').rstrip().split()
      vec = [float(n) for n in elems[-dim:]]
      word = ' '.join(elems[:-dim])
      char_embeddings[cnt] = vec
      chars.append(word)
      cnt += 1
  return char_embeddings, chars

def convert_to_unicode(data, decode_type='utf-8'):
  if isinstance(data, basestring):
    return data.decode(decode_type)
  elif isinstance(data, collections.Mapping):
    return dict(map(convert_to_unicode, data.iteritems()))
  elif isinstance(data, collections.Iterable):
    return type(data)(map(convert_to_unicode, data))
  else:
    return data

def gen_embeddings(word_dict, rev_word_dict, dim1, dim2, in_file=None, in_file_char=None,
                   save='./data/mix_emb'):
  """
      Generate an initial embedding matrix for `word_dict`.
      If an embedding file is not given or a word is not in the embedding file,
      a randomly initialized vector will be used.
  """
  word_dict = convert_to_unicode(word_dict, 'utf-8')
  rev_word_dict = convert_to_unicode(rev_word_dict, 'utf-8')

  num_words = max(word_dict.values()) + 1
  embeddings = np.zeros([num_words, dim1 + dim2])

  print('loading Glove embeddings...')
  if in_file is not None:
    pre_trained = 0
    with open(in_file) as infile: 
      for line in tqdm(infile, total=count_lines(in_file)):
        sp = line.decode('utf-8').rstrip().split()
        try:
          word = ' '.join(sp[:-dim1])
          if word in word_dict:
            vec = [float(n) for n in sp[-dim1:]]
            pre_trained += 1
            embeddings[word_dict[word]][:dim1] = vec 
        except Exception as e:
            print(line)
            print(word)
            print(vec)
            pass
  

  print('loading Char embeddings...')
  if in_file_char is not None:
    char_emb, chars = load_word2emb(in_file_char, dim2)

  char_pre_trained = 0
  all_pre_trained = 0
  print('Computing Char_to_Word embeddings...')
  for idx in tqdm(xrange(len(embeddings)), total=len(embeddings)):
    embeddings[idx][dim1:] = emb(rev_word_dict[idx], char_emb, chars, dim2)
    if np.any(embeddings[idx]):
      all_pre_trained += 1
      if np.any(embeddings[idx][dim1:]):
        char_pre_trained += 1

  if save:
    pkl.dump(embeddings, open('%s.pkl' % save, 'wb')) 

  print('Glove Pre-trained: %d (%.2f%%)' %
                 (pre_trained, pre_trained * 100.0 / num_words))
  print('Char Pre-trained: %d (%.2f%%)' %
                 (pre_trained, char_pre_trained * 100.0 / num_words))
  print('All Pre-trained: %d (%.2f%%)' %
                 (pre_trained, all_pre_trained * 100.0 / num_words))

  return embeddings
