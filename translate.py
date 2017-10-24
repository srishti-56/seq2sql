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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging
#import ipdb
import subprocess
import cPickle as pkl 

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import data_utils
import seq2seq_model
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("drop_out", 0.5, "drop out rate")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("emb_size", 400, "Size of embedding")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("optim_method", "sgd", "optim method sgd or adam.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_string("pretrain_embs", "./data/mix_emb.pkl", "pretrained embeddings")
tf.app.flags.DEFINE_string("gpu_ids", "2", "gpu_ids e.g. 2,3")

FLAGS = tf.app.flags.FLAGS


os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_ids

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = [(50, 15), (60, 25), (70, 35), (200, 100)]

def get_variable_by_name(name):
  return [v for v in tf.global_variables() if v.name == name][0]

def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        #target_ids = [source_ids.index(int(x)) for x in target.split()]
        source_ids.append(data_utils.EOS_ID)
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids, counter - 1])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.from_vocab_size,
      FLAGS.to_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.emb_size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      FLAGS.optim_method,
      drop_out=FLAGS.drop_out,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    if FLAGS.pretrain_embs:
      with open(FLAGS.pretrain_embs, 'rb') as f:
        emb = pkl.load(f)
        expend_emb = np.concatenate([emb,\
                     np.zeros([FLAGS.from_vocab_size - len(emb), FLAGS.emb_size])], axis=0)
        #encoder_emb = tf.get_default_graph().get_tensor_by_name("embedding_attention_seq2seq/rnn/embedding_wrapper/embedding:0")
        variable_emb = get_variable_by_name("embedding_attention_seq2seq/embedding:0")
        print(variable_emb)
        session.run(variable_emb.assign(expend_emb)) 
  return model


def train():
  """Train a en->fr translation model using WMT data."""
  from_train = None
  to_train = None
  from_dev = None
  to_dev = None
  if FLAGS.from_train_data and FLAGS.to_train_data:
    from_train_data = FLAGS.from_train_data
    to_train_data = FLAGS.to_train_data
    from_dev_data = from_train_data
    to_dev_data = to_train_data
    if FLAGS.from_dev_data and FLAGS.to_dev_data:
      from_dev_data = FLAGS.from_dev_data
      to_dev_data = FLAGS.to_dev_data
    from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
        FLAGS.data_dir,
        from_train_data,
        to_train_data,
        from_dev_data,
        to_dev_data,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size)

    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.from" % FLAGS.from_vocab_size)
    _, rev_fr_vocab = data_utils.initialize_vocabulary(en_vocab_path)
  else:
      # Prepare WMT data.
      print("Preparing WMT data in %s" % FLAGS.data_dir)
      from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_wmt_data(
          FLAGS.data_dir, FLAGS.from_vocab_size, FLAGS.to_vocab_size)

  train_graph = tf.Graph()
  eval_graph = tf.Graph()
  train_sess = tf.Session(graph=train_graph)
  eval_sess = tf.Session(graph=eval_graph)
  #eval_sess = tf_debug.LocalCLIDebugWrapperSession(eval_sess)


  with train_graph.as_default():
    # Create train model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    train_model = create_model(train_sess, False)

  with eval_graph.as_default():
    # Create eval model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    eval_model = create_model(eval_sess, True)

  #with tf.Session() as sess:

  # Read data into buckets and compute their sizes.
  print ("Reading development and training data (limit: %d)."
         % FLAGS.max_train_data_size)
  dev_set = read_data(from_dev, to_dev)
  train_set = read_data(from_train, to_train, FLAGS.max_train_data_size)
  train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
  train_total_size = float(sum(train_bucket_sizes))

  # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
  # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
  # the size if i-th training bucket, as used later.
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in xrange(len(train_bucket_sizes))]

  # This is the training loop.
  step_time, loss = 0.0, 0.0
  current_step = 0
  previous_losses = []
  while True:
    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number_01 = np.random.random_sample()
    bucket_id = min([i for i in xrange(len(train_buckets_scale))
                     if train_buckets_scale[i] > random_number_01])

    # Get a batch and make a step.
    start_time = time.time()
    encoder_inputs, decoder_inputs, target_weights, target_inputs, sent_ids = train_model.get_batch(
        train_set, bucket_id)
    _, step_loss, _ = train_model.step(train_sess, encoder_inputs, decoder_inputs,
                                 target_weights, target_inputs, bucket_id, False)
    step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
    loss += step_loss / FLAGS.steps_per_checkpoint
    current_step += 1

    # Once in a while, we save checkpoint, print statistics, and run evals.
    if current_step % FLAGS.steps_per_checkpoint == 0:
      # Print statistics for the previous epoch.
      perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
      print ("global step %d learning rate %.4f step-time %.2f perplexity "
             "%.2f" % (train_model.global_step.eval(session = train_sess), train_model.learning_rate.eval(session = train_sess),
                       step_time, perplexity))
      # Decrease learning rate if no improvement was seen over last 3 times.
      if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
        train_sess.run(train_model.learning_rate_decay_op)
      previous_losses.append(loss)
      # Save checkpoint and zero timer and loss.
      checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
      ckpt_path = train_model.saver.save(train_sess, checkpoint_path, global_step=train_model.global_step)
      eval_model.saver.restore(eval_sess, ckpt_path)
      step_time, loss = 0.0, 0.0
      # Run evals on development set and print their perplexity.
      print("run evals")
      ft = open('tmp.eval.ids','w')
      for bucket_id in xrange(len(_buckets)):
        if len(dev_set[bucket_id]) == 0:
          print("  eval: empty bucket %d" % (bucket_id))
          continue
        all_encoder_inputs, all_decoder_inputs, all_target_weights, all_target_inputs, all_sent_ids = eval_model.get_all_batch(
            dev_set, bucket_id)
        #ipdb.set_trace()
        for idx in xrange(len(all_encoder_inputs)):
          _, eval_loss, output_logits = eval_model.step(eval_sess, all_encoder_inputs[idx], all_decoder_inputs[idx],
                                       all_target_weights[idx], all_target_inputs[idx], bucket_id, True)
          batch_ids = all_sent_ids[idx]
          #eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
          #    "inf")
          #print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          #ipdb.set_trace()
          swap_inputs = np.array(all_encoder_inputs[idx])
          swap_inputs = swap_inputs.swapaxes(0,1) 

          outputs = [np.argmax(logit, axis=1) for logit in output_logits]
          swap_outputs = np.array(outputs)
          swap_outputs = swap_outputs.swapaxes(0,1) 

          out_ids = []
          for batch_id in xrange(len(swap_outputs)):
            out_ids.append(swap_inputs[batch_id][swap_outputs[batch_id]])
          #if data_utils.EOS_ID in outputs:
          #  t = [m[:m.index(data_utils.EOS_ID)] for m in t] 
          
          for batch_id in xrange(len(swap_outputs)):
            #print(" ".join([tf.compat.as_str(rev_fr_vocab[o]) for o in m]))
            ft.write(" ".join([tf.compat.as_str(rev_fr_vocab[o]) for o in out_ids[batch_id].tolist()]) + "|"+ str(batch_ids[batch_id]) + '\n')
      ft.close()
      print("converting output...")
      subprocess.call("python convert_to_json.py --din tmp.eval.ids --dout out.json --dsource /users1/ybsun/seq2sql/WikiSQL/annotated/dev.jsonl", shell=True)
      print("running evaluation script...")
      subprocess.call("python evaluate.py ../WikiSQL/data/dev.jsonl ../WikiSQL/data/dev.db  ./out.json", shell=True)

      print("finish evals")
      sys.stdout.flush()


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.from" % FLAGS.from_vocab_size)
    fr_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab%d.to" % FLAGS.to_vocab_size)
    en_vocab, rev_en_vocab = data_utils.initialize_vocabulary(en_vocab_path)
    #_, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    #sys.stdout.write("> ")
    #sys.stdout.flush()
    #sentence = sys.stdin.readline()
    #while sentence:
    id_counts = 0
    with open('./wikisql_in_nmt/dev.seq') as f, open('tmp.eval.ids.true','w') as ft:
      lines = f.readlines()
      for l in tqdm(lines, total = len(lines)):
       # Get token-ids for the input sentence.
       token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(l), en_vocab)
       token_ids.append(data_utils.EOS_ID)
       # Which bucket does it belong to?
       bucket_id = len(_buckets) - 1
       for i, bucket in enumerate(_buckets):
         if bucket[0] >= len(token_ids):
           bucket_id = i
           break
       else:
         logging.warning("Sentence truncated: %s", sentence)
     
       # Get a 1-element batch to feed the sentence to the model.
       encoder_inputs, decoder_inputs, target_weights, target_id, sent_id = model.get_batch(
           {bucket_id: [(token_ids, [], 1)]}, bucket_id)
       # Get output logits for the sentence.
       _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                        target_weights, target_id, bucket_id, True)
       # This is a greedy decoder - outputs are just argmaxes of output_logits.
       outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
       # If there is an EOS symbol in outputs, cut them at that point.
       #if data_utils.EOS_ID in outputs:
       #  outputs = outputs[:outputs.index(data_utils.EOS_ID)]
       # Print out French sentence corresponding to outputs.
       ft.write(" ".join([tf.compat.as_str(rev_en_vocab[int(encoder_inputs[output])]) for output in outputs])+ "|"+ str(id_counts) + '\n')
       id_counts += 1


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.global_variables_initializer())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
