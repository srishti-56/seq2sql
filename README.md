test_emb# seq2sql

## pre-processing

### data

- Use WikiSQL/annotate.py to annotate files in WikiSQL/data. The usage of annotate.py could be found at https://github.com/salesforce/WikiSQL

- Use WikiSQL/convert.py to extract input/outputs for seq2seq task from annotated files.

### Vocabs

Vocabs are generated automatically when you run the training script. If you want to create the vocabe your self, you can use code/create_vocab.py

### Embeddings

- The embeddings is a concatenate of Glove embeddings(http://nlp.stanford.edu/projects/glove) and Char embeddings(http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz).

- You can generate your own embedding file related to your vocab dict using code/test_emb.py
## Train

```python translate.py --from_train_data ./wikisql_in_nmt/train.seq --to_train_data ./wikisql_in_nmt/train.sql --from_dev_data ./wikisql_in_nmt/dev.seq --to_dev_data ./wikisql_in_nmt/dev.sql --train /your/path/to/save/model --steps_per_checkpoint 1000 --drop_out 0.3 --gpu_id 1 --optim_method sgd --train_emb False --pretrain_embs ./data/emb_50000.pkl```

## Inference

```python translate.py --from_train_data ./wikisql_in_nmt/train.seq --to_train_data ./wikisql_in_nmt/train.sql --from_dev_data ./wikisql_in_nmt/dev.seq --to_dev_data ./wikisql_in_nmt/dev.sql --train /your/path/to/save/model --steps_per_checkpoint 1000 --drop_out 0.3 --gpu_id 1 --optim_method sgd --train_emb False --pretrain_embs ./data/emb_50000.pkl --decode True```

## Evaluate

```python3 convert_to_json_split_py3.py --din tmp.eval.ids.true --dout out_split.json --dsource /users1/ybsun/seq2sql/WikiSQL/annotated/dev.jsonl --dtable ../WikiSQL/data/dev.tables.jsonl```
or
```python3 convert_to_json_split_py3.py --din tmp.eval.ids.true --dout out_split.json --dsource /users1/ybsun/seq2sql/WikiSQL/annotated/dev.jsonl --dtable ../WikiSQL/data/dev.tables.jsonl --strong True```

