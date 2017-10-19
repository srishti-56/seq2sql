from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import ujson as json

from lib.dbengine import DBEngine
from lib.query import Query
from lib.common import count_lines

def query_func(input):
    return

def get_output_with_gloss(seq_output, seq_input):
    sequence = {}
    sequence['words'] = seq_output
    sequence['gloss'] = [seq_input['gloss'][seq_input['words'].index(w)] for w in seq_output]
    sequence['after'] = [seq_input['after'][seq_input['words'].index(w)] for w in seq_output]
    return sequence

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--din', default='data', help='data directory')
    parser.add_argument('--dout', default='annotated', help='output directory')
    parser.add_argument('--dsource', default='annotated', help='output directory')
    args = parser.parse_args()

    data_source = []
    with open(args.dsource) as fs:
        data_source = fs.readlines()
    orig_list = []
    with open(args.din) as fi:
        for line in fi.readlines():
            line_split = line.split('|')
            source_idx = int(line_split[-1])
            if len(line_split[:-1]) > 1:
                seq_in = '|'.join(line_split[:-1])
            else:
                seq_in = line_split[0]
            json_line = {}
            json_line['sample_id'] = source_idx
            json_line['query'] = ""
            json_line['error'] = ""
            seq_in = seq_in.decode('utf-8')
            seq_in = seq_in.split(' ')
            if '_EOS' in seq_in:
                seq_in = seq_in[:seq_in.index('_EOS')]
            json_line['seq'] = ' '.join(seq_in)
            sj = json.loads(data_source[source_idx])
            try:
                output_with_gloss = get_output_with_gloss(seq_in, sj['seq_input'])
                q = Query.from_sequence(output_with_gloss, sj['table']).to_dict()
            except Exception as e:
                json_line['error'] = repr(e) 
            else:
                json_line['query'] = q
            orig_list.append(json_line) 
    sortedlist = sorted(orig_list, key=lambda k: k['sample_id'])
    with open(args.dout, 'wt') as fo:
        for item in sortedlist:
            fo.write(json.dumps(item) + '\n')
