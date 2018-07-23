# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import re

import torch
from multiprocessing import Pool, Manager, Process


SPACE_NORMALIZER = re.compile("\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class Tokenizer:

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize):
        with open(filename, 'r') as f:
            for line in f:
                for word in tokenize(line):
                    dict.add_symbol(word)
                dict.add_symbol(dict.eos_word)

    # binarize: single worker
    @staticmethod
    def binarize_single_worker(filename, dict, consumer, tokenize, append_eos, reverse_order):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            for line in f:
                ids = Tokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1
                consumer(ids)
                ntok += len(ids)

        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': len(replaced)}

    # binarize: multi-worker (slave)
    @staticmethod
    def binarize_child_worker(worker_id, num_of_worker, ids_queue, filename, tokenize, dict, append_eos,
                              reverse_order):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            for line_idx, line in enumerate(f):
                if line_idx % num_of_worker == worker_id:
                    ids = Tokenizer.tokenize(
                        line=line,
                        dict=dict,
                        tokenize=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                    )
                    nseq += 1
                    ntok += len(ids)
                    ids_queue.put(ids)
            # indicate end of queue
            ids_queue.put(None)

        return {'nseq': nseq, 'ntok': ntok, 'replaced': replaced}

    # binarize: multi-worker (master)
    @staticmethod
    def binarize_multi_worker(filename, dict, consumer, tokenize, append_eos, reverse_order, num_of_worker):
        n_seq_tok = [0, 0]
        replaced = Counter()
        manager = Manager()
        ids_queue = manager.Queue()

        def merge_result(worker_result):
            replaced.update(worker_result['replaced'])
            n_seq_tok[0] += worker_result['nseq']
            n_seq_tok[1] += worker_result['ntok']

        def process_queue(ids_queue):
            num_of_none = 0
            while True:
                top = ids_queue.get()
                if top is None:
                    num_of_none += 1
                    if num_of_none == num_of_worker:
                        # finish processing queue
                        break
                else:
                    consumer(top)

        # process that fetches tensors from queue and feeds to the consumer
        merge_proc = Process(target=process_queue, args=(ids_queue,))
        merge_proc.start()

        # concurrent processes that read file and put tensors in the queue
        pool = Pool(processes=num_of_worker)
        for worker_id in range(num_of_worker):
            pool.apply_async(Tokenizer.binarize_child_worker, (worker_id, num_of_worker, ids_queue, filename, tokenize,
                                                               dict, append_eos, reverse_order), callback=merge_result)
        pool.close()
        pool.join()
        merge_proc.join()

        return {'nseq': n_seq_tok[0], 'nunk': sum(replaced.values()), 'ntok': n_seq_tok[1], 'replaced': len(replaced)}

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line,
                 append_eos=True, reverse_order=False, num_of_worker=1):
        if num_of_worker == 1:
            return Tokenizer.binarize_single_worker(filename, dict, consumer, tokenize, append_eos, reverse_order)
        elif num_of_worker > 1:
            return Tokenizer.binarize_multi_worker(filename, dict, consumer, tokenize, append_eos, reverse_order,
                                                   num_of_worker)
        else:
            raise ValueError('Error: number of workers should be a positive number')

    @staticmethod
    def tokenize(line, dict, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False):
        words = tokenize(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dict.eos_index
        return ids
