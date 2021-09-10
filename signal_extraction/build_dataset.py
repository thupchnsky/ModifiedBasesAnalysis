#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Chao Pan <chaopan2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
    This file is to build dataset for neural network training and testing. It
    converts "real_start_end_idx_lst.npy" files to "pool_id.npy" files.

    Input:
        data_path: str, path of extracted signals;
        pool_path: str, path of raw fast5 files;
        out_path: str, path for output files.
        std_thres_lb: float, lower bound for std value of region of interest, for debug usage only;
        std_thres_ub: float, upper bound for std value of region of interest, for debug usage only.

    Usage:
        python build_dataset.py --data_path="processed_files/extracted_signals" --pool_path="raw_fast5"
        --out_path="processed_files/dataset" --std_thres_ub=10.0
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from collections import Counter
import json
import scipy.signal as sci_signal
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Building dataset")
    parser.add_argument("--data_path", type=str, default='./processed_flexible_level', help="Data directory")
    parser.add_argument("--pool_path", type=str, default='./', help="Pool directory")
    parser.add_argument("--out_path", type=str, default='./dataset_flexible_level', help="Output directory")
    parser.add_argument("--std_thres_lb", type=float, default=3.0, help="Lower bound for std")
    parser.add_argument("--std_thres_ub", type=float, default=20.0, help="Upper bound for std")
    args = parser.parse_args()
    sos = sci_signal.butter(50, 450, 'lp', fs=1000, output='sos')

    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    pool_list = os.listdir(args.data_path)
    # the length we would extract
    total_len = 1000
    for pool_id in tqdm(pool_list):
        if os.path.isdir(os.path.join(args.data_path, pool_id)):
            tmp_signal_lst = []
            file_list = os.listdir(os.path.join(args.data_path, pool_id))
            for file_id in file_list:
                tmp_pool_dir = os.listdir(os.path.join(args.pool_path, pool_id))
                fast5_file_id = file_id + '.fast5'
                if fast5_file_id in tmp_pool_dir:
                    fast5_dir = os.path.join(args.pool_path, pool_id, file_id + '.fast5')
                else:
                    fast5_dir = os.path.join(args.pool_path, pool_id, pool_id + '_fast5_pass', file_id + '.fast5')
                # read the data
                f = h5py.File(fast5_dir, 'r')
                raw_current = []
                # normalize the signals
                for key in list(f.keys()):
                    digitisation = f[key]['channel_id'].attrs['digitisation']
                    offset = f[key]['channel_id'].attrs['offset']
                    range_ = f[key]['channel_id'].attrs['range']
                    sampling_rate = f[key]['channel_id'].attrs['sampling_rate']
                    # convert to pA
                    # (based on https://github.com/nanoporetech/flappie/blob/master/src/fast5_interface.c#L297)
                    rescaled_pA = (np.array(f[key]['Raw']['Signal']) + offset) * (range_ / digitisation)
                    rescaled_pA = sci_signal.sosfilt(sos, rescaled_pA)
                    raw_current.append(rescaled_pA)
                extracted_signal_path = os.path.join(args.data_path, pool_id, file_id, 'real_start_end_idx_lst.npy')
                idx_info = np.load(extracted_signal_path)
                # add same oligo removal
                signal_idx_lst = idx_info[:, 0]
                signal_len_lst = idx_info[:, 2] - idx_info[:, 1]
                unique_idx_lst = np.unique(idx_info[:, 0])
                for i in unique_idx_lst:
                    idx_pos = np.where(signal_idx_lst == i)[0]
                    most_possible_idx = idx_pos[np.argmax(signal_len_lst[idx_pos])]
                    signal_idx = idx_info[most_possible_idx][0]
                    assert signal_idx == i
                    start_idx = idx_info[most_possible_idx][1]
                    end_idx = idx_info[most_possible_idx][2]
                    if end_idx + total_len <= len(raw_current[signal_idx]):
                        tmp_signal = raw_current[signal_idx][end_idx: end_idx + total_len]
                    elif start_idx + total_len <= len(raw_current[signal_idx]):
                        tmp_signal = raw_current[signal_idx][
                                     len(raw_current[signal_idx]) - total_len: len(raw_current[signal_idx])]
                    else:
                        tmp_signal = raw_current[signal_idx][start_idx: len(raw_current[signal_idx])]
                    if args.std_thres_lb < np.std(tmp_signal) < args.std_thres_ub:
                        if len(tmp_signal) < total_len:
                            tmp_signal = np.pad(tmp_signal, (total_len - len(tmp_signal), 0), mode='edge')
                        assert len(tmp_signal) == total_len
                        tmp_signal_lst.append(tmp_signal)
            print(pool_id, len(tmp_signal_lst))
            save_path = os.path.join(args.out_path, pool_id + '.npy')
            np.save(save_path, tmp_signal_lst)
