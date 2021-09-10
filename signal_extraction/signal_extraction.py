#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Chao Pan <chaopan2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
    This file is for automatic modified bases signal extraction. The overall
    idea is to first find the correct current level for polyA tail with kde,
    then find the polyT region and the oscillating part within polyT region
    as the signals corresponding to modified bases.

    Input:
        data_path: str, path of raw fast5 files;
        bs1, bs2: int, two intermediate parameters for debug usage only;
        std_A: float, upper bound for std value of polyA region;
        std_U: float, upper bound for std value of polyT region;
        out_dir: str, path for output files.

    Usage:
        python signal_extraction.py --data_path="raw_fast5" --out_dir="processed_files/extracted_signals"
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import scipy.signal as sci_signal
from tqdm import tqdm


def check_valid_polyA(polyA_val, sig, coarse_find_len=500, coarse_find_offset=100, detailed_find_len=100, bs2=300,
                      std_A = 4, std_U = 8):
    """Check if the polyA_val found is a correct estimation."""
    len_raw = len(sig)
    # Finite state machine scheme
    # state 0: we find nothing
    # state 1: we find polyA tail, trying to find polyT region
    # state 2 (termination state): we find polyT region, so the estimated polyA_val should
    # be correct and return
    state = 0
    start_idx = 0
    while start_idx < len_raw:
        # we try to first find polyA region
        if state == 0:
            tmp_seq = sig[start_idx: min(start_idx + coarse_find_len, len_raw)]
            # claim to find polyA region if both mean and std satisfy constraints
            # change to state 1
            if polyA_val - std_A <= np.mean(tmp_seq) <= polyA_val + std_A and np.std(tmp_seq) <= std_A:
                state = 1
                start_idx += coarse_find_len
            else:
                start_idx += coarse_find_offset
        # we have found polyA, next is to find polyT
        elif state == 1:
            tmp_seq = sig[start_idx: min(start_idx + detailed_find_len, len_raw)]
            # stay at state 1 if nothing changes
            if polyA_val - std_A <= np.mean(tmp_seq) <= polyA_val + std_A and np.std(tmp_seq) <= std_A:
                start_idx += detailed_find_len
            # claim to find a transition, because polyT should be lower than polyA
            elif np.mean(tmp_seq) < polyA_val - std_A:
                # see if the next seq falls into polyT region
                start_idx += detailed_find_len
                if start_idx < len(raw_current[i]):
                    tmp_seq_followed = sig[start_idx: min(start_idx + bs2, len_raw)]
                    # claim to find polyT, change to state 2 and return
                    if np.mean(tmp_seq_followed) <= polyA_val - 5 and np.std(tmp_seq_followed) < std_U:
                        return True, polyA_val, np.mean(tmp_seq_followed)
                    else:
                        state = 0
                else:
                    return False, None, None
            else:
                state = 0
                start_idx += detailed_find_len
    return False, None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extracting regions of interest")
    parser.add_argument("--data_path", type=str, default='./', help="Data directory")
    parser.add_argument("--bs1", type=int, default=200, help="Block size 1")
    parser.add_argument("--bs2", type=int, default=300, help="Block size 2")
    parser.add_argument("--std_A", type=float, default=2.0, help="Upper bound for std value of polyA")
    parser.add_argument("--std_U", type=float, default=2.0, help="Upper bound for std value of polyT")
    parser.add_argument("--out_dir", type=str, default='./processed/', help="Output directory")
    args = parser.parse_args()

    block_size_1 = args.bs1
    block_size_2 = args.bs2
    # state 1 setting
    std_high_st1 = args.std_A
    # state 2 setting
    std_high_st2 = args.std_U
    # how many steps to look ahead when saving the plots
    look_ahead = 3000

    # Butterworth digital filter
    sos = sci_signal.butter(50, 450, 'lp', fs=1000, output='sos')
    # get all directories under current folder
    dir_list = os.listdir(args.data_path)
    print('Pools will be processed:', dir_list)
    for dir_val in dir_list:
        # check if this is a path
        if os.path.isdir(os.path.join(args.data_path, dir_val)):
            pool_id = dir_val
            sub_dir_list = os.listdir(os.path.join(args.data_path, dir_val))
            if pool_id + '_fast5_pass' not in sub_dir_list:
                fast5_dirs = sub_dir_list
                append_flag = False
            else:
                fast5_dirs = os.listdir(os.path.join(args.data_path, dir_val, pool_id + '_fast5_pass'))
                append_flag = True
            out_dir = os.path.join(args.out_dir, pool_id)
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            for filename in fast5_dirs:
                if filename.endswith('.fast5'):
                    # read the file
                    if append_flag:
                        f = h5py.File(os.path.join(args.data_path, dir_val, pool_id + '_fast5_pass', filename), 'r')
                    else:
                        f = h5py.File(os.path.join(args.data_path, dir_val, filename), 'r')
                    raw_current = []
                    read_ids = []
                    # normalize the signals
                    for key in list(f.keys()):
                        digitisation = f[key]['channel_id'].attrs['digitisation']
                        offset = f[key]['channel_id'].attrs['offset']
                        range_ = f[key]['channel_id'].attrs['range']
                        sampling_rate = f[key]['channel_id'].attrs['sampling_rate']
                        # convert to pA
                        # (based on https://github.com/nanoporetech/flappie/blob/master/src/fast5_interface.c#L297)
                        rescaled_pA = (np.array(f[key]['Raw']['Signal'])+offset)*(range_/digitisation)
                        # filtering can achieve better performance
                        rescaled_pA = sci_signal.sosfilt(sos, rescaled_pA)
                        raw_current.append(rescaled_pA)
                        read_ids.append(str(key))
                    # start the identification process using a Finite State Machine
                    start_end_idx_lst = []
                    real_start_end_idx_lst = []
                    extracted_signals = []
                    # iterate over signals in each file
                    for i in tqdm(range(len(raw_current))):
                        len_raw = len(raw_current[i])
                        # too long reads are always bad ones
                        if len_raw >= 30000:
                            continue
                        # find possible polyA region by kernel density estimation
                        kernel = gaussian_kde(raw_current[i])
                        ind = np.linspace(raw_current[i].min(), raw_current[i].max(), 200)
                        kdepdf = kernel(ind)
                        peaks, peaks_property = find_peaks(kdepdf, height=3 * np.mean(kdepdf))
                        peak_heights = peaks_property['peak_heights']
                        peak_order_lst = list(np.argsort(peak_heights)[::-1])
                        # how many peaks to check can be decided here
                        peak_order_lst = peak_order_lst[0: min(len(peak_order_lst), 3)]

                        a_val = None
                        u_val = None
                        for peak_idx in peak_order_lst:
                            a_flag, a_val, u_val = check_valid_polyA(ind[peaks[peak_idx]], raw_current[i],
                                                                     bs2=block_size_2)
                            # print(a_flag, a_val, u_val)
                            if a_flag:
                                break
                        if a_val is None:
                            continue
                        else:
                            mean_low_st1 = a_val - args.std_A
                            mean_high_st1 = a_val + args.std_A
                            mean_low_st2 = u_val - args.std_U
                            mean_high_st2 = u_val + args.std_U
                            # start from state 0, meaning that we are trying to find polyA region
                            state = 0
                            start_idx = 0
                            tmp_start_idx = -1  # records the start idx of region of interest
                            while start_idx < len_raw:
                                # we are trying to first find polyA region
                                if state == 0:
                                    tmp_seq = raw_current[i][start_idx: min(start_idx + block_size_1, len_raw)]
                                    # claim to find polyA region if both mean and std satisfy constraints
                                    # change to state 1
                                    if mean_low_st1 <= np.mean(tmp_seq) <= mean_high_st1 and np.std(tmp_seq) <= std_high_st1:
                                        state = 1
                                    start_idx += block_size_1
                                # we have found polyA, next is to find polyT
                                elif state == 1:
                                    tmp_seq = raw_current[i][start_idx: min(start_idx + block_size_1, len_raw)]
                                    # stay at state 1 if nothing changes
                                    if mean_low_st1 <= np.mean(tmp_seq) <= mean_high_st1 and np.std(tmp_seq) < std_high_st1:
                                        state = 1
                                    # claim to find a transition, use both mean and median
                                    elif mean_low_st2 <= np.mean(tmp_seq) <= mean_high_st1 and mean_low_st2 <= np.median(
                                            tmp_seq) <= mean_high_st1:
                                        # see if the next seq falls into polyT region
                                        if start_idx + block_size_1 < len(raw_current[i]):
                                            tmp_seq_followed = raw_current[i][
                                                               start_idx + block_size_1: min(start_idx + block_size_1 * 2,
                                                                                             len_raw)]
                                            # claim to find polyT, change to state 2
                                            if mean_low_st2 <= np.mean(tmp_seq_followed) <= mean_high_st2 and np.std(
                                                    tmp_seq_followed) < std_high_st2:
                                                state = 2
                                                tmp_start_idx = start_idx
                                    # nothing interesting in this substring, change back to state 0
                                    else:
                                        state = 0
                                    start_idx += block_size_1
                                # we have found polyT, next is to find the end of it
                                elif state == 2:
                                    tmp_seq = raw_current[i][start_idx: min(start_idx + block_size_2, len_raw)]
                                    # stay at state 2
                                    if mean_low_st2 <= np.mean(tmp_seq) <= mean_high_st2 and np.std(tmp_seq) < std_high_st2:
                                        state = 2
                                        # if next block reaches the end, record the signal
                                        if start_idx + block_size_2 >= len_raw:
                                            state = 0
                                            start_end_idx_lst.append(
                                                (i, max(0, tmp_start_idx - look_ahead),
                                                 min(start_idx + look_ahead, len_raw)))
                                            real_start_end_idx_lst.append((i, tmp_start_idx, len_raw))
                                            extracted_signals.append(raw_current[i][tmp_start_idx:])
                                    else:
                                        # even if current substring does not fit polyT region, don't make decision
                                        # until next substring
                                        if start_idx + block_size_2 < len_raw:
                                            tmp_seq_followed = raw_current[i][
                                                               start_idx + block_size_2: min(start_idx + block_size_2 * 2,
                                                                                             len_raw)]
                                            # if next substring is fine, stay at state 2
                                            if mean_low_st2 <= np.mean(tmp_seq_followed) <= mean_high_st2 and np.std(
                                                    tmp_seq_followed) < std_high_st2:
                                                state = 2
                                            # otherwise return to state 0 and record the information
                                            else:
                                                state = 0
                                                start_end_idx_lst.append(
                                                    (i, max(0, tmp_start_idx - look_ahead),
                                                     min(start_idx + look_ahead, len_raw)))
                                                real_start_end_idx_lst.append((i, tmp_start_idx, start_idx))
                                                extracted_signals.append(raw_current[i][tmp_start_idx: start_idx])
                                        # if there is also no next substring
                                        else:
                                            state = 0
                                            start_end_idx_lst.append(
                                                (i, max(0, tmp_start_idx - look_ahead),
                                                 min(start_idx + look_ahead, len_raw)))
                                            real_start_end_idx_lst.append((i, tmp_start_idx, start_idx))
                                            extracted_signals.append(raw_current[i][tmp_start_idx: start_idx])
                                    start_idx += block_size_2
                    # print the progress
                    print('\n', filename, 'with', len(start_end_idx_lst), 'cases found')
                    # save the plots and statistics
                    if len(start_end_idx_lst) > 0:
                        # file name exclude the suffix
                        filename_prefix = filename.split('.')[0]
                        if not os.path.isdir(os.path.join(out_dir, filename_prefix)):
                            os.mkdir(os.path.join(out_dir, filename_prefix))
                        # store at most 30 plots
                        for i in range(min(30, len(start_end_idx_lst))):
                            tri_info = start_end_idx_lst[i]
                            signal = raw_current[tri_info[0]][tri_info[1]: tri_info[2]]
                            plt.plot(range(tri_info[1], tri_info[2]), signal)
                            plt.title(pool_id + ', ' + filename.split('.')[0] + ', read #: ' + str(tri_info[0]) + '\n' +
                                      'start: ' + str(real_start_end_idx_lst[i][1]) + ', end: ' +
                                      str(real_start_end_idx_lst[i][2]))
                            plt.ylabel('Current signal (pA)')
                            plt.xlabel('Real indices of signal')
                            plt.savefig(os.path.join(out_dir, filename_prefix, str(tri_info[0]) + '.png'))
                            plt.close()
                        # save the signals
                        np.save(os.path.join(out_dir, filename_prefix, 'start_end_idx_lst.npy'),
                                start_end_idx_lst, allow_pickle=True)
                        # files used to really build NN dataset
                        np.save(os.path.join(out_dir, filename_prefix, 'real_start_end_idx_lst.npy'),
                                real_start_end_idx_lst, allow_pickle=True)
                        np.save(os.path.join(out_dir, filename_prefix, 'extracted_signals.npy'),
                                extracted_signals, allow_pickle=True)
            # print the progress
            print(pool_id, 'finished.')
            print('='*20)
