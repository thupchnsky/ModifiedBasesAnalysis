#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Chao Pan <chaopan2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
    This file is to classify the samples from 66 classes where the number of
    samples is larger than 2000. There are 77 classes in total.

    Input:
        data_path: str, path of dataset;
        lr: float, learning rate of optimizer;
        wd: float, weight decay of optimizer;
        test_ratio: float, the fraction of validation set + test set within overall dataset;
        val_ratio: float, the fraction of validation set compared to test set;
        batch_size: int, number of samples in each batch. Needs to be decreased
            if CUDA is out of memory;
        max_epoch: int, the max number of training epochs;
        norm_type: str, use "StandardScaler" or "RobustScaler" to preprocess data;
        early_stop: int, the training will be stopped if number of consecutive epochs
            without validation accuracy improvement is larger than early_stop;
        chkpt: str, path of output;
        max_sample: int, max number of samples we would use for classification in each
            class;
        min_sample: int, min number of samples we would use for classification in each
            class. Classes with size less than this number will be ignored in case of
            imbalanced training;
        test_only: bool, whether to only perform testing procedure based on current stored
            checkpoint or not;
        num_trails: int, number of trails to repeat;
        cuda: int, which cuda to use if multiple cuda is available.

    Usage:
        To reprocedure the result shown in Figure 3(D), please change the data_path
        to the correct path in your project.
        python main_allclass_convplus.py --data_path="processed_files/dataset/"
        --lr=1e-3 --test_ratio=0.4 --batch_size=256 --max_epoch=200 --early_stop=25
        --chkpt="./chkpt/convplus/allclass_1e-3" --max_sample=3500 --num_trails=1
"""

import sys
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import argparse
from collections import defaultdict
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm


class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """

    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.

    The module adds (optionally) activation function and dropout
    layers right after a separable convolution layer.
    """

    def __init__(self, ni, no, kernel, stride, pad,
                 drop=None, bn=True,
                 activ=lambda: nn.PReLU()):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if bn:
            layers.append(nn.BatchNorm1d(no))
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


class Classifier(nn.Module):
    def __init__(self, raw_ni, no, drop=.5):
        super().__init__()

        self.raw = nn.Sequential(
            SepConv1d(raw_ni, 256, 5, 2, 1),
            SepConv1d(256, 256, 3, 1, 1, drop=drop),
            SepConv1d(256, 512, 5, 2, 1),
            SepConv1d(512, 512, 3, 1, 1, drop=drop),
            SepConv1d(512, 1024, 8, 4, 2),
            SepConv1d(1024, 1024, 3, 1, 1, drop=drop),
            SepConv1d(1024, 2048, 8, 4, 2),
            SepConv1d(2048, 2048, 3, 1, 1, drop=drop),
            SepConv1d(2048, 4096, 5, 2, 1),
            nn.AdaptiveAvgPool1d(4),
            Flatten(),
            nn.Dropout(drop), nn.Linear(4096*4, 1024), nn.PReLU(), nn.BatchNorm1d(1024),
            nn.Dropout(drop), nn.Linear(1024, 512), nn.PReLU(), nn.BatchNorm1d(512))

        self.out = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, no))

        self.init_weights(nn.init.kaiming_normal_)

    def init_weights(self, init_fn):
        def init(m):
            for child in m.children():
                if isinstance(child, nn.Conv1d):
                    init_fn(child.weights)

        init(self)

    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out


# The following is the implementation of the one-cycle policy schedule proposed
# in https://arxiv.org/abs/1803.09820.
def cosine(epoch, t_max, ampl):
    """Shifted and scaled cosine function."""

    t = epoch % t_max
    return (1 + np.cos(np.pi * t / t_max)) * ampl / 2


def inv_cosine(epoch, t_max, ampl):
    """A cosine function reflected on X-axis."""

    return 1 - cosine(epoch, t_max, ampl)


def one_cycle(epoch, t_max, a1=0.6, a2=1.0, pivot=0.3):
    """A combined schedule with two cosine half-waves."""

    pct = epoch / t_max
    if pct < pivot:
        return inv_cosine(epoch, pivot * t_max, a1)
    return cosine(epoch - pivot * t_max, (1 - pivot) * t_max, a2)


class Scheduler:
    """Updates optimizer's learning rates using provided scheduling function."""

    def __init__(self, opt, schedule):
        self.opt = opt
        self.schedule = schedule
        self.history = defaultdict(list)

    def step(self, t):
        for i, group in enumerate(self.opt.param_groups):
            lr = self.opt.defaults['lr'] * self.schedule(t)
            group['lr'] = lr
            self.history[i].append(lr)


def create_dataset_ON_multiclass(norm_type, data_path, max_sample_per_class=4000, min_sample_per_class=2000,
                                 test_ratio=0.5, val_ratio=0.5, batch_size=64):
    """Preprocess dataset and transform to dataloaders."""
    if len(data_path) <= 1:
        print('Only one class exists. No need for classification')
        return None
    start_offset = 0
    data_combined = np.load(data_path[start_offset])
    ori_sample_size = data_combined.shape[0]
    data_combined = data_combined[0: min(max_sample_per_class, ori_sample_size), :]
    label_combined = np.array([0] * min(max_sample_per_class, ori_sample_size))
    class_count = 1
    for i in range(start_offset + 1, len(data_path)):
        data_class = np.load(data_path[i])
        ori_sample_size = data_class.shape[0]
        if ori_sample_size < min_sample_per_class:
            continue
        else:
            data_combined = np.concatenate((data_combined, data_class[0: min(max_sample_per_class, ori_sample_size), :]), axis=0)
            label_combined = np.concatenate((label_combined, np.array([class_count] * min(max_sample_per_class, ori_sample_size))))
            class_count += 1
    label_combined = label_combined.astype('int')

    # normalization
    if norm_type == 'scalar':
        scaler = StandardScaler()
        data_combined = scaler.fit_transform(data_combined)
    elif norm_type == 'robust':
        scaler = RobustScaler()
        data_combined = scaler.fit_transform(data_combined)

    # train vs val+test
    x_train, x_test, y_train, y_test = train_test_split(data_combined, label_combined, test_size=test_ratio,
                                                        stratify=label_combined)
    # val vs test
    x_test_final, x_val, y_test_final, y_val = train_test_split(x_test, y_test, test_size=val_ratio, stratify=y_test)

    x_train = np.expand_dims(x_train, axis=1)
    x_val = np.expand_dims(x_val, axis=1)
    x_test_final = np.expand_dims(x_test_final, axis=1)
    print('Train size:', y_train.size, 'Val size:', y_val.size, 'Test size:', y_test_final.size)
    # create DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test_final).float(), torch.from_numpy(y_test_final)),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader, x_train.shape[0], x_val.shape[0], class_count


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time series binary classification")
    parser.add_argument("--data_path", type=str, default="./data/dataset/", help="Path for Data.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0, help="Weight decay.")
    parser.add_argument("--test_ratio", type=float, default=0.4, help="Test size.")
    parser.add_argument("--val_ratio", type=float, default=0.5, help="Validation size.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--max_epoch", type=int, default=1, help="Maximum number of training epochs.")
    parser.add_argument("--norm_type", type=str, default="scalar", help="Method for normalization.")
    parser.add_argument("--early_stop", type=int, default=10, help="Early stopping criterion.")
    parser.add_argument("--chkpt", type=str, default='./chkpt', help="Path to checkpoints.")
    parser.add_argument("--max_sample", type=int, default=4000, help="Max sample per class.")
    parser.add_argument("--min_sample", type=int, default=2000, help="Min sample per class.")
    parser.add_argument("--test_only", type=bool, default=False, help="Test only using existing checkpoint.")
    parser.add_argument("--num_trails", type=int, default=5, help="Number of trails to test.")
    parser.add_argument("--cuda", type=int, default=0, help="Which cuda to use.")
    args = parser.parse_args()

    if not os.path.exists(args.chkpt):
        os.makedirs(args.chkpt)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_path = os.listdir(args.data_path)
    real_path = [os.path.join(args.data_path, filename) for filename in data_path]

    for trail_idx in range(args.num_trails):
        train_loader, val_loader, test_loader, train_set_size, val_set_size, num_classes = create_dataset_ON_multiclass(
            args.norm_type, real_path, max_sample_per_class=args.max_sample, min_sample_per_class=args.min_sample,
            test_ratio=args.test_ratio, val_ratio=args.val_ratio, batch_size=args.batch_size)
        print('Total number of classes:', num_classes)
        lr = args.lr
        n_epochs = args.max_epoch
        iterations_per_epoch = len(train_loader)
        period = n_epochs * iterations_per_epoch
        best_acc = 0
        patience, trials = args.early_stop, 0
        iteration = 0
        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        model = Classifier(1, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=args.wd)
        # one-cycle learning rate scheduling
        # sched = Scheduler(opt, partial(one_cycle, t_max=period, pivot=0.3))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

        if not args.test_only:
            print('Start model training')
            for epoch in tqdm(range(1, n_epochs + 1)):
                # train
                model.train()
                epoch_loss = 0
                correct, total = 0, 0
                for x_t, y_t in train_loader:
                    iteration += 1
                    x_t, y_t = x_t.to(device), y_t.to(device)
                    # sched.step(iteration)  # update the learning rates
                    opt.zero_grad()
                    out = model(x_t)
                    preds = F.log_softmax(out, dim=1).argmax(dim=1)
                    loss = criterion(out, y_t)
                    epoch_loss += loss.item()
                    loss.backward()
                    opt.step()
                    total += y_t.size(0)
                    correct += (preds == y_t).sum().item()
                acc = correct / total
                epoch_loss /= train_set_size
                train_acc_history.append(acc)
                train_loss_history.append(epoch_loss)
                # validation
                model.eval()
                epoch_loss = 0
                correct, total = 0, 0
                for x_v, y_v in val_loader:
                    x_v, y_v = x_v.to(device), y_v.to(device)
                    out = model(x_v)
                    preds = F.log_softmax(out, dim=1).argmax(dim=1)
                    loss = criterion(out, y_v)
                    epoch_loss += loss.item()
                    total += y_v.size(0)
                    correct += (preds == y_v).sum().item()
                acc = correct / total
                epoch_loss /= val_set_size
                val_acc_history.append(acc)
                val_loss_history.append(epoch_loss)
                scheduler.step(epoch_loss)

                if epoch % 1 == 0:
                    print(f'Epoch: {epoch:3d}. Train loss: {train_loss_history[-1]:.4f}. '
                          f'Train acc.: {train_acc_history[-1]:2.2%}. '
                          f'Val loss: {val_loss_history[-1]:.4f}. '
                          f'Val acc.: {val_acc_history[-1]:2.2%}.')

                acc = val_acc_history[-1]
                if acc > best_acc:
                    trials = 0
                    best_acc = acc
                    torch.save(model.state_dict(), os.path.join(args.chkpt, f'{trail_idx}_best.pth'))
                    print(f'Epoch {epoch} best model saved with validation accuracy: {best_acc:2.2%}')
                else:
                    trials += 1
                    if trials >= patience:
                        print(f'Early stopping on epoch {epoch}')
                        break
            print('Done!')

            f, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(train_loss_history, c='r', label='train loss')
            ax[0].plot(val_loss_history, c='g', label='val loss')
            ax[0].legend()
            ax[0].set_title('Loss History')
            ax[0].set_xlabel('Epoch no.')
            ax[0].set_ylabel('Loss')

            ax[1].plot(smooth(train_acc_history, 5)[:-2], c='r', label='train acc')
            ax[1].plot(smooth(val_acc_history, 5)[:-2], c='g', label='val acc')
            ax[1].legend()
            ax[1].set_title('Accuracy History')
            ax[1].set_xlabel('Epoch no.')
            ax[1].set_ylabel('Accuracy')
            # save the figure
            plt.savefig(os.path.join(args.chkpt, f'{trail_idx}_training_curve.png'))

        test_results = []
        model.load_state_dict(torch.load(os.path.join(args.chkpt, f'{trail_idx}_best.pth')))
        model.eval()
        correct, total = 0, 0
        true_list_test, preds_list_test = [], []
        for x_t, y_t in test_loader:
            true_list_test.append(y_t.detach().numpy())
            x_t, y_t = x_t.to(device), y_t.to(device)
            out = model(x_t)
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_t.size(0)
            correct += (preds == y_t).sum().item()
            preds_list_test.append(preds.cpu().detach().numpy())
        acc = correct / total
        print('Test acc', acc)
        true_np_test, preds_np_test = np.concatenate(true_list_test), np.concatenate(preds_list_test)
        weighted_f1 = f1_score(true_np_test, preds_np_test, average='weighted')
        print('F1 score', weighted_f1)
        confuse_mat = confusion_matrix(true_np_test, preds_np_test, normalize='true')
        np.savez(os.path.join(args.chkpt, f'{trail_idx}_test_results.npz'), confuse_matrix=confuse_mat,
                 acc=acc, f1_score=weighted_f1, true_np_test=true_np_test, preds_np_test=preds_np_test)
