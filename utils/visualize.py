import os
import time
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, ticker


def plot_heatmap(m, pdb_id=None, rc=None, seq_q=None, seq_v=None):
    fig, ax = plt.subplots()
    ax.imshow(m, cmap='jet', interpolation='bicubic')
    if rc:
        for r in rc:
            x0 = r.x1
            y0 = r.y1
            w = r.x2 - x0
            h = r.y2 - y0
            rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
    if pdb_id:
        ax.set_title(pdb_id)
    if seq_q is not None and seq_v is not None:
        y_labels = seq_q
        x_labels = seq_v
        positions_y = np.arange(m.shape[0])
        positions_x = np.arange(m.shape[1])
        ax.yaxis.set_major_locator(ticker.FixedLocator(positions_y))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(y_labels))
        ax.xaxis.set_major_locator(ticker.FixedLocator(positions_x))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_labels))
        ax.set_yticklabels(y_labels)
        ax.set_xticklabels(x_labels)
    plt.show()


def plot_spectrum(m, pdb_id=None, rc=None, seq_q=None, seq_v=None):
    fig, ax = plt.subplots()
    y = m
    x = np.linspace(0, len(seq_q), num=len(seq_q))
    ax.plot(x, y)
    print(seq_q)
    if pdb_id:
        ax.set_title(pdb_id)
    if seq_q is not None and seq_v is not None:
        x_labels = seq_v
        positions_x = np.arange(x.shape[0])
        ax.xaxis.set_major_locator(ticker.FixedLocator(positions_x))
        ax.set_xticklabels(x_labels)
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(x_labels))

    plt.show()
