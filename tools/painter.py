import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

__all__ = ['draw_curve']


def draw_curve(arch, ckpt):
    log_path = '{}/log.txt'.format(ckpt)
    curve_path = '{}/{}_curve.png'.format(ckpt, arch)

    f = open(log_path, 'r')

    train_acc = []
    val_acc = []
    for i in f:
        if i.strip().split('\t')[0] == 'Learning Rate':
            continue
        train_acc.append(100 - float(i.strip().split('\t')[3]))
        val_acc.append(100 - float(i.strip().split('\t')[4]))

    plt.ylim(0, 100)
    plt.xlabel('epochs')
    plt.ylabel('top-1 error(%)')

    # plt.plot(train_acc)
    plt.plot(train_acc, color='red')
    plt.plot(val_acc, color='green')

    plt.legend(['{}-train-error'.format(arch), '{}-val-error'.format(arch)])
    plt.grid(True)

    plt.savefig(curve_path, dpi=150)
