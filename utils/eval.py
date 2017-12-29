from __future__ import print_function, absolute_import

import torch

__all__ = ['accuracy', 'seg_accuracy', 'intersectionAndUnion']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def seg_accuracy(batch_data, pred):
    (imgs, segs, infos) = batch_data
    _, preds = torch.max(pred.data.cpu(), dim=1)
    valid = (segs >= 0)
    acc = 1.0 * torch.sum(valid * (preds == segs)) / (torch.sum(valid) + 1e-10)
    return acc * 100.0, torch.sum(valid)


def intersectionAndUnion(batch_data, pred, numClass):
    (imgs, segs, infos) = batch_data
    _, preds = torch.max(pred.data.cpu(), dim=1)

    # compute area intersection
    intersect = preds.clone()
    intersect[torch.ne(preds, segs)] = -1

    area_intersect = torch.histc(intersect.float(),
                                 bins=numClass,
                                 min=0,
                                 max=numClass - 1)

    # compute area union:
    preds[torch.lt(segs, 0)] = -1
    area_pred = torch.histc(preds.float(),
                            bins=numClass,
                            min=0,
                            max=numClass - 1)
    area_lab = torch.histc(segs.float(),
                           bins=numClass,
                           min=0,
                           max=numClass - 1)
    area_union = area_pred + area_lab - area_intersect
    return area_intersect, area_union