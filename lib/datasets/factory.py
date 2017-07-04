# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

from datasets.baggage2 import baggage2
baggage2_devkit_path = "./data/baggage2"
for split in ['train', 'val', 'test']:
    name = '{}_{}'.format('baggage2', split)
    __sets[name] = (lambda split=split: baggage2(split, baggage2_devkit_path))

from datasets.baggage6 import baggage6
baggage6_devkit_path = "./data/baggage6"
for split in ['train', 'val', 'test']:
    name = '{}_{}'.format('baggage6', split)
    __sets[name] = (lambda split=split: baggage6(split, baggage6_devkit_path))

from datasets.dleb17 import dleb17
dleb17_devkit_path = "./data/dleb17"
for split in ['train', 'test']:
    name = '{}_{}'.format('dleb17', split)
    __sets[name] = (lambda split=split: dleb17(split, dleb17_devkit_path))


from datasets.xray_bags import xray_bags
xray_bags_devkit_path = "./data/xray_bags"
for split in ['train', 'val', 'test']:
        name = '{}_{}'.format('xray_bags', split)
        __sets[name] = (lambda split=split: xray_bags(split, xray_bags_devkit_path))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
