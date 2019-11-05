import numpy as np
import os
import shutil
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from pathlib import Path
import json

class COCOLoader(COCO):
    def __init__(self, data_dir):
        """
        Parameters
        ----------
        data_dir
        str: location of COCO data files
        """
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir.joinpath('annotations/instances_train2017.json')
        self.train_dir = self.data_dir.joinpath('train2017')
        super(COCOLoader, self).__init__(self.annotations_dir.as_posix())
        self.images = list(self.train_dir.glob('*.jpg'))
        #self.images = {os.path.splitext(os.path.basename(i.as_posix()))[0].lstrip('0'): i for i in self.images}
        self.images = {int(os.path.splitext(os.path.basename(i.as_posix()))[0]): i for i in self.images}
        return

    def random_subset(self, count):
        images = np.random.choice(list(self.images.keys()), size=count)
        return {i:self.images[i] for i in images}

    def create_subset(self, images, directory):
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir()
        train_subset = directory.joinpath('train2017')
        if not train_subset.exists():
            train_subset.mkdir()
        annotation_dir = directory.joinpath('annotations')
        if not annotation_dir.exists():
            annotation_dir.mkdir()
        for image in images:
            shutil.copy(self.images[image], train_subset.joinpath(os.path.basename(self.images[image])))
        annotation_ids = self.getAnnIds(images)
        annotations = self.loadAnns(annotation_ids)
        with open(annotation_dir.joinpath('instances_train2017.json'), 'w') as anno_file:
            anno_file.write(json.dumps(annotations))

class COCOSubsetter(object):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.instance_file = self.data_dir.joinpath('annotations/instances_train2017.json')
        self.train_dir = self.data_dir.joinpath('train2017')
        self.images = list(self.train_dir.glob('*.jpg'))
        self.images = {int(os.path.splitext(os.path.basename(i.as_posix()))[0]): i for i in self.images}
        return

    def random_subset(self, count):
        images = np.random.choice(list(self.images.keys()), size=count)
        return {i:self.images[i] for i in images}

    def load_annotations(self):
        with open(self.instance_file) as infile:
            self.instances = json.load(infile)

    def filter_subset(self, images):
        subset = dict()
        subset['info'] = self.instances['info']
        subset['licenses'] = self.instances['licenses']
        subset['categories'] = self.instances['categories']
        subset['annotations'] = [i for i in self.instances['annotations'] if i['image_id'] in images]
        subset['images'] = [i for i in self.instances['images'] if i['id'] in images]
        return subset




coco_loader = COCOSubsetter('/Users/jbsnyder/PycharmProjects/mrcnn-notebooks/data')

coco_loader.load_annotations()
images = coco_loader.random_subset(10)
coco_loader.filter_subset(list(images.keys()))