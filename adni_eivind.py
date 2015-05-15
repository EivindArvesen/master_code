#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset-class for using custom preprocessed ADNI MRI dataset with Pylearn2.

A simple general csv dataset wrapper for pylearn2.
Can do automatic one-hot encoding based on labels present in a file.
Based upon work from Zygmunt Zajac's (zygmunt@fastml.com) CSVDataset class in the Pylearn2 repo.

FIX based on answer from
http://stackoverflow.com/questions/27609843/pylearn2-csvdataset-typeerror
"""
__authors__ = "Eivind Arvesen"
__license__ = "3-clause BSD"
__email__ = "eivind.arvesen@gmail.com"

import numpy as np

from pylearn2.datasets import dense_design_matrix
from pylearn2.format.target_format import convert_to_one_hot
from pylearn2.utils.string_utils import preprocess
from pylearn2.config import yaml_parse


# Dynamically change class based on dataset size?
# http://stackoverflow.com/questions/9539052/python-dynamically-changing-base-classes-at-runtime-how-to


class CSVDatasetPlus(dense_design_matrix.DenseDesignMatrixPyTables):

    """A generic class for accessing CSV files.

    Labels, if present, should be in the first column
    if there's no labels, set expect_labels to False
    if there's no header line in your file, set expect_headers to False

    Parameters
    ----------
    path : str
      The path to the CSV file.

    task : str
      The type of task in which the dataset will be used -- either
      "classification" or "regression".  The task determines the shape of the
      target variable.  For classification, it is a vector; for regression, a
      matrix.

    expect_labels : bool
      Whether the CSV file contains a target variable in the first column.

    expect_headers : bool
      Whether the CSV file contains column headers.

    delimiter : str
      The CSV file's delimiter.

    start : int
      The first row of the CSV file to load.

    stop : int
      The last row of the CSV file to load.

    start_fraction : float
      The fraction of rows, starting at the beginning of the file, to load.

    end_fraction : float
      The fraction of rows, starting at the end of the file, to load.


    *** NEW, EIVIND, ADNI ***


    yaml_src : string
      Lorem ipsum

    one_hot : bool
      Lorem ipsum

    num_classes : int
      Lorem ipsum

    which_set : string
      Lorem ipsum
    """

    def __init__(self,
                 path='train.csv',
                 task='classification',
                 expect_labels=True,
                 expect_headers=True,
                 delimiter=',',
                 start=None,
                 stop=None,
                 start_fraction=None,
                 end_fraction=None,
                 yaml_src=None,
                 one_hot=True,
                 num_classes=4,
                 which_set=None):
        """
        .. todo:: ..

            WRITEME
        """
        self.path = path
        self.task = task
        self.expect_labels = expect_labels
        self.expect_headers = expect_headers
        self.delimiter = delimiter
        if which_set is not None:
            self.start = start
            self.stop = stop
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction

        self.view_converter = None

        if yaml_src is not None:
            self.yaml_src = yaml_parse.load_path(yaml_src)
        # self.yaml_src=yaml_parse.load_path("mlp.yaml")
        # eventually; triple-quoted yaml...
        self.one_hot = one_hot
        self.num_classes = num_classes

        if which_set is not None and which_set not in[
                                                     'train', 'test', 'valid']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test","valid"].')
        else:
            self.which_set = which_set
            if self.start is not None or self.stop is not None:
                raise ValueError("Use start/stop or which_set,"
                    " just not together.")

        if task not in ['classification', 'regression']:
            raise ValueError('task must be either "classification" or '
                             '"regression"; got ' + str(task))

        if start_fraction is not None:
            if end_fraction is not None:
                raise ValueError("Use start_fraction or end_fraction, "
                                 " not both.")
            if start_fraction <= 0:
                raise ValueError("start_fraction should be > 0")

            if start_fraction >= 1:
                raise ValueError("start_fraction should be < 1")

        if end_fraction is not None:
            if end_fraction <= 0:
                raise ValueError("end_fraction should be > 0")

            if end_fraction >= 1:
                raise ValueError("end_fraction should be < 1")

        if start is not None:
            if start_fraction is not None or end_fraction is not None:
                raise ValueError("Use start, start_fraction, or end_fraction,"
                                 " just not together.")

        if stop is not None:
            if start_fraction is not None or end_fraction is not None:
                raise ValueError("Use stop, start_fraction, or end_fraction,"
                                 " just not together.")

        # and go
        self.path = preprocess(self.path)
        X, y = self._load_data()

        # y=y.astype(int)
        # y=map(int, np.rint(y).astype(int))

        if self.task == 'regression':
            super(CSVDatasetPlus, self).__init__(X=X, y=y)
        else:
            # , y_labels=4 # y_labels=np.max(y)+1
            super(CSVDatasetPlus, self).__init__(
                X=X, y=y.astype(int), y_labels=self.num_classes)

    # new interface (get_...)?
    # see pylearn2/datasets/cifar10.py
    # see pylearn2/datasets/svhn.py

    def _load_data(self):
        """
        .. todo:: ..

            WRITEME
        """
        assert self.path.endswith('.csv')

        if self.expect_headers:
            data = np.loadtxt(self.path,
                              delimiter=self.delimiter,
                              skiprows=1, dtype=int)
        else:
            data = np.loadtxt(self.path, delimiter=self.delimiter, dtype=int)

        def take_subset(X, y):
            self.num_classes = np.unique(y).shape[0]
            if self.which_set is not None:
                total = X.shape[0]
                train = int((total/100.0)*70)
                test = int((total/100.0)*15)
                valid = int((total/100.0)*15)
                train += total - train - test - valid
                if self.which_set == 'train':
                    self.start = 0
                    self.stop = train -1
                elif self.which_set == 'test':
                    self.start = train
                    self.stop = train + test -1
                elif self.which_set == 'valid':
                    self.start = train + test
                    self.stop = train + test + valid -1

            if self.start_fraction is not None:
                n = X.shape[0]
                subset_end = int(self.start_fraction * n)
                X = X[0:subset_end, :]
                y = y[0:subset_end]
            elif self.end_fraction is not None:
                n = X.shape[0]
                subset_start = int((1 - self.end_fraction) * n)
                X = X[subset_start:, ]
                y = y[subset_start:]
            elif self.start is not None:
                X = X[self.start:self.stop, ]
                if y is not None:
                    y = y[self.start:self.stop]

            return X, y

        if self.expect_labels:
            y = data[:, 0]
            X = data[:, 1:]
            y = y.reshape((y.shape[0], 1))

        else:
            X = data
            y = None

        X, y = take_subset(X, y)

        if self.one_hot:
            y = convert_to_one_hot(
                y, dtype=int, max_labels=self.num_classes, mode="concatenate")

        return X, y  # .astype(float)
