#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the different MLP variation experiments.

Todo: Complete,
    write...
"""

__authors__ = "Eivind Arvesen"
__copyright__ = "Copyright 2015, Eivind Arvesen"
__credits__ = ["Eivind Arvesen"]
__license__ = "3-clause BSD"
__maintainer__ = "Eivind Arvesen"
__email__ = "eivind.arvesen@gmail.com"

# from pylearn2.utils import serial
# train_obj = serial.load_train_file('experimental.yaml')
import csv
import errno
import glob
import numpy as np
import os
import sys
from pylearn2.config import yaml_parse

experimental = []

experimental.append(
    {'title': "experimental",
    'yaml': """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.dataset_iterators.StratifiedDatasetValidationKFold {
        dataset: &train !obj:adni_eivind.CSVDatasetPlus {
            path: &path %(dataset)s,
            which_set: "train",
            task: "classification",
        },
        n_folds: 10 # 10 fold cross-validation!
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: %(nvis)d, # 32 # 184320
        layers: [
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h0',
                dim: 16, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .05, # .005
            },
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h1',
                dim: 8, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .005, # .005
            },
            # !obj:pylearn2.models.mlp.Sigmoid {
            #     layer_name: 'h1',
            #     dim: 400, # dimension, i.e. hidden neurons
            #     sparse_init: 15, # number of non-sparse weights per neuron
            #     # irange: 0. # .005
            # },
            # !obj:pylearn2.models.maxout.Maxout {
            #          layer_name: 'h2',
            #          num_units: 240,
            #          num_pieces: 5,
            #          irange: .005,
            #          max_col_norm: 1.9365,
            # },
            # !obj:pylearn2.models.mlp.Linear {
            #     layer_name: 'h3',
            #     dim: 100, # dimension, i.e. hidden neurons
            #     irange: .05,
            # },
            # !obj:pylearn2.models.mlp.RectifiedLinear {
            #     layer_name: 'h4',
            #     dim: 400, # dimension, i.e. hidden neurons
            #     irange: .005,
            # # Rather than using weight decay, we constrain the norms of the weight vectors
            # max_col_norm: 2.
            # },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: %(n_classes)d,# 2 # *num_classes, # output neurons, i.e. classes
                irange: 0. # .005
                # sparse_init: 15 # either this, or previous line...
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: 5e-3,
        batch_size: 10, # 10 # 1162
        # learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
        #     init_momentum: 5e-1,
        # },
        monitoring_dataset:
            {
                'train' : *train,
                'test' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "test",
                    task: "classification",
                },
                'valid' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "valid",
                    task: "classification",
                }
            },
        # The SumOfCosts allows us to add together a few terms to make a complicated cost function.
        # cost : !obj:pylearn2.costs.cost.SumOfCosts {
        #     costs: [
        #         # First: Dropout.
        #         !obj:pylearn2.costs.mlp.dropout.Dropout {
        #             input_include_probs: { 'h0' : .8 , 'h1' : .8  }, # .5
        #             input_scales: { 'h0' : 1. , 'h1' : 1.  } # 2.
        #         },
        #         # The second term of our cost function is a little bit of weight decay (L1).
        #         !obj:pylearn2.costs.mlp.L1WeightDecay {
        #             coeffs: [ .0001  ]
        #         },
        #         # Finally, we use more weight decay (L2)
        #         !obj:pylearn2.costs.mlp.WeightDecay {
        #             coeffs: [ .0005  ]
        #         },
        #     ],
        # },
        # # Couldn't work this out... Error possibly stemming from CV-usage...
        # cost: !obj:pylearn2.costs.mlp.WeightDecay {
        #     coeffs: { 'h0': .0005, 'y': .0005 }
        # },
        # Dropout...
        # cost: !obj:pylearn2.costs.mlp.dropout.Dropout {
        #     input_include_probs: { 'h0' : .5 }, # .5
        #     input_scales: { 'h0' : 2. } # 2.
        # },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: %(max_epochs)d #10000
                },
                !obj:pylearn2.termination_criteria.And {
                    criteria: [
                        !obj:pylearn2.termination_criteria.MonitorBased {
                            channel_name: "valid_y_misclass",
                            prop_decrease: 0., # 0.01 # 0.001
                            N: %(prop_decrease_over)d # 100 # 1000
                        },
                        !obj:pylearn2.termination_criteria.ChannelTarget {
                            channel_name: "valid_y_misclass",
                            target: %(min_misclass)f # Stop at 0, no point in overfitting
                        }
                    ]
                }
            ]
        },
        # update_callbacks: !obj:pylearn2.training_algorithms.sgd.ExponentialDecay {
        #     decay_factor: 1.000004,
        #     min_lr: .000001
        # }
    },
    cv_extensions: [
        !obj:pylearn2.cross_validation.train_cv_extensions.MonitorBasedSaveBestCV {
            channel_name: 'valid_y_misclass',
            save_path: %(save_path_best)s,
            save_folds: True
        },
    ],
    save_path: %(save_path_full)s,
    save_freq: 1,
    save_folds: True
}
"""})


experiments = []


experiments.append(
    {'title': "Baseline",
    'yaml': """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.dataset_iterators.StratifiedDatasetValidationKFold {
        dataset: &train !obj:adni_eivind.CSVDatasetPlus {
            path: &path %(dataset)s,
            which_set: "train",
            task: "classification",
        },
        n_folds: 3 # 10 fold cross-validation!
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: %(nvis)d, # 32 # 184320
        layers: [
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h0',
                dim: 20, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .05, # .005
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: %(n_classes)d,# 2 # *num_classes, # output neurons, i.e. classes
                irange: 0. # .005
                # sparse_init: 15 # either this, or previous line...
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: 5e-3,
        batch_size: 10, # 10 # 1162
        monitoring_dataset:
            {
                'train' : *train,
                'test' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "test",
                    task: "classification",
                },
                'valid' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "valid",
                    task: "classification",
                }
            },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: %(max_epochs)d #10000
                },
                !obj:pylearn2.termination_criteria.And {
                    criteria: [
                        !obj:pylearn2.termination_criteria.MonitorBased {
                            channel_name: "valid_y_misclass",
                            prop_decrease: 0., # 0.01 # 0.001
                            N: %(prop_decrease_over)d # 100 # 1000
                        },
                        !obj:pylearn2.termination_criteria.ChannelTarget {
                            channel_name: "valid_y_misclass",
                            target: %(min_misclass)f # Stop at 0, no point in overfitting
                        }
                    ]
                }
            ]
        },
    },
    cv_extensions: [
        !obj:pylearn2.cross_validation.train_cv_extensions.MonitorBasedSaveBestCV {
            channel_name: 'valid_y_misclass',
            save_path: %(save_path_best)s,
            save_folds: True
        },
    ],
    save_path: %(save_path_full)s,
    save_freq: 1,
    save_folds: True
}
"""})
experiments.append(
    {'title': "Baseline_faster",
    'yaml': """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.dataset_iterators.StratifiedDatasetValidationKFold {
        dataset: &train !obj:adni_eivind.CSVDatasetPlus {
            path: &path %(dataset)s,
            which_set: "train",
            task: "classification",
        },
        n_folds: 3 # 10 fold cross-validation!
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: %(nvis)d, # 32 # 184320
        layers: [
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h0',
                dim: 20, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .05, # .005
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: %(n_classes)d,# 2 # *num_classes, # output neurons, i.e. classes
                irange: 0. # .005
                # sparse_init: 15 # either this, or previous line...
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: 5e-2,
        batch_size: 10, # 10 # 1162
        monitoring_dataset:
            {
                'train' : *train,
                'test' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "test",
                    task: "classification",
                },
                'valid' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "valid",
                    task: "classification",
                }
            },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: %(max_epochs)d #10000
                },
                !obj:pylearn2.termination_criteria.And {
                    criteria: [
                        !obj:pylearn2.termination_criteria.MonitorBased {
                            channel_name: "valid_y_misclass",
                            prop_decrease: 0., # 0.01 # 0.001
                            N: %(prop_decrease_over)d # 100 # 1000
                        },
                        !obj:pylearn2.termination_criteria.ChannelTarget {
                            channel_name: "valid_y_misclass",
                            target: %(min_misclass)f # Stop at 0, no point in overfitting
                        }
                    ]
                }
            ]
        },
    },
    cv_extensions: [
        !obj:pylearn2.cross_validation.train_cv_extensions.MonitorBasedSaveBestCV {
            channel_name: 'valid_y_misclass',
            save_path: %(save_path_best)s,
            save_folds: True
        },
    ],
    save_path: %(save_path_full)s,
    save_freq: 1,
    save_folds: True
}
"""})
experiments.append(
    {'title': "More_neurons",
    'yaml': """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.dataset_iterators.StratifiedDatasetValidationKFold {
        dataset: &train !obj:adni_eivind.CSVDatasetPlus {
            path: &path %(dataset)s,
            which_set: "train",
            task: "classification",
        },
        n_folds: 3 # 10 fold cross-validation!
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: %(nvis)d, # 32 # 184320
        layers: [
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h0',
                dim: 50, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .005, # .005
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: %(n_classes)d,# 2 # *num_classes, # output neurons, i.e. classes
                irange: 0. # .005
                # sparse_init: 15 # either this, or previous line...
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: 5e-3,
        batch_size: 10, # 10 # 1162
        monitoring_dataset:
            {
                'train' : *train,
                'test' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "test",
                    task: "classification",
                },
                'valid' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "valid",
                    task: "classification",
                }
            },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: %(max_epochs)d #10000
                },
                !obj:pylearn2.termination_criteria.And {
                    criteria: [
                        !obj:pylearn2.termination_criteria.MonitorBased {
                            channel_name: "valid_y_misclass",
                            prop_decrease: 0., # 0.01 # 0.001
                            N: %(prop_decrease_over)d # 100 # 1000
                        },
                        !obj:pylearn2.termination_criteria.ChannelTarget {
                            channel_name: "valid_y_misclass",
                            target: %(min_misclass)f # Stop at 0, no point in overfitting
                        }
                    ]
                }
            ]
        },
    },
    cv_extensions: [
        !obj:pylearn2.cross_validation.train_cv_extensions.MonitorBasedSaveBestCV {
            channel_name: 'valid_y_misclass',
            save_path: %(save_path_best)s,
            save_folds: True
        },
    ],
    save_path: %(save_path_full)s,
    save_freq: 1,
    save_folds: True
}
"""})
experiments.append(
    {'title': "Two_layers",
    'yaml': """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.dataset_iterators.StratifiedDatasetValidationKFold {
        dataset: &train !obj:adni_eivind.CSVDatasetPlus {
            path: &path %(dataset)s,
            which_set: "train",
            task: "classification",
        },
        n_folds: 3 # 10 fold cross-validation!
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: %(nvis)d, # 32 # 184320
        layers: [
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h0',
                dim: 16, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .05, # .005
            },
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h1',
                dim: 8, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .005, # .005
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: %(n_classes)d,# 2 # *num_classes, # output neurons, i.e. classes
                irange: 0. # .005
                # sparse_init: 15 # either this, or previous line...
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: 5e-3,
        batch_size: 10, # 10 # 1162
        monitoring_dataset:
            {
                'train' : *train,
                'test' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "test",
                    task: "classification",
                },
                'valid' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "valid",
                    task: "classification",
                }
            },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: %(max_epochs)d #10000
                },
                !obj:pylearn2.termination_criteria.And {
                    criteria: [
                        !obj:pylearn2.termination_criteria.MonitorBased {
                            channel_name: "valid_y_misclass",
                            prop_decrease: 0., # 0.01 # 0.001
                            N: %(prop_decrease_over)d # 100 # 1000
                        },
                        !obj:pylearn2.termination_criteria.ChannelTarget {
                            channel_name: "valid_y_misclass",
                            target: %(min_misclass)f # Stop at 0, no point in overfitting
                        }
                    ]
                }
            ]
        },
    },
    cv_extensions: [
        !obj:pylearn2.cross_validation.train_cv_extensions.MonitorBasedSaveBestCV {
            channel_name: 'valid_y_misclass',
            save_path: %(save_path_best)s,
            save_folds: True
        },
    ],
    save_path: %(save_path_full)s,
    save_freq: 1,
    save_folds: True
}
"""})
experiments.append(
    {'title': "Two_layers_big_batch",
    'yaml': """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.dataset_iterators.StratifiedDatasetValidationKFold {
        dataset: &train !obj:adni_eivind.CSVDatasetPlus {
            path: &path %(dataset)s,
            which_set: "train",
            task: "classification",
        },
        n_folds: 3 # 10 fold cross-validation!
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: %(nvis)d, # 32 # 184320
        layers: [
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h0',
                dim: 16, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .05, # .005
            },
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h1',
                dim: 8, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .005, # .005
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: %(n_classes)d,# 2 # *num_classes, # output neurons, i.e. classes
                irange: 0. # .005
                # sparse_init: 15 # either this, or previous line...
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: 5e-3,
        batch_size: 100, # 10 # 1162
        monitoring_dataset:
            {
                'train' : *train,
                'test' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "test",
                    task: "classification",
                },
                'valid' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "valid",
                    task: "classification",
                }
            },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: %(max_epochs)d #10000
                },
                !obj:pylearn2.termination_criteria.And {
                    criteria: [
                        !obj:pylearn2.termination_criteria.MonitorBased {
                            channel_name: "valid_y_misclass",
                            prop_decrease: 0., # 0.01 # 0.001
                            N: %(prop_decrease_over)d # 100 # 1000
                        },
                        !obj:pylearn2.termination_criteria.ChannelTarget {
                            channel_name: "valid_y_misclass",
                            target: %(min_misclass)f # Stop at 0, no point in overfitting
                        }
                    ]
                }
            ]
        },
    },
    cv_extensions: [
        !obj:pylearn2.cross_validation.train_cv_extensions.MonitorBasedSaveBestCV {
            channel_name: 'valid_y_misclass',
            save_path: %(save_path_best)s,
            save_folds: True
        },
    ],
    save_path: %(save_path_full)s,
    save_freq: 1,
    save_folds: True
}
"""})
experiments.append(
    {'title': "Larger_layers",
    'yaml': """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.dataset_iterators.StratifiedDatasetValidationKFold {
        dataset: &train !obj:adni_eivind.CSVDatasetPlus {
            path: &path %(dataset)s,
            which_set: "train",
            task: "classification",
        },
        n_folds: 3 # 10 fold cross-validation!
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: %(nvis)d, # 32 # 184320
        layers: [
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h0',
                dim: 150, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .05, # .005
            },
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h1',
                dim: 300, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .005, # .005
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: %(n_classes)d,# 2 # *num_classes, # output neurons, i.e. classes
                irange: 0. # .005
                # sparse_init: 15 # either this, or previous line...
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: 5e-3,
        batch_size: 10, # 10 # 1162
        monitoring_dataset:
            {
                'train' : *train,
                'test' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "test",
                    task: "classification",
                },
                'valid' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "valid",
                    task: "classification",
                }
            },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: %(max_epochs)d #10000
                },
                !obj:pylearn2.termination_criteria.And {
                    criteria: [
                        !obj:pylearn2.termination_criteria.MonitorBased {
                            channel_name: "valid_y_misclass",
                            prop_decrease: 0., # 0.01 # 0.001
                            N: %(prop_decrease_over)d # 100 # 1000
                        },
                        !obj:pylearn2.termination_criteria.ChannelTarget {
                            channel_name: "valid_y_misclass",
                            target: %(min_misclass)f # Stop at 0, no point in overfitting
                        }
                    ]
                }
            ]
        },
    },
    cv_extensions: [
        !obj:pylearn2.cross_validation.train_cv_extensions.MonitorBasedSaveBestCV {
            channel_name: 'valid_y_misclass',
            save_path: %(save_path_best)s,
            save_folds: True
        },
    ],
    save_path: %(save_path_full)s,
    save_freq: 1,
    save_folds: True
}
"""})
experiments.append(
    {'title': "Three_layers",
    'yaml': """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.dataset_iterators.StratifiedDatasetValidationKFold {
        dataset: &train !obj:adni_eivind.CSVDatasetPlus {
            path: &path %(dataset)s,
            which_set: "train",
            task: "classification",
        },
        n_folds: 3 # 10 fold cross-validation!
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: %(nvis)d, # 32 # 184320
        layers: [
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h0',
                dim: 30, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .05, # .005
            },
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h1',
                dim: 90, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .005, # .005
            },
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h2',
                dim: 60, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .0005, # .005
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: %(n_classes)d,# 2 # *num_classes, # output neurons, i.e. classes
                irange: 0. # .005
                # sparse_init: 15 # either this, or previous line...
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: 5e-3,
        batch_size: 10, # 10 # 1162
        monitoring_dataset:
            {
                'train' : *train,
                'test' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "test",
                    task: "classification",
                },
                'valid' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "valid",
                    task: "classification",
                }
            },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: %(max_epochs)d #10000
                },
                !obj:pylearn2.termination_criteria.And {
                    criteria: [
                        !obj:pylearn2.termination_criteria.MonitorBased {
                            channel_name: "valid_y_misclass",
                            prop_decrease: 0., # 0.01 # 0.001
                            N: %(prop_decrease_over)d # 100 # 1000
                        },
                        !obj:pylearn2.termination_criteria.ChannelTarget {
                            channel_name: "valid_y_misclass",
                            target: %(min_misclass)f # Stop at 0, no point in overfitting
                        }
                    ]
                }
            ]
        },
    },
    cv_extensions: [
        !obj:pylearn2.cross_validation.train_cv_extensions.MonitorBasedSaveBestCV {
            channel_name: 'valid_y_misclass',
            save_path: %(save_path_best)s,
            save_folds: True
        },
    ],
    save_path: %(save_path_full)s,
    save_freq: 1,
    save_folds: True
}
"""})
experiments.append(
    {'title': "Three_layers_dropout",
    'yaml': """
!obj:pylearn2.cross_validation.TrainCV {
    dataset_iterator: !obj:pylearn2.cross_validation.dataset_iterators.StratifiedDatasetValidationKFold {
        dataset: &train !obj:adni_eivind.CSVDatasetPlus {
            path: &path %(dataset)s,
            which_set: "train",
            task: "classification",
        },
        n_folds: 3 # 10 fold cross-validation!
    },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: %(nvis)d, # 32 # 184320
        layers: [
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h0',
                dim: 30, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .05, # .005
            },
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h1',
                dim: 90, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .005, # .005
            },
            !obj:pylearn2.models.mlp.Tanh {
                layer_name: 'h2',
                dim: 60, # dimension, i.e. hidden neurons
                # init_bias: 0. # all biases initialized to this.
                irange: .0005, # .005
            },
            !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: %(n_classes)d,# 2 # *num_classes, # output neurons, i.e. classes
                irange: 0. # .005
                # sparse_init: 15 # either this, or previous line...
            }
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: 5e-3,
        batch_size: 10, # 10 # 1162
        monitoring_dataset:
            {
                'train' : *train,
                'test' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "test",
                    task: "classification",
                },
                'valid' : !obj:adni_eivind.CSVDatasetPlus {
                    path: *path,
                    which_set: "valid",
                    task: "classification",
                }
            },
        cost: !obj:pylearn2.costs.mlp.dropout.Dropout {
            input_include_probs: { 'h0' : .5, 'h1' : .5, 'h2' : .5  }, # .8
            input_scales: { 'h0' : 2., 'h1' : 2., 'h2' : 2. } # 1.
        },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria: [
                !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: %(max_epochs)d #10000
                },
                !obj:pylearn2.termination_criteria.And {
                    criteria: [
                        !obj:pylearn2.termination_criteria.MonitorBased {
                            channel_name: "valid_y_misclass",
                            prop_decrease: 0., # 0.01 # 0.001
                            N: %(prop_decrease_over)d # 100 # 1000
                        },
                        !obj:pylearn2.termination_criteria.ChannelTarget {
                            channel_name: "valid_y_misclass",
                            target: %(min_misclass)f # Stop at 0, no point in overfitting
                        }
                    ]
                }
            ]
        },
    },
    cv_extensions: [
        !obj:pylearn2.cross_validation.train_cv_extensions.MonitorBasedSaveBestCV {
            channel_name: 'valid_y_misclass',
            save_path: %(save_path_best)s,
            save_folds: True
        },
    ],
    save_path: %(save_path_full)s,
    save_freq: 1,
    save_folds: True
}
"""})


"""
TODO:
    Other costs (weight decay, momentum, etc.)
    Variations on base configurations...
    Multiprocessing?

    Cross-validation options...
"""

# Take first found CSV in directory as dataset
dataset = glob.glob("*.csv")[0]  # 'ADNI.csv' / 'TEST.csv'

reader = csv.reader(open(dataset))
line = reader.next()
nvis = len(line)-1

classes = []
for row in reader:
    classes.append(row[0])
classes = np.array(classes)
n_classes = np.unique(classes).shape[0]

for idx, experiment in enumerate(experiments): # experiments
    directory = "Results/Pylearn2/" + experiment['title']
    try:
        os.makedirs(directory)
    except OSError, e:
        if e.errno != errno.EEXIST:
            pass

    filename_best = directory + "/0" + str(idx) + "-" + experiment['title'] + \
        "_best.pkl"
    filename_full = directory + "/0" + str(idx) + "-" + experiment['title'] + \
        "_full.pkl"

    try:
        train_obj = yaml_parse.load(experiment['yaml'] % {
            'save_path_best': filename_best,
            'save_path_full': filename_full,
            'dataset': dataset,
            'nvis': nvis,
            'n_classes': n_classes,
            'prop_decrease_over': 25,
            'max_epochs': 10000,
            'min_misclass': 0.01
        })
    except Exception, e:
        print 'Error in "' + experiment['title'] + '":'
        print e
        sys.exit(1)
    train_obj.main_loop()
