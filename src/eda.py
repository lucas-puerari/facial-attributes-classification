import os
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt

from utils.pipeline import plot_labels_distribution


ROOT_DIR = os.getcwd()
CONFIG_FILE = os.path.join(ROOT_DIR, 'src', 'config.yml')

with open(CONFIG_FILE) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

DATASET_DIR = os.path.join(ROOT_DIR, config['project']['dataset'])
LABELS_FILE = os.path.join(DATASET_DIR, config['project']['labels'])
EDA_DIR = os.path.join(ROOT_DIR, config['project']['eda'])

# Labels
LABELS  = config['dataset']['labels']

plot_labels_distribution(LABELS_FILE, LABELS, f'{EDA_DIR}/labels-distribution')
