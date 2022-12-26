import PIL
import json
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_dataset_image(image, filename):
    image = tf.cond(
        tf.math.less_equal(tf.math.reduce_max(image), 1),
        lambda: tf.math.multiply(image, 255),
        lambda: image
    )

    image = image.numpy().astype('uint8')
    image = PIL.Image.fromarray(image)
    image.save(f'{filename}.png')


def plot_confusion_matrix(validate_result, LABELS, filename):
    fig = plt.figure(figsize = (15, 8))
    for i, (label, result) in enumerate(zip(LABELS, validate_result)):
        cm = confusion_matrix(result[0], result[1])
        labels = [f'Not_{label}', label]

        plt.subplot(2, int(len(LABELS) / 2), i + 1)
        sns.heatmap(
            cm, annot = True, fmt='d', cbar = False, cmap = 'Blues',
            xticklabels = labels, yticklabels = labels, linecolor = 'black', linewidth = 1
        )
        plt.title(labels[1])

    plt.tight_layout()
    fig.savefig(f'{filename}.png', dpi=fig.dpi)


def plot_labels_distribution(LABELS_FILE, LABELS, filename):
    with open(LABELS_FILE) as f:
        labels_file = json.load(f)

    labels_items = list(labels_file.items())
    labels_items = [image_labels for (image_name, image_labels) in labels_items]
        
    labels_values = [list(item.values()) for item in labels_items]
    labels_values = np.asarray(labels_values, dtype=int)

    labels_counts = np.sum(labels_values, axis=0)
    trasposed_labels_counts = len(labels_items) - labels_counts

    fig, ax = plt.subplots()

    line_up = ax.bar(range(len(labels_counts)), labels_counts, label='1')
    line_down = ax.bar(range(len(trasposed_labels_counts)), trasposed_labels_counts, bottom=labels_counts, label='0')
    ax.legend(handles=[line_up, line_down])

    plt.title(f'Dataset count: {len(labels_items)}')
    plt.xticks(range(0, len(LABELS)), LABELS, rotation='vertical')

    plt.tight_layout()
    fig.savefig(f'{filename}.png', dpi=fig.dpi)
