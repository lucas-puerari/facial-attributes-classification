import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yaml
import shutil
import datetime
from tqdm import tqdm
import tensorflow as tf

from architectures.cnn import CNN
from utils import  preparing, preprocessing, augmentation, normalization, plot_dataset_image, plot_confusion_matrix

print('TensorFlow version:', tf.__version__)


ROOT_DIR = os.getcwd()
CONFIG_FILE = os.path.join(ROOT_DIR, 'src', 'config.yml')

with open(CONFIG_FILE) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

DATASET_DIR = os.path.join(ROOT_DIR, config['project']['dataset'])
IMAGES_DIR = os.path.join(DATASET_DIR, config['project']['images'])
LABELS_FILE = os.path.join(DATASET_DIR, config['project']['labels'])
MODELS_DIR = os.path.join(ROOT_DIR, config['project']['models'])
LOGS_DIR = os.path.join(ROOT_DIR, config['project']['logs'])

# Dataset
SAMPLES_LIMIT = config['dataset']['samples_limit']
# Image
IMAGE_FORMAT = config['dataset']['image']['format']
IMAGE_SIZE = config['dataset']['image']['size']
IMAGE_CHANNELS = config['dataset']['image']['channels']
# Labels
LABELS = config['dataset']['labels']
# Model
FILTERS = config['architecture']['filters']
LATENT_DIM = config['architecture']['latent_dim']
# Training
SPLIT_THRESHOLDS = config['training']['splits']
OPTIMIZER = config['training']['optimizer']['name']
LEARNING_RATE = config['training']['optimizer']['learning_rate']
LOSS = config['training']['loss']
EVALUATOR = config['training']['evaluator']
EPOCHS = config['training']['epochs']
BATCH_SIZE = config['training']['batch_size']

# Load images
pattern = os.path.join(IMAGES_DIR, f'*.{IMAGE_FORMAT}')
images = tf.data.Dataset.list_files(pattern, shuffle=True)
images = images.take(SAMPLES_LIMIT) if SAMPLES_LIMIT is not None else images
length = tf.data.experimental.cardinality(images).numpy()
print(f'Dataset: {length}')

# Preparing
dataset = images.map(
    lambda image: tf.numpy_function(preparing, [image, LABELS_FILE], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE   
)

# Preprocessing
dataset = dataset.map(
    lambda image, labels: tf.py_function(preprocessing, [image, IMAGE_SIZE, labels], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

# Split
train_index = round(length * SPLIT_THRESHOLDS[0])
test_index = round(length * SPLIT_THRESHOLDS[1])
validate_index = round(length * SPLIT_THRESHOLDS[2])

train_set = dataset.take(train_index)
test_set = dataset.skip(train_index).take(test_index)
validate_set = dataset.skip(train_index + test_index).take(validate_index)

train_set_len = tf.data.experimental.cardinality(train_set).numpy()
test_set_len = tf.data.experimental.cardinality(test_set).numpy()
validate_set_len = tf.data.experimental.cardinality(validate_set).numpy()

print('Training set:', train_set_len)
print('Test set:', test_set_len)
print('Validation set:', validate_set_len)

train_set = train_set.cache()
test_set = test_set.cache()
validate_set = validate_set.cache()

# Augmentation
train_set = train_set.map(
    lambda image, labels: tf.py_function(augmentation, [image, labels], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

# Normalization
train_set = train_set.map(
    lambda image, labels: tf.py_function(normalization, [image, labels], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

test_set = test_set.map(
    lambda image, labels: tf.py_function(normalization, [image, labels], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

validate_set = validate_set.map(
    lambda image, labels: tf.py_function(normalization, [image, labels], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

# Batching and prefetching
train_set = train_set.shuffle(train_set_len).batch(BATCH_SIZE)
test_set = test_set.batch(BATCH_SIZE)
validate_set = validate_set.batch(BATCH_SIZE)

train_set = train_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_set = test_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validate_set = validate_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Setup model
model = CNN(
    image_size = IMAGE_SIZE,
    channels_num = IMAGE_CHANNELS,
    filters = FILTERS,
    latent_dim = LATENT_DIM, 
    classes_num = len(LABELS)
)

model.compile(
    optimizer=OPTIMIZER,
    learning_rate=LEARNING_RATE,
    loss=LOSS,
    evaluator=EVALUATOR
)

# Setup training
experiment_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

model_dir = os.path.join(MODELS_DIR, experiment_dir)
os.makedirs(model_dir)

shutil.copyfile(CONFIG_FILE, os.path.join(model_dir, 'config.yml'))

log_dir = os.path.join(LOGS_DIR, experiment_dir)
summary_writer = tf.summary.create_file_writer(log_dir)

plot_dataset_image(next(iter(train_set))[0][0], f'{log_dir}/train_image')
plot_dataset_image(next(iter(test_set))[0][0], f'{log_dir}/test_image')
plot_dataset_image(next(iter(validate_set))[0][0], f'{log_dir}/validate_image')

# Training
for epoch in tqdm(range(EPOCHS)):

    # Train
    for batch, train_batch in enumerate(train_set):
        model.train_step(train_batch)

    # Test
    for batch, test_batch in enumerate(test_set):
        model.test_step(test_batch)

    # Save best model
    if epoch > (EPOCHS / 2):
        model.save_best_model(model_dir)

    # Log
    with summary_writer.as_default():
        model.log(epoch)

    # Reset losses
    model.reset_losses_state()


validate_result = [[[], []] for _ in list(range(0, len(LABELS)))]

# Validation
for batch, validate_batch in enumerate(validate_set):
    images, labels = validate_batch

    prediction = model(images, training=False)
    prediction = tf.math.round(prediction)

    for i in range(0, len(LABELS)):
        validate_result[i][0] = validate_result[i][0] + list(tf.gather(labels, i, axis=1).numpy())
        validate_result[i][1] = validate_result[i][1] + list(tf.gather(prediction, i, axis=1).numpy())


# Plot confusion matrix
cm_file = os.path.join(model_dir, 'cm')
plot_confusion_matrix(validate_result, LABELS, cm_file)