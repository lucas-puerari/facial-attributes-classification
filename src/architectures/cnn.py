import numpy as np
import tensorflow as tf
from keras import Model, layers, optimizers, losses, metrics, activations


class WeightedLoss(losses.Loss):
    def __init__(self, name):
        super().__init__(name=name)
        self.loss = losses.get({
            "class_name": name,
            "config": {
                "reduction": "none",
                "label_smoothing": 0.1,
            }
        })

    def call(self, labels, prediction):
        batch_size = labels.shape[0]
        non_zeros = tf.math.count_nonzero(labels, axis=0)

        zero_weight = tf.cast(batch_size - non_zeros, tf.float32) / batch_size
        non_zeros_weight = tf.cast(non_zeros, tf.float32) / batch_size

        weights = (tf.cast(1.0, tf.float32) - labels) * zero_weight + labels * non_zeros_weight
        weights = tf.transpose(weights)

        loss = self.loss(labels, prediction)

        return loss * weights


class DownSampling(Model):

    def __init__(self, name, filter):
        super(DownSampling, self).__init__()
        self._name = name

        self.conv = layers.Conv2D(filter, (3, 3), strides=1, padding='same')
        self.bn = layers.BatchNormalization()
        self.act = layers.LeakyReLU()
        self.pool = layers.MaxPooling2D((2, 2))

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)

        return x


class CNN(Model):

    def __init__(self, image_size, channels_num, filters, latent_dim, classes_num):
        super(CNN, self).__init__()
        self._name = 'cnn'
        self.channels_num = channels_num
        self.best_loss = np.inf
        self.classes_num = classes_num

        input_shape = (None, image_size, image_size, channels_num)

        self.inp = layers.InputLayer(input_shape=input_shape[1:])

        self.downsamplings = []
        for id, filter in enumerate(filters):
            self.downsamplings.append(
                DownSampling(f'down_sampling_{id}', filter)
            )

        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(units=latent_dim)
        self.act1 = layers.LeakyReLU()
        self.dense2 = layers.Dense(units=classes_num)
        self.act2 = layers.Activation(activations.sigmoid)

        self.build(input_shape=input_shape)
        self.summary()

    def call(self, input):
        x = self.inp(input)

        for ds in self.downsamplings:
            x = ds(x)

        x = self.flat(x)
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dense2(x)
        out = self.act2(x)

        return out

    def compile(self, optimizer, learning_rate, loss, evaluator):
        super(CNN, self).compile()

        self.optimizer = optimizers.get({
            "class_name": optimizer,
            "config": {
                "learning_rate": learning_rate
            }
        })
        # self.loss = WeightedLoss(loss)
        self.loss = losses.get({
            "class_name": loss,
            "config": {
                "reduction": "none",
            }
        })
        self.evaluator = metrics.get(evaluator)

        self.training_loss_tracker = metrics.Mean()
        self.test_loss_tracker = metrics.Mean()
        self.training_evaluator_tracker = metrics.Mean()
        self.test_evaluator_tracker = metrics.Mean()

    @property
    def metrics(self):
        return [
            self.training_loss_tracker,
            self.test_loss_tracker,
            self.training_evaluator_tracker,
            self.test_evaluator_tracker
        ]

    @tf.function
    def train_step(self, train_batch):
        images, labels = train_batch

        with tf.GradientTape() as tape:
            prediction = self(images)
            loss = self.loss(labels, prediction)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.training_loss_tracker.update_state(loss)

        accuracy = self.evaluator(labels, prediction)
        self.training_evaluator_tracker.update_state(accuracy)

        return {
            "loss": self.training_loss_tracker.result(),
            "evaluator": self.training_evaluator_tracker.result()
        }

    @tf.function
    def test_step(self, test_batch):
        images, labels = test_batch

        prediction = self(images)
        loss = self.loss(labels, prediction)
        
        self.test_loss_tracker.update_state(loss)

        accuracy = self.evaluator(labels, prediction)
        self.test_evaluator_tracker.update_state(accuracy)

        return {
            "loss": self.test_loss_tracker.result(),
            "evaluator": self.test_evaluator_tracker.result()
        }

    def save_best_model(self, model_dir):
        loss_to_monitor = self.test_loss_tracker.result()

        if self.best_loss > loss_to_monitor:
            self.best_loss = loss_to_monitor
            self.save(model_dir)

    def log(self, epoch):
        tf.summary.scalar(
            'Decoded Loss/Training',
            self.training_loss_tracker.result(),
            step=epoch
        )
        tf.summary.scalar(
            'Decoded Loss/Test',
            self.test_loss_tracker.result(),
            step=epoch
        )
        tf.summary.scalar(
            'Accuracy/Training',
            self.training_evaluator_tracker.result(),
            step=epoch
        )
        tf.summary.scalar(
            'Accuracy/Test',
            self.test_evaluator_tracker.result(),
            step=epoch
        )

    def reset_losses_state(self):
        self.training_loss_tracker.reset_states()
        self.test_loss_tracker.reset_states()
        self.training_evaluator_tracker.reset_states()
        self.test_evaluator_tracker.reset_states()
