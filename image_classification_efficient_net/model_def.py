"""
This example shows how you could use Keras `Sequence`s and multiprocessing/multithreading for Keras
models in Determined.

Useful References:
    http://docs.determined.ai/latest/keras.html
    https://keras.io/utils/

Based off of: https://medium.com/@nickbortolotti/iris-species-categorization-using-tf-keras-tf-data-
              and-differences-between-eager-mode-on-and-off-9b4693e0b22
"""
from typing import List
import os
from determined import keras

import tensorflow as tf
import filelock
import tensorflow_datasets as tfds
from keras.models import Sequential
from keras import layers


"""
#IMG_SIZE is determined by EfficientNet model choice
"""
IMG_SIZE = 224
# Constants about the data set.
NUM_CLASSES = 120

batch_size = 64


# ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
# ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
# print((ds_train))

# The first row of each data set is not a typical CSV header with column labels, but rather a
# dataset descriptor of the following format:
#
# <num observations>,<num features>,<species 0 label>,<species 1 label>,<species 2 label>
#
# The remaining rows then contain observations, with the four features followed by label.  The
# label values in the observation rows take on the values 0, 1, or 2 which correspond to the
# three species in the header.  Define the columns explicitly here so that we can more easily
# separate features and labels below.
LABEL_HEADER = "Species"
DS_COLUMNS = [
    "SepalLength",
    "SepalWidth",
    "PetalLength",
    "PetalWidth",
    LABEL_HEADER,
]



class EfficientNetTrial(keras.TFKerasTrial):
    def __init__(self, context: keras.TFKerasTrialContext) -> None:
        self.context = context
        self.download_directory = "/tmp/data"
        # dataset_name = "stanford_dogs"
        # print("11111")
        # (ds_train, ds_test), ds_info = tfds.load(
        #     dataset_name, split=["train", "test"], with_info=True, as_supervised=True
        # )
        # print("22222")
        # '''
        # Resize images
        # '''
        # size = (IMG_SIZE, IMG_SIZE)
        # print("888888888")
        # self.DS_TRAIN_DATA = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
        # print("999999")
        # self.DS_TEST_DATA = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
        # print("0000000")

    def build_model(self):
        """
        Define model for iris classification.

        This is a simple model with one hidden layer to predict iris species (setosa, versicolor, or
        virginica) based on four input features (length and width of sepals and petals).
        """
        img_augmentation = Sequential(
            [
                layers.RandomRotation(factor=0.15),
                layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                layers.RandomFlip(),
                layers.RandomContrast(factor=0.1),
            ],
            name="img_augmentation",
        )
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = img_augmentation(inputs)
        model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
        # Freeze the pretrained weights
        model.trainable = False
        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = layers.BatchNormalization()(x)

        top_dropout_rate = self.context.get_hparam("dropout_rate")
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

        # Compile
        model = tf.keras.Model(inputs, outputs, name="EfficientNet")

        model = self.context.wrap_model(model)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.context.get_hparam("learning_rate"))

        optimizer = self.context.wrap_optimizer(optimizer)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        return model

    # def keras_callbacks(self) -> List[tf.keras.callbacks.Callback]:
    #     return [keras.callbacks.TensorBoard(update_freq="batch", profile_batch=0, histogram_freq=1)]

    # One-hot / categorical encoding
    def input_preprocess(self, image, label):
        label = tf.one_hot(label, NUM_CLASSES)
        size = (IMG_SIZE, IMG_SIZE)
        return tf.image.resize(image, size), label
    def build_training_data_loader(self):
        os.makedirs(self.download_directory, exist_ok=True)

        # Use a file lock so only one worker on each node does the download
        with filelock.FileLock(os.path.join(self.download_directory, "download.lock")):
            dataset, ds_info = tfds.load(
                "stanford_dogs",
                split="train",
                with_info=True,
                as_supervised=True,
                data_dir=self.download_directory,
            )

        # ds_train_data = self.DS_TRAIN_DATA.map(self.input_preprocess)
        # print("ds_train_data 1")
        # ds_train_data = ds_train_data.batch(batch_size=batch_size, drop_remainder=True)
        # print("ds_train_data 2")
        # ds_train_data = ds_train_data.prefetch(tf.data.AUTOTUNE)
        # print("ds_train_data 3")
        #size = (IMG_SIZE, IMG_SIZE)
        #dataset_labels = dataset.map(lambda image_label: (tf.image.resize(image_label['image'], size), image_label['label']))
        train = dataset.map(self.input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train = self.context.wrap_dataset(train)
        train_dataset = (
            train.cache()
                .shuffle(1000)
                .batch(self.context.get_per_slot_batch_size())

        )
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_dataset

    def build_validation_data_loader(self) :
        os.makedirs(self.download_directory, exist_ok=True)

        # Use a file lock so only one worker on each node does the download
        with filelock.FileLock(os.path.join(self.download_directory, "download.lock")):
            dataset, ds_info = tfds.load(
                "stanford_dogs",
                split="test",
                with_info=True,
                as_supervised=True,
                data_dir=self.download_directory,
            )

        # ds_train_data = self.DS_TRAIN_DATA.map(self.input_preprocess)
        # print("ds_train_data 1")
        # ds_train_data = ds_train_data.batch(batch_size=batch_size, drop_remainder=True)
        # print("ds_train_data 2")
        # ds_train_data = ds_train_data.prefetch(tf.data.AUTOTUNE)
        # print("ds_train_data 3")
        #size = (IMG_SIZE, IMG_SIZE)
        #dataset_labels = dataset.map(lambda image, label: (tf.image.resize(image, size), label))
        test = dataset.map(self.input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test = self.context.wrap_dataset(test)
        test_dataset = (
            test.cache()
                .shuffle(1000)
                .batch(self.context.get_per_slot_batch_size())

        )
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return test_dataset
