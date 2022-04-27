"""
Title: main.py

Created on: 2/2/2022

Author: 187-Shogun

Encoding: UTF-8

Description: Build a Malaria Cell Classification Model. It predicts between healthy and
infected cells based on image data.
"""


from tqdm import tqdm
from pytz import timezone
from datetime import datetime
import tensorflow as tf
import seaborn as sn
import pandas as pd
import os
import shutil
import random
import sys


# Global variables to interact with the script:
AUTOTUNE = tf.data.AUTOTUNE
TRAIN_DIR = os.path.join(os.getcwd(), 'data', 'train')
TEST_DIR = os.path.join(os.getcwd(), 'data', 'test')
LOGS_DIR = os.path.join(os.getcwd(), 'logs')
CM_DIR = os.path.join(os.getcwd(), 'reports')
MODELS_DIR = os.path.join(os.getcwd(), 'models')
BATCH_SIZE = 32
IM_HEIGHT = 200
IM_WIDTH = 200
EPOCHS = 100
PATIENCE = 5
RANDOM_SEED = 420


def download_zip():
    """ Download zipfile from GCS into local system. """
    url = 'https://storage.googleapis.com/open-ml-datasets/malaria-cells-dataset/cell_images.zip'
    tf.keras.utils.get_file('malaria.zip', url, extract=True, cache_subdir=os.path.join(os.getcwd(), 'data'))


def get_model_version_name(model_name: str):
    ts = datetime.now(timezone('America/Costa_Rica'))
    run_id = ts.strftime("%Y%m%d-%H%M%S")
    return f"{model_name}_v{run_id}"


def reset_folders():
    # Destroy and recreate directories:
    shutil.rmtree(TRAIN_DIR, ignore_errors=True)
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    shutil.rmtree(CM_DIR, ignore_errors=True)
    os.makedirs(os.path.join(TRAIN_DIR, 'Infected'))
    os.makedirs(os.path.join(TRAIN_DIR, 'Uninfected'))
    os.makedirs(os.path.join(TEST_DIR, 'Infected'))
    os.makedirs(os.path.join(TEST_DIR, 'Uninfected'))
    os.makedirs(CM_DIR)


def split_raw_dataset(test_split: float = 0.2, reset: bool = True):
    # Check if work is already done:
    if reset is False:
        return None
    else:
        # Download data if required:
        download_zip()

        # Check total number of images available:
        infected = os.listdir(os.path.join(os.getcwd(), 'data', 'cell_images', 'Parasitized'))
        uninfected = os.listdir(os.path.join(os.getcwd(), 'data', 'cell_images', 'Uninfected'))
        total_images = len(infected) + len(uninfected)
        total_test_images = int(total_images * test_split)

        # Randomly select images from the raw dataset:
        random.seed(RANDOM_SEED)
        random.shuffle(infected)
        train_infected = infected[total_test_images:]
        train_uninfected = uninfected[total_test_images:]
        test_infected = infected[:total_test_images]
        test_uninfected = uninfected[:total_test_images]

        # Build directories:
        reset_folders()
        for img in tqdm(train_infected, desc='Building Training Infected folder', file=sys.stdout):
            source = os.path.join(os.getcwd(), 'data', 'cell_images', 'Parasitized', img)
            destination = os.path.join(TRAIN_DIR, 'Infected', img)
            shutil.copy(source, destination)
        for img in tqdm(train_uninfected, desc='Building Training Uninfected folder', file=sys.stdout):
            source = os.path.join(os.getcwd(), 'data', 'cell_images', 'Uninfected', img)
            destination = os.path.join(TRAIN_DIR, 'Uninfected', img)
            shutil.copy(source, destination)
        for img in tqdm(test_infected, desc='Building Test Infected folder', file=sys.stdout):
            source = os.path.join(os.getcwd(), 'data', 'cell_images', 'Parasitized', img)
            destination = os.path.join(TEST_DIR, 'Infected', img)
            shutil.copy(source, destination)
        for img in tqdm(test_uninfected, desc='Building Test Uninfected folder', file=sys.stdout):
            source = os.path.join(os.getcwd(), 'data', 'cell_images', 'Uninfected', img)
            destination = os.path.join(TEST_DIR, 'Uninfected', img)
            shutil.copy(source, destination)


def get_dataset(batch_size: int, im_height: int, im_width: int, subset: str = 'training', validation: float = 0.2):
    return tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=validation,
        subset=subset,
        seed=RANDOM_SEED,
        image_size=(im_height, im_width),
        batch_size=batch_size,
    )


def get_test_dataset(batch_size: int, im_height: int, im_width: int):
    return tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        seed=RANDOM_SEED,
        image_size=(im_height, im_width),
        batch_size=batch_size,
    )


def get_baseline_nn():
    model_layers = [
        tf.keras.layers.Flatten(input_shape=(IM_HEIGHT, IM_WIDTH, 3)),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
    model = tf.keras.models.Sequential(model_layers, name='BaselineModel')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics='accuracy'
    )
    return model


def get_custom_nn():
    model_layers = [
        tf.keras.layers.Flatten(input_shape=(IM_HEIGHT, IM_WIDTH, 3)),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
    model = tf.keras.models.Sequential(model_layers, name='CustomModelOne')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics='accuracy'
    )
    return model


def get_best_cnn():
    input_shape = (IM_HEIGHT, IM_WIDTH, 3)
    model_layers = [
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomContrast(0.3),
        tf.keras.layers.Conv2D(64, 3, input_shape=input_shape, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
    model = tf.keras.models.Sequential(model_layers, name='CustomModel-BestCNNSoFar')
    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9, nesterov=True),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics='accuracy'
    )
    return model


def get_custom_cnn():
    input_shape = (IM_HEIGHT, IM_WIDTH, 3)
    model_layers = [
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomContrast(0.3),
        tf.keras.layers.Conv2D(128, 3, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
    model = tf.keras.models.Sequential(model_layers, name='CustomModel-CNN')
    model.compile(
        optimizer=tf.keras.optimizers.SGD(momentum=0.9, nesterov=True),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics='accuracy'
    )
    return model


def plot_confision_matrix(model, test_dataset, version_name):
    # Fetch predictions and true labels:
    predictions = []
    labels = []
    for x, y in tqdm(test_dataset, desc='Fetching predictions...', file=sys.stdout):
        predictions += list(model.predict(x).reshape(-1))
        labels += list(y.numpy().astype(float))

    # Build a confusion matrix and save the plot in a PNG file:
    matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()
    df = pd.DataFrame(matrix)
    df.columns = test_dataset.class_names
    df.index = test_dataset.class_names
    cf = sn.heatmap(df, annot=True, fmt="d")
    cf.set(xlabel='Actuals', ylabel='Predicted')
    cf.get_figure().savefig(version_name)

    # Compute precision and recall:
    precision = tf.keras.metrics.Precision()
    precision.update_state(labels, predictions)
    print(f"Model's Precision: {precision.result().numpy()}")
    recall = tf.keras.metrics.Recall()
    recall.update_state(labels, predictions)
    print(f"Model's Recall: {recall.result().numpy()}")


def evaluate_existing_model():
    # Load test dataset and model from H5 file:
    test_ds = get_test_dataset(BATCH_SIZE, IM_HEIGHT, IM_WIDTH)
    best_model = 'CustomModel-CNN_v.20220207-232113.h5'
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, best_model))

    # Fetch predictions and true labels:
    predictions = []
    labels = []
    for x, y in tqdm(test_ds, desc="Fetching predictions"):
        predictions += list(model.predict(x).reshape(-1))
        labels += list(y.numpy().astype(float))

    # Compute accuracy, precision and recall:
    print("Evaluating model on test data:")
    test_score = model.evaluate(test_ds)
    print(f"Model's Accuracy: {test_score}")
    precision = tf.keras.metrics.Precision()
    precision.update_state(labels, predictions)
    print(f"Model's Precision: {precision.result().numpy()}")
    recall = tf.keras.metrics.Recall()
    recall.update_state(labels, predictions)
    print(f"Model's Recall: {recall.result().numpy()}")


def main():
    # Prepare the raw data to be loaded:
    split_raw_dataset()

    # Load data into a generator:
    train_ds = get_dataset(BATCH_SIZE, IM_HEIGHT, IM_WIDTH)
    val_ds = get_dataset(BATCH_SIZE, IM_HEIGHT, IM_WIDTH, subset='validation')
    test_ds = get_test_dataset(BATCH_SIZE, IM_HEIGHT, IM_WIDTH)

    # Normalize data prior training:
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    # Cache the datasets:
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Start training a single model:
    model = get_custom_cnn()
    version_name = get_model_version_name(model.name)
    tb_logs = tf.keras.callbacks.TensorBoard(os.path.join(LOGS_DIR, version_name))
    early_stop = tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=.5, patience=5)
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[tb_logs, early_stop, lr_scheduler])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))

    # Evaluate single model:
    test_score = model.evaluate(test_ds)
    print(f"Test Score: {test_score}")
    plot_confision_matrix(model, test_ds, os.path.join(CM_DIR, f"{version_name}.png"))
    return {}


if __name__ == "__main__":
    main()
