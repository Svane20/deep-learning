import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import itertools
import zipfile
import os
import datetime

from typing import Tuple, List


# Function to import an image and resize it to be able to used with TensorFlow model
def load_and_preprocess_image(filename: str, image_shape: int = 224, scale: bool = True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into (224, 224, 3).

    Parameters
    -----------
    filename : str
        Path to the image file.

    image_shape : int, default: 224
        Desired shape of the image.

    scale : bool, default: True
        If True, scales the image to be between 0 and 1.

    Returns
    --------
    image : Tensor:
        Processed image.
    """

    # Read in the image
    image = tf.io.read_file(filename)

    # Decode it into a tensor
    image = tf.image.decode_jpeg(image)

    # Resize the image
    image = tf.image.resize(image, [image_shape, image_shape])

    if scale:
        # Rescale the image (get all values between 0 and 1)
        image = image / 255.

    return image


# The following confusion matrix code is a remix of Scikit-Learn's plot_confusion_matrix function -
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
def make_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          classes: List[str] | None = None,
                          fig_size: Tuple[int, int] = (10, 10),
                          text_size: int = 15,
                          norm: bool = False,
                          save_fig: bool = False) -> None:
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Parameters
    ---------
    y_true : np.ndarray
        Array of truth labels (must be same shape as y_pred).

    y_pred : np.ndarray
        Array of predicted labels (must be same shape as y_true).

    classes : optional, default: None
        Array of class labels (e.g. string form). If `None`, integer labels are used.

    fig_size : tuple, default: (10, 10)
        Size of output figure.

    text_size : int, default: 15
        Size of output figure text.

    norm : bool, default: False
        Normalize the confusion matrix or not.

    save_fig : bool, default: False
        Save the confusion matrix plot or not.

    Examples
    --------
        make_confusion_matrix(y_true=test_labels,
                              y_pred=y_pred,
                              classes=class_names,
                              fig_size=(15, 15),
                              text_size=10)
    """

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalizing
    n_classes = cm.shape[0]  # Find the number of classes

    # Plot the figure
    fig, ax = plt.subplots(figsize=fig_size)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # Colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # Create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # Axes will labeled with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    # Save the figure to the current working directory
    if save_fig:
        fig.savefig("confusion_matrix.png")


# Function to predict on images and plot the results
def predict_and_plot(model: tf.keras.Model, filename: str, class_names: List[str]) -> None:
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.

    Parameters
    -----------
    model : tf.keras.Model
        A trained model.

    class_names : list
        A list of class names.

    filename : str
        The target image filepath.
    """

    # Import the target image and preprocess it
    img = load_and_preprocess_image(filename)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1:  # check for multi-class
        pred_class = class_names[pred.argmax()]  # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]  # if only one output, round

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False);


def plot_loss_curves(history: tf.keras.callbacks.History) -> None:
    """
    Plots separate loss curves for training and validation metrics.

    Parameters
    -----------
    history : tf.keras.callbacks.History
        The output of the `fit` method of a TensorFlow model.
    """

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    # Plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend();


def create_tensorboard_callback(dir_name: str, experiment_name: str) -> tf.keras.callbacks.TensorBoard:
    """
    Creates a TensorBoard callback instance to store log files.

    Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

    Parameters
    -----------
    dir_name : str
        Target directory for storing log files.

    experiment_name : str
        Name of the experiment.

    Returns
    --------
    TensorBoard : tf.keras.callbacks.TensorBoard
        A TensorBoard callback instance.
    """

    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    print(f"Saving TensorBoard log files to: {log_dir}")

    return tensorboard_callback


def compare_histories(original_history: tf.keras.callbacks.History,
                      new_history: tf.keras.callbacks.History,
                      initial_epochs: int = 5) -> None:
    """
    Compares two TensorFlow model training histories.

    Parameters
    -----------
    original_history : tf.keras.callbacks.History
        The history object given by a model which has been trained for a given number of epochs.

    new_history : tf.keras.callbacks.History The history object given by a model which has been trained for a given
    number of epochs on top of the original model.

    initial_epochs : int, default: 5
        The number of epochs the original model was trained for.
    """

    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label="Training Accuracy")
    plt.plot(total_loss, label="Training Loss")
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label="Start Fine Tuning")  # Re-shift plot around epochs
    plt.legend(loc="lower right")
    plt.title("Training Accuracy and Loss")

    # Plot validation
    plt.subplot(2, 1, 2)
    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    plt.plot(total_val_acc, label="Validation Accuracy")
    plt.plot(total_val_loss, label="Validation Loss")
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(),
             label="Start Fine Tuning")  # Re-shift plot around epochs
    plt.legend(loc="upper right")
    plt.title("Validation Accuracy and Loss");


# Function to unzip a zipfile into current working directory
def unzip_data(filename: str) -> None:
    """
    Unzips filename into the current working directory.

    Parameters
    -----------
    filename : str
        Name of the file to unzip.
    """

    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()


# Function to walk through a directory and list all the files
def walk_through_dir(dir_path: str) -> None:
    """
    Walks through dir_path returning its contents.

    Parameters
    -----------
    dir_path : str
        Target directory.
    """

    for directory_path, directory_names, filenames in os.walk(dir_path):
        print(f"There are {len(directory_names)} directories and {len(filenames)} images in '{directory_path}'.")


def calculate_results(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.

    Parameters
    -----------
    y_true : np.ndarray
        True labels in the form of a 1D array.

    y_pred : np.ndarray
        Predicted labels in the form of a 1D array.

    Returns
    --------
    results : dict
        Dictionary of accuracy, precision, recall and f1 score.
    """

    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100

    # Calculate model precision, recall and f1 score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}

    return model_results
