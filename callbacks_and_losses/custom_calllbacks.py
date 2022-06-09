from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, TensorBoard
from data import validation_image_generator, corrected_validation_image_generator

import numpy as np


class EarlyStoppingAtMinValLoss(Callback):
    """Stop training when the validation loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        paths: (2, n_images) array containing the paths to the raw images in row 0 and paths to gt images in row 1
        file_path: file name to save weights of the model with minimum validation loss
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, paths, file_path="best_val_loss.hdf5", patience=0, rgb_preprocessor=None):
        super(EarlyStoppingAtMinValLoss, self).__init__()

        self.patience = patience

        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.evaluate_steps = paths.shape[1]
        self.paths = paths
        self.file_path = file_path
        self.rgb_preprocessor = rgb_preprocessor
        self.bad_ending = False

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
        self.image_generator = validation_image_generator(self.paths, batch_size=1,
                                                          rgb_preprocessor=self.rgb_preprocessor)

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.model.evaluate(x=self.image_generator, steps=self.evaluate_steps, verbose=0)

        result_string = ""
        for idx, metric in enumerate(self.model.metrics_names):
            result_string += "val_{}: {:.4f} - ".format(metric, metrics[idx])
            if metric == "loss":
                current = metrics[idx]
            logs['val_%s' % metric] = metrics[idx]
        if self.params["verbose"] > 0:
            print("\n" + result_string.strip('- '))

        if metrics[-1] < 0.01 and metrics[-2] < 0.01:
            self.model.stop_training = True
            self.bad_ending = True
            return 0

        if np.less(current, self.best):
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            if self.file_path:
                if self.params["verbose"] > 0:
                    print('Epoch %05d: validation loss improved from %0.5f to %0.5f,  saving model to %s\n' % (
                epoch + 1, self.best, current, self.file_path))
                self.model.save_weights(self.file_path, overwrite=True)
            else:
                if self.params["verbose"] > 0:
                    print('Epoch %05d: validation loss improved from %0.5f to %0.5f\n' % (
                        epoch + 1, self.best, current))
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                # print('Restoring model weights from the end of the best epoch.')
                # self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


class CorrectedEarlyStoppingAtMinValLoss(Callback):
    """Stop training when the validation loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        paths: (2, n_images) array containing the paths to the raw images in row 0 and paths to gt images in row 1
        file_path: file name to save weights of the model with minimum validation loss
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """

    def __init__(self, paths, corrected_paths, file_path="best_val_loss.hdf5", patience=0, rgb_preprocessor=None):
        super(CorrectedEarlyStoppingAtMinValLoss, self).__init__()

        self.patience = patience

        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.evaluate_steps = paths.shape[1]
        self.paths = paths
        self.corrected_paths = corrected_paths
        self.file_path = file_path
        self.rgb_preprocessor = rgb_preprocessor
        self.bad_ending = False

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
        self.image_generator = corrected_validation_image_generator(self.paths, self.corrected_paths, batch_size=1,
                                                          rgb_preprocessor=self.rgb_preprocessor)

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.model.evaluate(x=self.image_generator, steps=self.evaluate_steps, verbose=0)

        result_string = ""
        for idx, metric in enumerate(self.model.metrics_names):
            result_string += "val_{}: {:.4f} - ".format(metric, metrics[idx])
            if metric == "loss":
                current = metrics[idx]
            logs['val_%s' % metric] = metrics[idx]
        if self.params["verbose"] > 0:
            print("\n" + result_string.strip('- '))

        if metrics[-1] < 0.01 and metrics[-2] < 0.01:
            self.model.stop_training = True
            self.bad_ending = True
            return 0

        if np.less(current, self.best):
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            if self.file_path:
                if self.params["verbose"] > 0:
                    print('Epoch %05d: validation loss improved from %0.5f to %0.5f,  saving model to %s\n' % (
                epoch + 1, self.best, current, self.file_path))
                self.model.save_weights(self.file_path, overwrite=True)
            else:
                if self.params["verbose"] > 0:
                    print('Epoch %05d: validation loss improved from %0.5f to %0.5f\n' % (
                        epoch + 1, self.best, current))
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                # print('Restoring model weights from the end of the best epoch.')
                # self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
