"""
MSGP-MLP regression on Kin40k data.
Reference: https://arxiv.org/abs/1511.02222

This example showcases semi-stochastic training
of GP-MLP model from scratch. Note that the original
paper used full-batch pretraining-finetuning scheme.
"""
from __future__ import print_function

import os

import numpy as np
np.random.seed(42)

# Keras
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# KGP
from kgp.models import Model
from kgp.layers import GP

# Dataset interfaces
from kgp.datasets.kin40k import load_data

# Model assembling and executing
from kgp.utils.experiment import train

# Metrics & losses
from kgp.losses import gen_gp_loss
from kgp.metrics import root_mean_squared_error as RMSE


def standardize_data(X_train, X_test, X_valid, X_unlabeled):
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)

    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    X_valid -= X_mean
    X_valid /= X_std
    X_unlabeled -= X_mean
    X_unlabeled /= X_std

    return X_train, X_test, X_valid, X_unlabeled


def assemble_mlp(input_shape, output_shape, batch_size, nb_train_samples):
    """Assemble a simple MLP model.
    """
    inputs = Input(shape=input_shape)
    hidden = Dense(1024, activation='relu', name='dense1')(inputs)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(512, activation='relu', name='dense2')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(64, activation='relu', name='dense3')(hidden)
    hidden = Dropout(0.25)(hidden)
    hidden = Dense(2, activation='relu', name='dense4')(hidden)
    gp = GP(hyp={
                'lik': np.log(0.3),
                'mean': [],
                'cov': [[0.5], [1.0]],
                'ss_strength': 1.0
            },
            inf='infGrid', dlik='dlikGrid_ssdkl',
            opt={'cg_maxit': 2000, 'cg_tol': 1e-6},
            mean='meanZero', cov='covSEiso',
            update_grid=1,
            grid_kwargs={'eq': 1, 'k': 70.},
            batch_size=batch_size,
            nb_train_samples=nb_train_samples)
    outputs = [gp(hidden)]
    return Model(inputs=inputs, outputs=outputs)


def main():
    # Load data
    X_train, y_train = np.random.rand(1000, 50), np.random.rand(1000)
    X_test, y_test = np.random.rand(300, 50), np.random.rand(300)
    X_unlabeled = np.random.rand(500, 50)
    # X_train, y_train = load_data(stop=90.)
    # X_test, y_test = load_data(start=90.)
    X_valid, y_valid = X_test, y_test
    X_train, X_test, X_valid, X_unlabeled = standardize_data(X_train, X_test, X_valid, X_unlabeled)
    data = {
        'train': (X_train, y_train),
        'valid': (X_valid, y_valid),
        'test': (X_test, y_test),
        'unlabeled': X_unlabeled
    }

    # Model & training parameters
    input_shape = data['train'][0].shape[1:]
    output_shape = data['train'][1].shape[1:]
    batch_size = 2**10
    epochs = 500

    # Construct & compile the model
    model = assemble_mlp(input_shape, output_shape, batch_size,
                         nb_train_samples=len(X_train))
    loss = [gen_gp_loss(gp) for gp in model.output_layers]
    model.compile(optimizer=Adam(1e-4), loss=loss)


    model_checkpoint_cb = ModelCheckpoint('checkpoints/best_model.h5', save_best_only=True)
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10)
    callbacks = [model_checkpoint_cb, early_stopping_cb]

    # Train the model
    history = train(model, data, callbacks=callbacks, gp_n_iter=5,
                    checkpoint='best_model', checkpoint_monitor='val_loss',
                    epochs=epochs, batch_size=batch_size, verbose=1)

    # Load saved weights
    if os.path.isfile('checkpoints/best_model.h5'):
        model.load_weights('checkpoints/best_model.h5', by_name=True)

    # Test the model
    X_test, y_test = data['test']
    y_preds = model.predict(X_test)
    rmse_predict = RMSE(y_test, y_preds)
    print('Test RMSE:', rmse_predict)


if __name__ == '__main__':
    main()
