import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as k
import utils.utils as utils
import numpy as np
import pandas as pd


class RULCostModel:
    def __init__(self, maintenance_cost, safe_interval=0, field="machine"):
        self.maintenance_cost = maintenance_cost
        self.safe_interval = safe_interval
        self.field = field
    def cost(self, machine, pred, thr, return_margin=False):
        # Merge machine and prediction data
        tmp = np.array([machine, pred]).T
        tmp = pd.DataFrame(data=tmp,
                           columns=[self.field, 'pred'])
        # Cost computation
        cost = 0
        nfails = 0
        slack = 0
        for _, gtmp in tmp.groupby(self.field):
            idx = np.nonzero(gtmp['pred'].values < thr)[0]
            if len(idx) == 0:
                cost += self.maintenance_cost
                nfails += 1
            else:
                cost -= max(0, idx[0] - self.safe_interval)
                slack += len(gtmp) - idx[0]
        if not return_margin:
            return cost
        else:
            return cost, nfails, slack
            
class MLPRegressor(keras.Model):
    def __init__(self, input_shape, hidden=[]):
        super(MLPRegressor, self).__init__()
        # Build the model
        self.lrs = [layers.Dense(h, activation='relu') for h in hidden]
        self.lrs.append(layers.Dense(1, activation='linear'))

    def call(self, data):
        x = data
        for layer in self.lrs:
            x = layer(x)
        return x

class CstBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, in_cols, batch_size=32, seed=42):
        super(CstBatchGenerator).__init__()
        self.data = data
        self.in_cols = in_cols
        self.dpm = utils.split_by_field(data, 'machine')
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        # Build the first sequence of batches
        self.__build_batches()

    def __len__(self):
        return len(self.batches)

    # def __getitem__(self, index):
    #     idx = self.batches[index]
    #     mcn = self.machines[index]
    #     x = self.data[self.in_cols].loc[idx].values
    #     y = self.data['rul'].loc[idx].values
    #     return x, y


    def __getitem__(self, index):
        idx = self.batches[index]
        # mcn = self.machines[index]
        x = self.data[self.in_cols].loc[idx].values
        y = self.data['rul'].loc[idx].values
        flags = (y != -1)
        info = np.vstack((y, flags, idx)).T
        return x, info

    def on_epoch_end(self):
        self.__build_batches()

    def __build_batches(self):
        self.batches = []
        self.machines = []
        # Randomly sort the machines
        # self.rng.shuffle(mcns)
        # Loop over all machines
        mcns = list(self.dpm.keys())
        for mcn in mcns:
            # Obtain the list of indices
            index = self.dpm[mcn].index
            # Padding
            padsize = self.batch_size - (len(index) % self.batch_size)
            padding = self.rng.choice(index, padsize)
            idx = np.hstack((index, padding))
            # Shuffle
            self.rng.shuffle(idx)
            # Split into batches
            bt = idx.reshape(-1, self.batch_size)
            # Sort each batch individually
            bt = np.sort(bt, axis=1)
            # Store
            self.batches.append(bt)
            self.machines.append(np.repeat([mcn], len(bt)))
        # Concatenate all batches
        self.batches = np.vstack(self.batches)
        self.machines = np.hstack(self.machines)
        # Shuffle the batches
        bidx = np.arange(len(self.batches))
        self.rng.shuffle(bidx)
        self.batches = self.batches[bidx, :]
        self.machines = self.machines[bidx]

class LagDualRULRegressor(MLPRegressor):
    def __init__(self, input_shape, maxrul, alpha=None, hidden=[]):
        super(LagDualRULRegressor, self).__init__(input_shape, hidden)
        # Weights
        self.alpha_fixed = True if alpha else None
        if self.alpha_fixed:
            self.alpha = alpha
        else:
            self.alpha = tf.Variable(0., name='alpha')
        self.maxrul = maxrul
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    def __custom_loss(self, data, sign=1):
        x, info = data
        y_true = info[:, 0:1]
        flags = info[:, 1:2]
        idx = info[:, 2:3]

        y_pred = self(x, training=True)
        mse = k.mean(flags * k.square(y_pred-y_true))

        # Compute the constraint regularization term
        delta_pred = y_pred[1:] - y_pred[:-1]
        delta_rul = -(idx[1:] - idx[:-1]) / self.maxrul
        deltadiff = delta_pred - delta_rul
        cst =  k.mean(k.square(deltadiff))
        loss = mse + self.alpha * cst
        return sign*loss, mse, cst

    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            loss, mse, cst = self.__custom_loss(data, sign=1)

        # Separate training variables
        tr_vars = self.trainable_variables
        wgt_vars = tr_vars[:-1]

        grads = tape.gradient(loss, wgt_vars)
        self.optimizer.apply_gradients(zip(grads, wgt_vars))


        if not self.alpha_fixed:
            mul_vars = tr_vars[-1:]

            
            
            with tf.GradientTape() as tape:
                loss, mse, cst = self.__custom_loss(data, sign=-1)

            grads = tape.gradient(loss, mul_vars)
            self.optimizer.apply_gradients(zip(grads, mul_vars))


        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]

class LagDualRULRegressorMulti(MLPRegressor):
    def __init__(self, input_shape, maxrul, shape, alpha=None, hidden=[]):
        super(LagDualRULRegressorMulti, self).__init__(input_shape, hidden)
        # Weights
        self.alpha_fixed = True if alpha else None
        if self.alpha_fixed:
            self.alpha = [alpha]*shape
        else:
            self.alpha = tf.Variable(tf.zeros(shape), name='alpha')
        self.maxrul = maxrul
        # Loss trackers
        self.ls_tracker = keras.metrics.Mean(name='loss')
        self.mse_tracker = keras.metrics.Mean(name='mse')
        self.cst_tracker = keras.metrics.Mean(name='cst')

    def __custom_loss(self, data, sign=1):
        x, info = data
        y_true = info[:, 0:1]
        flags = info[:, 1:2]
        idx = info[:, 2:3]

        y_pred = self(x, training=True)
        mse = k.mean(flags * k.square(y_pred-y_true))

        # Compute the constraint regularization term
        delta_pred = y_pred[1:] - y_pred[:-1]
        delta_rul = -(idx[1:] - idx[:-1]) / self.maxrul
        deltadiff = delta_pred - delta_rul

        cst = k.square(deltadiff)
        loss = mse + k.mean(self.alpha * cst)
        return sign*loss, mse, k.mean(cst)

    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            loss, mse, cst = self.__custom_loss(data, sign=1)

        # Separate training variables
        tr_vars = self.trainable_variables
        wgt_vars = tr_vars[:-1]

        grads = tape.gradient(loss, wgt_vars)
        self.optimizer.apply_gradients(zip(grads, wgt_vars))

        if not self.alpha_fixed:
            mul_vars = tr_vars[-1:]


            
            with tf.GradientTape() as tape:
                loss, mse, cst = self.__custom_loss(data, sign=-1)

            grads = tape.gradient(loss, mul_vars)
            self.optimizer.apply_gradients(zip(grads, mul_vars))


        # Track the loss change
        self.ls_tracker.update_state(loss)
        self.mse_tracker.update_state(mse)
        self.cst_tracker.update_state(cst)
        return {'loss': self.ls_tracker.result(),
                'mse': self.mse_tracker.result(),
                'cst': self.cst_tracker.result()}

    @property
    def metrics(self):
        return [self.ls_tracker,
                self.mse_tracker,
                self.cst_tracker]

def build_model(Model, loss, *params):
    model = Model(*params)
    model.compile(optimizer='Adam', loss=loss, run_eagerly=False)
    return model

