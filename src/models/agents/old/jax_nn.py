# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MNIST classifier example."""

from typing import Iterator, Mapping, Tuple
from sklearn.model_selection import train_test_split

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn import datasets
from sklearn import metrics
from jax import value_and_grad

class JaxNN():
    def __init__(self):
        self.opt = optax.adam(1e-4)
        self.net = hk.without_apply_rng(hk.transform(self.net_fn))

        self.params = None
        self.opt_state = None

    def net_fn(self, x):
        """Standard LeNet-300-100 MLP network."""

        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(133*30), jax.nn.relu,
            #hk.Linear(100), jax.nn.relu,
            hk.Linear(1),
        ])
        return mlp(x)

    def fit(self, X, y, batch_size = 64, epochs = 100):

        if(self.params is None):
            self.params = self.net.init(jax.random.PRNGKey(42), X)
            self.opt_state = self.opt.init(self.params)

        @jax.jit
        def update(
                params: hk.Params,
                opt_state: optax.OptState,
                x, y,
        ) -> Tuple[hk.Params, optax.OptState]:
            """Learning rule (stochastic gradient descent)."""
            grads = jax.grad(loss)(params, x, y)
            updates, opt_state = self.opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state

        @jax.jit
        def mse(weights: hk.Params, x, y) -> jnp.ndarray:
            """Compute the loss of the network, including L2."""
            y_hat = self.net.apply(weights, x)

            preds = y_hat.squeeze()
            return jnp.power(y - preds, 2).mean()

        @jax.jit
        def update(
                params: hk.Params,
                opt_state: optax.OptState,
                x, y,
        ) -> Tuple[hk.Params, optax.OptState]:
            """Learning rule (stochastic gradient descent)."""
            #grads = jax.grad(loss)(params, x, y)

            loss, grads = jax.value_and_grad(mse)(params,x,y)
            updates, opt_state = self.opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state,loss

        # We maintain avg_params, the exponential moving average of the "live" params.
        # avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
        @jax.jit
        def ema_update(params, avg_params):
            return optax.incremental_update(params, avg_params, step_size=0.001)

        # We maintain avg_params, the exponential moving average of the "live" params.
        # avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
        @jax.jit
        def ema_update(params, avg_params):
            return optax.incremental_update(params, avg_params, step_size=0.001)

        params = self.params
        opt_state = self.opt_state
        for i in range(1, epochs + 1):
            batches = jnp.arange(
                (X.shape[0] // batch_size) + 1)  ### Batch Indices

            for batch in batches:
                if batch != batches[-1]:
                    start, end = int(batch * batch_size), int(
                        batch * batch_size + batch_size)
                else:
                    start, end = int(batch * batch_size), None

                X_batch, Y_batch = X[start:end], y[
                                                       start:end]  ## Single batch of data

                params, opt_state,loss = update(params, opt_state, X_batch, Y_batch)
            y_hat = self.net.apply(params, X)
            self.params = params
            self.opt_state = opt_state
            mse = metrics.mean_squared_error(y, y_hat)
            #print(mse)

    def predict(self,X, verbose = True):
        y = self.net.apply(self.params, X)
        return y




if __name__ == "__main__":
    # Make the network and optimiser.

    # Training loss (cross-entropy).

    X, Y = datasets.load_boston(return_X_y=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,
                                                        random_state=123)

    X_train, X_test, Y_train, Y_test = jnp.array(X_train, dtype=jnp.float32), \
                                       jnp.array(X_test, dtype=jnp.float32), \
                                       jnp.array(Y_train, dtype=jnp.float32), \
                                       jnp.array(Y_test, dtype=jnp.float32)
    samples, features = X_train.shape
    print(samples, features)

    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    clf = JaxNN()
    clf.fit(X_train, Y_train)
