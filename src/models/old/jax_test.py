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


def net_fn(x):
    """Standard LeNet-300-100 MLP network."""

    mlp = hk.Sequential([
        hk.Flatten(),
        hk.Linear(300), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(1),
    ])
    return mlp(x)




def main(_):
    # Make the network and optimiser.
    net = hk.without_apply_rng(hk.transform(net_fn))
    opt = optax.adam(1e-4)

    # Training loss (cross-entropy).
    @jax.jit
    def loss(weights: hk.Params, x, y) -> jnp.ndarray:
        """Compute the loss of the network, including L2."""
        y_hat = net.apply(weights, x)

        preds = y_hat.squeeze()
        return jnp.power(y - preds, 2).mean()

    @jax.jit
    def update(
            params: hk.Params,
            opt_state: optax.OptState,
            x, y,
    ) -> Tuple[hk.Params, optax.OptState]:
        """Learning rule (stochastic gradient descent)."""
        grads = jax.grad(loss)(params, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    # We maintain avg_params, the exponential moving average of the "live" params.
    # avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
    @jax.jit
    def ema_update(params, avg_params):
        return optax.incremental_update(params, avg_params, step_size=0.001)

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

    params = avg_params = net.init(jax.random.PRNGKey(42), X_train)
    opt_state = opt.init(params)

    # Train/eval loop.
    print(X_train.shape)
    exit()
    from sklearn import metrics
    for step in range(10001):
        params, opt_state = update(params, opt_state, X_train, Y_train)
        y_hat = net.apply(params, X_train)
        if (step%100 == 0):
            mse = metrics.mean_squared_error(Y_train, y_hat)
            print(mse)
        #print(opt_state)



if __name__ == "__main__":
    app.run(main)
