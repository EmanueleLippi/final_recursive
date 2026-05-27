"""Native TensorFlow 2 models for the Girsanov-like recursive FBSDE solver."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from .io_utils import _as_blob_dict, save_blob_npz
from .tf_backend import assert_modern_tensorflow, tf

assert_modern_tensorflow()


class _SessionShim:
    """Tiny compatibility shim for old orchestration code that only closes/runs vars."""

    def __init__(self, owner: "FBSNN") -> None:
        self._owner = owner

    def run(self, fetches, feed_dict=None):
        if feed_dict:
            raise RuntimeError("Native TF2 models do not accept TF1 feed_dict execution.")
        return self._materialize(fetches)

    def close(self) -> None:
        self._owner.close()

    def _materialize(self, value):
        if isinstance(value, (list, tuple)):
            return [self._materialize(item) for item in value]
        if isinstance(value, dict):
            return {key: self._materialize(item) for key, item in value.items()}
        if hasattr(value, "numpy"):
            return value.numpy()
        return value


class FBSNN(tf.Module, ABC):
    """Forward-Backward Stochastic Neural Network implemented with TF2 tapes."""

    def __init__(
        self,
        Xi_generator,
        T,
        M,
        N,
        D,
        layers,
        clip_grad_norm=1.0,
        use_antithetic_sampling=True,
        same_xi_antithetic_sampling=False,
        dynamic_loss_dt_normalization=False,
        dynamic_loss_weight=1.0,
        terminal_y_loss_weight=1.0,
        terminal_z_loss_weight=1.0,
        terminal_z_component_weights=None,
        structural_z_loss_weight=0.0,
        structural_z_component_weights=None,
        log_device_placement=False,
    ):
        super().__init__(name=self.__class__.__name__)
        self.Xi_generator = Xi_generator
        self.T = np.float32(T)
        self.M = int(M)
        self.N = int(N)
        self.D = int(D)
        self.layers = list(layers)
        self.clip_grad_norm = clip_grad_norm
        self.use_antithetic_sampling = bool(use_antithetic_sampling)
        self.same_xi_antithetic_sampling = bool(same_xi_antithetic_sampling)
        self.dynamic_loss_dt_normalization = bool(dynamic_loss_dt_normalization)
        self.dynamic_loss_weight = np.float32(dynamic_loss_weight)
        self.terminal_y_loss_weight = np.float32(terminal_y_loss_weight)
        self.terminal_z_loss_weight = np.float32(terminal_z_loss_weight)
        self.structural_z_loss_weight = np.float32(structural_z_loss_weight)
        for name in (
            "dynamic_loss_weight",
            "terminal_y_loss_weight",
            "terminal_z_loss_weight",
            "structural_z_loss_weight",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.terminal_z_component_weights_np = self._prepare_component_weights(
            terminal_z_component_weights,
            default_value=1.0,
            name="terminal_z_component_weights",
        )
        self.structural_z_component_weights_np = self._prepare_component_weights(
            structural_z_component_weights,
            default_value=0.0,
            name="structural_z_component_weights",
        )
        self.terminal_z_component_weights_tf = tf.constant(
            self.terminal_z_component_weights_np, dtype=tf.float32
        )
        self.structural_z_component_weights_tf = tf.constant(
            self.structural_z_component_weights_np, dtype=tf.float32
        )
        self._legacy_loss_composition = self._is_legacy_loss_config()
        self.log_device_placement = bool(log_device_placement)
        self.const = np.float32(getattr(self, "const", 1.0))
        self.const_tf = tf.Variable(
            self.const,
            trainable=False,
            dtype=tf.float32,
            name="const",
        )

        self.weights, self.biases = self.initialize_NN(self.layers)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-3)
        self.sess = _SessionShim(self)
        self._checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            weights=self.weights,
            biases=self.biases,
            const=self.const_tf,
        )

    def _prepare_component_weights(self, values, default_value: float, name: str) -> np.ndarray:
        if values is None:
            weights = np.full((self.D,), np.float32(default_value), dtype=np.float32)
        else:
            if isinstance(values, str):
                raw_values = [item.strip() for item in values.split(",") if item.strip() != ""]
                weights = np.asarray(raw_values, dtype=np.float32)
            else:
                weights = np.asarray(values, dtype=np.float32).reshape(-1)
            if weights.size == 0:
                weights = np.full((self.D,), np.float32(default_value), dtype=np.float32)
            elif weights.size == 1:
                weights = np.full((self.D,), np.float32(weights[0]), dtype=np.float32)
            elif weights.size != self.D:
                raise ValueError(
                    f"{name} must contain 1 or {self.D} values, got {weights.size}"
                )
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"{name} must contain only finite values")
        if np.any(weights < 0.0):
            raise ValueError(f"{name} must be non-negative")
        return weights.reshape(1, self.D).astype(np.float32)

    def _is_legacy_loss_config(self) -> bool:
        return bool(
            not self.dynamic_loss_dt_normalization
            and np.isclose(float(self.dynamic_loss_weight), 1.0)
            and np.isclose(float(self.terminal_y_loss_weight), 1.0)
            and np.isclose(float(self.terminal_z_loss_weight), 1.0)
            and np.isclose(float(self.structural_z_loss_weight), 0.0)
            and np.allclose(self.terminal_z_component_weights_np, 1.0)
            and np.allclose(self.structural_z_component_weights_np, 0.0)
        )

    @property
    def trainable_variables(self):
        return list(self.weights) + list(self.biases)

    def close(self) -> None:
        """Compatibility hook; variables are released by normal Python lifetime."""
        return None

    def save_model(self, path):
        save_path = self._checkpoint.save(file_prefix=path)
        print(f"Model saved in path: {save_path}")

    def load_model(self, path):
        self._checkpoint.restore(path).expect_partial()
        print(f"Model restored from path: {path}")

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]], name=f"W_{l}")
            b = tf.Variable(
                tf.zeros([1, layers[l + 1]], dtype=tf.float32),
                dtype=tf.float32,
                name=f"b_{l}",
            )
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size, name: Optional[str] = None):
        in_dim = int(size[0])
        out_dim = int(size[1])
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal(
                [in_dim, out_dim],
                stddev=np.float32(xavier_stddev),
                dtype=tf.float32,
            ),
            dtype=tf.float32,
            name=name,
        )

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = tf.cast(X, tf.float32)
        for l in range(0, num_layers - 2):
            H = tf.sin(tf.add(tf.matmul(H, weights[l]), biases[l]))
        return tf.add(tf.matmul(H, weights[-1]), biases[-1])

    def net_u(self, t, X):
        X = tf.cast(X, tf.float32)
        t = tf.cast(t, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X)
            u = self.neural_net(tf.concat([t, X], 1), self.weights, self.biases)
        Du = tape.gradient(u, X)
        if Du is None:
            Du = tf.zeros_like(X)
        return u, Du

    def Dg_tf(self, X):
        X = tf.cast(X, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X)
            value = self.g_tf(X)
        grad = tape.gradient(value, X)
        if grad is None:
            grad = tf.zeros_like(X)
        return grad

    def loss_function(self, t, W, Xi, const_value=None, return_components=False):
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        W = tf.convert_to_tensor(W, dtype=tf.float32)
        Xi = tf.convert_to_tensor(Xi, dtype=tf.float32)
        if const_value is not None:
            self.const_tf.assign(tf.cast(const_value, tf.float32))

        loss_dynamic = tf.constant(0.0, dtype=tf.float32)
        loss_dynamic_normalized = tf.constant(0.0, dtype=tf.float32)
        X_list = []
        Y_list = []
        Z_list = []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        X0 = Xi
        Y0, Du0 = self.net_u(t0, X0)
        sigma0 = self.sigma_tf(t0, X0, Y0)
        Z0 = tf.squeeze(tf.matmul(tf.expand_dims(Du0, 1), sigma0), axis=1)

        X_list.append(X0)
        Y_list.append(Y0)
        Z_list.append(Z0)

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]

            dW = W1 - W0
            sigma_dW = tf.squeeze(tf.matmul(sigma0, tf.expand_dims(dW, -1)), axis=-1)
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + sigma_dW

            Y1_tilde = (
                Y0
                + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0)
                + tf.reduce_sum(Z0 * dW, axis=1, keepdims=True)
            )

            Y1, Du1 = self.net_u(t1, X1)
            sigma1 = self.sigma_tf(t1, X1, Y1)
            Z1 = tf.squeeze(tf.matmul(tf.expand_dims(Du1, 1), sigma1), axis=1)

            dynamic_residual_sq = tf.square(Y1 - Y1_tilde)
            loss_dynamic += tf.reduce_sum(dynamic_residual_sq)
            if self.dynamic_loss_dt_normalization:
                dt_step = tf.maximum(t1 - t0, tf.constant(1.0e-8, dtype=tf.float32))
                loss_dynamic_normalized += tf.reduce_mean(dynamic_residual_sq / dt_step)
            else:
                loss_dynamic_normalized += tf.reduce_mean(dynamic_residual_sq)

            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1
            sigma0 = sigma1

            X_list.append(X0)
            Y_list.append(Y0)
            Z_list.append(Z0)

        terminal_y_residual = Y1 - self.g_tf(X1)
        loss_terminal_y = tf.reduce_sum(tf.square(terminal_y_residual))

        Dg = self.Dg_tf(X1)
        Z_terminal = tf.squeeze(tf.matmul(tf.expand_dims(Dg, 1), sigma1), axis=1)
        terminal_z_residual = Z1 - Z_terminal
        loss_terminal_z = tf.reduce_sum(tf.square(terminal_z_residual))

        X = tf.stack(X_list, axis=1)
        Y = tf.stack(Y_list, axis=1)
        Z = tf.stack(Z_list, axis=1)
        structural_weights = tf.reshape(self.structural_z_component_weights_tf, [1, 1, self.D])
        loss_structural_z = tf.reduce_mean(tf.square(Z) * structural_weights)
        loss_structural_z_raw = tf.reduce_sum(tf.square(Z) * structural_weights)

        scale = tf.cast(self.N, tf.float32)
        if self._legacy_loss_composition:
            loss = loss_dynamic + loss_terminal_y + loss_terminal_z + loss_structural_z_raw
            loss_dynamic_component = loss_dynamic / scale
            loss_terminal_y_component = loss_terminal_y / scale
            loss_terminal_z_component = loss_terminal_z / scale
            loss_structural_z_component = loss_structural_z_raw / scale
            weighted_dynamic_component = loss_dynamic_component
            weighted_terminal_y_component = loss_terminal_y_component
            weighted_terminal_z_component = loss_terminal_z_component
            weighted_structural_z_component = loss_structural_z_component
            loss_scaled = loss / scale
        else:
            terminal_y_mse = tf.reduce_mean(tf.square(terminal_y_residual))
            weighted_terminal_z_residual = (
                tf.square(terminal_z_residual) * self.terminal_z_component_weights_tf
            )
            terminal_z_weighted_mse = tf.reduce_mean(weighted_terminal_z_residual)
            loss_dynamic_component = loss_dynamic_normalized / scale
            loss_terminal_y_component = terminal_y_mse
            loss_terminal_z_component = terminal_z_weighted_mse
            loss_structural_z_component = loss_structural_z
            weighted_dynamic_component = self.dynamic_loss_weight * loss_dynamic_component
            weighted_terminal_y_component = (
                self.terminal_y_loss_weight * loss_terminal_y_component
            )
            weighted_terminal_z_component = (
                self.terminal_z_loss_weight * loss_terminal_z_component
            )
            weighted_structural_z_component = (
                self.structural_z_loss_weight * loss_structural_z_component
            )
            loss_scaled = (
                weighted_dynamic_component
                + weighted_terminal_y_component
                + weighted_terminal_z_component
                + weighted_structural_z_component
            )

        if not return_components:
            return loss_scaled, X, Y, Z

        components = {
            "loss_total": loss_scaled,
            "loss_dynamic": loss_dynamic_component,
            "loss_terminal_y": loss_terminal_y_component,
            "loss_terminal_z": loss_terminal_z_component,
            "loss_structural_z": loss_structural_z_component,
            "loss_weighted_dynamic": weighted_dynamic_component,
            "loss_weighted_terminal_y": weighted_terminal_y_component,
            "loss_weighted_terminal_z": weighted_terminal_z_component,
            "loss_weighted_structural_z": weighted_structural_z_component,
        }
        for i in range(self.D):
            component_average_scale = tf.constant(1.0, dtype=tf.float32)
            if not self._legacy_loss_composition:
                component_average_scale = 1.0 / tf.cast(self.D, tf.float32)
            terminal_z_component = tf.reduce_sum(tf.square(terminal_z_residual[:, i])) / scale
            if not self._legacy_loss_composition:
                terminal_z_component = tf.reduce_mean(tf.square(terminal_z_residual[:, i]))
            terminal_z_weight = self.terminal_z_component_weights_tf[0, i]
            structural_z_component = tf.reduce_sum(tf.square(Z[:, :, i])) / scale
            if not self._legacy_loss_composition:
                structural_z_component = tf.reduce_mean(tf.square(Z[:, :, i]))
            structural_z_weight = self.structural_z_component_weights_tf[0, i]
            components[f"loss_terminal_z_component_{i}"] = terminal_z_component
            components[f"loss_weighted_terminal_z_component_{i}"] = (
                self.terminal_z_loss_weight
                * terminal_z_weight
                * terminal_z_component
                * component_average_scale
            )
            components[f"loss_structural_z_component_{i}"] = structural_z_component
            components[f"loss_weighted_structural_z_component_{i}"] = (
                self.structural_z_loss_weight
                * structural_z_weight
                * structural_z_component
                * component_average_scale
            )

        return loss_scaled, X, Y, Z, components

    def _sample_brownian_increments(self, dt: float) -> np.ndarray:
        M = self.M
        N = self.N
        D = self.D
        DW = np.zeros((M, N + 1, D), dtype=np.float32)
        if self.use_antithetic_sampling and M > 1:
            half_M = M // 2
            DW_half = np.sqrt(dt) * np.random.normal(size=(half_M, N, D))
            DW[:half_M, 1:, :] = DW_half
            DW[half_M : 2 * half_M, 1:, :] = -DW_half
            if M % 2 == 1:
                DW[-1, 1:, :] = np.sqrt(dt) * np.random.normal(size=(N, D))
        else:
            DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))
        return DW

    def _sample_initial_states(self) -> np.ndarray:
        M = self.M
        D = self.D
        if self.same_xi_antithetic_sampling and self.use_antithetic_sampling and M > 1:
            half_M = M // 2
            Xi_batch = np.zeros((M, D), dtype=np.float32)
            Xi_half = self.Xi_generator(half_M, D).astype(np.float32)
            Xi_batch[:half_M, :] = Xi_half
            Xi_batch[half_M : 2 * half_M, :] = Xi_half
            if M % 2 == 1:
                Xi_batch[-1, :] = self.Xi_generator(1, D).astype(np.float32)[0]
            return Xi_batch
        return self.Xi_generator(M, D).astype(np.float32)

    def _build_minibatch(self, t_start: float = 0.0):
        M = self.M
        N = self.N

        Dt = np.zeros((M, N + 1, 1), dtype=np.float32)
        dt = float(self.T) / float(N)
        Dt[:, 1:, :] = dt
        DW = self._sample_brownian_increments(dt)
        t = np.cumsum(Dt, axis=1)
        if float(t_start) != 0.0:
            t = np.float32(t_start) + t
        W = np.cumsum(DW, axis=1)
        Xi_batch = self._sample_initial_states()
        return t.astype(np.float32), W.astype(np.float32), Xi_batch

    def fetch_minibatch(self):
        return self._build_minibatch(t_start=0.0)

    def _get_snapshot(self):
        return [v.numpy().copy() for v in self.trainable_variables]

    def _restore_snapshot(self, weights):
        for variable, value in zip(self.trainable_variables, weights):
            variable.assign(value)

    def _set_optimizer_learning_rate(self, learning_rate: float) -> None:
        lr = np.float32(learning_rate)
        if hasattr(self.optimizer.learning_rate, "assign"):
            self.optimizer.learning_rate.assign(lr)
        else:
            self.optimizer.learning_rate = float(lr)

    def _set_const(self, const_value) -> None:
        self.const = np.float32(const_value)
        self.const_tf.assign(np.float32(const_value))

    @tf.function(reduce_retracing=True)
    def _train_step_tensor(self, t_batch, W_batch, Xi_batch, const_value):
        with tf.GradientTape() as tape:
            loss_value, _, _, _ = self.loss_function(
                t_batch,
                W_batch,
                Xi_batch,
                const_value=const_value,
            )
        variables = self.trainable_variables
        gradients = tape.gradient(loss_value, variables)
        grads_and_vars = [(g, v) for g, v in zip(gradients, variables) if g is not None]
        if self.clip_grad_norm is not None and grads_and_vars:
            grads, vars_ = zip(*grads_and_vars)
            clipped, _ = tf.clip_by_global_norm(grads, self.clip_grad_norm)
            grads_and_vars = list(zip(clipped, vars_))
        self.optimizer.apply_gradients(grads_and_vars)
        return loss_value

    def train(
        self,
        N_Iter,
        learning_rate,
        const_value=None,
        eval_every=50,
        val_batches=8,
        early_stopping_metric="loss",
        patience=None,
        min_delta=1e-3,
        restore_best=False,
    ):
        self._set_optimizer_learning_rate(float(learning_rate))
        start_time = time.time()
        last_loss = None
        current_const = np.float32(self.const if const_value is None else const_value)
        self._set_const(current_const)

        best_score = np.inf
        best_iter = -1
        best_snapshot = None
        no_improve_iters = 0
        stopped_early = False

        for it in range(int(N_Iter)):
            t_batch, W_batch, Xi_batch = self.fetch_minibatch()
            self._train_step_tensor(
                tf.convert_to_tensor(t_batch, dtype=tf.float32),
                tf.convert_to_tensor(W_batch, dtype=tf.float32),
                tf.convert_to_tensor(Xi_batch, dtype=tf.float32),
                tf.constant(current_const, dtype=tf.float32),
            )

            if it % 50 == 0:
                elapsed = time.time() - start_time
                loss_value, _, Y_value, _ = self.loss_function(
                    t_batch,
                    W_batch,
                    Xi_batch,
                    const_value=current_const,
                )
                last_loss = float(loss_value.numpy())
                mean_Y0 = np.mean(Y_value.numpy()[:, 0, 0])
                print(
                    "It: %d, Loss: %.3e, Mean Y0: %.3f, Time: %.2f, Learning Rate: %.3e"
                    % (it, last_loss, mean_Y0, elapsed, float(learning_rate))
                )
                start_time = time.time()

            if (it % int(eval_every) == 0) or (it == int(N_Iter) - 1):
                eval_stats = self.evaluate(const_value=current_const, n_batches=val_batches)
                if early_stopping_metric == "loss":
                    score = eval_stats["mean_loss"]
                else:
                    raise ValueError(f"Unsupported early_stopping_metric='{early_stopping_metric}'")

                if (best_score - score) > min_delta:
                    best_score = float(score)
                    best_iter = int(it)
                    best_snapshot = self._get_snapshot()
                    no_improve_iters = 0
                else:
                    no_improve_iters += int(eval_every)

                if patience is not None and no_improve_iters >= int(patience):
                    print(f"[EarlyStop] it={it}, best_it={best_iter}, best_score={best_score:.6e}")
                    stopped_early = True
                    break

        if restore_best and best_snapshot is not None:
            self._restore_snapshot(best_snapshot)
            print(f"[RestoreBest] best_it={best_iter}, best_score={best_score:.6e}")

        return {
            "const": float(current_const),
            "learning_rate": float(learning_rate),
            "n_iter": int(N_Iter),
            "last_loss": last_loss,
            "best_iter": int(best_iter),
            "best_score": float(best_score),
            "stopped_early": bool(stopped_early),
        }

    def evaluate(self, const_value=None, n_batches=5):
        current_const = np.float32(self.const if const_value is None else const_value)
        self._set_const(current_const)
        losses = []
        losses_per_sample = []
        y0s = []
        component_values: Dict[str, List[float]] = {}

        for _ in range(int(n_batches)):
            t_batch, W_batch, Xi_batch = self.fetch_minibatch()
            loss_value, _, y_value, _, components = self.loss_function(
                t_batch,
                W_batch,
                Xi_batch,
                const_value=current_const,
                return_components=True,
            )
            loss_value = float(loss_value.numpy())
            losses.append(loss_value)
            losses_per_sample.append(loss_value / float(self.M))
            y0s.append(list(y_value.numpy()[:, 0, 0]))
            for key, value in components.items():
                component_values.setdefault(key, []).append(float(value.numpy()))

        stats = {
            "const": float(current_const),
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "mean_loss_per_sample": float(np.mean(losses_per_sample)),
            "std_loss_per_sample": float(np.std(losses_per_sample)),
            "mean_y0": float(np.mean(y0s)),
            "std_y0": float(np.std(y0s)),
            "n_batches": int(n_batches),
        }
        for key, values in component_values.items():
            stats[f"mean_{key}"] = float(np.mean(values))
            stats[f"std_{key}"] = float(np.std(values))
            stats[f"mean_{key}_per_sample"] = float(np.mean(values) / float(self.M))
        return stats

    def predict(self, Xi_star, t_star, W_star, const_value=None):
        current_const = np.float32(self.const if const_value is None else const_value)
        self._set_const(current_const)
        _, X_star, Y_star, Z_star = self.loss_function(
            t_star,
            W_star,
            Xi_star,
            const_value=current_const,
        )
        return X_star.numpy(), Y_star.numpy(), Z_star.numpy()

    def get_weight_bias_arrays(self) -> List[np.ndarray]:
        return [v.numpy().astype(np.float32) for v in (self.weights + self.biases)]

    @abstractmethod
    def phi_tf(self, t, X, Y, Z):
        pass

    @abstractmethod
    def g_tf(self, X):
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        M = tf.shape(X)[0]
        D = tf.shape(X)[1]
        return tf.zeros([M, D], dtype=tf.float32)

    @abstractmethod
    def sigma_tf(self, t, X, Y):
        M = tf.shape(X)[0]
        D = tf.shape(X)[1]
        return tf.linalg.diag(tf.ones([M, D], dtype=tf.float32))


fbsde_NN = FBSNN


class NN_Quadratic_Coupled(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, parameters, **kwargs):
        for key in (
            "dynamic_loss_dt_normalization",
            "dynamic_loss_weight",
            "same_xi_antithetic_sampling",
            "terminal_y_loss_weight",
            "terminal_z_loss_weight",
            "terminal_z_component_weights",
            "structural_z_loss_weight",
            "structural_z_component_weights",
        ):
            if key in parameters and key not in kwargs:
                kwargs[key] = parameters[key]
        self.mu1 = parameters["mu1"]
        self.mu2 = parameters["mu2"]
        self.c1 = parameters["c1"]
        self.c2 = parameters["c2"]
        self.c3 = parameters["c3"]
        self.c4 = parameters["c4"]
        self.gamma = parameters["gamma"]
        self.s1 = parameters["s1"]
        self.s2 = parameters["s2"]
        self.s3 = parameters["s3"]
        self.x_max = parameters["x_max"]
        self.v_min = parameters["v_min"]
        self.v_max = parameters["v_max"]
        self.d = parameters["d"]
        self.const = parameters["const"]
        super().__init__(Xi, T, M, N, D, layers, **kwargs)

    def psi(self, X_state):
        return tf.maximum(
            0.0,
            tf.minimum(
                1.0,
                tf.minimum(X_state / self.d, (self.x_max - X_state) / self.d),
            ),
        )

    def psi3(self, V):
        return tf.maximum(0.0, tf.minimum(1.0, (self.v_max - V) / self.d))

    def psi4(self, V):
        return tf.maximum(0.0, tf.minimum(1.0, (V - self.v_min) / self.d))

    def f(self, X, Z):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        Z_S, Z_H, Z_V, _ = tf.split(Z, num_or_size_splits=4, axis=1)
        s1 = tf.cast(self.s1, tf.float32)
        gamma = tf.cast(self.gamma, tf.float32)
        exp_S = tf.exp(-S)
        return -0.5 * V * self.psi(-exp_S * Z_S / (gamma * s1))

    def mu_tf(self, t, X, Y, Z):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        mu1 = tf.cast(self.mu1, tf.float32)
        mu2 = tf.cast(self.mu2, tf.float32)
        c1 = tf.cast(self.c1, tf.float32)
        c2 = tf.cast(self.c2, tf.float32)
        c3 = tf.cast(self.c3, tf.float32)
        c4 = tf.cast(self.c4, tf.float32)
        x_max = tf.cast(self.x_max, tf.float32)
        const = tf.cast(self.const_tf, tf.float32)

        dS = mu1 * (c1 - S)
        dH = mu2 * (c2 - H)
        dV = (
            self.f(X, const * Z) * self.psi(X_state)
            + c3 * self.psi(-X_state) * self.psi3(V)
            - c4 * self.psi(X_state - x_max) * self.psi4(V)
        )
        dX = V
        return tf.concat([dS, dH, dV, dX], axis=1)

    def g_tf(self, X):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        gamma = tf.cast(self.gamma, tf.float32)
        exp_S = tf.exp(S)
        return -gamma * exp_S * X_state + V ** 2 + V * X_state

    def phi_tf(self, t, X, Y, Z):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        Z_S, Z_H, Z_V, _ = tf.split(Z, num_or_size_splits=4, axis=1)

        mu1 = tf.cast(self.mu1, tf.float32)
        c1 = tf.cast(self.c1, tf.float32)
        s1 = tf.cast(self.s1, tf.float32)
        s3 = tf.cast(self.s3, tf.float32)
        c3 = tf.cast(self.c3, tf.float32)
        c4 = tf.cast(self.c4, tf.float32)
        x_max = tf.cast(self.x_max, tf.float32)
        gamma = tf.cast(self.gamma, tf.float32)
        const = tf.cast(self.const_tf, tf.float32)

        exp_S = tf.exp(S)

        term1 = -gamma * exp_S * X_state * mu1 * (c1 - S)
        term2 = (2 * V + X_state) * (
            self.f(X, Z) * self.psi(X_state)
            + c3 * self.psi(-X_state) * self.psi3(V)
            - c4 * self.psi(X_state - x_max) * self.psi4(V)
        )
        term3 = -gamma * exp_S * V + (0.5 * (Z_V / s3 - X_state)) ** 2
        term4 = -0.5 * gamma * exp_S * X_state * s1 ** 2 + s3 ** 2
        term5 = (Z_V / s3) * (self.f(X, const * Z) - self.f(X, Z)) * self.psi(X_state)

        return term1 + term2 + term3 + term4 + term5

    def sigma_tf(self, t, X, Y):
        S, H, V, X_state = tf.split(X, num_or_size_splits=4, axis=1)
        s1 = tf.cast(self.s1, tf.float32)
        s2 = tf.cast(self.s2, tf.float32)
        s3 = tf.cast(self.s3, tf.float32)

        zeros = tf.zeros_like(S)
        ones = tf.ones_like(S)

        r1 = tf.concat([s1 * ones, zeros, zeros, zeros], axis=1)
        r2 = tf.concat([zeros, s2 * ones, zeros, zeros], axis=1)
        r3 = tf.concat([zeros, zeros, s3 * ones, zeros], axis=1)
        r4 = tf.concat([zeros, zeros, zeros, zeros], axis=1)

        return tf.stack([r1, r2, r3, r4], axis=1)


class NN_Quadratic_Coupled_Recursive(NN_Quadratic_Coupled):
    """
    Recursive block model:
    - absolute time inside [t_start, t_end]
    - optional time normalization
    - terminal value supplied by a frozen next-block network blob when present
    """

    def __init__(
        self,
        Xi_generator,
        T,
        M,
        N,
        D,
        layers,
        parameters,
        t_start,
        t_end,
        T_total,
        terminal_blob=None,
        normalize_time_input=True,
        x_norm_mean=None,
        x_norm_std=None,
    ):
        self.t_start = np.float32(t_start)
        self.t_end = np.float32(t_end)
        self.T_total = np.float32(T_total)
        self.normalize_time_input = bool(normalize_time_input)

        x_mean = (
            np.zeros((1, D), dtype=np.float32)
            if x_norm_mean is None
            else np.asarray(x_norm_mean, dtype=np.float32).reshape(1, D)
        )
        x_std = (
            np.ones((1, D), dtype=np.float32)
            if x_norm_std is None
            else np.asarray(x_norm_std, dtype=np.float32).reshape(1, D)
        )
        self.x_norm_mean_np = x_mean
        self.x_norm_std_np = np.maximum(x_std, 1.0e-3).astype(np.float32)

        self.terminal_blob = _as_blob_dict(terminal_blob)
        self._terminal_weights_tf = None
        self._terminal_biases_tf = None
        self._terminal_x_mean_tf = None
        self._terminal_x_std_tf = None
        self._terminal_T_total_tf = None
        self._terminal_use_time = False

        self._x_norm_mean_tf = tf.constant(self.x_norm_mean_np, dtype=tf.float32)
        self._x_norm_std_tf = tf.constant(self.x_norm_std_np, dtype=tf.float32)
        self._T_total_tf = tf.constant(self.T_total, dtype=tf.float32)

        super().__init__(Xi_generator, T, M, N, D, layers, parameters)
        self._build_terminal_constants_if_needed()

    def _normalize_t(self, t):
        if not self.normalize_time_input:
            return t
        return 2.0 * (t / self._T_total_tf) - 1.0

    def _normalize_x(self, X):
        return (X - self._x_norm_mean_tf) / self._x_norm_std_tf

    def net_u(self, t, X):
        X = tf.cast(X, tf.float32)
        t = tf.cast(t, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X)
            t_in = self._normalize_t(t)
            X_in = self._normalize_x(X)
            u = self.neural_net(tf.concat([t_in, X_in], 1), self.weights, self.biases)
        Du = tape.gradient(u, X)
        if Du is None:
            Du = tf.zeros_like(X)
        return u, Du

    def fetch_minibatch(self):
        return self._build_minibatch(t_start=float(self.t_start))

    def _build_terminal_constants_if_needed(self):
        if self.terminal_blob is None:
            return
        if self._terminal_weights_tf is not None:
            return

        n_layers = int(self.terminal_blob["n_layers"])
        self._terminal_weights_tf = []
        self._terminal_biases_tf = []
        for i in range(n_layers):
            self._terminal_weights_tf.append(
                tf.constant(self.terminal_blob[f"W_{i}"], dtype=tf.float32)
            )
            self._terminal_biases_tf.append(
                tf.constant(self.terminal_blob[f"b_{i}"], dtype=tf.float32)
            )

        self._terminal_x_mean_tf = tf.constant(
            self.terminal_blob.get("x_norm_mean", np.zeros((1, self.D), dtype=np.float32)),
            dtype=tf.float32,
        )
        self._terminal_x_std_tf = tf.constant(
            np.maximum(
                self.terminal_blob.get("x_norm_std", np.ones((1, self.D), dtype=np.float32)),
                1.0e-3,
            ),
            dtype=tf.float32,
        )
        self._terminal_T_total_tf = tf.constant(
            np.float32(self.terminal_blob.get("T_total", self.T_total)), dtype=tf.float32
        )
        self._terminal_use_time = bool(int(self.terminal_blob.get("normalize_time_input", 1)))

    def _terminal_u(self, t_abs, X):
        self._build_terminal_constants_if_needed()
        t_in = t_abs
        if self._terminal_use_time:
            t_in = 2.0 * (t_abs / self._terminal_T_total_tf) - 1.0
        X_in = (X - self._terminal_x_mean_tf) / self._terminal_x_std_tf
        return self.neural_net(
            tf.concat([t_in, X_in], 1), self._terminal_weights_tf, self._terminal_biases_tf
        )

    def g_tf(self, X):
        if self.terminal_blob is None:
            return super().g_tf(X)
        t_eval = tf.ones([tf.shape(X)[0], 1], dtype=tf.float32) * tf.constant(
            self.t_end, dtype=tf.float32
        )
        return self._terminal_u(t_eval, X)

    def Dg_tf(self, X):
        if self.terminal_blob is None:
            return super().Dg_tf(X)
        X = tf.cast(X, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X)
            value = self.g_tf(X)
        grad = tape.gradient(value, X)
        if grad is None:
            grad = tf.zeros_like(X)
        return grad

    def export_parameter_blob(self) -> Dict[str, np.ndarray]:
        values = self.get_weight_bias_arrays()
        n_layers = len(self.weights)
        blob = {
            "n_layers": np.array(n_layers, dtype=np.int32),
            "layers": np.asarray(self.layers, dtype=np.int32),
            "t_start": np.asarray(self.t_start, dtype=np.float32),
            "t_end": np.asarray(self.t_end, dtype=np.float32),
            "T_total": np.asarray(self.T_total, dtype=np.float32),
            "normalize_time_input": np.asarray(int(self.normalize_time_input), dtype=np.int32),
            "x_norm_mean": np.asarray(self.x_norm_mean_np, dtype=np.float32),
            "x_norm_std": np.asarray(self.x_norm_std_np, dtype=np.float32),
        }
        for i in range(n_layers):
            blob[f"W_{i}"] = values[i].astype(np.float32)
            blob[f"b_{i}"] = values[n_layers + i].astype(np.float32)
        return blob

    def import_parameter_blob(self, blob_or_path, strict=True):
        blob = _as_blob_dict(blob_or_path)
        if blob is None:
            return
        n_layers = len(self.weights)
        if strict and int(blob["n_layers"]) != n_layers:
            raise ValueError(
                f"n_layers mismatch: model={n_layers}, blob={int(blob['n_layers'])}"
            )
        for i in range(n_layers):
            w_key = f"W_{i}"
            b_key = f"b_{i}"
            if w_key in blob:
                self.weights[i].assign(np.asarray(blob[w_key], dtype=np.float32))
            elif strict:
                raise KeyError(f"Missing key {w_key} in blob")
            if b_key in blob:
                self.biases[i].assign(np.asarray(blob[b_key], dtype=np.float32))
            elif strict:
                raise KeyError(f"Missing key {b_key} in blob")

    def save_parameter_blob(self, path: str) -> None:
        save_blob_npz(self.export_parameter_blob(), path)

    def load_parameter_blob(self, path: str, strict=True) -> None:
        self.import_parameter_blob(path, strict=strict)
