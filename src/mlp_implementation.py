"""MLP de una o mas capas ocultas con retropropagacion manual para XOR."""

import argparse
import math
from numbers import Integral
from typing import Dict, List, Optional, Sequence, Union

import numpy as np


class SimpleMLP:
    """Perceptron multicapa con salida lineal y descenso por gradiente batch.

    ``hidden_size`` acepta un entero (una capa oculta) o una secuencia de
    enteros (por ejemplo, ``(4, 4)`` para dos capas ocultas). La opcion
    ``initialization="guide"`` reproduce la escala 0.01 del ejemplo de la
    guia. Por defecto se usan He para capas ocultas ReLU y Xavier para
    Sigmoide/salida lineal.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, Sequence[int]],
        output_size: int,
        activation: str = "relu",
        seed: int = 42,
        initialization: str = "scaled",
    ) -> None:
        if isinstance(input_size, bool) or input_size < 1:
            raise ValueError("input_size debe ser un entero positivo.")
        if isinstance(output_size, bool) or output_size < 1:
            raise ValueError("output_size debe ser un entero positivo.")
        if isinstance(hidden_size, Integral):
            hidden_sizes = (int(hidden_size),)
        else:
            hidden_sizes = tuple(int(size) for size in hidden_size)
        if not hidden_sizes or any(size < 1 for size in hidden_sizes):
            raise ValueError("Debe indicarse al menos una capa oculta positiva.")
        if activation not in ("relu", "sigmoid"):
            raise ValueError("activation debe ser 'relu' o 'sigmoid'.")
        if initialization not in ("scaled", "guide"):
            raise ValueError("initialization debe ser 'scaled' o 'guide'.")

        self.input_size = int(input_size)
        self.hidden_sizes = hidden_sizes
        self.output_size = int(output_size)
        self.activation_name = activation
        self.initialization = initialization
        self.layer_sizes = (self.input_size,) + hidden_sizes + (self.output_size,)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._rng = np.random.RandomState(seed)

        for layer_index, (fan_in, fan_out) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ):
            is_hidden_layer = layer_index < len(self.layer_sizes) - 2
            if initialization == "guide":
                scale = 0.01
            elif is_hidden_layer and activation == "relu":
                scale = math.sqrt(2.0 / fan_in)  # Inicializacion He.
            else:
                scale = math.sqrt(2.0 / (fan_in + fan_out))  # Xavier/Glorot.

            self.weights.append(self._rng.randn(fan_in, fan_out) * scale)
            self.biases.append(np.zeros((1, fan_out), dtype=float))

    @staticmethod
    def _feature_matrix(X: np.ndarray, input_size: int) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2 or features.shape[1] != input_size:
            raise ValueError(
                "X debe tener forma (numero_de_muestras, input_size)."
            )
        if not np.isfinite(features).all():
            raise ValueError("X contiene valores no finitos.")
        return features

    def _target_matrix(self, y: np.ndarray, sample_count: int) -> np.ndarray:
        targets = np.asarray(y, dtype=float)
        if targets.ndim == 1 and self.output_size == 1:
            targets = targets.reshape(-1, 1)
        if targets.ndim != 2 or targets.shape != (sample_count, self.output_size):
            raise ValueError(
                "y debe tener forma (numero_de_muestras, output_size)."
            )
        if not np.isfinite(targets).all():
            raise ValueError("y contiene valores no finitos.")
        return targets

    def _activate(self, values: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            return np.maximum(0.0, values)
        return 1.0 / (1.0 + np.exp(-np.clip(values, -500.0, 500.0)))

    def _activation_derivative(
        self, pre_activation: np.ndarray, activation: np.ndarray
    ) -> np.ndarray:
        if self.activation_name == "relu":
            # En z = 0 se define la derivada de ReLU como cero.
            return (pre_activation > 0.0).astype(float)
        return activation * (1.0 - activation)

    def _forward_with_cache(
        self, X: np.ndarray
    ) -> tuple:
        features = self._feature_matrix(X, self.input_size)
        activations = [features]
        pre_activations = []

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for layer_index, (weights, biases) in enumerate(
                zip(self.weights, self.biases)
            ):
                z = activations[-1] @ weights + biases
                pre_activations.append(z)
                if layer_index < len(self.weights) - 1:
                    activations.append(self._activate(z))
                else:
                    # La guia utiliza una salida lineal para este ejemplo.
                    activations.append(z)

        return activations[-1], activations, pre_activations

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Calcula la salida de la red sin actualizar sus parametros."""
        output, _, _ = self._forward_with_cache(X)
        return output

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.1,
        epochs: int = 1000,
        loss_target: Optional[float] = None,
        stability_patience: int = 20,
    ) -> Dict[str, object]:
        """Entrena la red y devuelve perdida, estado e historial por epoca.

        Si se da ``loss_target``, la red se detiene cuando mantiene esa perdida
        y clasificacion perfecta durante ``stability_patience`` epocas seguidas.
        La perdida es MSE y su gradiente se calcula con el factor 2 exacto.
        """
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate debe ser positivo y finito.")
        if epochs < 1:
            raise ValueError("epochs debe ser positivo.")
        if stability_patience < 1:
            raise ValueError("stability_patience debe ser positivo.")
        if loss_target is not None and (
            not math.isfinite(loss_target) or loss_target <= 0.0
        ):
            raise ValueError("loss_target debe ser positivo y finito.")

        features = self._feature_matrix(X, self.input_size)
        targets = self._target_matrix(y, features.shape[0])
        history: List[float] = []
        stable_epochs = 0
        stabilized_epoch: Optional[int] = None
        diverged_epoch: Optional[int] = None
        final_loss = float("nan")
        status = "max_epochs"

        for epoch in range(1, epochs + 1):
            output, activations, pre_activations = self._forward_with_cache(features)
            if not np.isfinite(output).all():
                final_loss = float("inf")
                history.append(final_loss)
                status = "diverged"
                diverged_epoch = epoch
                break

            final_loss = float(np.mean(np.square(output - targets)))
            history.append(final_loss)
            if not math.isfinite(final_loss):
                status = "diverged"
                diverged_epoch = epoch
                break

            predicted_labels = output >= 0.5
            expected_labels = targets >= 0.5
            is_correct = bool(np.array_equal(predicted_labels, expected_labels))
            meets_target = loss_target is not None and final_loss <= loss_target and is_correct

            if meets_target:
                stable_epochs += 1
                if stable_epochs >= stability_patience:
                    status = "converged"
                    stabilized_epoch = epoch
                    break
            else:
                stable_epochs = 0

            # d(MSE)/d(output); ``targets.size`` incluye muestras y salidas.
            delta = (2.0 / targets.size) * (output - targets)
            gradients_w: List[Optional[np.ndarray]] = [None] * len(self.weights)
            gradients_b: List[Optional[np.ndarray]] = [None] * len(self.biases)

            with np.errstate(over="ignore", invalid="ignore"):
                for layer_index in reversed(range(len(self.weights))):
                    gradients_w[layer_index] = activations[layer_index].T @ delta
                    gradients_b[layer_index] = np.sum(delta, axis=0, keepdims=True)

                    if layer_index > 0:
                        hidden_activation = activations[layer_index]
                        derivative = self._activation_derivative(
                            pre_activations[layer_index - 1], hidden_activation
                        )
                        delta = (
                            delta @ self.weights[layer_index].T
                        ) * derivative

                for layer_index in range(len(self.weights)):
                    grad_w = gradients_w[layer_index]
                    grad_b = gradients_b[layer_index]
                    if grad_w is None or grad_b is None:
                        raise RuntimeError("No se pudo calcular un gradiente.")
                    self.weights[layer_index] -= learning_rate * grad_w
                    self.biases[layer_index] -= learning_rate * grad_b

            if not self._parameters_are_finite():
                status = "diverged"
                diverged_epoch = epoch
                break

        return {
            "status": status,
            "epochs_completed": len(history),
            "stabilized_epoch": stabilized_epoch,
            "diverged_epoch": diverged_epoch,
            "final_loss": final_loss,
            "loss_history": history,
        }

    def _parameters_are_finite(self) -> bool:
        return all(np.isfinite(value).all() for value in self.weights + self.biases)


def _xor_data() -> tuple:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    return X, y


def _accuracy(output: np.ndarray, targets: np.ndarray) -> float:
    if not np.isfinite(output).all():
        return 0.0
    predicted = output >= 0.5
    expected = targets >= 0.5
    return float(np.mean(predicted == expected))


def _format_loss(value: float) -> str:
    return f"{value:.8g}" if math.isfinite(value) else "no finita"


def run_baseline() -> None:
    X, y = _xor_data()
    model = SimpleMLP(
        input_size=2,
        hidden_size=4,
        output_size=1,
        activation="relu",
        seed=42,
        initialization="scaled",
    )
    result = model.train(X, y, learning_rate=0.1, epochs=1000)
    predictions = model.forward(X)

    print("=== MLP base para XOR ===")
    print("Arquitectura: 2 -> 4 (ReLU) -> 1 (lineal)")
    print("Epocas ejecutadas: {}".format(result["epochs_completed"]))
    print("Perdida MSE: {}".format(_format_loss(float(result["final_loss"]))))
    print("Exactitud XOR: {:.1f}%".format(100.0 * _accuracy(predictions, y)))
    print("Predicciones [00, 01, 10, 11]:")
    print(np.round(predictions, 6))


def _train_experiment(
    hidden_size: Union[int, Sequence[int]],
    activation: str,
    learning_rate: float,
    initialization: str,
    max_epochs: int,
) -> Dict[str, object]:
    X, y = _xor_data()
    model = SimpleMLP(
        input_size=2,
        hidden_size=hidden_size,
        output_size=1,
        activation=activation,
        seed=42,
        initialization=initialization,
    )
    initial_output = model.forward(X)
    initial_loss = float(np.mean(np.square(initial_output - y)))
    result = model.train(
        X,
        y,
        learning_rate=learning_rate,
        epochs=max_epochs,
        loss_target=0.001,
        stability_patience=20,
    )
    predictions = model.forward(X)
    result["initial_loss"] = initial_loss
    result["peak_loss"] = max(result["loss_history"])
    result["accuracy"] = _accuracy(predictions, y)
    result["predictions"] = predictions
    return result


def _print_experiment(label: str, result: Dict[str, object]) -> None:
    status_names = {
        "converged": "criterio alcanzado",
        "max_epochs": "limite de epocas",
        "diverged": "divergio",
    }
    status = status_names.get(str(result["status"]), str(result["status"]))
    stabilized = result["stabilized_epoch"]
    epoch_text = (
        str(stabilized)
        if stabilized is not None
        else "no ({} epocas)".format(result["epochs_completed"])
    )
    print(
        "{}: estado={}, estabilizacion={}, MSE inicial/pico/final={}/{}/{}, "
        "exactitud={:.1f}%".format(
            label,
            status,
            epoch_text,
            _format_loss(float(result["initial_loss"])),
            _format_loss(float(result["peak_loss"])),
            _format_loss(float(result["final_loss"])),
            100.0 * float(result["accuracy"]),
        )
    )


def run_experiments() -> None:
    print("=== Diagnostico de la escala 0.01 del ejemplo ===")
    guide_result = _train_experiment(4, "relu", 0.1, "guide", 1000)
    _print_experiment("inicializacion=0.01", guide_result)

    print("=== Actividad 1: tasa de aprendizaje (ReLU, He, maximo 1000 epocas) ===")
    for rate in (0.9, 0.1, 0.0001):
        result = _train_experiment(4, "relu", rate, "scaled", 1000)
        _print_experiment("learning_rate={}".format(rate), result)

    print("\n=== Actividad 2: ReLU frente a Sigmoide ===")
    activation_results = {}
    for activation in ("relu", "sigmoid"):
        result = _train_experiment(4, activation, 0.1, "scaled", 10000)
        activation_results[activation] = result
        _print_experiment("activacion={}".format(activation), result)

    relu_epoch = activation_results["relu"]["stabilized_epoch"]
    sigmoid_epoch = activation_results["sigmoid"]["stabilized_epoch"]
    if relu_epoch is not None and sigmoid_epoch is not None:
        print(
            "Epocas adicionales con Sigmoide: {} ({:.2f}x ReLU).".format(
                sigmoid_epoch - relu_epoch, sigmoid_epoch / relu_epoch
            )
        )

    print("\n=== Actividad 3: segunda capa oculta (ReLU, He) ===")
    for hidden_sizes in ((4,), (4, 4)):
        result = _train_experiment(hidden_sizes, "relu", 0.1, "scaled", 10000)
        _print_experiment("capas_ocultas={}".format(hidden_sizes), result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena un MLP con backpropagation manual para resolver XOR."
    )
    parser.add_argument(
        "--experiments",
        action="store_true",
        help="ejecuta tambien las tres actividades practicas de la guia",
    )
    args = parser.parse_args()

    run_baseline()
    if args.experiments:
        print()
        run_experiments()


if __name__ == "__main__":
    main()
