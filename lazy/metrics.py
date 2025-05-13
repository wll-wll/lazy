import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def metric_cal(y_true, y_pred=None, metric_type='mse'):
    metrics = {
        'mse': lambda y_true, y_pred: mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)),
        'mae': mean_absolute_error,
        'snr': lambda y_true, y_pred: snr(y_true, y_pred),
        'normalized_mse': normalized_mse,
    }

    if metric_type not in metrics:
        raise ValueError(f"Invalid metric_type '{metric_type}'. Supported metrics are: {list(metrics.keys())}")

    return metrics[metric_type](y_true, y_pred)


def snr(clean_signal, signal_with_noise=None):
    if signal_with_noise is not None:
        noise = clean_signal - signal_with_noise
        signal_power = np.mean(clean_signal ** 2)
        noise_power = np.mean(noise ** 2)
        snr_value = 10 * np.log10(signal_power / noise_power)
    else:
        signal_mean = np.mean(clean_signal)
        noise_std = np.std(clean_signal)
        snr_value = 20 * np.log10(abs(signal_mean) / noise_std)

    if np.isinf(snr_value):
        return 100
    return snr_value


def normalized_mse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(y_true ** 2)
    return numerator / denominator


def adjusted_r2(y_true, y_pred, X_test):
    r2 = r2_score(y_true, y_pred)
    n, p = X_test.shape
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))
