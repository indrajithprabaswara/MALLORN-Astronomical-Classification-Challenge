import csv
import math
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path('Data')
FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']


def load_metadata(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_lightcurves(meta_rows, filename):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in meta_rows:
        grouped[row['split']][row['object_id']] = []
    for split, objects in grouped.items():
        with (DATA_DIR / split / filename).open(newline='') as handle:
            reader = csv.DictReader(handle)
            for entry in reader:
                if entry['object_id'] in objects:
                    objects[entry['object_id']].append(entry)
    return grouped


def slope(times, values):
    n = len(values)
    if n < 2:
        return 0.0
    sum_t = sum(times)
    sum_f = sum(values)
    sum_tf = sum(t * v for t, v in zip(times, values))
    sum_tt = sum(t * t for t in times)
    denom = n * sum_tt - sum_t * sum_t
    if denom == 0:
        return 0.0
    return (n * sum_tf - sum_t * sum_f) / denom

def lightcurve_features(meta_rows, split_data):
    features = {}
    for row in meta_rows:
        object_id = row['object_id']
        lc = split_data[row['split']][object_id]
        fluxes = [float(entry['Flux']) for entry in lc if entry['Flux']]
        times = [float(entry['Time (MJD)']) for entry in lc if entry['Flux']]
        errs = [float(entry['Flux_err']) for entry in lc if entry['Flux_err']]
        n = len(fluxes)
        if n == 0:
            base = [0.0] * (14 + 2 * len(FILTERS))
        else:
            mean = sum(fluxes) / n
            std = math.sqrt(sum((v - mean) ** 2 for v in fluxes) / n)
            max_flux = max(fluxes)
            min_flux = min(fluxes)
            pos_frac = sum(1 for v in fluxes if v > 0) / n
            span = (max(times) - min(times)) if times else 0.0
            abs_mean = sum(abs(v) for v in fluxes) / n
            mean_err = sum(errs) / len(errs) if errs else 0.0
            slope_all = slope(times, fluxes)
            base = [
                n, mean, std, max_flux, min_flux, max_flux - min_flux,
                pos_frac, span, abs_mean, mean_err, slope_all,
                float(row['Z']) if row['Z'] else 0.0,
                float(row['Z_err']) if row['Z_err'] else 0.0,
                float(row['EBV']) if row['EBV'] else 0.0
            ]
            for flt in FILTERS:
                flt_flux = [float(entry['Flux']) for entry in lc if entry['Filter'] == flt and entry['Flux']]
                count = len(flt_flux)
                mean_f = sum(flt_flux) / count if count else 0.0
                base.extend([count, mean_f])
        features[object_id] = base
    return features

def standardize(vectors):
    length = len(next(iter(vectors.values())))
    means = [0.0] * length
    for vec in vectors.values():
        for i, val in enumerate(vec):
            means[i] += val
    n = len(vectors)
    means = [val / n for val in means]
    stds = [0.0] * length
    for vec in vectors.values():
        for i, val in enumerate(vec):
            diff = val - means[i]
            stds[i] += diff * diff
    stds = [math.sqrt(val / n) if val > 0 else 1.0 for val in stds]
    scaled = {key: [(val - means[i]) / stds[i] if stds[i] else 0.0 for i, val in enumerate(vec)] for key, vec in vectors.items()}
    return scaled, means, stds


def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)


def train_logistic(data, feature_len, epochs=400, lr=0.2, pos_weight=15.0, l2=0.01):
    weights = [0.0] * (feature_len + 1)
    n = len(data)
    for epoch in range(epochs):
        grad = [0.0] * (feature_len + 1)
        for vec, label in data:
            z = weights[0]
            for i, val in enumerate(vec):
                z += weights[i + 1] * val
            pred = sigmoid(z)
            weight = pos_weight if label == 1 else 1.0
            error = (pred - label) * weight
            grad[0] += error
            for i, val in enumerate(vec):
                grad[i + 1] += error * val
        rate = lr / n
        for i in range(feature_len + 1):
            weights[i] -= rate * (grad[i] + l2 * weights[i])
    return weights


def predict_prob(weights, vec):
    z = weights[0]
    for i, val in enumerate(vec):
        z += weights[i + 1] * val
    return sigmoid(z)

def train_and_evaluate():
    train_meta = load_metadata(DATA_DIR / 'train_log.csv')
    train_lightcurves = load_lightcurves(train_meta, 'train_full_lightcurves.csv')
    train_features = lightcurve_features(train_meta, train_lightcurves)
    scaled_train, feature_means, feature_stds = standardize(train_features)
    train_vectors = [(scaled_train[row['object_id']], int(row['target'])) for row in train_meta]
    feature_len = len(next(iter(scaled_train.values())))
    weights = train_logistic(train_vectors, feature_len)
    probs = [predict_prob(weights, vec) for vec, _ in train_vectors]
    labels = [label for _, label in train_vectors]
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in [i / 100 for i in range(5, 95, 5)]:
        tp = fp = fn = tn = 0
        for prob, label in zip(probs, labels):
            pred = 1 if prob >= threshold else 0
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    print(f"Training F1 (best threshold {best_threshold:.2f}): {best_f1:.3f}")
    return weights, feature_means, feature_stds, best_threshold


def generate_submission(weights, feature_means, feature_stds, threshold):
    test_meta = load_metadata(DATA_DIR / 'test_log.csv')
    test_lightcurves = load_lightcurves(test_meta, 'test_full_lightcurves.csv')
    test_features = lightcurve_features(test_meta, test_lightcurves)
    scaled_test = {
        obj: [(val - feature_means[i]) / feature_stds[i] if feature_stds[i] else 0.0 for i, val in enumerate(vec)]
        for obj, vec in test_features.items()
    }
    predictions = {}
    for row in test_meta:
        prob = predict_prob(weights, scaled_test[row['object_id']])
        predictions[row['object_id']] = 1 if prob >= threshold else 0
    with Path('submission.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['object_id', 'prediction'])
        for row in test_meta:
            writer.writerow([row['object_id'], predictions[row['object_id']]])
    print('submission.csv written with', sum(predictions.values()), 'positive predictions')


if __name__ == '__main__':
    weights, means, stds, threshold = train_and_evaluate()
    generate_submission(weights, means, stds, threshold)
