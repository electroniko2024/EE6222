"""Assignment 1: reproduce all experiments with python experiment.py."""
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

SEED = 6222
OUT = Path(__file__).resolve().parent / "results"
DIMS = {"Raw": [64], "PCA": [1, 2, 3, 5, 9, 12, 16, 24,
                                32, 40, 48, 64],
        "LDA": list(range(1, 10))}
PARAMS = {"kNN": [1, 3, 5, 7, 9],
          "Ridge": [0.01, 0.1, 1.0, 10.0, 100.0]}


def classifier(name, param):
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=int(param),
            weights="uniform", metric="euclidean", algorithm="brute")
    return RidgeClassifier(alpha=param, solver="svd", fit_intercept=True)


def features(name, Xfit, yfit, Xother):
    # Fit the maximum basis once; every smaller basis is a prefix.
    if name == "Raw":
        return Xfit, Xother, None
    transform = (PCA(n_components=64, svd_solver="full", whiten=False)
        if name == "PCA" else
        LinearDiscriminantAnalysis(solver="svd", n_components=9))
    transform.fit(Xfit, yfit)
    return transform.transform(Xfit), transform.transform(Xother), transform


def write_csv(filename, rows):
    with (OUT / filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(correct, total):
    # Approximate marginal 95% interval, conditional on this fitted model.
    z, p = 1.959963984540054, correct / total
    denom = 1 + z*z/total
    centre = (p + z*z/(2*total)) / denom
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denom
    return [float(centre-half), float(centre+half)]


def main():
    OUT.mkdir(exist_ok=True)
    data = load_digits()
    X, y = data.data.astype(float) / 16.0, data.target
    ids = np.arange(len(y))
    development, test = train_test_split(ids, test_size=0.2,
        stratify=y, random_state=SEED)
    train, validation = train_test_split(development, test_size=0.25,
        stratify=y[development], random_state=SEED+1)
    assert not (set(train) & set(validation) or set(train) & set(test)
                or set(validation) & set(test))
    assert set(train) | set(validation) | set(test) == set(ids)
    np.savez_compressed(OUT / "split_indices.npz", train=train,
        validation=validation, development=development, test=test)
    np.savez_compressed(OUT / "digits_data.npz", X=data.data, y=y)

    # Selection phase: no test features or test labels enter fitting/scoring.
    validation_rows, chosen = [], {}
    for representation, dimensions in DIMS.items():
        A, B, _ = features(representation, X[train], y[train], X[validation])
        for name, parameters in PARAMS.items():
            candidates = []
            for d in dimensions:
                for order, param in enumerate(parameters):
                    model = classifier(name, param).fit(A[:, :d], y[train])
                    score = float(accuracy_score(y[validation],
                                                model.predict(B[:, :d])))
                    row = dict(representation=representation, classifier=name,
                               dimension=d, parameter=param, accuracy=score)
                    validation_rows.append(row)
                    candidates.append((score, d, order, param))
            best = min(candidates, key=lambda v: (-v[0], v[1], v[2]))
            chosen[(representation, name)] = dict(dimension=best[1],
                parameter=best[3], validation_accuracy=best[0])
    write_csv("validation_grid.csv", validation_rows)
    # Persist every selection before inspecting test outcomes.
    selections = [{"representation": rep, "classifier": clf, **value}
                  for (rep, clf), value in chosen.items()]
    (OUT / "selections.json").write_text(json.dumps(selections, indent=2))

    # Evaluation phase: refit on the entire non-test training pool.
    curves, summary, predictions, cached = [], [], {}, {}
    pca_variance = None
    for representation, dimensions in DIMS.items():
        A, B, transform = features(representation, X[development],
                                   y[development], X[test])
        cached[representation] = (A, B)
        if representation == "PCA":
            pca_variance = np.cumsum(transform.explained_variance_ratio_)
        for name in PARAMS:
            selection = chosen[(representation, name)]
            for d in dimensions:
                model = classifier(name, selection["parameter"])
                model.fit(A[:, :d], y[development])
                pred = model.predict(B[:, :d])
                correct = int(np.sum(pred == y[test]))
                score = float(accuracy_score(y[test], pred))
                curves.append(dict(representation=representation,
                    classifier=name, dimension=d,
                    parameter=selection["parameter"], accuracy=score,
                    error_rate=1-score, correct=correct, total=len(test)))
                if d == selection["dimension"]:
                    key = representation + "-" + name
                    predictions[key] = pred
                    summary.append(dict(representation=representation,
                        classifier=name, **selection, test_accuracy=score,
                        correct=correct, errors=len(test)-correct,
                        compression=1-d/64,
                        ci95=wilson(correct, len(test))))
                    np.savetxt(OUT / (key + "_confusion.csv"),
                        confusion_matrix(y[test], pred), fmt="%d", delimiter=",")
    write_csv("test_curves.csv", curves)
    np.savez_compressed(OUT / "test_predictions.npz", y_true=y[test],
                        test_indices=test, **predictions)

    # Raw and full, unwhitened PCA must agree at the SAME hyperparameter.
    checks = {}
    for name in PARAMS:
        param = chosen[("Raw", name)]["parameter"]
        raw_a, raw_b = cached["Raw"]
        pca_a, pca_b = cached["PCA"]
        raw_pred = classifier(name, param).fit(raw_a, y[development]).predict(raw_b)
        pca_pred = classifier(name, param).fit(pca_a, y[development]).predict(pca_b)
        agreements = int(np.sum(raw_pred == pca_pred))
        assert agreements == len(test), (name, agreements)
        checks[name] = dict(parameter=param, agreements=agreements,
                            total=len(test))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = {"kNN": "#1466a5", "Ridge": "#c65e22"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for ax, representation in zip(axes, ["PCA", "LDA"]):
        for name in PARAMS:
            points = [r for r in curves if r["representation"] == representation
                      and r["classifier"] == name]
            ax.plot([r["dimension"] for r in points],
                [100*r["accuracy"] for r in points], "o-",
                color=colors[name], markersize=4, label=name)
            raw = next(r for r in summary if r["representation"] == "Raw"
                       and r["classifier"] == name)
            ax.axhline(100*raw["test_accuracy"], color=colors[name],
                       linestyle="--", alpha=0.65, label="Raw " + name)
            selected = next(r for r in summary
                if r["representation"] == representation and r["classifier"] == name)
            ax.scatter(selected["dimension"], 100*selected["test_accuracy"],
                s=120, facecolors="none", edgecolors=colors[name], linewidths=2)
        ax.set(title=representation, xlabel="Retained dimension d",
               ylabel="Test accuracy (%)", ylim=(0, 102))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_curves.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.plot(np.arange(1, 65), pca_variance*100, color="#1466a5")
    for d in [9, 16, 24, 32]:
        ax.scatter(d, pca_variance[d-1]*100, color="#1466a5")
        ax.annotate(f"{d}: {pca_variance[d-1]*100:.1f}%",
            (d, pca_variance[d-1]*100), xytext=(5, -16),
            textcoords="offset points", fontsize=8)
    ax.set(xlabel="PCA dimension d", ylabel="Cumulative variance (%)",
           ylim=(0, 104), xlim=(1, 64))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "pca_variance.png", dpi=220)
    plt.close(fig)

    pairs = []
    for row in summary:
        if row["representation"] == "Raw":
            continue
        base = predictions["Raw-" + row["classifier"]] == y[test]
        reduced = predictions[row["representation"]+"-"+row["classifier"]] == y[test]
        pairs.append(dict(representation=row["representation"],
            classifier=row["classifier"],
            reduced_only_correct=int(np.sum(reduced & ~base)),
            raw_only_correct=int(np.sum(base & ~reduced))))
    versions = dict(python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, sklearn=sklearn.__version__,
        matplotlib=matplotlib.__version__)
    results = dict(seed=SEED, split_sizes=dict(train=len(train),
        validation=len(validation), development=len(development), test=len(test)),
        class_counts={key:np.bincount(y[value], minlength=10).tolist()
            for key, value in [("train",train),("validation",validation),("test",test)]},
        dataset_sha256=hashlib.sha256(data.data.tobytes()+y.tobytes()).hexdigest(),
        versions=versions, summary=summary, paired_counts=pairs,
        pca_cumulative_variance=pca_variance.tolist(), equivalence_checks=checks)
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(dict(split=results["split_sizes"], summary=summary), indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
