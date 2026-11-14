"""Module 3 runner. Simulated data only."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..core import paths
from ..core.config import load_config
from ..core.io import load_quotes, load_truth, nb_observational, time_split, write_json
from ..core.plotting import save_fig
from .explain import (partial_dependence_ln_ratio, plot_calibration, plot_price_response, price_response,
                      save_shap_beeswarm, shap_importance)
from .features import GLMDesign
from .models import NaiveBenchmark, calibration_table, fit_gbm, fit_glm, metrics


def main() -> dict:
    cfg = load_config()
    k = cfg["market.n_competitors"]
    df = load_quotes()
    truth = load_truth().set_index("quote_id")
    nb = nb_observational(df)
    train, val, test = time_split(nb, cfg)
    proc = paths.data_processed()
    models_dir = proc / "models"
    models_dir.mkdir(exist_ok=True)

    naive = NaiveBenchmark(train)
    glm = fit_glm(train, k)
    t0 = time.time()
    gbm, search = fit_gbm(train, val, k, cfg)
    fit_secs = time.time() - t0

    gd_val, gd_test = GLMDesign(val, k), GLMDesign(test, k)
    preds = {
        "validation": {"naive": naive.predict(val), "glm": glm.predict(gd_val), "gbm": gbm.predict(val)},
        "test": {"naive": naive.predict(test), "glm": glm.predict(gd_test), "gbm": gbm.predict(test)},
    }
    rows = []
    for split, frame in [("validation", val), ("test", test)]:
        for name, p in preds[split].items():
            rows.append({"split": split, "model": name, **metrics(frame["converted"].to_numpy(), p)})
    met = pd.DataFrame(rows)
    met.to_csv(proc / "m3_metrics__simulated_data.csv", index=False)
    search.to_csv(proc / "m3_gbm_search__simulated_data.csv", index=False)
    glm.coef_table().to_csv(proc / "m3_glm_coefficients__simulated_data.csv", index=False)
    glm.save(models_dir / "conversion_glm.json")
    gbm.save(models_dir)

    cal = {n: calibration_table(test["converted"].to_numpy(), p) for n, p in preds["test"].items()}
    pd.concat([t.assign(model=n) for n, t in cal.items()]).to_csv(proc / "m3_calibration__simulated_data.csv", index=False)
    save_fig(plot_calibration(cal), "m3_calibration")

    imp, sv, X = shap_importance(gbm, test, k)
    imp.to_csv(proc / "m3_shap_importance__simulated_data.csv", index=False)
    save_shap_beeswarm(sv, X, paths.figures() / "m3_shap_beeswarm__simulated_data.png")

    u = truth.loc[test["quote_id"], "latent_intent"].to_numpy()
    pr = price_response(test, k, glm, gbm, u, cfg)
    pr.to_csv(proc / "m3_price_response__simulated_data.csv", index=False)
    save_fig(plot_price_response(pr), "m3_price_response")
    pdp = partial_dependence_ln_ratio(test, k, glm, gbm)
    pdp.to_csv(proc / "m3_partial_dependence__simulated_data.csv", index=False)

    t = met[met["split"] == "test"].set_index("model")
    out = {
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "gbm_backend": gbm.params["backend"], "gbm_best_params": gbm.params, "gbm_search_seconds": fit_secs,
        "test_auc": t["roc_auc"].to_dict(), "test_log_loss": t["log_loss"].to_dict(), "test_brier": t["brier"].to_dict(),
        "top_shap_features": imp.head(5)["feature"].tolist(),
    }
    write_json(out, "m3_summary.json")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
