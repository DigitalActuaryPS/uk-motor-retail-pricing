"""Fit and evaluate the renewal models. Runs before the optimiser, which needs the renewal response."""
from __future__ import annotations

import json

import pandas as pd

from ..core import paths
from ..core.config import load_config
from ..core.io import load_quotes, renewals, time_split, write_json
from ..core.plotting import save_fig
from ..conversion.explain import plot_calibration
from ..conversion.models import NaiveBenchmark, calibration_table, metrics
from .features import RenewalDesign
from .models import RenewalGLM, fit_renewal_glm, fit_renewal_gbm


def main() -> dict:
    cfg = load_config()
    k = cfg["market.n_competitors"]
    ren = renewals(load_quotes())
    train, val, test = time_split(ren, cfg)
    proc = paths.data_processed()
    mdir = proc / "models"
    mdir.mkdir(exist_ok=True)

    naive = NaiveBenchmark(train)
    glm = fit_renewal_glm(train, k)
    gbm, search = fit_renewal_gbm(train, val, k, cfg)
    glm.save(mdir / "renewal_glm.json")
    gbm.save(mdir, "renewal_gbm")
    search.to_csv(proc / "m6_renewal_gbm_search__simulated_data.csv", index=False)
    glm.coef_table().to_csv(proc / "m6_renewal_glm_coefficients__simulated_data.csv", index=False)

    rows, cal = [], {}
    for split, frame in [("validation", val), ("test", test)]:
        preds = {"naive": naive.predict(frame), "glm": glm.predict_df(frame), "gbm": gbm.predict(frame)}
        for n, p in preds.items():
            rows.append({"split": split, "model": n, **metrics(frame["converted"].to_numpy(), p)})
        if split == "test":
            cal = {n: calibration_table(frame["converted"].to_numpy(), p) for n, p in preds.items()}
    met = pd.DataFrame(rows)
    met.to_csv(proc / "m6_renewal_metrics__simulated_data.csv", index=False)
    save_fig(plot_calibration(cal), "m6_renewal_calibration")
    t = met[met["split"] == "test"].set_index("model")
    out = {"train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
           "test_auc": t["roc_auc"].to_dict(), "test_log_loss": t["log_loss"].to_dict(), "test_brier": t["brier"].to_dict(),
           "gbm_backend": gbm.params["backend"]}
    write_json(out, "m6_renewal_model_summary.json")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
