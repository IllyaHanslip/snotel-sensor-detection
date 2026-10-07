"""
Denoising model: ARIMA-residual noise detector.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from stations_config import get_primary_station

# cur/sta = current/station, proc = processed, fig = figures, p = path
cur_sta = get_primary_station()

proc_dir = Path("data/processed") / cur_sta["triplet"].replace(":", "_")
fig_dir = Path("figures") / cur_sta["triplet"].replace(":", "_")
data_p = proc_dir / "snotel_preprocessed.csv"
lab_p = proc_dir / "manual_labels.csv"

filter_var = "SNWD"  # var = variable (the one being filtered)
# osr = order search range
osr_p = (0, 1, 2, 3, 4, 5)
osr_d = (0, 1, 2)
osr_q = (0, 1, 2, 3, 4, 5)
max_comb_ord = 5   # max combination order
score_win = 7   # score window
out_thr = 3.0   # outlier threshold


def load_full_series():
    """Load the preprocessed SNWD series for the primary station."""
    df = pd.read_csv(data_p, parse_dates = ["date"], index_col = "date")
    return df[filter_var]


def load_labels():
    """Load the manual good/noise labels for the primary station."""
    labels = pd.read_csv(lab_p, parse_dates = ["date"])
    labels["is_noise"] = labels["label"] == "noise"
    return labels


def select_order_by_aic(series, p_range = osr_p, d_range = osr_d,
                         q_range = osr_q, max_comb_ord = max_comb_ord):
    """Grid-search ARIMA orders (p = AR order, d = differencing order,
    q = MA order) by AIC, capped at p+q <= max_comb_ord. BIC is also
    computed per candidate and compared to the AIC pick as a
    diagnostic only; AIC is what selects the order actually used."""
    series_filled = series.interpolate(limit_direction = "both")

    print(f"\nSearching {len(p_range)} x {len(d_range)} x {len(q_range)} "
          f"candidate ARIMA orders by AIC "
          f"(ranking restricted to p+q <= {max_comb_ord})...")
    print(f"{'order':>12} {'aic':>12} {'bic':>12}")

    rows = []
    for d in d_range:
        for p in p_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue
                try:
                    # m = model, fit = fitted
                    m_fit = ARIMA(series_filled, order = (p, d, q)).fit()
                    rows.append({
                        "p": p, "d": d, "q": q,
                        "aic": m_fit.aic, "bic": m_fit.bic,
                    })
                    print(f"{f'({p},{d},{q})':>12} {m_fit.aic:>12.1f} "
                          f"{m_fit.bic:>12.1f}")
                except Exception as err:
                    print(f"{f'({p},{d},{q})':>12} {'skipped':>12} "
                          f"({err})")

    results = pd.DataFrame(rows).sort_values("aic").reset_index(drop = True)
    within_cap = results[results["p"] + results["q"] <= max_comb_ord]

    best = within_cap.iloc[0]
    best_ord = (int(best["p"]), int(best["d"]), int(best["q"]))
    uncap_best = results.iloc[0]
    uncap_best_ord = (
        int(uncap_best["p"]), int(uncap_best["d"]), int(uncap_best["q"])
    )

    print(f"\nBest order by AIC among p+q <= {max_comb_ord}: {best_ord} "
          f"(AIC = {best['aic']:.1f})")
    if uncap_best_ord != best_ord:
        print(f"(Without the order cap, {uncap_best_ord} would score "
              f"better, AIC = {uncap_best['aic']:.1f}, but sits right "
              f"at the edge of the search grid with no sign of AIC "
              f"levelling off - the order cap above is applied "
              f"specifically to avoid choosing that kind of result.)")

    bic_best = within_cap.sort_values("bic").iloc[0]
    bic_best_ord = (
        int(bic_best["p"]), int(bic_best["d"]), int(bic_best["q"])
    )
    if bic_best_ord != best_ord:
        print(f"(Best order by BIC (within the same cap) would instead "
              f"be {bic_best_ord}, BIC = {bic_best['bic']:.1f} - BIC's "
              f"heavier complexity penalty prefers a simpler model "
              f"here; AIC is used as the selection criterion.)")

    return best_ord, results


def fit_arima_and_predict(series, order):
    """Fit an ARIMA model of the given order and return its in-sample
    predictions."""
    series_filled = series.interpolate(limit_direction = "both")

    print(f"Fitting ARIMA{order} to {len(series_filled)} days of data...")
    m = ARIMA(series_filled, order = order)
    m_fit = m.fit()
    print(m_fit.summary().tables[0])

    pred = m_fit.get_prediction().predicted_mean
    pred.index = series.index
    return pred


def compute_deviation_scores(series, pred, win_sz = score_win):
    """Turn actual-vs-predicted residuals into a local-deviation
    suspicion score. min_periods = 3: see baseline_filter.py."""
    resi = (series - pred).abs()  # resi = residual

    resi_ord = resi.rolling(   # ord = ordinary (unscaled)
        window = win_sz, center = True, min_periods = 3
    ).median()
    resi_ord_scaled = resi_ord * 1.4826

    dev_score = np.where(  # dev = deviation
        resi_ord_scaled > 0,
        resi / resi_ord_scaled,
        np.where(resi > 0, np.inf, 0.0),
    )
    return pd.Series(dev_score, index = series.index), resi


def compute_metrics(actual_is_noise, pred_is_noise):
    """Compute accuracy, precision, recall, and F1 from predicted vs
    actual noise flags."""
    # tp/fp/fn/tn = true/false positive/negative
    tp = (actual_is_noise & pred_is_noise).sum()
    fp = (~actual_is_noise & pred_is_noise).sum()
    fn = (actual_is_noise & ~pred_is_noise).sum()
    tn = (~actual_is_noise & ~pred_is_noise).sum()

    total = tp + fp + fn + tn
    acc = (tp + tn) / total  # acc = accuracy
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # prec = precision
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

    return {
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "accuracy": acc, "precision": prec, "recall": recall, "f1": f1,
    }


def print_metrics_report(metrics, n_lab):  # n_lab = number labeled days
    """Print a confusion matrix and headline metrics to the console."""
    print("\nConfusion matrix:")
    print("                    predicted good   predicted noise")
    print(f"  actual good            {metrics['true_negative']:>6}"
          f"           {metrics['false_positive']:>6}")
    print(f"  actual noise           {metrics['false_negative']:>6}"
          f"           {metrics['true_positive']:>6}")

    print(f"\nAccuracy:  {metrics['accuracy']:.2f}  (fraction of all "
          f"{n_lab} labeled days scored correctly)")
    print(f"Precision: {metrics['precision']:.2f}  (of the days the "
          f"model called 'noise', how many really were)")
    print(f"Recall:    {metrics['recall']:.2f}  (of the days that "
          f"really were 'noise', how many the model caught)")
    print(f"F1 score:  {metrics['f1']:.2f}  (a single balance of "
          f"precision and recall)")


def plot_results(series, pred, labels, pred_is_noise, out_dir = fig_dir):
    """Plot the raw series against ARIMA's predictions, marking caught,
    missed, and wrongly-flagged days."""
    merged = labels.copy()
    merged["predicted"] = merged["date"].map(pred)
    merged["predicted_noise"] = merged["date"].map(pred_is_noise).fillna(False)

    caught_ok = merged[merged["is_noise"] & merged["predicted_noise"]]
    miss_noise = merged[merged["is_noise"] & ~merged["predicted_noise"]]
    flag_bad = merged[~merged["is_noise"] & merged["predicted_noise"]]

    plt.figure(figsize = (14, 5))
    plt.plot(series.index, series.values, linewidth = 0.6,
             color = "steelblue", label = "raw SNWD", zorder = 1)
    plt.plot(pred.index, pred.values, linewidth = 0.6, color = "gray",
             alpha = 0.6, label = "ARIMA prediction", zorder = 1)
    plt.scatter(caught_ok["date"], caught_ok["value"], color = "green",
                s = 25,
                label = f"noise, correctly caught ({len(caught_ok)})",
                zorder = 3)
    plt.scatter(miss_noise["date"], miss_noise["value"], color = "orange",
                s = 25, label = f"noise, missed ({len(miss_noise)})",
                zorder = 3)
    plt.scatter(flag_bad["date"], flag_bad["value"], color = "red", s = 25,
                label = f"good day, wrongly flagged ({len(flag_bad)})",
                zorder = 3)

    plt.title(f"{cur_sta['name']}: ARIMA-residual model vs manual labels")
    plt.xlabel("Date")
    plt.ylabel("SNWD (in)")
    plt.legend(fontsize = 8)
    plt.tight_layout()

    out_path = out_dir / "arima_model_evaluation.png"
    plt.savefig(out_path, dpi = 150)
    plt.close()
    print(f"\nSaved evaluation plot to {out_path}")


def main():
    """Run the full ARIMA pipeline: fit, score, evaluate, and plot."""
    series = load_full_series()
    labels = load_labels()

    arima_ord, osr_results = select_order_by_aic(series)
    osr_csv = proc_dir / "arima_order_search_results.csv"
    osr_results.to_csv(osr_csv, index = False)
    print(f"Saved full order search results to {osr_csv}")

    pred = fit_arima_and_predict(series, order = arima_ord)
    dev_scores, resi = compute_deviation_scores(series, pred)
    pred_is_noise = dev_scores > out_thr

    result = pd.DataFrame({
        "predicted_value": pred,
        "residual": resi,
        "deviation_score": dev_scores,
        "predicted_noise": pred_is_noise,
    })
    result.index.name = "date"

    merged = labels.merge(result, on = "date", how = "left")
    metrics = compute_metrics(merged["is_noise"], merged["predicted_noise"])

    n_noise = labels["is_noise"].sum()
    n_good = (~labels["is_noise"]).sum()
    print(f"\nEvaluated on {len(labels)} manually labeled days "
          f"({n_noise} noise, {n_good} good)")
    print_metrics_report(metrics, len(labels))

    out_csv = proc_dir / "arima_model_results.csv"
    merged.to_csv(out_csv, index = False)
    print(f"\nSaved full per-day results to {out_csv}")

    plot_results(series, pred, labels, pred_is_noise)


if __name__ == "__main__":
    main()
