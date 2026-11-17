"""Source-backed experiment readout, with successes and rejected components."""
import json
import pandas as pd
import numpy as np
from .v3_plan import ROOT,OUT,BASE


def report():
    r=json.loads((OUT/'run.json').read_text());v=json.loads((OUT/'verification.json').read_text())
    assert r['phase']=='evaluated' and v['status']=='verified'
    a=r['fresh_test']['reference'];b=r['fresh_test']['ranker'];c=r['fresh_test']['v3']
    lo,hi=r['ranking_difference']['95_percent_date_bootstrap_interval']
    metadata=json.loads((ROOT/'artifacts/gocap-40m-recommended.json').read_text())
    lines=['# Five 40m model improvements — experiment 3','',
        f'**The direct hourly ranker improves the reserved-date comparison:** median Spearman '
        f'correlation rises from **{a["median_spearman"]:.3f} to {b["median_spearman"]:.3f}**. '
        f'The paired difference is **{r["ranking_difference"]["difference"]:+.3f}**, with a 95% date-bootstrap '
        f'interval **[{lo:+.3f}, {hi:+.3f}]**. Mean loss from the strongest observed hour falls from '
        f'**{a["mean_best_hour_shortfall_db"]:.2f} to {b["mean_best_hour_shortfall_db"]:.2f} dB**.','',
        'The recommended package combines this new ranker with the established SNR model and intervals. '
        'The WSPR SNR challengers and conditional intervals were tested and were not promoted. '
        'The user confirmed archive-only operation; physical 100 W/dipole calibration remains unverified.','',
        '| Requested improvement | Implemented experiment | Outcome |','|---|---|---|',
        '| Direct hourly learning | Pairwise XGBoost ranker; only positive observed hours compete | Promoted for WSPR hourly ranking |',
        '| Coherent training sequences | Outcome-blind station-pair/day sampling across 96 planned historical dates | 203,931 hourly aggregates on 93 positive dates; 18,207 eligible rank-training groups |',
        '| WSPR-focused versus pooled learning | WSPR-only, pooled, and shared model with cross-fitted source residual heads | Existing WSPR SNR model won validation; source heads remain experimental |',
        '| Richer geometry and solar history | Nine-point path illumination; bearing; sunrise/sunset timing; IGRF degree-one magnetic latitude; completed 6/24/72h ap and previous 27-day SFI | Included in selected ranker; matched feature ablation below |',
        '| Better uncertainty and reference calibration | 10th/90th conditional quantiles plus per-source/mode split calibration; documented measurement intake | Conditional intervals rejected on final interval score; no documented physical observations supplied |','',
        '## Data, fitting and final boundary','',
        f'The coherent training dataset covers 93 positive-reception dates from 2010, 2013, 2016, 2019, '
        f'2022 and 2024. The 96-date plan was fixed before queries. Empty responses remain unknown. '
        'The matched-calendar scattered control has 564,335 hourly rows on 94 positive dates, '
        'so it is a comparison of sampling strategies on the same planned calendar, not perfectly '
        'matched actual day support or row counts. Pooled challengers add 117,714 historical FT8/FT4/CW '
        'training rows.','',
        f'Ranking training uses **{r["rank_training_hours"]:,} observed hours** in '
        f'**{r["rank_training_groups"]:,} fixed-station-pair/day groups**, with at least four reported hours. '
        'Reported WSPR power is normalized to a common 1 W reference before forming strength-order labels. '
        'Features contain neither station identifiers, report counts, measured SNR nor ranking labels. '
        'Pairwise loss compares received strengths; absent hours never receive labels.','',
        'Model and checkpoint decisions use the original 2024 validation dates. Interval calibration '
        'uses separate dates. The final dates are the third day of each month in 2025, disjoint from '
        'every original split and the previously used tuning dates. All component choices and strict '
        'exclusion fits were frozen before these final scores.','',
        f'The new final SNR sample contains **{a["rows"]:,} positive hourly aggregates** on twelve dates. '
        f'The denser independent pair/day cohort yields **{b["ranking_eligible_cases"]:,} eligible fixed-path/day '
        f'curves** with at least six observed hours and three distinct strengths. '
        f'The paired correlation comparison uses {r["ranking_difference"]["paired_cases"]:,} curves; '
        'the reference has one constant prediction curve whose correlation is undefined. '
        'Best-hour loss retains all eligible curves, breaking ties in favor of the earliest observed hour.','',
        '## Reserved-date outcomes','',
        '| Metric | Established reference | Selected v3 component |','|---|---:|---:|',
        f'| WSPR SNR MAE | {a["mae_db"]:.2f} dB | {c["mae_db"]:.2f} dB (reference retained) |',
        f'| Median observed-hour rank correlation | {a["median_spearman"]:.3f} | {b["median_spearman"]:.3f} |',
        f'| Mean loss from best observed hour | {a["mean_best_hour_shortfall_db"]:.2f} dB | {b["mean_best_hour_shortfall_db"]:.2f} dB |','',
        'Bootstrap uncertainty resamples all cases from each UTC date together, using twelve date '
        'clusters and 5,000 replicates. This supports the observed ranking gain within this sampled '
        'reception population. Twelve dates in one year do not establish universal performance, '
        'open-path probabilities, or accuracy for unseen solar combinations.','',
        '![Reserved-date performance](results/ml40/v3/comparison.png)','',
        '## Model and feature comparisons','',
        '| WSPR SNR challenger | Validation balanced MAE | New-date MAE | New-date median rank correlation |',
        '|---|---:|---:|---:|']
    for name,s in r['fresh_ablations'].items():
        val=r['validation'][name]['by_mode']['WSPR']['balanced_mae_db']
        lines.append(f'| {name} | {val:.2f} dB | {s["mae_db"]:.2f} dB | {s["median_spearman"]:.3f} |')
    lines+=['', 'These normalized SNR challengers did not beat the established WSPR point model. '
        'Coherent sequences help provide relative-hour comparisons, but did not improve absolute SNR '
        'in these regressors. Unknown antenna/noise effects, sampling differences and station populations '
        'remain possible explanations rather than established causes.','',
        'The geometry/history ablation below holds the ranker architecture at 240 trees and depth 3 '
        'and changes only the feature set. These are diagnostics; the separately selected final ranker '
        'uses 120 trees and depth 5.','',
        '| Matched ranker | Validation correlation | New-date correlation | New-date best-hour loss |',
        '|---|---:|---:|---:|']
    for name,s in r['fresh_rank_feature_ablations'].items():
        val=next(x for x in r['ranking_trials'] if x['name']==name)
        lines.append(f'| {name} | {val["median_spearman"]:.3f} | {s["median_spearman"]:.3f} | {s["best_hour_shortfall_db"]:.2f} dB |')
    lines+=['','## Strict excluded-station and geographic tests','',
        'Designated callsigns or two-character Maidenhead fields are excluded in both endpoint roles '
        'before fitting the stress models. The comparison uses the previously verified reference '
        'exclusion model and a refitted v3 ranker; their training samples differ, so this tests '
        'the complete model procedures rather than isolating a single feature.','',
        '| Exclusion | Fresh ranking cases | Reference correlation | V3 correlation |','|---|---:|---:|---:|']
    for name,s in r['strict_fresh_stress'].items():lines.append(f'| {name} | {s["eligible_rank_cases"]:,} | {s["reference_rank"]:.3f} | {s["v3_rank"]:.3f} |')
    lines+=['','The geographic result has fewer cases and should be treated cautiously. These stress '
        'checks support further use of the ranker; they do not identify physical antenna gains.','',
        '## Conditional uncertainty experiment','',
        'The challenger predicts condition-dependent 10th and 90th SNR quantiles, sorts crossed bounds, '
        'includes the selected point estimate within the bounds, then applies a nonnegative per-source/mode '
        'conformal expansion determined from separate calibration dates.','',
        '| Interval measure | Existing bands | Conditional challenger |','|---|---:|---:|',
        f'| Observed coverage (nominal 80%) | {a["coverage"]:.1%} | {c["coverage"]:.1%} |',
        f'| Mean total width | {a["mean_width_db"]:.2f} dB | {c["mean_width_db"]:.2f} dB |',
        f'| Mean 80% interval score, lower is better | {a["mean_interval_score"]:.2f} | {c["mean_interval_score"]:.2f} |','',
        'The interval score penalizes width and observations outside the bounds. Similar coverage '
        'with wider, worse-scoring intervals does not justify replacing the existing bands. '
        '[Coverage by distance, illumination and geomagnetic activity](results/ml40/v3/interval_segments.csv) '
        'is retained for diagnosis. These intervals describe detected receptions, not unseen sub-threshold signals.','',
        '## Other modes and physical calibration','',
        'The FT8/FT4/CW comparisons below reuse previously published test data and are regression checks, '
        'not independent new-mode validation. The CW source-head choice was selected on validation, '
        'but is not included in the recommended package without stronger independent evidence.','',
        '| Mode | Test dates | Reference MAE | Experimental profile MAE |','|---|---:|---:|---:|']
    for mode,s in r['published_mode_regressions'].items():lines.append(f'| {mode} | {s["dates"]} | {s["reference"]["mae_db"]:.2f} dB | {s["v3"]["mae_db"]:.2f} dB |')
    lines+=['','A [documented-reference intake schema](results/ml40/v3/reference_measurement_schema.json) '
        'and validator are implemented. Actual positive measurements must include calibrated received '
        'signal power, 2500 Hz noise power, feedpoint transmit power, documented antennas and measurement '
        'provenance. The user chose archive-only work, so '
        '[the physical calibration audit](results/ml40/v3/physical_calibration_audit.json) records zero '
        'reference measurements and makes no physical calibration claim.','',
        '## Query the recommended package','',
        f'The [recommended artifact](artifacts/gocap-40m-recommended.joblib) is '
        f'**{metadata["bytes"]/1024/1024:.2f} MiB** and loads without reception archives. '
        'It retains established SNR/interval outputs and adds the verified WSPR ranker.','',
        '```sh','.venv/bin/python -m ml40.v3_inference \\',
        '  artifacts/gocap-40m-recommended.joblib \\',
        '  JN76 FN31 2026-11-17T00:00Z \\',
        '  --sfi 150 --kp 3 --ap 10 --power-watts 100 --hours 24','```','',
        'Higher `relative_hour_rank_score` means a preferred hour within the same path/day. It is not '
        'a dB value or contact probability and is not comparable between different paths. The ranker '
        'was evaluated only among reported hours; a 24-hour scenario query is not proof that every '
        'unreported hour would support reception. FT8/FT4/CW queries return no dedicated rank score.','',
        'With no history file, the scenario assumes steady Kp over the completed preceding 72 hours '
        'and steady SFI over the preceding 27 completed days. `--ap` is independent previous-day Ap. '
        'A one-hour query can instead accept `--history-file` containing `kp_3h` (24 oldest-first values), '
        '`ending_boundary_utc` (the last completed three-hour boundary) and optional '
        '`sfi_previous27_days` (27 oldest-first values). Last values must match `--kp` and `--sfi`. '
        'Future or misaligned history is rejected.','',
        'Magnetic coordinates use degree-one IGRF-14 centered-dipole coefficients, not a full '
        'field-line/auroral coordinate calculation. Epochs outside 1900–2030, antipodal shortest-path '
        'ambiguity and marginal training-range violations are surfaced. Arbitrary date/solar queries '
        'remain conditional scenarios, not solar forecasts.','',
        '## Reproduce and inspect','',
        '```sh','.venv/bin/pip install -r requirements-v3-lock.txt',
        '.venv/bin/python -m ml40.v3_collect','.venv/bin/python -m ml40.v3_train',
        '.venv/bin/python -m ml40.v3_stress','.venv/bin/python -m ml40.v3_calibration',
        '.venv/bin/python -m ml40.v3_evaluate','.venv/bin/python -m ml40.v3_verify',
        '.venv/bin/python -m ml40.v3_deploy','.venv/bin/python -m ml40.v3_scenario_check',
        '.venv/bin/python -m ml40.v3_report','```','',
        '- [Frozen experiment plan](results/ml40/v3/plan.json)',
        '- [Validation, feature comparisons and final metrics](results/ml40/v3/run.json)',
        '- [Paired actual-path ranking results](results/ml40/v3/paired_hour_rankings.csv)',
        '- [Independent verification](results/ml40/v3/verification.json)',
        '- [Recommended-package verification](results/ml40/v3/deployment_verification.json)',
        '- [Recommended model metadata](artifacts/gocap-40m-recommended.json)',
        '- [Experimental model metadata](artifacts/gocap-40m-v3.json)','',
        'Reception archives and solar-index provenance remain as documented in [README.md](README.md). '
        'Methods: [XGBoost pairwise ranking](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html), '
        '[scikit-learn quantile regression](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html), '
        '[NOAA/IAGA IGRF-14](https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field).','']
    diagnostic=OUT/'scenario_history_diagnostic.json'
    if diagnostic.exists():
        s=json.loads(diagnostic.read_text())['results']['steady_history_assumption']
        lines+=['## Steady-history query diagnostic','',
            f'With each hour\'s historical Kp/SFI/Ap retained but the preceding history assumed steady, '
            f'the new ranker has median correlation **{s["new_rank_correlation"]:.3f}**, versus '
            f'**{s["reference_rank_correlation"]:.3f}** for the reference under the same assumption. '
            f'Its mean best-observed-hour loss is **{s["new_best_hour_loss_db"]:.2f} dB**. '
            'This post-freeze diagnostic does not select a model; it illustrates the cost of replacing '
            'actual history with scenario assumptions. It still supplies the actual current-hour indices, '
            'so it does not measure solar-forecast accuracy. '
            '[Inspect the diagnostic](results/ml40/v3/scenario_history_diagnostic.json).','']
    (ROOT/'V3_RESULTS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');colors=['#667085','#00796B']
    values=[[a['median_spearman'],b['median_spearman']],
        [a['mean_best_hour_shortfall_db'],b['mean_best_hour_shortfall_db']],
        [a['mean_interval_score'],c['mean_interval_score']]]
    for ax,vals,title,ylabel,labels in zip(axes,values,
        ['Hourly ranking improves','Best observed hour selection','Conditional intervals rejected'],
        ['Median Spearman (higher is better)','Mean SNR loss (dB; lower is better)','80% interval score (lower is better)'],
        [['Reference','Ranker'],['Reference','Ranker'],['Reference','Conditional']]):
        bars=ax.bar(labels,vals,color=colors);ax.bar_label(bars,fmt='%.3f' if ax is axes[0] else '%.2f',padding=3)
        ax.set(title=title,ylabel=ylabel,ylim=(0,max(vals)*1.2));ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    print('Wrote V3_RESULTS.md and comparison.png')
if __name__=='__main__':report()
