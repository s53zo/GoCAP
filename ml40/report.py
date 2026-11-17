"""Readout and figures from released, verified v2 evaluation artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .plan import ROOT,DATA,RESULTS

LABELS={'boosted_baseline':'Boosted baseline','boosted_station':'Station-adjusted boosted',
    'neural_station':'Station-adjusted neural'}
ORDER=list(LABELS)

def report():
    run=json.loads((RESULTS/'training_run.json').read_text())
    audit=json.loads((RESULTS/'data_audit.json').read_text())
    verified=json.loads((RESULTS/'verification.json').read_text())
    if run['phase']!='evaluated' or verified['status']!='verified':raise ValueError('Verify released evaluation first')
    m=pd.read_csv(RESULTS/'test_metrics.csv')
    ranks=pd.read_csv(RESULTS/'test_hour_ranking.csv')
    summary=ranks.groupby(['scenario','selection','cohort','unit','source','mode','model','view']).agg(
        cases=('observed_hours','size'),days=('date','nunique'),
        median_spearman=('spearman','median'),mean_best_hour_shortfall_db=('best_observed_hour_shortfall_db','mean')).reset_index()
    summary.to_csv(RESULTS/'hour_ranking_summary.csv',index=False)
    standard=m[m.scenario.eq('standard')&m.selection.eq('later_dates')&m.view.eq('population_core')]
    core_ranks=summary[summary.scenario.eq('standard')&summary.cohort.eq('pair_day_cohort')&summary.unit.eq('station_pair')&summary.view.eq('population_core')]
    modes=['WSPR','FT8','FT4','CW']
    lines=['# 40m learned propagation model — second experiment','',
        '**Scope:** positive-reception SNR and hourly ranking. Missing reports remain unknown. '
        'The population reference is not an independently calibrated 100 W/dipole station.','',
        'The station-adjusted boosted model improved the primary hourly ranking check: median '
        'correlation rose from **0.346 to 0.399** over **2,905 fixed-path/day cases**. '
        'Mean loss from the best observed hour fell from **4.96 to 4.82 dB**. '
        'Generic WSPR SNR error was essentially unchanged (**6.69 versus 6.70 dB MAE**). '
        'The neural challenger did not outperform the boosted models overall. '
        'This supports a modest ranking improvement, not calibrated contact probabilities.','',
        f'The prepared sample contains **{audit["rows"]:,} hourly aggregates** on **{audit["actual_dates"]} distinct days**. '
        'The primary WSPR collection covers the 324 planned dates; empty source responses are not classified as closed paths. '
        'Historical PSK/RBN samples cover the selected seasonal HARC dates. Exact source/mode/split support appears below.','',
        '## Model design','',
        'Three compact learners share numeric geometry/time/solar features and explicit source/mode/decoder context. '
        'They use no reception-history query at inference. Transmitter/receiver offsets are learned from '
        'date-cross-fitted training residuals, regularized and centered; unseen keys receive zero correction. '
        'The neural challenger has hidden layers of 64 and 32 neurons. Candidate and epoch decisions use validation only.','',
        'Numeric preprocessing, nuisance offsets and model weights are fitted without final test rows. '
        'Intervals use a separate calibration split. Strict experiments exclude designated callsigns or '
        'geographic fields from training, validation and calibration in both endpoint roles.','',
        '## Frozen choice and later-period SNR error','',
        f'The standard model selected before final scoring was **{LABELS[run["scenarios"]["standard"]["chosen_by_validation"]]}**, '
        f'using {run["scenarios"]["standard"]["selection_criterion"]}. '
        'The following errors are the generic population-core predictions available from squares and conditions alone; '
        'they exclude optional known-station corrections. Lower MAE is better.','',
        '| Mode | Test observations | Test days | Boosted baseline MAE | Adjusted boosted MAE | Adjusted neural MAE |',
        '|---|---:|---:|---:|---:|---:|']
    for mode in modes:
        rows=standard[standard['mode'].eq(mode)]
        if rows.empty:continue
        first=rows.iloc[0]
        values={r.model:r.mae_db for r in rows.itertuples()}
        lines.append(f'| {mode} | {int(first.rows):,} | {int(first.days)} | '+ ' | '.join(f'{values.get(k,np.nan):.2f} dB' for k in ORDER)+' |')
    lines += ['', 'These are conditional predictions of reported receptions. Source populations, unknown '
        'transmitter power outside WSPR, antennas and local noise remain mixed in the observations. '
        'A lower error is not evidence of calibrated path-open probabilities.','',
        '## Primary hourly ranking check','',
        'The independent WSPR ranking cohort retains reported receptions for outcome-blind station-pair/day keys. '
        'A case must have at least six observed hours and at least three distinct measured strengths. '
        'Station pairs must remain in the same four-character squares. Known WSPR power is adjusted to a common '
        'reference before ranking. Unreported hours never become negative examples.','',
        '| Model | Fixed-path/day cases | Test days | Median Spearman rank correlation | Mean loss from best observed hour |',
        '|---|---:|---:|---:|---:|']
    for name in ORDER:
        a=core_ranks[core_ranks.model.eq(name)]
        if len(a):
            r=a.iloc[0];lines.append(f'| {LABELS[name]} | {int(r.cases):,} | {int(r.days)} | {r.median_spearman:.3f} | {r.mean_best_hour_shortfall_db:.2f} dB |')
    lines += ['', '## WSPR generalization stress checks','',
        'Generic SNR predictions on reserved dates; compare models within each row because the '
        'held-out populations differ. Designated stations and fields are excluded from fitting '
        'in both endpoint roles.','',
        '| Test population | Observations | Baseline MAE | Adjusted boosted MAE |',
        '|---|---:|---:|---:|']
    for scenario,selection,label in [('station_holdout','both_endpoints_unseen','Both stations unseen'),
                                    ('geographic_holdout','held_geographic_field','Excluded geographic fields')]:
        subset=m[m.scenario.eq(scenario)&m.selection.eq(selection)&m.view.eq('population_core')&m['mode'].eq('WSPR')]
        if len(subset):
            vals=subset.set_index('model');first=subset.iloc[0]
            lines.append(f'| {label} | {int(first.rows):,} | {vals.loc["boosted_baseline","mae_db"]:.2f} dB | {vals.loc["boosted_station","mae_db"]:.2f} dB |')
    lines += ['', '![Model comparison](results/ml40/model-comparison.png)','',
        'The saved ranking table also includes square-pair aggregation, the scattered main sample, '
        'strict unseen-station and geographic-field cases, and source-specific HARC ranking checks. '
        'Per-hour actual and predicted values are retained for the primary coherent cohort.','',
        '## Uncertainty calibration','',
        'An 80% nominal interval is formed from per-source/mode absolute prediction residuals on '
        'the separate calibration dates. Temporal distribution shifts and station changes mean that '
        '80% future coverage is not guaranteed; the table reports measured final-test coverage.','',
        '| Mode | Model | Observed coverage | Mean width |',
        '|---|---|---:|---:|']
    for mode in modes:
        for name in ORDER:
            a=standard[standard['mode'].eq(mode)&standard.model.eq(name)]
            if len(a):
                r=a.iloc[0];lines.append(f'| {mode} | {LABELS[name]} | {r.interval_coverage:.1%} | {r.mean_interval_width_db:.1f} dB |')
    lines += ['', 'Intervals describe the observed reception population. They do not recover the unseen '
        'sub-threshold distribution. Theoretical WSPR/FT8/FT4/CW/SSB margins can be calculated from a '
        'predicted SNR, but inherit these measurement and calibration limits.','',
        '## Source and split support','',
        '| Source | Mode | Split | Hourly observations |', '|---|---|---|---:|']
    for row in audit['by_source_mode_split']:
        lines.append(f'| {row["source"]} | {row["mode"]} | {row["split"]} | {row["rows"]:,} |')
    lines += ['', 'WSPR has broad calendar coverage; PSK/RBN have a smaller seasonal panel. '
        'The source-specific test-day counts matter when assessing generalization. '
        'Legacy WSPR code 0 and pre-2023 mode ambiguities retain explicit decoder context. '
        'HARC coordinates can mix reported and lookup locations without a per-record provenance flag. '
        'Explicit UTC fields are cross-checked against original RBN events; source epoch discrepancies '
        'and unmatched modes are retained in archive audits.','',
        '## Compact inference and reproducibility','',
        'Load the selected artifact with `ml40.inference`. It computes features from squares, UTC time '
        'and the supplied SFI/Kp/Ap scenario, and predicts without loading the reception dataset. '
        'Its solar-history default holds Kp steady over the previous completed day and uses Ap as '
        'independent previous-day context. This is a stated scenario assumption, not a solar forecast. '
        'Marginal training-range checks do not establish support for every joint solar scenario.','',
        '| Standard artifact | Size |', '|---|---:|']
    for name,info in run['scenarios']['standard']['files'].items():
        lines.append(f'| {LABELS[name]} | {info["bytes"]/1024:.0f} KiB |')
    lines += ['', 'The portable population-core export is [gocap-40m.joblib](artifacts/gocap-40m.joblib) '
        '(about 507 KiB), with [checksum and metadata](artifacts/gocap-40m.json). '
        'It omits station lookup tables and preserves the selected core predictions.','',
        '- [Frozen sample plan and commands](ML40_PLAN.md)',
        '- [Training run, validation decisions and artifact checksums](results/ml40/training_run.json)',
        '- [Final error and coverage metrics](results/ml40/test_metrics.csv)',
        '- [Hour-ranking summary](results/ml40/hour_ranking_summary.csv)',
        '- [Data audit](results/ml40/data_audit.json)',
        '- [Independent verification](results/ml40/verification.json)',
        '- [Original pilot readout](RESULTS.md)','',
        'Reception sources: WSPRNet/WSPR Live, PSK Reporter, RBN, and HamSCI/CEDAR Madrigal. '
        'Geomagnetic and F10.7 history: GFZ and DRAO/Natural Resources Canada. '
        'Models: scikit-learn histogram gradient boosting and MLPRegressor. '
        'Access scope and attribution remain as documented in README.md.','']
    (ROOT/'ML40_RESULTS.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    x=np.arange(len(modes));colors=['#667085','#00796B','#B35600']
    for i,name in enumerate(ORDER):
        v=[standard[standard['mode'].eq(mode)&standard.model.eq(name)].mae_db.mean() for mode in modes]
        axes[0].bar(x+(i-1)*.25,v,.23,label=LABELS[name],color=colors[i])
    axes[0].set(xticks=x,xticklabels=modes,ylabel='SNR MAE (dB; lower is better)',title='Generic predictions on later dates',ylim=(0,None))
    values=[core_ranks[core_ranks.model.eq(name)].median_spearman.mean() for name in ORDER]
    bars=axes[1].bar(np.arange(3),values,color=colors)
    axes[1].bar_label(bars,fmt='%.3f',padding=3)
    axes[1].set(xticks=np.arange(3),xticklabels=['Boosted','Adjusted boosted','Adjusted neural'],
        ylabel='Median Spearman correlation',title='WSPR fixed-path hourly ranking',ylim=(0,max(.6,np.nanmax(values)+.1)))
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8,frameon=False)
    fig.savefig(RESULTS/'model-comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print('Wrote ML40_RESULTS.md and model-comparison.png')

if __name__=='__main__':report()
