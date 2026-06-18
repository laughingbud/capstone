const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, ImageRun, TableOfContents,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
} = require("docx");

const ACCENT = "2E5BA8";
const HEAD = "1F3864";
const GREY = "595959";

// ---- helpers ----------------------------------------------------------
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const P = (t, opts = {}) => new Paragraph({ spacing: { after: 120 }, children: parseRuns(t), ...opts });
// supports **bold** inline markers
function parseRuns(t) {
  if (typeof t !== "string") return t;
  const out = [];
  t.split(/(\*\*[^*]+\*\*)/g).forEach((seg) => {
    if (!seg) return;
    if (seg.startsWith("**") && seg.endsWith("**")) out.push(new TextRun({ text: seg.slice(2, -2), bold: true }));
    else out.push(new TextRun(seg));
  });
  return out;
}
const bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: parseRuns(t) });
const num = (t) => new Paragraph({ numbering: { reference: "nums", level: 0 }, spacing: { after: 60 }, children: parseRuns(t) });
const caption = (t) => new Paragraph({ spacing: { after: 200 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: t, italics: true, size: 18, color: GREY })] });

const border = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border,
  insideHorizontal: border, insideVertical: border };

function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const mkCell = (txt, w, opts = {}) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders,
    shading: opts.head ? { fill: ACCENT, type: ShadingType.CLEAR } : (opts.zebra ? { fill: "EEF3FA", type: ShadingType.CLEAR } : undefined),
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({ alignment: opts.num ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: [new TextRun({ text: String(txt), bold: !!opts.head, color: opts.head ? "FFFFFF" : "000000", size: 18 })] })],
  });
  const headRow = new TableRow({ tableHeader: true, children: headers.map((h, i) => mkCell(h, widths[i], { head: true, num: i > 0 })) });
  const dataRows = rows.map((r, ri) => new TableRow({ children: r.map((c, i) => mkCell(c, widths[i], { num: i > 0, zebra: ri % 2 === 1 })) }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [headRow, ...dataRows] });
}

function image(path, widthIn, imgW = 1186, imgH = 704) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, children: [
    new ImageRun({ type: "png", data: fs.readFileSync(path),
      transformation: { width: Math.round(widthIn * 96), height: Math.round(widthIn * 96 * imgH / imgW) },
      altText: { title: "chart", description: "chart", name: "chart" } }),
  ] });
}

const CW = 9360; // content width (US Letter, 1" margins)

// ---- content ----------------------------------------------------------
const children = [];

// Title page
children.push(
  new Paragraph({ spacing: { before: 2600, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Systematic Trading Strategies on the NSE F&O Universe", bold: true, size: 48, color: HEAD })] }),
  new Paragraph({ spacing: { before: 200, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Intraday and Daily Cross-Sectional & Time-Series Strategies, with Walk-Forward Validation, Realistic Cost / Market-Impact Modelling and Capacity Analysis", italics: true, size: 24, color: GREY })] }),
  new Paragraph({ spacing: { before: 1400, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Research & Methodology Report", bold: true, size: 26 })] }),
  new Paragraph({ spacing: { before: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Framework: QuantResearch.py", size: 22, color: GREY })] }),
  new Paragraph({ spacing: { before: 1600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Capstone Project — Group 7132", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "June 2026", size: 22, color: GREY })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// TOC
children.push(H1("Contents"));
children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// 1. Executive summary
children.push(H1("1. Executive Summary"));
children.push(P("This report documents a complete, dependency-light research framework (QuantResearch.py) built on a two-year minute-level dataset of NSE single-stock and index futures, and the empirical conclusions it produced. The framework spans the full pipeline: ticker classification, cached data loading and resampling, a library of cross-sectional and time-series strategies, a look-ahead-safe backtester with volatility targeting and end-of-day flattening, anchored walk-forward validation, a realistic transaction-cost and market-impact model, turnover controls, liquidity-weighted sizing, a full performance-metric suite, persisted results, and capacity analysis."));
children.push(P("The central finding is a clean horizon effect that survives scrutiny. **At intraday frequencies, short-horizon cross-sectional mean reversion dominates**; **at the daily frequency, time-series momentum dominates** and reversion loses. However, the two regimes differ enormously in implementability:"));
children.push(bullet("**Intraday mean reversion** posts spectacular frictionless Sharpe ratios (50+ at one-minute bars) but is almost entirely an artefact of microstructure / bid-ask bounce: it turns the book over ~200,000 times a year and is annihilated by realistic transaction costs and market impact. Its realistic capacity is tiny — on the order of ₹130–160 million (~US$1.5–1.9 million)."));
children.push(bullet("**Daily time-series momentum** earns a far more modest but believable Sharpe (~0.9 out-of-sample, net of costs), turns over only ~80–225 times a year, and remains profitable to a book size of roughly ₹100 billion (~US$1.2 billion) — about 600× the capacity of the best intraday strategy."));
children.push(P("The practical conclusion: **the genuinely deployable, scalable strategy is daily time-series momentum on index and single-name futures**, while the intraday reversal edge is a high-Sharpe, very-low-capacity niche whose profitability hinges entirely on execution cost."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// 2. Data & universe
children.push(H1("2. Data and Universe"));
children.push(P("The dataset is intraday (one-minute) OHLCV+OI bars for NSE futures, organised as one CSV per ticker-contract-month across 24 monthly folders spanning January 2022 to December 2023."));
children.push(H2("2.1 Instrument classification"));
children.push(P("Four symbols are index futures — **BANKNIFTY, FINNIFTY, MIDCPNIFTY, NIFTY** — and the remaining ~200 are single-name equity futures. TickerClassifier encodes this split and routes time-series strategies to the index universe and cross-sectional strategies to the equity universe."));
children.push(H2("2.2 Contracts, frequencies and features"));
children.push(bullet("**Contracts:** F1 (front month, used throughout), F2, F3."));
children.push(bullet("**Frequencies:** native one-minute, resampled on demand to 5-minute, 15-minute, hourly, daily, weekly or monthly with correct OHLC / summed-volume / last-OI aggregation."));
children.push(bullet("**Features:** Open, High, Low, Close, Volume, Open Interest."));
children.push(H2("2.3 Data-quality issue discovered and fixed"));
children.push(P("The vendor files mix date conventions: most months use MM/DD/YYYY with slashes, but the August–December 2023 files use DD-MM-YYYY with dashes. A single hard-coded parser silently dropped every day after the 12th and scrambled the rest, making the history look truncated to ~10 weeks. A per-file auto-detecting parser fixed this; the corrected F1 minute series spans 2022-01-03 to 2023-12-29 — 24 months, 493 trading days, ~184,000 bars per ticker. All results in this report use the corrected data."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// 3. Architecture
children.push(H1("3. Framework Architecture"));
children.push(P("QuantResearch.py is a single, self-contained module (numpy / pandas / scipy only) exposed through a QuantLab facade. Its components:"));
children.push(table(
  ["Component", "Responsibility"],
  [
    ["TickerClassifier", "Index vs single-name equity classification"],
    ["MarketData", "Discover, load, resample OHLCV+OI; parquet caching"],
    ["Strategy library", "Cross-sectional & time-series signal generators"],
    ["Backtester", "Look-ahead-safe P&L, vol targeting, EOD flat, costs, impact, turnover controls, liquidity sizing"],
    ["WalkForwardValidator", "Anchored/rolling parameter optimisation, stitched OOS"],
    ["performance_metrics", "Full return/risk metric suite (frequency-aware)"],
    ["execution_stats", "Leverage, turnover, realised cost, cost drag"],
    ["save_run / compare_runs", "Persist & track results across runs"],
    ["plot_dashboard / plot_capacity_frontier", "2×3 dashboard and capacity charts"],
  ],
  [2600, 6760],
));
children.push(P("", { spacing: { after: 60 } }));
children.push(P("**Performance:** reading ~4,900 minute CSVs and resampling the full universe takes ~80 seconds; the resampled per-ticker frames are cached to parquet, so subsequent loads take ~0.2 seconds (a ~400× speed-up). Results are persisted to timestamped folders plus an always-current latest/ snapshot and a cumulative runs_log.csv."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// 4. Strategy library
children.push(H1("4. Strategy Library"));
children.push(P("Strategies are run on F1 contracts. Time-series strategies trade each instrument on its own signal and are run on both the index universe (suffix _index) and the single-name universe (suffix _ss). Cross-sectional strategies rank the equity universe and build dollar-neutral long/short books."));
children.push(H2("4.1 Time-series strategies (per instrument)"));
children.push(bullet("**Momentum** — long if trailing return is positive (lookback tunable)."));
children.push(bullet("**Mean reversion** — fade rolling z-score extremes."));
children.push(bullet("**Seasonality** — trade the sign of the expanding-mean return for the current calendar bucket (hour-of-day intraday, day-of-week daily); look-ahead-safe."));
children.push(bullet("**Regime-adaptive (bonus)** — switch between momentum and mean reversion based on the rolling Hurst exponent (trend if H>0.5, revert otherwise)."));
children.push(H2("4.2 Cross-sectional strategies (equity universe)"));
children.push(bullet("**Momentum** — long past winners, short past losers (top/bottom quantile, dollar-neutral)."));
children.push(bullet("**Mean reversion** — long recent losers, short recent winners."));
children.push(bullet("**Seasonality** — rank names by historical mean return in the current calendar bucket."));
children.push(bullet("**Residual momentum (bonus)** — momentum on returns residual to a rolling regression on the equal-weight market (market-neutral)."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// 5. Methodology
children.push(H1("5. Methodology"));
children.push(H2("5.1 Backtesting"));
children.push(P("The backtester is vectorised and look-ahead-safe: target weights decided on bar t are lagged one bar before being applied to forward returns. Per-asset P&L is retained so it sums exactly to the portfolio return, enabling per-asset Sharpe attribution."));
children.push(bullet("**Volatility targeting** — weights are scaled each bar by target_vol / trailing realised vol (using only past data), so strategies run at a constant risk level and are comparable."));
children.push(bullet("**Intraday EOD flattening** — the book is forced flat on the last bar of each day, so no overnight gap is earned and no position is carried overnight (verified: last-bar weights are exactly zero)."));
children.push(H2("5.2 Walk-forward validation"));
children.push(P("The sample is cut into consecutive out-of-sample (OOS) blocks; each strategy's parameters are optimised on the preceding in-sample window and applied to the next OOS block. The stitched OOS streams are scored as one series, giving an honest, overfit-resistant estimate. All headline numbers in this report are walk-forward OOS."));
children.push(H2("5.3 Performance metric suite"));
children.push(P("Each run reports: Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, average win / loss return, CAGR, annualised volatility, skewness and kurtosis. Annualisation is inferred from the bar spacing, so it is correct for daily and any intraday frequency. Execution columns are reported alongside: gross leverage, turnover per bar, annualised turnover, the assumed spread, the realised cost (in bps, derived from gross vs net so it captures impact), and the annualised cost drag."));
children.push(H2("5.4 Transaction cost and market-impact model"));
children.push(P("Cost per name per bar = |Δweight| × (spread_bps + impact_coef_bps × √participation) / 10,000, where participation = |Δweight| × capital / ADV_value. The linear spread is the half-spread/slippage always paid; the square-root term is Almgren-style market impact that penalises large trades in thin names. With impact disabled it reduces to a linear cost. ADV is each name's average daily traded value (price × volume)."));
children.push(H2("5.5 Turnover controls and liquidity sizing"));
children.push(bullet("**Signal smoothing** — EMA the target weights so positions move gradually."));
children.push(bullet("**Rebalance throttle** — act only every N bars, holding the target in between."));
children.push(bullet("**No-trade band** — hysteresis: only trade a name when its target drifts beyond a relative band (EOD-aware so intraday stays flat overnight)."));
children.push(bullet("**Turnover penalty** — subtract a penalty × turnover term from the in-sample selection score, so the walk-forward chooses lower-churn parameters."));
children.push(bullet("**Liquidity-weighted sizing** — tilt position sizes by ADV^power, rescaling each leg to preserve gross exposure and dollar-neutrality; positive power leans toward liquid names, negative toward less-liquid names."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// 6. Results
children.push(H1("6. Results"));

children.push(H2("6.1 Daily strategies (full universe, 15% vol target, walk-forward OOS)"));
children.push(table(
  ["Strategy", "Sharpe", "Calmar", "MaxDD", "CAGR", "Vol"],
  [
    ["ts_momentum_ss", "0.93", "1.28", "-14%", "18%", "0.20"],
    ["ts_momentum_index", "0.87", "0.84", "-18%", "15%", "0.18"],
    ["xs_momentum", "0.61", "0.73", "-11%", "8%", "0.14"],
    ["xs_mean_reversion", "0.41", "0.57", "-8%", "5%", "0.13"],
    ["ts_seasonality_index", "0.32", "0.26", "-15%", "4%", "0.16"],
    ["xs_residual_momentum", "-0.42", "-0.45", "-14%", "-6%", "0.14"],
    ["xs_seasonality", "-0.94", "-0.62", "-18%", "-11%", "0.12"],
    ["ts_mean_reversion_index", "-1.15", "-0.88", "-39%", "-34%", "0.32"],
    ["ts_regime_adaptive_index", "-1.99", "-0.91", "-33%", "-30%", "0.17"],
  ],
  [3000, 1272, 1272, 1272, 1272, 1272],
));
children.push(caption("Daily momentum (time-series and cross-sectional) is profitable out-of-sample; reversion and seasonality are not."));
children.push(image("results/latest/daily/dashboard_daily.png", 6.4, 2146, 1196));
children.push(caption("Figure 1. Daily strategy dashboard: equity curves, drawdowns, rolling Sharpe, metric bars, cost-degradation and risk/return map."));

children.push(H2("6.2 The horizon effect"));
children.push(P("Running the same library at minute frequency flips the ranking: cross-sectional mean reversion is the dominant intraday strategy, and momentum loses. This is a textbook horizon effect — short-term reversal (overreaction / microstructure) at high frequency, trend at low frequency — and its clean appearance is evidence the framework is capturing real structure rather than artefacts."));
children.push(table(
  ["Strategy (xs_mean_reversion)", "Sharpe", "Ann. turnover", "Realised cost"],
  [
    ["1-min, spread-only 0.5bp", "54.4", "~212,000x", "0.50bp"],
    ["15-min, spread-only 0.5bp", "3.0", "~14,800x", "0.50bp"],
    ["15-min, with sqrt impact (5% ADV)", "-5.7", "~9,400x", "1.69bp"],
  ],
  [4360, 1666, 1667, 1667],
));
children.push(caption("Slowing from 1-min to 15-min strips the microstructure illusion; realistic impact then erases the residual edge."));

children.push(H2("6.3 The frictionless illusion and cost sensitivity"));
children.push(P("At one-minute bars the intraday reversal edge is enormous gross but rebalances essentially the entire book every minute (~212,000 turns/year, ~1,060%/year cost drag at 0.5bp). Persisting a Sharpe-vs-cost ladder for every run makes this explicit: the strategy's break-even cost is below ~1bp, far under any realistic all-in cost. The high Sharpe is mostly bid-ask bounce, not tradeable alpha."));

children.push(H2("6.4 Turnover controls"));
children.push(P("Applied to intraday xs_mean_reversion, the controls cut turnover sharply but at a roughly one-for-one Sharpe cost, because the alpha itself lives at the high-frequency horizon. Smoothing and throttling are the effective levers; the no-trade band barely helps a fully-rotating basket. The walk-forward turnover penalty also helps: penalising turnover shifts parameter selection to a longer lookback (e.g. 2→10), cutting turnover ~50% for a ~20% Sharpe give-up."));

children.push(H2("6.5 The alpha–liquidity conflict and liquidity-weighted sizing"));
children.push(P("Mapping intraday xs_mean_reversion by liquidity quartile shows the alpha is monotonically concentrated in less-liquid names (spread-only Sharpe: most-liquid quartile -2.0 → least-liquid +12.2), while impact cost rises as liquidity falls. The two implementability levers therefore conflict: restricting to the most liquid names removes the alpha. Liquidity-weighted sizing confirms this — at a small book, a mild tilt toward less-liquid names (power = -0.5) maximises OOS Sharpe (3.3 vs 2.0 equal-weight), the opposite of the usual instinct. The decisive lever is trading a smaller book, not trading more liquid names."));

children.push(H2("6.6 Capacity analysis"));
children.push(P("Capacity is defined as the book size at which impact-aware OOS Sharpe crosses zero, found by sweeping book size with the impact model."));
children.push(H3("Intraday xs_mean_reversion (5-min vs 15-min)"));
children.push(image("results/latest/capacity/capacity_frontier.png", 6.0));
children.push(caption("Figure 2. Intraday capacity frontier. 5-min has higher small-book Sharpe but decays faster; curves cross in the capacity zone."));
children.push(table(
  ["Bar size", "Sharpe @ small book", "Capacity (INR)", "% of median ADV"],
  [
    ["5-min", "~27 @ ₹10M", "~₹161M", "7.8%"],
    ["15-min", "~8 @ ₹10M", "~₹129M", "6.2%"],
  ],
  [2340, 2340, 2340, 2340],
));
children.push(caption("Both intraday capacities are tiny (~₹130–160M, ~US$1.5–1.9M)."));

children.push(H3("Daily momentum strategies"));
children.push(image("results/latest/capacity_daily/capacity_frontier_daily.png", 6.0));
children.push(caption("Figure 3. Daily capacity frontier. Sharpe is flat from ₹10M to ~₹1bn because daily turnover makes impact negligible until very large books."));
children.push(table(
  ["Strategy", "Ann. turnover", "Capacity (INR)", "≈ USD"],
  [
    ["ts_momentum_ss", "~205x", "~₹99bn", "~$1.2bn"],
    ["ts_momentum_index", "~80x", "~₹97bn", "~$1.2bn"],
    ["xs_momentum", "~225x", "~₹2.75bn", "~$33M"],
  ],
  [3000, 2120, 2120, 2120],
));
children.push(caption("Daily momentum has ~600× the capacity of the best intraday strategy."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// 7. Key findings
children.push(H1("7. Key Findings"));
children.push(num("**Horizon effect is real and clean:** reversal wins intraday, momentum wins daily — consistent with overreaction at short horizons and trend at longer horizons."));
children.push(num("**Frictionless Sharpe is dangerously misleading intraday:** the one-minute reversal Sharpe of ~54 is mostly bid-ask bounce; honest accounting (impact + capacity) is essential."));
children.push(num("**Costs, not signals, are the binding constraint intraday:** break-even cost is sub-1bp and realistic impact pushes every intraday strategy negative at meaningful size."));
children.push(num("**Alpha and liquidity conflict for reversal:** the edge lives in less-liquid names, so 'trade only liquid names' destroys it; 'trade smaller' is the lever that works."));
children.push(num("**Daily time-series momentum is the implementable winner:** modest but believable OOS Sharpe (~0.9), low turnover, and capacity ~₹100bn — roughly 600× the intraday reversal."));
children.push(num("**Walk-forward + impact-aware selection self-regularises:** when impact bites, the optimiser automatically picks lower-turnover parameters."));

children.push(H1("8. Limitations"));
children.push(bullet("**Two-year sample (2022–2023).** A single, fairly trending macro regime; results may not generalise across bull/bear/sideways regimes. No out-of-period (e.g. 2024+) test."));
children.push(bullet("**Idealised execution.** Fills are assumed at the bar close; the impact model is a parametric square-root law with assumed coefficients, not calibrated to realised fills. No modelling of queue position, partial fills, adverse selection, latency, or borrow/short availability."));
children.push(bullet("**ADV proxy.** Average daily traded value is computed as price × futures volume without lot-size/multiplier normalisation, so participation and absolute capacity figures are approximate (relative comparisons are robust)."));
children.push(bullet("**Futures-specific frictions omitted.** Roll costs, expiry effects, margin, financing and contract-switching between F1 and the next front month are not modelled; F1 is treated as a continuous series."));
children.push(bullet("**No portfolio-level construction.** Strategies are evaluated individually; there is no multi-strategy combination, correlation-aware allocation, or risk-budgeting across signals."));
children.push(bullet("**Simple signals and grids.** Each strategy uses one or two parameters over coarse grids; no feature engineering, ensembling, or machine-learning models, and no formal multiple-testing / data-snooping correction."));
children.push(bullet("**Costs assumed symmetric and static.** Spread and impact coefficients are constant across names and time, rather than time-varying with volatility or order-book depth."));
children.push(bullet("**Survivorship / universe drift.** The universe is taken as the available files; additions, deletions and ban-period exclusions of single-stock futures are not explicitly handled."));

children.push(H1("9. What Else Could Be Most Useful to Explore"));
children.push(H2("9.1 Highest priority"));
children.push(num("**Out-of-sample period test (2024+).** Re-run the committed daily momentum configuration on fresh data to confirm the edge persists out of the development sample."));
children.push(num("**Multi-strategy portfolio.** Combine daily ts_momentum (index + single-name) and xs_momentum into one risk-weighted, correlation-aware book and report its impact-aware metrics and capacity — likely the most deployable end product."));
children.push(num("**Execution realism.** Model fills at next-bar open (or VWAP over a participation schedule), calibrate the impact coefficient to realised slippage, and add a participation cap (e.g. ≤5–10% of ADV) — turning capacity from an estimate into a constraint."));
children.push(H2("9.2 Signal and regime research"));
children.push(bullet("**Regime overlay (the project's original thesis):** use the Hurst / volatility regime to switch the whole book between daily momentum and intraday reversion, rather than running each standalone."));
children.push(bullet("**Better intraday alpha:** test whether intraday reversion has a longer-horizon, lower-turnover form (e.g. open-to-close overreaction, or signals refreshed a few times per day) that survives impact at larger size."));
children.push(bullet("**Volatility / OI features:** the loader already provides Open Interest and volume; OI changes and volume spikes are natural conditioning variables left unused here."));
children.push(bullet("**Cross-frequency blends:** daily momentum core with a small intraday reversion satellite sized to its low capacity."));
children.push(H2("9.3 Robustness and infrastructure"));
children.push(bullet("**Parameter-stability and deflated-Sharpe analysis** to guard against the coarse-grid overfitting risk."));
children.push(bullet("**Transaction-cost sensitivity as a first-class deliverable** per strategy (already persisted) used as a go/no-go gate against measured live costs."));
children.push(bullet("**Walk-forward with rolling (not just anchored) windows** and more splits, to test stationarity of the chosen parameters."));
children.push(bullet("**F2/F3 and roll-aware continuous series** to study term-structure and reduce expiry noise."));

children.push(H1("10. Reproducibility"));
children.push(P("All results are persisted under results/latest/ as committed snapshots, each containing metrics.csv, asset_sharpe.csv (per-name Sharpe), cost_sensitivity.csv (Sharpe vs cost ladder), returns/gross_returns parquet, and meta.json (full run config including cost, impact, turnover and liquidity settings):"));
children.push(bullet("**daily/** — full-universe daily strategies (11)."));
children.push(bullet("**intraday/** and **intraday15/** — 1-minute and 15-minute intraday strategies."));
children.push(bullet("**intraday_lowturn/**, **intraday_realistic/** — turnover-controlled and impact-aware intraday runs."));
children.push(bullet("**xs_mr_15m_best/** — best implementable intraday reversal config (15-min, small book, liquidity tilt)."));
children.push(bullet("**capacity/** and **capacity_daily/** — capacity-frontier curves and charts."));
children.push(P("Typical usage: lab = QuantLab(\"Data\", target_vol=0.15); lab.load(frequency=\"daily\"); res = lab.run_all(); lab.report(res); lab.plot(res); lab.save(res). Capacity: lab.capacity_frontier(\"ts_momentum\", universe=\"index\")."));

// ---- document ----------------------------------------------------------
const doc = new Document({
  creator: "QuantResearch",
  title: "Systematic Trading Strategies on the NSE F&O Universe",
  styles: {
    default: { document: { run: { font: "Arial", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: HEAD, font: "Arial" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: ACCENT, font: "Arial" },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, color: "000000", font: "Arial" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] },
      { reference: "nums", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT,
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT, space: 4 } },
      children: [new TextRun({ text: "Systematic Trading Strategies — NSE F&O", size: 16, color: GREY })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Page ", size: 16, color: GREY }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: GREY }),
                 new TextRun({ text: " of ", size: 16, color: GREY }), new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: GREY })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("Capstone_Strategy_Report.docx", buf); console.log("wrote Capstone_Strategy_Report.docx"); });
