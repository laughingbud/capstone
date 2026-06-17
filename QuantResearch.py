"""
QuantResearch.py
================
A self-contained research framework for the NSE F&O intraday dataset shipped in
this repository (``Data/Futures IEOD-<Month> <Year>/<TICKER>_<F1|F2|F3>.csv``).

It provides one cohesive workflow exposed through the :class:`QuantLab` facade:

1.  **Ticker identification** -- distinguish the four index futures
    (``BANKNIFTY``, ``FINNIFTY``, ``MIDCPNIFTY``, ``NIFTY``) from single-name
    equity futures.  -> :class:`TickerClassifier`
2.  **Historical data** -- discover, load and resample the OHLCV+OI panel with
    selectable *tickers*, *contract* (F1/F2/...), *frequency* (minutely, hourly,
    daily, ...) and *features*.  -> :class:`MarketData`
3.  **Strategies (F1 contracts)** -- a library of cross-sectional strategies for
    single-name equities and time-series strategies for the indices, each
    covering momentum, mean-reversion, seasonality plus a bonus regime/residual
    idea.  -> :mod:`strategies`
4.  **Backtesting & validation** -- a vectorised backtester and an anchored /
    rolling **walk-forward** optimiser that tunes parameters in-sample and
    reports stitched out-of-sample performance.  -> :class:`Backtester`,
    :class:`WalkForwardValidator`
5.  **Reporting** -- the full metric suite (sharpe, sortino, calmar, max
    drawdown, win rate, profit factor, avg win/loss, cagr, vol, skew, kurtosis).
    -> :func:`performance_metrics`

The whole thing is dependency-light (numpy / pandas / scipy, plus optional
matplotlib for plotting) and reads the data straight from the local ``Data``
directory, so it runs outside Colab.

Example
-------
>>> lab = QuantLab(data_dir="Data")
>>> lab.load(frequency="daily")                 # all F1 tickers, daily bars
>>> ts = lab.run_timeseries("ts_momentum")      # on the 4 index futures
>>> xs = lab.run_crosssectional("xs_momentum")  # on the equity universe
>>> lab.report([ts, xs])                         # tabular metric comparison
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Ticker identification
# ---------------------------------------------------------------------------
class TickerClassifier:
    """Classify a ticker as an *index* future or a *single-name equity* future.

    Only four symbols in this universe are index futures; everything else is a
    single-name equity.  The class is intentionally tiny and stateless so it can
    be used as a pure utility.
    """

    INDEX_TICKERS: frozenset = frozenset(
        {"BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTY"}
    )

    @classmethod
    def is_index(cls, ticker: str) -> bool:
        return ticker.upper() in cls.INDEX_TICKERS

    @classmethod
    def is_equity(cls, ticker: str) -> bool:
        return not cls.is_index(ticker)

    @classmethod
    def classify(cls, ticker: str) -> str:
        """Return ``"index"`` or ``"equity"`` for *ticker*."""
        return "index" if cls.is_index(ticker) else "equity"

    @classmethod
    def split(cls, tickers: Iterable[str]) -> Tuple[List[str], List[str]]:
        """Partition *tickers* into ``(indices, equities)`` preserving order."""
        indices, equities = [], []
        for t in tickers:
            (indices if cls.is_index(t) else equities).append(t)
        return indices, equities


# ---------------------------------------------------------------------------
# 2. Historical data: discovery, loading, resampling
# ---------------------------------------------------------------------------
#: Human friendly frequency name -> pandas offset alias used by ``resample``.
FREQUENCY_ALIASES: Dict[str, str] = {
    "minutely": "1min",
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "hourly": "1h",
    "1h": "1h",
    "daily": "1D",
    "1d": "1D",
    "weekly": "1W",
    "monthly": "1ME",
}

#: Canonical feature set and how each column aggregates under resampling.
ALL_FEATURES: List[str] = ["Open", "High", "Low", "Close", "Volume", "OI"]
_OHLC_AGG: Dict[str, str] = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
    "OI": "last",
}

#: Raw csv header (after stripping ``<>`` / whitespace) -> canonical feature.
_RAW_COLUMN_MAP: Dict[str, str] = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "o/i": "OI",
    "oi": "OI",
}

_FILENAME_RE = re.compile(r"^(?P<ticker>.+)_(?P<contract>F\d+)\.csv$", re.IGNORECASE)


class MarketData:
    """Index the on-disk CSV universe and load it as a tidy OHLCV+OI panel.

    Parameters
    ----------
    data_dir:
        Root folder that contains the ``Futures IEOD-<Month> <Year>`` sub
        folders.  Defaults to ``"Data"`` relative to the working directory.
    """

    def __init__(self, data_dir: str = "Data") -> None:
        self.data_dir = data_dir
        # {(ticker, contract): [csv_path, ...]} sorted chronologically-ish.
        self._index: Dict[Tuple[str, str], List[str]] = {}
        self.data: Dict[str, pd.DataFrame] = {}
        self.discover()

    # -- discovery ---------------------------------------------------------
    def discover(self) -> "MarketData":
        """Scan ``data_dir`` and build the ``(ticker, contract) -> files`` map."""
        self._index.clear()
        pattern = os.path.join(self.data_dir, "**", "*.csv")
        for path in glob.glob(pattern, recursive=True):
            # Skip the explicitly flagged "Missing Data" folders.
            if "Missing" in os.path.basename(os.path.dirname(path)):
                continue
            m = _FILENAME_RE.match(os.path.basename(path))
            if not m:
                continue  # e.g. the stray ``MIDCPNIFTY-1.csv``
            key = (m.group("ticker").upper(), m.group("contract").upper())
            self._index.setdefault(key, []).append(path)
        return self

    def available_tickers(self, contract: str = "F1") -> List[str]:
        """Tickers that have at least one file for *contract*."""
        contract = contract.upper()
        return sorted(t for (t, c) in self._index if c == contract)

    def available_contracts(self, ticker: str) -> List[str]:
        ticker = ticker.upper()
        return sorted(c for (t, c) in self._index if t == ticker)

    def index_tickers(self, contract: str = "F1") -> List[str]:
        return [t for t in self.available_tickers(contract) if TickerClassifier.is_index(t)]

    def equity_tickers(self, contract: str = "F1") -> List[str]:
        return [t for t in self.available_tickers(contract) if TickerClassifier.is_equity(t)]

    # -- loading -----------------------------------------------------------
    @staticmethod
    def _read_raw(path: str) -> pd.DataFrame:
        """Read one raw csv into a DatetimeIndexed OHLCV+OI frame."""
        df = pd.read_csv(path)
        # Normalise the ``<open>`` style headers.
        df.columns = [c.strip().strip("<>").strip().lower() for c in df.columns]
        # Combine date (dd/mm/yyyy) + time into a single index.
        dt = pd.to_datetime(
            df["date"].astype(str).str.replace("-", "/", regex=False)
            + " "
            + df["time"].astype(str),
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        df = df.rename(columns=_RAW_COLUMN_MAP)
        keep = [c for c in ALL_FEATURES if c in df.columns]
        out = df[keep].copy()
        out.index = dt
        out = out[out.index.notna()].sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out

    def _load_one(self, ticker: str, contract: str) -> pd.DataFrame:
        key = (ticker.upper(), contract.upper())
        files = self._index.get(key)
        if not files:
            raise KeyError(f"No data for ticker={ticker} contract={contract}")
        frames = [self._read_raw(p) for p in files]
        df = pd.concat(frames, axis=0).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    @staticmethod
    def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample a 1-minute OHLCV+OI frame to *freq* (pandas offset alias)."""
        agg = {c: _OHLC_AGG[c] for c in df.columns if c in _OHLC_AGG}
        if freq in ("1min", "1T"):
            out = (
                df.groupby(df.index.normalize(), group_keys=False)
                .apply(lambda x: x.resample("1min").agg(agg))
                .sort_index()
            )
            if "Close" in out.columns:
                out.loc[out["Close"].isna()] = np.nan
            return out
        out = df.resample(freq).agg(agg)
        # Drop empty buckets (weekends / non-trading periods).
        return out.dropna(how="all").dropna(subset=["Close"])

    def load(
        self,
        tickers: Optional[Sequence[str]] = None,
        contract: str = "F1",
        frequency: str = "minutely",
        features: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load historical bars.

        Parameters
        ----------
        tickers:
            Symbols to load; ``None`` (default) loads *every* ticker available
            for *contract*.
        contract:
            Contract month, e.g. ``"F1"`` (front), ``"F2"``, ``"F3"``.
        frequency:
            Bar size -- ``"minutely"``, ``"hourly"``, ``"daily"`` ... (see
            :data:`FREQUENCY_ALIASES`).
        features:
            Subset of ``Open, High, Low, Close, Volume, OI``; ``None`` keeps all.
        start, end:
            Optional ISO date strings to clip the sample.

        Returns
        -------
        dict[str, pandas.DataFrame]
            One DatetimeIndexed frame per ticker (also cached on ``self.data``).
        """
        if frequency not in FREQUENCY_ALIASES:
            raise ValueError(
                f"Unknown frequency {frequency!r}; choose from {list(FREQUENCY_ALIASES)}"
            )
        freq = FREQUENCY_ALIASES[frequency]
        if tickers is None:
            tickers = self.available_tickers(contract)
        features = list(features) if features else ALL_FEATURES

        loaded: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            t = t.upper()
            try:
                df = self._load_one(t, contract)
            except KeyError:
                print(f"[MarketData] skip {t}: no {contract} file")
                continue
            df = self._resample(df, freq)
            cols = [c for c in features if c in df.columns]
            df = df[cols]
            if start is not None:
                df = df[df.index >= pd.Timestamp(start)]
            if end is not None:
                df = df[df.index <= pd.Timestamp(end)]
            if not df.empty:
                loaded[t] = df
        self.data = loaded
        return loaded

    # -- convenience views -------------------------------------------------
    def panel(self, field: str = "Close") -> pd.DataFrame:
        """Return a wide ``time x ticker`` frame of one *field* from cache."""
        if not self.data:
            raise RuntimeError("No data loaded; call load() first.")
        cols = {t: df[field] for t, df in self.data.items() if field in df.columns}
        return pd.DataFrame(cols).sort_index()


# ---------------------------------------------------------------------------
# 3. Performance metrics & reporting
# ---------------------------------------------------------------------------
def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Estimate the number of return observations per year from the index.

    Works for any frequency (intraday or daily) by looking at the median spacing
    of *trading* timestamps and scaling by ~252 trading days a year.
    """
    if len(index) < 3:
        return 252.0
    deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
    med = np.median(deltas[deltas > 0]) if np.any(deltas > 0) else 86400.0
    if med >= 86400 * 0.9:          # daily or coarser
        days = med / 86400.0
        return 252.0 / max(days, 1.0)
    # Intraday: ~6.25h NSE session, 252 sessions.
    seconds_per_session = 6.25 * 3600
    bars_per_session = max(seconds_per_session / med, 1.0)
    return bars_per_session * 252.0


def performance_metrics(
    returns: pd.Series,
    periods_per_year: Optional[float] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute the full performance/risk metric suite for a return series.

    Parameters
    ----------
    returns:
        Per-period (simple) strategy returns with a DatetimeIndex.
    periods_per_year:
        Annualisation factor; inferred from the index when ``None``.
    risk_free_rate:
        Annualised risk-free rate.

    Returns
    -------
    dict
        sharpe, sortino, calmar, max_drawdown, win_rate, profit_factor,
        avg_win_return, avg_loss_return, cagr, volatility, skewness, kurtosis
        (plus a few extras: total_return, n_periods).
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return {k: np.nan for k in _METRIC_KEYS}

    ppy = periods_per_year or infer_periods_per_year(r.index)
    rf_per = (1 + risk_free_rate) ** (1 / ppy) - 1

    mean, std = r.mean(), r.std()
    ann_vol = std * np.sqrt(ppy)
    ann_ret = (1 + r).prod() ** (ppy / len(r)) - 1            # geometric / CAGR

    sharpe = (mean - rf_per) / std * np.sqrt(ppy) if std > 0 else np.nan
    downside = r[r < rf_per]
    dd_std = downside.std()
    sortino = (mean - rf_per) / dd_std * np.sqrt(ppy) if dd_std and dd_std > 0 else np.nan

    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    wins, losses = r[r > 0], r[r < 0]
    win_rate = len(wins) / len(r)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    gross_profit, gross_loss = wins.sum(), losses.sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.nan

    return {
        "cagr": ann_ret,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win_return": avg_win,
        "avg_loss_return": avg_loss,
        "skewness": stats.skew(r, bias=False) if len(r) > 2 else np.nan,
        "kurtosis": stats.kurtosis(r, bias=False) if len(r) > 3 else np.nan,  # excess
        "total_return": equity.iloc[-1] - 1,
        "n_periods": float(len(r)),
    }


_METRIC_KEYS = [
    "cagr", "volatility", "sharpe", "sortino", "calmar", "max_drawdown",
    "win_rate", "profit_factor", "avg_win_return", "avg_loss_return",
    "skewness", "kurtosis", "total_return", "n_periods",
]


@dataclass
class BacktestResult:
    """Container for a single strategy's backtest output."""

    name: str
    returns: pd.Series                      # per-period NET portfolio returns
    weights: Optional[pd.DataFrame] = None  # time x asset target weights
    metrics: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    gross_returns: Optional[pd.Series] = None   # before transaction costs
    turnover: Optional[pd.Series] = None        # per-bar gross weight change

    @property
    def equity_curve(self) -> pd.Series:
        return (1 + self.returns.fillna(0)).cumprod()

    def summary(self) -> pd.Series:
        return pd.Series(self.metrics, name=self.name)

    def net_returns_at_cost(self, cost_bps: float) -> pd.Series:
        """Recompute net returns at an arbitrary *cost_bps* (no re-run needed).

        Uses the stored gross returns and turnover, so the cost/Sharpe trade-off
        can be explored cheaply after a single backtest.
        """
        if self.gross_returns is None or self.turnover is None:
            raise RuntimeError("gross_returns/turnover not stored on this result.")
        return self.gross_returns - self.turnover * (cost_bps / 1e4)


def report(results: Sequence[BacktestResult], sort_by: str = "sharpe") -> pd.DataFrame:
    """Build a tidy metric-comparison table across several backtests."""
    rows = {res.name: res.metrics for res in results}
    tbl = pd.DataFrame(rows).T
    cols = [c for c in _METRIC_KEYS if c in tbl.columns]
    tbl = tbl[cols]
    if sort_by in tbl.columns:
        tbl = tbl.sort_values(sort_by, ascending=False)
    return tbl


#: Default transaction-cost ladder (basis points) for sensitivity analysis.
DEFAULT_COST_LEVELS: List[float] = [1, 2, 5, 7, 10, 15, 20, 25]


def sharpe_vs_cost(
    result: BacktestResult,
    cost_levels: Optional[Sequence[float]] = None,
    metric: str = "sharpe",
) -> pd.Series:
    """Recompute *metric* across a ladder of per-bar transaction costs.

    Cheap: it reuses the stored gross returns and turnover, so no strategy is
    re-run.  Indispensable for intraday work, where high turnover makes a
    strategy's edge highly sensitive to costs.

    Returns
    -------
    pandas.Series indexed by cost in bps (name = strategy name).
    """
    cost_levels = list(cost_levels) if cost_levels is not None else DEFAULT_COST_LEVELS
    if result.gross_returns is None or result.turnover is None:
        return pd.Series(dtype=float, name=result.name)
    out = {
        c: performance_metrics(result.net_returns_at_cost(c)).get(metric, np.nan)
        for c in cost_levels
    }
    return pd.Series(out, name=result.name)


def plot_dashboard(
    results: Sequence[BacktestResult],
    metric: str = "sharpe",
    rolling_window: Optional[int] = None,
    log_equity: bool = True,
    cost_levels: Optional[Sequence[float]] = None,
    figsize: Tuple[float, float] = (18, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Render a 2x3 performance dashboard comparing several backtests.

    Panels:
      1. Cumulative equity curves (log scale by default).
      2. Underwater / drawdown curves.
      3. Rolling annualised Sharpe ratio.
      4. Bar chart of a chosen summary *metric* across strategies.
      5. **Sharpe degradation vs transaction cost** -- each strategy's Sharpe
         recomputed across a cost ladder (1, 2, 5, ... bps); the steeper the
         line, the more fragile the edge.  A dot marks each strategy's
         cost-zero crossing (break-even cost).
      6. **Risk/return map** -- annualised vol (x) vs CAGR (y) scatter, marker
         size proportional to Calmar; the quick "which strategy is actually
         worth running" view.

    Parameters
    ----------
    results:
        Backtests to compare.
    metric:
        Summary metric for the bar chart and the cost-sensitivity panel.
    rolling_window:
        Window (bars) for the rolling Sharpe; inferred (~half a year) when None.
    log_equity:
        Plot the equity curve on a log scale.
    cost_levels:
        Transaction-cost ladder (bps) for the degradation panel.
    save_path:
        If given, the figure is also written to disk.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("Nothing to plot.")
    cost_levels = list(cost_levels) if cost_levels is not None else DEFAULT_COST_LEVELS

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    ax_eq, ax_dd, ax_sharpe = axes[0]
    ax_bar, ax_cost, ax_scatter = axes[1]

    # Intraday series have overnight gaps that distort a datetime x-axis, so for
    # sub-daily data we plot against a positional bar index instead.
    probe = next((res.returns for res in results if len(res.returns) > 2), None)
    intraday_x = probe is not None and infer_periods_per_year(probe.index) > 300

    def _xaxis(series: pd.Series):
        return np.arange(len(series)) if intraday_x else series.index

    for res in results:
        r = res.returns.fillna(0)
        ppy = infer_periods_per_year(r.index) if len(r) > 2 else 252.0
        # Auto window ~ half a year, capped so short OOS series still show a curve.
        win = rolling_window or max(min(int(ppy / 2), len(r) // 3), 20)

        equity = (1 + r).cumprod()
        ax_eq.plot(_xaxis(equity), equity.values, label=res.name, linewidth=1.3)

        drawdown = equity / equity.cummax() - 1
        ax_dd.plot(_xaxis(drawdown), drawdown.values, label=res.name, linewidth=1.0)

        roll = r.rolling(win)
        rolling_sharpe = (roll.mean() / roll.std()) * np.sqrt(ppy)
        ax_sharpe.plot(_xaxis(rolling_sharpe), rolling_sharpe.values,
                       label=res.name, linewidth=1.0)

    _xlabel = "bar # (overnight gaps removed)" if intraday_x else None
    for ax in (ax_eq, ax_dd, ax_sharpe):
        if _xlabel:
            ax.set_xlabel(_xlabel, fontsize=8)
    ax_eq.set_title("Cumulative growth of $1 (out-of-sample)")
    if log_equity:
        ax_eq.set_yscale("log")
    ax_eq.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax_eq.legend(fontsize=7, loc="upper left", ncol=2)
    ax_eq.grid(alpha=0.3)

    ax_dd.set_title("Drawdown")
    ax_dd.axhline(0.0, color="grey", lw=0.8)
    ax_dd.grid(alpha=0.3)

    ax_sharpe.set_title("Rolling Sharpe")
    ax_sharpe.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax_sharpe.grid(alpha=0.3)

    # -- panel 4: metric bar chart ----------------------------------------
    names = [res.name for res in results]
    values = [res.metrics.get(metric, np.nan) for res in results]
    order = np.argsort(np.nan_to_num(values, nan=-np.inf))
    names_s = [names[i] for i in order]
    values_s = [values[i] for i in order]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values_s]
    ax_bar.barh(names_s, values_s, color=colors)
    ax_bar.set_title(f"{metric} by strategy")
    ax_bar.axvline(0.0, color="grey", lw=0.8)
    ax_bar.grid(alpha=0.3, axis="x")

    # -- panel 5: Sharpe vs transaction cost ------------------------------
    cmap = plt.get_cmap("tab10")
    any_cost = False
    all_vals: List[float] = []
    for i, res in enumerate(results):
        series = sharpe_vs_cost(res, cost_levels, metric=metric)
        if series.empty:
            continue
        any_cost = True
        all_vals.extend([v for v in series.values if np.isfinite(v)])
        ax_cost.plot(series.index, series.values, marker="o", ms=4,
                     color=cmap(i % 10), label=res.name, linewidth=1.2)
    if any_cost:
        ax_cost.axhline(0.0, color="grey", lw=0.8, ls="--")
        ax_cost.set_xlabel("transaction cost (bps per bar)")
        ax_cost.set_ylabel(metric)
        ax_cost.legend(fontsize=7, ncol=2)
        # Strategies can plunge to large negatives at high cost while the action
        # of interest (the break-even crossing) sits near zero.  A symlog scale
        # keeps the near-zero band linear and readable yet still shows the
        # collapse, instead of one strategy squashing all the others.
        if all_vals:
            hi = max(all_vals)
            linthresh = max(2.0, abs(hi))
            ax_cost.set_yscale("symlog", linthresh=linthresh)
    else:
        ax_cost.text(0.5, 0.5, "no gross/turnover stored\n(run via Backtester)",
                     ha="center", va="center", transform=ax_cost.transAxes)
    ax_cost.set_title(f"{metric} degradation vs cost")
    ax_cost.grid(alpha=0.3)

    # -- panel 6: risk / return map ---------------------------------------
    for i, res in enumerate(results):
        vol = res.metrics.get("volatility", np.nan)
        cagr = res.metrics.get("cagr", np.nan)
        calmar = res.metrics.get("calmar", np.nan)
        size = 60 + 240 * min(abs(calmar) / 3.0, 1.0) if np.isfinite(calmar) else 60
        ax_scatter.scatter(vol, cagr, s=size, color=cmap(i % 10),
                           alpha=0.75, edgecolors="k", linewidths=0.5)
        ax_scatter.annotate(res.name, (vol, cagr), fontsize=7,
                            xytext=(4, 4), textcoords="offset points")
    ax_scatter.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax_scatter.set_xlabel("annualised volatility")
    ax_scatter.set_ylabel("CAGR")
    ax_scatter.set_title("Risk / return  (marker size ~ Calmar)")
    ax_scatter.grid(alpha=0.3)

    fig.suptitle(title or "Strategy backtest dashboard", fontsize=14, y=0.995)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[plot_dashboard] saved to {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 4. Strategy library
# ---------------------------------------------------------------------------
class Strategy:
    """Base strategy: turns a price panel into target weights.

    A strategy returns a ``time x asset`` weight matrix.  Positive weights are
    long, negative short.  The :class:`Backtester` lags weights by one bar
    before applying them to forward returns, so implementations may use the most
    recent bar without introducing look-ahead.

    Sub-classes implement :meth:`generate_weights`.  ``kind`` is either
    ``"timeseries"`` (each asset traded on its own signal) or
    ``"crosssectional"`` (assets ranked against each other, dollar-neutral).
    """

    kind: str = "timeseries"
    name: str = "strategy"

    def __init__(self, **params: Any) -> None:
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

    def generate_weights(self, close: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({p})"


# -- helpers -----------------------------------------------------------------
def _zscore(x: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = x.rolling(window).mean()
    std = x.rolling(window).std()
    return (x - mean) / std.replace(0, np.nan)


def _cross_sectional_weights(scores: pd.DataFrame, quantile: float = 0.2) -> pd.DataFrame:
    """Long top-quantile / short bottom-quantile, dollar-neutral per row."""
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    ranks = scores.rank(axis=1, pct=True)
    n_valid = scores.notna().sum(axis=1)
    longs = ranks.ge(1 - quantile) & scores.notna()
    shorts = ranks.le(quantile) & scores.notna()
    # Equal weight within each leg, scaled so gross exposure = 1 (0.5 long/0.5 short).
    n_long = longs.sum(axis=1).replace(0, np.nan)
    n_short = shorts.sum(axis=1).replace(0, np.nan)
    weights = weights.add(longs.div(n_long, axis=0).mul(0.5), fill_value=0)
    weights = weights.add(shorts.div(n_short, axis=0).mul(-0.5), fill_value=0)
    weights[n_valid < 4] = 0.0  # need a minimum cross-section
    return weights.fillna(0.0)


def _hurst(series: np.ndarray) -> float:
    """Lagged-difference dispersion Hurst estimate of a 1-D array."""
    s = series[~np.isnan(series)]
    n = len(s)
    if n < 20:
        return np.nan
    lags = range(2, min(20, n // 2))
    tau = []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        tau.append(np.sqrt(np.std(diff)) if diff.size else np.nan)
    tau = np.array(tau)
    valid = (tau > 0) & np.isfinite(tau)
    if valid.sum() < 3:
        return np.nan
    poly = np.polyfit(np.log(np.array(list(lags))[valid]), np.log(tau[valid]), 1)
    return poly[0] * 2.0


# === Time-series strategies (intended for the index futures) ================
class TSMomentum(Strategy):
    """Time-series momentum: go long if the trailing return is positive."""

    kind, name = "timeseries", "ts_momentum"

    def generate_weights(self, close: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.params.get("lookback", 20))
        signal = np.sign(close.pct_change(lookback, fill_method=None))
        return signal.fillna(0.0)


class TSMeanReversion(Strategy):
    """Time-series mean reversion: fade extreme rolling z-scores."""

    kind, name = "timeseries", "ts_mean_reversion"

    def generate_weights(self, close: pd.DataFrame) -> pd.DataFrame:
        window = int(self.params.get("window", 20))
        z_entry = float(self.params.get("z_entry", 1.0))
        z = _zscore(close, window)
        w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        w[z >= z_entry] = -1.0   # too high -> short
        w[z <= -z_entry] = 1.0   # too low  -> long
        return w.fillna(0.0)


class TSSeasonality(Strategy):
    """Time-of-day / day-of-week seasonality.

    For each calendar bucket (intraday hour or weekday) we trade in the
    direction of that bucket's historical *expanding* mean return, so no future
    information leaks into the signal.
    """

    kind, name = "timeseries", "ts_seasonality"

    def generate_weights(self, close: pd.DataFrame) -> pd.DataFrame:
        rets = close.pct_change(fill_method=None)
        intraday = infer_periods_per_year(close.index) > 300  # finer than daily
        bucket = close.index.hour if intraday else close.index.dayofweek
        bucket = pd.Index(bucket, name="bucket")
        w = {}
        for col in rets.columns:
            s = rets[col]
            grp = s.groupby(bucket)
            # expanding mean per bucket, shifted to use only past observations
            exp_mean = grp.transform(lambda x: x.shift(1).expanding().mean())
            w[col] = np.sign(exp_mean)
        return pd.DataFrame(w, index=close.index).fillna(0.0)


class TSRegimeAdaptive(Strategy):
    """**Bonus** -- regime-adaptive trend/reversion switch via the Hurst exponent.

    The Hurst exponent ``H`` of the recent price path tells us whether the
    series is trending (``H > 0.5``) or mean-reverting (``H < 0.5``).  We run
    momentum in trending regimes and mean-reversion otherwise, echoing the
    project's "detect regime, then pick the strategy" thesis.
    """

    kind, name = "timeseries", "ts_regime_adaptive"

    def generate_weights(self, close: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.params.get("lookback", 20))
        hurst_window = int(self.params.get("hurst_window", 100))
        mom = np.sign(close.pct_change(lookback, fill_method=None))
        z = _zscore(close, lookback)
        rev = -np.sign(z)
        w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        # Evaluate Hurst only every ``step`` bars (over the full hurst_window of
        # preceding prices) and forward-fill: the exponent drifts slowly, so this
        # keeps the otherwise-heavy computation tractable on minute-level data
        # without materially changing the regime label.
        step = max(hurst_window // 4, 1)
        for col in close.columns:
            vals = close[col].values
            h_sparse = pd.Series(np.nan, index=close.index)
            for pos in range(hurst_window, len(vals) + 1, step):
                h_sparse.iloc[pos - 1] = _hurst(vals[pos - hurst_window:pos])
            h = h_sparse.ffill()
            trending = h > 0.5
            w[col] = np.where(trending, mom[col], rev[col])
        return w.fillna(0.0)


# === Cross-sectional strategies (intended for the equity universe) ==========
class XSMomentum(Strategy):
    """Cross-sectional momentum: long past winners, short past losers."""

    kind, name = "crosssectional", "xs_momentum"

    def generate_weights(self, close: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.params.get("lookback", 20))
        quantile = float(self.params.get("quantile", 0.2))
        scores = close.pct_change(lookback, fill_method=None)
        return _cross_sectional_weights(scores, quantile)


class XSMeanReversion(Strategy):
    """Cross-sectional reversal: long recent losers, short recent winners."""

    kind, name = "crosssectional", "xs_mean_reversion"

    def generate_weights(self, close: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.params.get("lookback", 5))
        quantile = float(self.params.get("quantile", 0.2))
        scores = -close.pct_change(lookback, fill_method=None)   # invert -> losers score high
        return _cross_sectional_weights(scores, quantile)


class XSSeasonality(Strategy):
    """Cross-sectional seasonality.

    Rank stocks by their historical *expanding* mean return for the current
    calendar bucket (intraday hour or weekday); go long the names that tend to
    do best in that bucket and short the worst.
    """

    kind, name = "crosssectional", "xs_seasonality"

    def generate_weights(self, close: pd.DataFrame) -> pd.DataFrame:
        quantile = float(self.params.get("quantile", 0.2))
        rets = close.pct_change(fill_method=None)
        intraday = infer_periods_per_year(close.index) > 300
        bucket = pd.Index(
            rets.index.hour if intraday else rets.index.dayofweek, name="bucket"
        )
        scores = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
        for col in rets.columns:
            scores[col] = (
                rets[col].groupby(bucket).transform(lambda x: x.shift(1).expanding().mean())
            )
        return _cross_sectional_weights(scores, quantile)


class XSResidualMomentum(Strategy):
    """**Bonus** -- market-neutral residual momentum.

    Each stock's return is regressed on the equal-weight market over the
    lookback window; ranking on the *residual* cumulative return isolates
    idiosyncratic momentum and strips out broad market beta, which historically
    delivers steadier, lower-drawdown cross-sectional performance than raw
    price momentum.
    """

    kind, name = "crosssectional", "xs_residual_momentum"

    def generate_weights(self, close: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.params.get("lookback", 20))
        quantile = float(self.params.get("quantile", 0.2))
        rets = close.pct_change(fill_method=None)
        market = rets.mean(axis=1)
        var_m = market.rolling(lookback).var()
        scores = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
        for col in rets.columns:
            cov = rets[col].rolling(lookback).cov(market)
            beta = cov / var_m.replace(0, np.nan)
            resid = rets[col] - beta * market
            scores[col] = resid.rolling(lookback).sum()  # cumulative residual
        return _cross_sectional_weights(scores, quantile)


#: Registry so strategies can be referenced by name from :class:`QuantLab`.
STRATEGIES: Dict[str, type] = {
    cls.name: cls
    for cls in [
        TSMomentum, TSMeanReversion, TSSeasonality, TSRegimeAdaptive,
        XSMomentum, XSMeanReversion, XSSeasonality, XSResidualMomentum,
    ]
}

#: Sensible parameter grids for walk-forward optimisation, keyed by strategy.
DEFAULT_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "ts_momentum": {"lookback": [5, 10, 20, 40, 60]},
    "ts_mean_reversion": {"window": [10, 20, 40], "z_entry": [1.0, 1.5, 2.0]},
    "ts_seasonality": {},
    "ts_regime_adaptive": {"lookback": [10, 20, 40], "hurst_window": [60, 100]},
    "xs_momentum": {"lookback": [5, 10, 20, 40], "quantile": [0.2, 0.3]},
    "xs_mean_reversion": {"lookback": [2, 5, 10], "quantile": [0.2, 0.3]},
    "xs_seasonality": {"quantile": [0.2, 0.3]},
    "xs_residual_momentum": {"lookback": [10, 20, 40], "quantile": [0.2, 0.3]},
}


# ---------------------------------------------------------------------------
# 5. Backtesting & walk-forward validation
# ---------------------------------------------------------------------------
class Backtester:
    """Vectorised, look-ahead-safe backtester.

    Parameters
    ----------
    cost_bps:
        Round-trip transaction cost in basis points charged on turnover
        (change in weights) each bar.  Defaults to ``1.0`` bp.
    target_vol:
        Annualised volatility target.  When set, the strategy's weights are
        scaled each bar by ``target_vol / recent_realised_vol`` so the portfolio
        runs at a roughly constant risk level (a.k.a. volatility targeting).
        The scaling uses only past returns, so it introduces no look-ahead.
        ``None`` (default) disables it and trades the raw strategy weights.
    vol_window:
        Look-back (in bars) for the realised-volatility estimate used by the
        vol target.
    max_leverage:
        Upper bound on the vol-target leverage multiplier, to stop the sizing
        from exploding when recent realised vol collapses.
    intraday:
        When ``True`` the book is forced flat at the end of every trading day:
        the weight on the last bar of each session is set to zero, so the
        overnight close->open gap return is never earned and no position is
        carried overnight.  This is the right mode for minute-level strategies.
    """

    def __init__(
        self,
        cost_bps: float = 1.0,
        target_vol: Optional[float] = None,
        vol_window: int = 20,
        max_leverage: float = 3.0,
        intraday: bool = False,
    ) -> None:
        self.cost_bps = cost_bps
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.intraday = intraday

    def _vol_target_weights(
        self, weights: pd.DataFrame, asset_rets: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Scale *weights* to the annualised ``target_vol`` using past returns."""
        ppy = infer_periods_per_year(weights.index)
        # Realised vol of the *unscaled* strategy, known up to bar t.
        base_gross = (weights.shift(1).fillna(0.0) * asset_rets).sum(axis=1)
        realised = base_gross.rolling(self.vol_window).std() * np.sqrt(ppy)
        leverage = (self.target_vol / realised).replace([np.inf, -np.inf], np.nan)
        # Shift so the multiplier applied at t only uses information through t-1.
        leverage = leverage.shift(1).clip(upper=self.max_leverage).fillna(0.0)
        return weights.mul(leverage, axis=0), leverage

    @staticmethod
    def _flatten_eod(weights: pd.DataFrame) -> pd.DataFrame:
        """Zero the weights on the last bar of each trading day (go flat EOD)."""
        days = pd.Series(weights.index.normalize(), index=weights.index)
        is_last_of_day = days != days.shift(-1)   # True on each day's final bar
        w = weights.copy()
        w.loc[is_last_of_day.values] = 0.0
        return w

    def run(self, strategy: Strategy, close: pd.DataFrame, name: Optional[str] = None) -> BacktestResult:
        raw_weights = strategy.generate_weights(close).reindex(close.index)
        asset_rets = close.pct_change(fill_method=None).reindex(close.index)

        leverage = None
        weights = raw_weights
        if self.target_vol is not None:
            weights, leverage = self._vol_target_weights(raw_weights, asset_rets)
        if self.intraday:
            # Flat into the close -> lagged weight on the next day's first bar is
            # zero, so the overnight gap return is excluded.
            weights = self._flatten_eod(weights)

        # Lag weights by one bar: decide on bar t, earn return over t -> t+1.
        lagged = weights.shift(1).fillna(0.0)
        gross = (lagged * asset_rets).sum(axis=1)
        # Transaction costs on turnover (includes the EOD flatten / next-open re-entry).
        turnover = (weights - weights.shift(1)).abs().sum(axis=1).fillna(0.0)
        net = (gross - turnover * (self.cost_bps / 1e4)).fillna(0.0)
        # Drop the first bar (no prior weight) and align gross/turnover to net.
        gross, turnover, net = gross.iloc[1:], turnover.iloc[1:], net.iloc[1:]

        meta = {"params": dict(strategy.params), "kind": strategy.kind,
                "intraday": self.intraday}
        if leverage is not None:
            meta["target_vol"] = self.target_vol
            meta["avg_leverage"] = float(leverage.replace(0.0, np.nan).mean())
        res = BacktestResult(
            name=name or strategy.name,
            returns=net,
            weights=weights,
            metrics=performance_metrics(net),
            meta=meta,
            gross_returns=gross,
            turnover=turnover,
        )
        return res


class WalkForwardValidator:
    """Anchored or rolling walk-forward parameter optimisation.

    The sample is cut into consecutive out-of-sample (OOS) blocks.  For each
    block, every parameter combination is scored on the preceding in-sample
    (IS) window; the best combination is then applied to the OOS block.  The OOS
    return streams are stitched together and scored as one series, giving an
    honest, overfit-resistant performance estimate.

    Parameters
    ----------
    n_splits:
        Number of OOS blocks.
    train_span:
        Fraction of the sample used for the first IS window (anchored mode) or
        the size of each rolling IS window (rolling mode).
    mode:
        ``"anchored"`` (expanding IS) or ``"rolling"`` (fixed-length IS).
    scoring:
        Metric maximised during optimisation (default ``"sharpe"``).
    """

    def __init__(
        self,
        n_splits: int = 4,
        train_span: float = 0.5,
        mode: str = "anchored",
        scoring: str = "sharpe",
        cost_bps: float = 1.0,
        target_vol: Optional[float] = None,
        vol_window: int = 20,
        max_leverage: float = 3.0,
        intraday: bool = False,
    ) -> None:
        self.n_splits = n_splits
        self.train_span = train_span
        self.mode = mode
        self.scoring = scoring
        self.backtester = Backtester(
            cost_bps=cost_bps,
            target_vol=target_vol,
            vol_window=vol_window,
            max_leverage=max_leverage,
            intraday=intraday,
        )

    @staticmethod
    def _param_combos(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        if not grid:
            return [{}]
        keys = list(grid)
        from itertools import product
        return [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]

    def run(
        self,
        strategy_cls: type,
        close: pd.DataFrame,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        name: Optional[str] = None,
    ) -> BacktestResult:
        param_grid = param_grid if param_grid is not None else \
            DEFAULT_PARAM_GRIDS.get(strategy_cls.name, {})
        combos = self._param_combos(param_grid)
        n = len(close)
        if n < 50:
            raise ValueError("Not enough observations for walk-forward.")

        start_oos = int(n * self.train_span)
        block = max((n - start_oos) // self.n_splits, 1)
        oos_returns: List[pd.Series] = []
        oos_gross: List[pd.Series] = []
        oos_turnover: List[pd.Series] = []
        chosen: List[Dict[str, Any]] = []

        for i in range(self.n_splits):
            oos_lo = start_oos + i * block
            oos_hi = n if i == self.n_splits - 1 else start_oos + (i + 1) * block
            if oos_lo >= n:
                break
            is_lo = 0 if self.mode == "anchored" else max(oos_lo - start_oos, 0)
            is_slice = close.iloc[is_lo:oos_lo]
            # Give the OOS slice some warm-up history so rolling signals are valid.
            warmup = min(oos_lo, 250)
            oos_slice = close.iloc[oos_lo - warmup:oos_hi]

            best_score, best_params = -np.inf, combos[0]
            for params in combos:
                strat = strategy_cls(**params)
                is_res = self.backtester.run(strat, is_slice)
                score = is_res.metrics.get(self.scoring, np.nan)
                if np.isfinite(score) and score > best_score:
                    best_score, best_params = score, params

            strat = strategy_cls(**best_params)
            oos_res = self.backtester.run(strat, oos_slice)
            # Keep only the genuine OOS portion (drop the warm-up returns).
            cutoff = close.index[oos_lo]
            oos_returns.append(oos_res.returns.loc[oos_res.returns.index >= cutoff])
            if oos_res.gross_returns is not None:
                oos_gross.append(oos_res.gross_returns.loc[oos_res.gross_returns.index >= cutoff])
                oos_turnover.append(oos_res.turnover.loc[oos_res.turnover.index >= cutoff])
            chosen.append(best_params)

        def _stitch(parts: List[pd.Series]) -> Optional[pd.Series]:
            if not parts:
                return None
            s = pd.concat(parts).sort_index()
            return s[~s.index.duplicated(keep="first")]

        stitched = _stitch(oos_returns)
        if stitched is None or stitched.empty:
            raise ValueError(
                "Walk-forward produced no out-of-sample returns; adjust train_span, "
                "n_splits, or sample size."
            )
        return BacktestResult(
            name=name or f"{strategy_cls.name}_wf",
            returns=stitched,
            metrics=performance_metrics(stitched),
            meta={
                "mode": self.mode,
                "n_splits": self.n_splits,
                "chosen_params": chosen,
                "kind": strategy_cls.kind,
                "intraday": self.backtester.intraday,
            },
            gross_returns=_stitch(oos_gross),
            turnover=_stitch(oos_turnover),
        )


# ---------------------------------------------------------------------------
# 6. Facade
# ---------------------------------------------------------------------------
class QuantLab:
    """One-stop research facade tying the whole pipeline together.

    >>> lab = QuantLab("Data", target_vol=0.15)   # 15% annualised vol target
    >>> lab.load(frequency="daily")               # F1 by default
    >>> res = lab.run_timeseries("ts_momentum")
    >>> lab.report([res])
    >>> lab.plot([res])                           # dashboard
    """

    def __init__(
        self,
        data_dir: str = "Data",
        cost_bps: float = 1.0,
        target_vol: Optional[float] = None,
        vol_window: int = 20,
        max_leverage: float = 3.0,
        intraday: bool = False,
    ) -> None:
        self.market = MarketData(data_dir)
        self.classifier = TickerClassifier
        self.cost_bps = cost_bps
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.intraday = intraday
        self.backtester = Backtester(
            cost_bps=cost_bps, target_vol=target_vol,
            vol_window=vol_window, max_leverage=max_leverage, intraday=intraday,
        )
        self.contract = "F1"
        self.frequency = "daily"

    # -- data --------------------------------------------------------------
    def load(
        self,
        tickers: Optional[Sequence[str]] = None,
        contract: str = "F1",
        frequency: str = "daily",
        features: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        self.contract, self.frequency = contract, frequency
        return self.market.load(tickers, contract, frequency, features, start, end)

    def _close_panel(self, kind: str) -> pd.DataFrame:
        close = self.market.panel("Close")
        idx_cols, eq_cols = self.classifier.split(close.columns)
        sub = close[idx_cols] if kind == "timeseries" else close[eq_cols]
        return sub.dropna(how="all")

    # -- single strategy ---------------------------------------------------
    def run_strategy(self, name: str, walk_forward: bool = True, **params: Any) -> BacktestResult:
        if name not in STRATEGIES:
            raise KeyError(f"Unknown strategy {name!r}; choose from {list(STRATEGIES)}")
        cls = STRATEGIES[name]
        close = self._close_panel(cls.kind)
        if walk_forward:
            grid = {k: [v] for k, v in params.items()} if params else None
            validator = WalkForwardValidator(
                cost_bps=self.cost_bps, target_vol=self.target_vol,
                vol_window=self.vol_window, max_leverage=self.max_leverage,
                intraday=self.intraday,
            )
            return validator.run(cls, close, grid, name=name)
        return self.backtester.run(cls(**params), close, name=name)

    def run_timeseries(self, name: str = "ts_momentum", **kw: Any) -> BacktestResult:
        return self.run_strategy(name, **kw)

    def run_crosssectional(self, name: str = "xs_momentum", **kw: Any) -> BacktestResult:
        return self.run_strategy(name, **kw)

    # -- batches -----------------------------------------------------------
    def run_all(self, walk_forward: bool = True) -> List[BacktestResult]:
        """Run every registered strategy on its appropriate universe."""
        results = []
        for name in STRATEGIES:
            try:
                results.append(self.run_strategy(name, walk_forward=walk_forward))
            except Exception as exc:                       # keep the batch going
                print(f"[QuantLab] {name} failed: {exc}")
        return results

    @staticmethod
    def report(results: Sequence[BacktestResult], sort_by: str = "sharpe") -> pd.DataFrame:
        return report(results, sort_by=sort_by)

    @staticmethod
    def plot(results: Sequence[BacktestResult], **kwargs: Any):
        """Render the 2x3 performance dashboard (see :func:`plot_dashboard`)."""
        return plot_dashboard(results, **kwargs)


if __name__ == "__main__":  # pragma: no cover - quick smoke run
    lab = QuantLab("Data")
    print("Index tickers :", lab.market.index_tickers())
    print("Equity sample :", lab.market.equity_tickers()[:8], "...")
    lab.load(frequency="daily")
    results = lab.run_all(walk_forward=True)
    print(lab.report(results).round(3).to_string())
