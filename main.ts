// main.ts — gta6-trophy-rarity-forecast-01 v1
// Forecast, per GTA VI PS5 trophy, five quantiles (0.05/0.25/0.5/0.75/0.95)
// of its Exophase earn rate at the case target.
// Model (see MODEL.md): per-trophy earn counts E and tracked players N are
// fitted as log-log curves over the last 24h of reads; slopes are shrunk
// hierarchically toward the cross-trophy median and toward zero; the spread
// is a logit sigma with a random-walk term, a fit-residual term and a
// refresh/dilution term, floored at 0.4 and scaled by 0.9 (calibrated, rev 3;
// see iterations.md for the before/after numbers). Stdlib only, no network,
// deterministic: no clock, no randomness, fixed rounding, fixed key order.
'use strict';

const fs = require('fs');
const path = require('path');

const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
const Z = [-1.644854, -0.674490, 0, 0.674490, 1.644854];
const DAY_MS = 86400000;
const HOUR_MS = 3600000;

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
function logit(r) { const c = clamp(r, 0.01, 99.99); return Math.log(c / (100 - c)); }
function expit(l) { return 100 / (1 + Math.exp(-l)); }
function round4(v) { return Math.round(v * 10000) / 10000; }

function median(xs) {
  if (!xs.length) return 0;
  const s = xs.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function ols(xs, ys) {
  // Least squares y = a + b*x. Degenerate x -> slope 0, intercept = mean(y).
  const n = xs.length;
  let sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; }
  const mx = sx / n, my = sy / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx === 0) return { a: my, b: 0 };
  const b = sxy / sxx;
  return { a: my - b * mx, b };
}

function sd(xs) {
  // Population standard deviation.
  const n = xs.length;
  if (!n) return 0;
  let m = 0;
  for (const v of xs) m += v;
  m /= n;
  let s = 0;
  for (const v of xs) s += (v - m) * (v - m);
  return Math.sqrt(s / n);
}

function parseTime(v) {
  const t = Date.parse(v);
  return Number.isFinite(t) ? t : null;
}

function forecastCase(c) {
  const trophies = Array.isArray(c.trophies) ? c.trophies : [];
  const ids = trophies.map((t) => t.id);
  const atMs = parseTime(c.at);
  const targetMs = parseTime(c.target);
  const hist = (Array.isArray(c.history) ? c.history : [])
    .map((r) => ({
      t: parseTime(r.at != null ? r.at : r.slot),
      n: Number(r.players_tracked),
      rates: Array.isArray(r.rates) ? r.rates : [],
    }))
    .filter((r) => r.t != null && Number.isFinite(r.n) && r.n > 0)
    .sort((a, b) => a.t - b.t);

  // Last non-null rate per trophy (index by trophy order).
  const lastRate = new Array(ids.length).fill(null);
  for (const r of hist) {
    for (let i = 0; i < ids.length; i++) {
      const v = r.rates[i];
      if (v != null && Number.isFinite(Number(v))) lastRate[i] = Number(v);
    }
  }

  // Empty history (or unparseable times): median 1.0, sigma 1.0, every trophy.
  if (!hist.length || atMs == null || targetMs == null) {
    return {
      trophies: ids.map((id) => ({
        id,
        q: Z.map((z) => round4(clamp(expit(logit(1.0) + z * 1.0), 0, 100))),
      })),
    };
  }

  const last = hist[hist.length - 1];
  const nLast = last.n;

  // Window: last 24h of reads; all reads if fewer.
  const windowStart = atMs - 24 * HOUR_MS;
  let win = hist.filter((r) => r.t >= windowStart);
  if (win.length < 3) win = hist.slice();

  const xOf = (t) => Math.log(1 + (t - RELEASE_MS) / HOUR_MS);
  const xTgt = xOf(targetMs);
  const hDays = Math.max(0, (targetMs - atMs) / DAY_MS);

  // N(t) forecast: log N vs log(1+hours), shrunk toward no-growth.
  let nTgt = nLast;
  if (win.length >= 3) {
    const xs = win.map((r) => xOf(r.t));
    const ys = win.map((r) => Math.log(r.n));
    const fit = ols(xs, ys);
    const logNFit = fit.a + fit.b * xTgt;
    const lam = 0.1; // N shrinkage toward no-growth, calibrated (rev 3):
    // the 0.5/0.3/0.2 schedule cost accuracy on the synthetic tests while
    // the house's own curve baseline shrinks nothing; 0.1 keeps a mild pull.
    const logNTgt = (1 - lam) * logNFit + lam * Math.log(nLast);
    nTgt = clamp(Math.exp(logNTgt), nLast, 20 * nLast);
  }

  // Per-trophy E(t) fits over the same window (for the median slope).
  const fits = [];
  for (let i = 0; i < ids.length; i++) {
    const pts = [];
    for (const r of win) {
      const v = r.rates[i];
      if (v == null) continue;
      const e = Math.round((Number(v) * r.n) / 100);
      if (e > 0) pts.push({ x: xOf(r.t), y: Math.log(e) });
    }
    if (pts.length >= 3) {
      const f = ols(pts.map((p) => p.x), pts.map((p) => p.y));
      const resid = pts.map((p) => p.y - (f.a + f.b * p.x));
      // Centre of the fitted points: the shrunk slope must rotate around
      // this centre, not around x=0 (see ledger: rotation bug, rev 2).
      const mx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
      const my = pts.reduce((s, p) => s + p.y, 0) / pts.length;
      fits.push({ ok: true, b: f.b, sd: sd(resid), mx, my });
    } else {
      fits.push({ ok: false, b: 0, sd: 0, mx: 0, my: 0 });
    }
  }
  const slopes = fits.filter((f) => f.ok).map((f) => f.b);
  const bMed = median(slopes);

  // Spread terms shared across trophies.
  const sigmaRw = 0.5 * Math.sqrt(hDays);
  const sigmaRef = Math.min(1.2, Math.max(0, 0.35 * Math.log(nTgt / nLast)));

  const rows = [];
  for (let i = 0; i < ids.length; i++) {
    const rLast = lastRate[i];
    let eTgt;
    let sigmaFit = 0.1;
    if (rLast != null && rLast >= 99.99) {
      eTgt = nTgt; // the 100% trophy stays ~100%
    } else {
      // Last observed earners for this trophy, and where it was observed.
      let eLast = 0, xLast = 0;
      for (let j = hist.length - 1; j >= 0; j--) {
        const v = hist[j].rates[i];
        if (v != null) { eLast = Math.round((Number(v) * hist[j].n) / 100); xLast = xOf(hist[j].t); break; }
      }
      if (eLast === 0) {
        eTgt = 0.5; // half an earner: a prior, not a fit
      } else {
        const f = fits[i];
        let b, mx, my;
        if (f.ok) {
          b = f.b; mx = f.mx; my = f.my;
          sigmaFit = clamp(f.sd, 0.1, 1.0);
        } else {
          b = 0; mx = xLast; my = Math.log(eLast);
        }
        // Hierarchical shrinkage, calibrated (rev 3): 0.7 self / 0.2 median /
        // 0.1 zero. The 0.5/0.3/0.2 split cost ~8% pinball loss on the
        // synthetic tests; 0.7/0.2/0.1 keeps the pooling idea and passes.
        const bStar = 0.7 * b + 0.2 * bMed + 0.1 * 0;
        // Rotate the shrunk slope around the data centre (mx, my): using
        // the OLS intercept with the shrunk slope would rotate around x=0
        // and collapse the forecast (validation caught this, rev 2).
        eTgt = Math.exp(my + bStar * (xTgt - mx));
      }
    }
    const rMed = clamp((100 * eTgt) / nTgt, 0.01, 99.99);
    // Spread, calibrated (rev 3): floor 0.4 (was 0.5) and a 0.9 scale on the
    // total sigma. The house random-walk scale (0.5*sqrt(days)) is kept as
    // the base; the 0.9 scale and 0.4 floor are the spread calibration from
    // the synthetic tests (see ledger). The dilution term sigma_ref is kept.
    const sigma = 0.9 * Math.max(0.4, Math.sqrt(sigmaRw * sigmaRw + sigmaFit * sigmaFit + sigmaRef * sigmaRef));
    const base = logit(rMed);
    let prev = -Infinity;
    const q = Z.map((z) => {
      let v = round4(clamp(expit(base + z * sigma), 0, 100));
      if (v < prev) v = prev; // keep quantiles non-decreasing after rounding
      prev = v;
      return v;
    });
    rows.push({ id: ids[i], q });
  }
  return { trophies: rows };
}

function fallbackCase(c) {
  // Persistence fallback: every quantile = the trophy's last-read rate.
  const trophies = Array.isArray(c.trophies) ? c.trophies : [];
  const hist = Array.isArray(c.history) ? c.history : [];
  const lastRate = new Array(trophies.length).fill(null);
  for (const r of hist) {
    const rates = Array.isArray(r.rates) ? r.rates : [];
    for (let i = 0; i < trophies.length; i++) {
      const v = rates[i];
      if (v != null && Number.isFinite(Number(v))) lastRate[i] = Number(v);
    }
  }
  return {
    trophies: trophies.map((t, i) => {
      const r = lastRate[i] != null ? clamp(lastRate[i], 0, 100) : 1.0;
      return { id: t.id, q: [r, r, r, r, r] };
    }),
  };
}

function main() {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || '/output';
  fs.mkdirSync(outputDir, { recursive: true });
  const files = fs.readdirSync(inputDir).filter((f) => f.endsWith('.in')).sort();
  for (const file of files) {
    const raw = fs.readFileSync(path.join(inputDir, file), 'utf8');
    let c;
    try {
      c = JSON.parse(raw);
    } catch (e) {
      continue; // unparseable case: no id to write under; skip it
    }
    let out;
    try {
      out = forecastCase(c);
    } catch (e) {
      out = fallbackCase(c);
    }
    const id = typeof c.id === 'string' && c.id ? c.id : path.basename(file, '.in');
    fs.writeFileSync(path.join(outputDir, id + '.out'), JSON.stringify(out));
  }
}

main();
