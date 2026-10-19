/*
 * FinPassport — agent core
 * An AI financial agent for international students (SAR -> BDT corridor).
 *
 * Pipeline for every request:
 *   1. Understand   : detectIntent()       -> intent + slots
 *   2. Plan         : *Flow() functions    -> ordered tool calls
 *   3. Guard        : evaluate*()          -> permission tier + risk checks (deterministic policy engine)
 *   4. Ask          : pending approvals    -> one-tap confirm (Tier 1) or approval + OTP (Tier 2)
 *   5. Act          : execute*()           -> sandbox ledger
 *   6. Explain/Log  : log()                -> append-only audit trail with reasons
 *
 * The policy engine sits between the planner and the tools. The planner (rule-based here,
 * an LLM in production) can only PROPOSE actions; it can never skip a check.
 *
 * Runs in the browser (window.FinPassport) and in Node (module.exports) with no dependencies.
 * Everything is simulated: no real money, accounts or providers.
 */
(function (root) {
  'use strict';

  // Trained models (ml/ -> src/models.js): rate forecast, federated fraud model, provider learner.
  const MODELS = (typeof module !== 'undefined' && module.exports)
    ? (function () { try { return require('./models.js'); } catch (e) { return null; } })()
    : (root.FinPassportModels || null);

  // ---------------------------------------------------------------- utilities
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const round = (x, d = 2) => Math.round(x * Math.pow(10, d)) / Math.pow(10, d);
  const fmtSAR = (n) => 'SAR ' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtBDT = (n) => 'BDT ' + Math.round(n).toLocaleString('en-US');
  const fmtUSD = (n) => 'USD ' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtRate = (r) => r.toFixed(2);

  const BASE_DATE = Date.UTC(2026, 9, 20); // simulation "today": 20 Oct 2026
  const DAY_MS = 86400000;
  const SAR_PER_USD = 3.75; // SAR is pegged to USD
  const FX_SPREAD = 0.005;
  const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];

  // Simulated providers (fictional names). Fee in SAR, margin = FX markup on the mid-market rate.
  const PROVIDERS = [
    { id: 'gulfx', name: 'GulfXpress', fee: 10, margin: 0.004, speed: 'Same day', payout: 'Bank account' },
    { id: 'quick', name: 'QuickRemit', fee: 15, margin: 0.006, speed: 'Within minutes', payout: 'Bank or mobile wallet' },
    { id: 'desh', name: 'DeshLink', fee: 0, margin: 0.015, speed: 'Same day', payout: 'Mobile wallet' },
    { id: 'bank', name: 'Bank wire', fee: 25, margin: 0.012, speed: '1-2 business days', payout: 'Bank account' }
  ];

  const DEFAULT_SETTINGS = {
    singleLimit: 1000,     // SAR; above this a transfer is Tier 2
    dailyCap: 5000,        // SAR; above this a transfer is blocked
    velocityMax: 3,        // transfers per day before Tier 2
    anomalyMultiple: 3,    // x average transfer before Tier 2
    otpAttempts: 3
  };

  // Simulated sanctions / watchlist (fictional entities)
  const WATCHLIST = ['shadow exchange llc', 'darkpool traders'];

  const TIER_NAMES = {
    0: 'Tier 0 · Read only',
    1: 'Tier 1 · Confirm',
    2: 'Tier 2 · Approve + code',
    3: 'Tier 3 · Blocked'
  };

  function percentile(arr, p) {
    const s = arr.slice().sort((a, b) => a - b);
    const idx = (s.length - 1) * p;
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return s[lo] + (s[hi] - s[lo]) * (idx - lo);
  }

  // ------------------------------------------------------------------- state
  function createState(opts) {
    opts = opts || {};
    const seed = opts.seed != null ? opts.seed : 20261020;
    const rnd = mulberry32(seed);

    let hist, future;
    if (MODELS && MODELS.forecast && MODELS.forecast.series) {
      // 90 days of the simulated series the forecast model was trained on, plus the 14 days that follow
      hist = MODELS.forecast.series.history.slice();
      future = MODELS.forecast.series.future.slice();
    } else {
      // Fallback: 60 days of simple simulated history (SAR is USD-pegged, USD/BDT ~122 => ~32.5)
      hist = [];
      let r = 32.5;
      for (let i = 0; i < 60; i++) {
        r += (32.5 - r) * 0.15 + (rnd() - 0.5) * 0.16;
        hist.push(round(r, 4));
      }
      const today = round(percentile(hist.slice(-30), 0.3), 4);
      hist[hist.length - 1] = today;
      future = [0.03, 0.10, 0.18, 0.26, 0.21, 0.12, 0.05, -0.02, 0.08, 0.14].map((d) => round(today + d, 4));
    }

    const startIndex = hist.length - 1;
    const st = {
      seed, rnd,
      rates: hist.concat(future),
      startIndex,
      dayIndex: startIndex,
      balances: { SAR: 4200, BDT: 0, USD: 300 },
      recipients: [{
        id: 'r1', name: 'Ammu', relation: 'Mother', country: 'Bangladesh',
        account: 'Dhaka bank account ****4821',
        aliases: ['ammu', 'mother', 'mom', 'amma', 'my mother', 'my mom', 'home', 'family'],
        trusted: true
      }],
      // Past transfers (dayIndex relative to the series) -> average 1,000 SAR
      transfers: [
        { day: startIndex - 40, amount: 900, recipient: 'Ammu', provider: 'Bank wire', received: 0, ref: 'FP-HIST-1' },
        { day: startIndex - 25, amount: 1000, recipient: 'Ammu', provider: 'Bank wire', received: 0, ref: 'FP-HIST-2' },
        { day: startIndex - 10, amount: 1100, recipient: 'Ammu', provider: 'DeshLink', received: 0, ref: 'FP-HIST-3' }
      ],
      alerts: [],
      scheduled: [],
      pending: {},
      log: [],
      settings: Object.assign({}, DEFAULT_SETTINGS, opts.settings || {}),
      // Demo switches for the fraud model (a stolen login looks like this)
      session: { newDevice: false, night: false },
      // Provider learner: posterior per provider, learned from community receipts, updated after every transfer
      providerModel: MODELS && MODELS.provider ? JSON.parse(JSON.stringify(MODELS.provider.providers)) : null,
      receipts: MODELS && MODELS.provider ? MODELS.provider.receipts : 0,
      seq: 0
    };
    return st;
  }

  function currentDate(st) { return new Date(BASE_DATE + (st.dayIndex - st.startIndex) * DAY_MS); }
  function dateForIndex(st, idx) { return new Date(BASE_DATE + (idx - st.startIndex) * DAY_MS); }
  function fmtDate(d) {
    return d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
  }
  function isoDate(d) { return d.toISOString().slice(0, 10); }

  function rateNow(st) { return st.rates[st.dayIndex]; }
  function window30(st) { return st.rates.slice(st.dayIndex - 29, st.dayIndex + 1); }

  function log(st, actor, action, tier, decision, detail) {
    st.seq += 1;
    const entry = {
      seq: st.seq,
      simDate: isoDate(currentDate(st)),
      time: new Date().toISOString(),
      actor, action,
      tier: tier == null ? null : tier,
      decision,
      detail: detail || ''
    };
    st.log.push(entry);
    return entry;
  }

  // ------------------------------------------------------------------- tools
  // Tier 0 tools: read only, no money movement.
  function toolGetRate(st) {
    const rate = rateNow(st);
    log(st, 'agent', 'tool:get_rate', 0, 'auto', `1 SAR = ${fmtRate(rate)} BDT (mid-market, simulated)`);
    return rate;
  }

  function rateInsight(st) {
    const w = window30(st);
    const today = rateNow(st);
    const below = w.filter((x) => x < today).length;
    const pct = below / (w.length - 1);
    const avg = w.reduce((a, b) => a + b, 0) / w.length;
    const min = Math.min.apply(null, w), max = Math.max.apply(null, w);
    let target = round(percentile(w, 0.75), 2);
    if (target <= today) target = round(today + 0.05, 2);
    const verdict = pct >= 0.7 ? 'good' : pct >= 0.4 ? 'fair' : 'low';
    const usual = usualAmount(st);
    const gainAtTarget = Math.max(0, (target - today) * usual);
    log(st, 'agent', 'tool:rate_insight', 0, 'auto', `better than ${Math.round(pct * 100)}% of last 30 days, verdict=${verdict}`);
    return {
      today: round(today, 4), avg: round(avg, 4), min: round(min, 4), max: round(max, 4),
      percentile: round(pct, 2), verdict, target, usual, gainAtTarget: round(gainAtTarget, 0),
      series: w.map((x) => round(x, 4)), date: isoDate(currentDate(st))
    };
  }

  function quote(st, provider, amountSAR) {
    const mid = rateNow(st);
    const rate = mid * (1 - provider.margin);
    const received = Math.max(0, amountSAR - provider.fee) * rate;
    const hiddenCostSAR = Math.max(0, amountSAR - provider.fee) * provider.margin;
    return {
      id: provider.id, name: provider.name, speed: provider.speed, payout: provider.payout,
      fee: provider.fee, margin: provider.margin, rate: round(rate, 4),
      received: round(received, 0), totalCostSAR: round(provider.fee + hiddenCostSAR, 2)
    };
  }

  // ---- Model 3: provider learner (contextual Thompson sampling / Bayesian linear regression)
  function transferContext(st, amountSAR, recipient) {
    const dow = currentDate(st).getUTCDay();
    return { amount: amountSAR, wallet: recipient && recipient.wallet ? 1 : 0, weekend: dow === 5 || dow === 6 ? 1 : 0, big: amountSAR > 2000 ? 1 : 0 };
  }
  const ctxVec = (c) => [1, c.wallet, c.weekend, Math.log(c.amount / 1000), c.big];
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  function solve(A, b) { // Gaussian elimination with partial pivoting (5x5)
    const n = b.length, M = A.map((row, i) => row.slice().concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      const tmp = M[c]; M[c] = M[p]; M[p] = tmp;
      for (let r = c + 1; r < n; r++) {
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
      x[r] = s / M[r][r];
    }
    return x;
  }
  function learnedCut(st, id, c) {
    if (!st.providerModel || !st.providerModel[id]) return 0;
    const m = st.providerModel[id];
    return Math.max(0, dot(solve(m.A, m.b), ctxVec(c)));
  }
  // The simulated "truth" behind receipts (unknown to the agent; used only to simulate what arrives).
  function hiddenDeduction(id, c, mid) {
    if (id === 'gulfx') return 0.009 * c.big + 0.004 * c.wallet;     // compliance charge on large amounts
    if (id === 'quick') return 0.001 + 0.011 * c.wallet;             // cash-out fee on mobile wallets
    if (id === 'desh') return 0.007 * c.weekend;                     // weekend intermediary rate
    if (id === 'bank') return 350 / (c.amount * mid);               // receiving bank takes ~BDT 350
    return 0;
  }
  function gaussian(st) {
    const u = Math.max(1e-12, st.rnd()), v = st.rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function learnFromReceipt(st, id, c, observedCut) {
    if (!st.providerModel || !st.providerModel[id]) return;
    const m = st.providerModel[id], x = ctxVec(c), w = 1 / Math.pow(MODELS.provider.noise, 2);
    for (let i = 0; i < 5; i++) {
      m.b[i] += x[i] * observedCut * w;
      for (let j = 0; j < 5; j++) m.A[i][j] += x[i] * x[j] * w;
    }
    st.receipts += 1;
  }

  function toolCompareProviders(st, amountSAR, recipient) {
    const c = transferContext(st, amountSAR, recipient);
    const rows = PROVIDERS.map((p) => {
      const q = quote(st, p, amountSAR);
      const cut = learnedCut(st, p.id, c);
      return Object.assign(q, { expectedCut: round(cut, 5), expected: round(q.received * (1 - cut), 0) });
    }).sort((a, b) => b.expected - a.expected);
    const best = rows[0], worst = rows[rows.length - 1];
    const bestQuote = rows.slice().sort((a, b) => b.received - a.received)[0];
    const result = {
      amountSAR, mid: round(rateNow(st), 4), rows, best, worst, bestQuote,
      saving: round(best.expected - worst.expected, 0), context: c, receipts: st.receipts, learned: !!st.providerModel
    };
    log(st, 'agent', 'tool:compare_providers', 0, 'auto', `${fmtSAR(amountSAR)}: AI pick ${best.name} (expects ${fmtBDT(best.expected)}), best quote ${bestQuote.name}`);
    return result;
  }

  // Smallest SAR amount that delivers `bdt` with the best provider.
  function sarNeededFor(st, bdt) {
    const mid = rateNow(st);
    let bestNeed = Infinity;
    PROVIDERS.forEach((p) => {
      const need = bdt / (mid * (1 - p.margin)) + p.fee;
      if (need < bestNeed) bestNeed = need;
    });
    return Math.ceil(bestNeed);
  }

  function usualAmount(st) {
    if (!st.transfers.length) return 1000;
    const recent = st.transfers.slice(-5);
    const avg = recent.reduce((a, t) => a + t.amount, 0) / recent.length;
    return Math.round(avg / 100) * 100;
  }

  // ---- Model 2: federated fraud model (10-16-1 network trained with FedAvg across 3 providers)
  const RISKY_MEMO = /\b(deposit|agent|admission|visa|processing fee|urgent|refundable|guarantee|release fee)\b/i;

  function fraudFeatures(st, amountSAR, recipient, memo) {
    const avg = st.transfers.length ? st.transfers.reduce((a, t) => a + t.amount, 0) / st.transfers.length : amountSAR;
    const known = !!recipient.trusted;
    return [
      Math.max(0.15, amountSAR / avg),             // amount_vs_usual
      known ? 0 : 1,                               // new_recipient
      known ? 0 : 0.35,                            // recipient_account_new (unknown for new people: base rate)
      st.session.night ? 1 : 0,                    // night_time
      countToday(st),                              // transfers_last_24h
      st.session.newDevice ? 1 : 0,                // new_device
      RISKY_MEMO.test(memo || '') ? 1 : 0,         // risky_memo_words
      recipient.wallet ? 1 : 0,                    // mobile_wallet_payout
      0,                                           // first_time_customer
      amountSAR % 1000 === 0 ? 1 : 0               // amount_round_thousand
    ];
  }

  function fraudForward(F, z) {
    let out = F.b2;
    for (let j = 0; j < F.b1.length; j++) {
      let s = F.b1[j];
      for (let i = 0; i < z.length; i++) s += z[i] * F.W1[i][j];
      out += Math.tanh(s) * F.W2[j];
    }
    return 1 / (1 + Math.exp(-out));
  }

  function fraudScore(features) {
    if (!MODELS || !MODELS.fraud) return null;
    const F = MODELS.fraud;
    const x = features.slice();
    x[0] = Math.log(x[0]); x[4] = Math.log1p(x[4]);
    const z = x.map((v, i) => (v - F.mu[i]) / F.sd[i]);
    const score = fraudForward(F, z);
    // Reason codes by occlusion: how much the score falls when a feature is set to a typical value
    const contrib = z.map((_, i) => {
      const z2 = z.slice(); z2[i] = F.baseline_z[i];
      return { feature: F.features[i], reason: F.reasons[F.features[i]], delta: score - fraudForward(F, z2) };
    }).filter((r) => r.delta > 0.02).sort((a, b) => b.delta - a.delta).slice(0, 3);
    return { score: round(score, 3), reasons: contrib.map((r) => r.reason), thresholds: F.thresholds };
  }

  // ---- Model 1: probabilistic rate forecast (21 linear quantile regressions)
  function forecastFeatures(st, t) {
    const lr = (i) => Math.log(st.rates[i]);
    const w = [];
    for (let i = t - 29; i <= t; i++) w.push(lr(i));
    const mean = w.reduce((a, b) => a + b, 0) / w.length;
    const diffs = [];
    for (let i = t - 9; i <= t; i++) diffs.push(lr(i) - lr(i - 1));
    const dm = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    const vol = Math.sqrt(diffs.reduce((a, b) => a + (b - dm) * (b - dm), 0) / diffs.length);
    const d = dateForIndex(st, t);
    const dow = (d.getUTCDay() + 6) % 7; // Mon = 0, as in pandas
    const dim = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 0)).getUTCDate();
    const f = [lr(t) - lr(t - 1), lr(t) - lr(t - 3), lr(t) - lr(t - 7), lr(t) - mean, vol];
    for (let k = 0; k < 7; k++) f.push(dow === k ? 1 : 0);
    f.push(dim - d.getUTCDate() < 3 ? 1 : 0, d.getUTCDate() <= 3 ? 1 : 0);
    return f;
  }

  function forecastReturns(st, t) {
    const F = MODELS.forecast;
    const z = forecastFeatures(st, t).map((v, i) => (v - F.mu[i]) / F.sd[i]);
    const out = {};
    F.horizons.forEach((h) => {
      out[h] = {};
      F.quantiles.forEach((q) => { const m = F.models[String(h)][String(q)]; out[h][q] = m.b + dot(m.w, z); });
    });
    return out;
  }

  function toolForecast(st) {
    if (!MODELS || !MODELS.forecast || st.dayIndex < 30) return null;
    const F = MODELS.forecast;
    const t = st.dayIndex, today = rateNow(st);
    const ret = forecastReturns(st, t);
    const days = F.horizons.map((h) => ({
      h, date: isoDate(dateForIndex(st, t + h)),
      q10: round(today * Math.exp(ret[h][0.1]), 4), q50: round(today * Math.exp(ret[h][0.5]), 4), q90: round(today * Math.exp(ret[h][0.9]), 4)
    }));
    let k = 0;
    F.horizons.forEach((h, i) => { if (ret[h][0.5] > ret[F.horizons[k]][0.5]) k = i; });
    const hBest = F.horizons[k], r = ret[hBest];
    const wait = r[0.5] > F.policy.min_gain && r[0.1] > F.policy.max_downside_q10;
    // chance the rate ends below today at that horizon, read off the three quantiles
    const pts = [[r[0.1], 0.1], [r[0.5], 0.5], [r[0.9], 0.9]];
    let pWorse;
    if (0 <= pts[0][0]) pWorse = 0.05; else if (0 >= pts[2][0]) pWorse = 0.95;
    else { const seg = 0 < pts[1][0] ? [pts[0], pts[1]] : [pts[1], pts[2]]; pWorse = seg[0][1] + (0 - seg[0][0]) / (seg[1][0] - seg[0][0]) * (seg[1][1] - seg[0][1]); }
    const usual = usualAmount(st);
    const result = {
      today: round(today, 4), date: isoDate(currentDate(st)), days,
      history: st.rates.slice(t - 13, t + 1).map((x) => round(x, 4)),
      decision: { wait, h: hBest, date: isoDate(dateForIndex(st, t + hBest)), expectedRate: days[k].q50, low: days[k].q10, high: days[k].q90,
        gainBDT: round(usual * today * (Math.exp(r[0.5]) - 1), 0), pWorse: round(pWorse, 2), usual }
    };
    log(st, 'agent', 'tool:forecast_rate', 0, 'auto', `${wait ? 'wait ' + hBest + ' day(s)' : 'send now'}; median ${fmtRate(days[k].q50)} (80% ${fmtRate(days[k].q10)}-${fmtRate(days[k].q90)})`);
    return result;
  }

  // ----------------------------------------------------------- policy engine
  function isWatchlisted(name) {
    const n = String(name || '').toLowerCase().trim();
    return WATCHLIST.some((w) => n === w || n.includes(w));
  }

  function sentToday(st) {
    return st.transfers.filter((t) => t.day === st.dayIndex).reduce((a, t) => a + t.amount, 0);
  }
  function countToday(st) {
    return st.transfers.filter((t) => t.day === st.dayIndex).length;
  }

  /**
   * Decide the permission tier for a transfer.
   * Tier 1: one-tap confirm. Tier 2: explicit approval + one-time code. Tier 3: blocked.
   */
  function evaluateTransfer(st, amountSAR, recipient, memo) {
    const s = st.settings;
    const checks = [];
    let tier = 1;
    const block = [];
    const escalate = [];

    const balOk = amountSAR <= st.balances.SAR;
    checks.push({ name: 'Balance', ok: balOk, note: balOk ? `${fmtSAR(st.balances.SAR)} available` : `Only ${fmtSAR(st.balances.SAR)} available` });
    if (!balOk) block.push('Not enough SAR in your wallet.');

    const listed = isWatchlisted(recipient.name);
    checks.push({ name: 'Sanctions screening', ok: !listed, note: listed ? 'Recipient matches a watchlist entry' : 'No match' });
    if (listed) block.push('The recipient matches a sanctions watchlist entry.');

    const today = sentToday(st);
    const capOk = today + amountSAR <= s.dailyCap;
    checks.push({ name: 'Daily limit', ok: capOk, note: `${fmtSAR(today)} sent today of ${fmtSAR(s.dailyCap)}` });
    if (!capOk) block.push(`This would go over your daily limit of ${fmtSAR(s.dailyCap)}.`);

    const trusted = !!recipient.trusted;
    checks.push({ name: 'Known recipient', ok: trusted, warn: !trusted, note: trusted ? recipient.account : 'First transfer to this person' });
    if (!trusted) escalate.push('This is a new recipient.');

    const underLimit = amountSAR <= s.singleLimit;
    checks.push({ name: 'Single-transfer limit', ok: underLimit, warn: !underLimit, note: `Limit ${fmtSAR(s.singleLimit)} without extra approval` });
    if (!underLimit) escalate.push(`The amount is above your ${fmtSAR(s.singleLimit)} single-transfer limit.`);

    const avg = st.transfers.length ? st.transfers.reduce((a, t) => a + t.amount, 0) / st.transfers.length : 0;
    const normal = !avg || amountSAR <= s.anomalyMultiple * avg;
    checks.push({ name: 'Usual pattern', ok: normal, warn: !normal, note: avg ? `Your average is ${fmtSAR(avg)}` : 'No history yet' });
    if (!normal) escalate.push(`The amount is more than ${s.anomalyMultiple}x your usual transfer.`);

    const count = countToday(st);
    const velOk = count < s.velocityMax;
    checks.push({ name: 'Transfer frequency', ok: velOk, warn: !velOk, note: `${count} transfer(s) today` });
    if (!velOk) escalate.push('You have already made several transfers today.');

    // AI fraud score from the federated model
    const fr = fraudScore(fraudFeatures(st, amountSAR, recipient, memo));
    if (fr) {
      const why = fr.reasons.length ? fr.reasons.join(', ') : 'nothing unusual';
      const high = fr.score >= fr.thresholds.block, mid = fr.score >= fr.thresholds.review;
      checks.push({ name: 'AI fraud score', ok: !mid, warn: mid && !high, note: `${fr.score.toFixed(2)} · ${why}`, ai: true });
      if (high) block.push(`The AI fraud model scored this ${fr.score.toFixed(2)} out of 1: ${why}. This matches known scam patterns.`);
      else if (mid) escalate.push(`The AI fraud model scored this ${fr.score.toFixed(2)}: ${why}.`);
    }

    if (block.length) tier = 3;
    else if (escalate.length) tier = 2;
    const decision = tier === 3 ? 'block' : tier === 2 ? 'approve' : 'confirm';
    const reasons = tier === 3 ? block : tier === 2 ? escalate : ['Known recipient, normal amount, within your limits.'];
    return { tier, decision, reasons, checks, fraud: fr };
  }

  // -------------------------------------------------------- language parsing
  const CUR = { sar: 'SAR', sr: 'SAR', riyal: 'SAR', riyals: 'SAR', bdt: 'BDT', tk: 'BDT', taka: 'BDT', usd: 'USD', dollar: 'USD', dollars: 'USD' };

  function parseMoney(text) {
    const t = String(text).replace(/(\d),(?=\d{3}\b)/g, '$1');
    const re = /(\$)?\s?(\d+(?:\.\d+)?)\s?(k\b)?\s?(sar|sr|riyals?|bdt|tk|taka|usd|dollars?)?\b/gi;
    const out = [];
    let m;
    while ((m = re.exec(t))) {
      let v = parseFloat(m[2]);
      if (m[3]) v *= 1000;
      const c = m[1] ? 'USD' : (m[4] ? CUR[m[4].toLowerCase()] : null);
      out.push({ value: v, currency: c });
    }
    return out;
  }

  const DATE_RE_DM = /\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\b/i;
  const DATE_RE_MD = /\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b/i;
  const DATE_RE_IN = /\bin\s+(\d{1,3})\s+days?\b/i;

  function parseDate(text, st) {
    let m = text.match(DATE_RE_IN);
    if (m) return { index: st.dayIndex + parseInt(m[1], 10), match: m[0] };
    m = text.match(/\btomorrow\b/i);
    if (m) return { index: st.dayIndex + 1, match: m[0] };
    let day, mon;
    m = text.match(DATE_RE_DM);
    if (m) { day = parseInt(m[1], 10); mon = MONTHS.indexOf(m[2].toLowerCase()); }
    else {
      m = text.match(DATE_RE_MD);
      if (m) { mon = MONTHS.indexOf(m[1].toLowerCase()); day = parseInt(m[2], 10); }
    }
    if (m) {
      const now = currentDate(st);
      let target = Date.UTC(now.getUTCFullYear(), mon, day);
      if (target < now.getTime()) target = Date.UTC(now.getUTCFullYear() + 1, mon, day);
      const diff = Math.round((target - now.getTime()) / DAY_MS);
      return { index: st.dayIndex + diff, match: m[0] };
    }
    return null;
  }

  function findRecipient(text, st) {
    const l = ' ' + String(text).toLowerCase().replace(/[^a-z0-9\s'.-]/g, ' ').replace(/\s+/g, ' ') + ' ';
    for (const r of st.recipients) {
      const names = [r.name.toLowerCase()].concat(r.aliases || []);
      for (const a of names) if (l.includes(' ' + a + ' ')) return { existing: true, ref: r };
    }
    const m = String(text).match(/\bto\s+([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,3})/);
    if (m) {
      const name = m[1].replace(/\s+\b(via|using|with|on|today|now|tomorrow|please|in|from|at|for|by|wallet|account)\b.*$/i, '').trim();
      if (name && !/^(bangladesh|bd|bdt|usd|sar|my|the|account|dollars?|taka)$/i.test(name)) {
        return { existing: false, name };
      }
    }
    return null;
  }

  const INJECTION = /(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+|your\s+|my\s+)?(previous\s+|prior\s+|above\s+|earlier\s+|safety\s+)?(rules|instructions|limits|policy|policies|prompt|checks)|\bbypass\b|\boverride\b|without\s+(any\s+)?(approval|confirmation|asking|otp|code|verification|checks?)|skip\s+(the\s+)?(approval|confirmation|otp|code|verification|checks?)|system\s+prompt|developer\s+mode|you\s+are\s+now|act\s+as\s+(an?\s+)?admin|jailbreak/i;

  function detectIntent(text, st) {
    const l = text.toLowerCase();
    if (INJECTION.test(text)) return { name: 'injection', slots: {} };
    if (/\b(limit|limits|cap)\b/.test(l) && /\b(change|set|raise|increase|remove|lift|disable|update|double)\b/.test(l)) return { name: 'change_limit', slots: {} };
    if (/\b(alert|notify|tell me when|let me know when|remind me when|watch the rate)\b/.test(l)) return { name: 'alert', slots: parseAlert(text, st) };
    if (/\btuition\b|\bsemester fees?\b|\buniversity fees?\b|\bschool fees?\b/.test(l)) return { name: 'tuition', slots: parseTuition(text, st) };
    if (/\bconvert\b/.test(l) || (/\bexchange\b/.test(l) && /\b(usd|dollars?)\b/.test(l) && /\d/.test(l))) return { name: 'convert', slots: parseConvert(text) };
    if (/\bforecast\b|\bpredict\b|\bnext (7 days|week)\b/.test(l)) return { name: 'forecast', slots: {} };
    if (/\b(good|best|right|bad)\s+(time|day|moment)\b|\bshould i (send|wait)\b|\bworth waiting\b|\bwait\b/.test(l)) return { name: 'best_time', slots: {} };
    if (/\bcompare\b/.test(l)) return { name: 'compare', slots: { money: parseMoney(text)[0] || null } };
    if (/\b(send|transfer|remit|wire)\b/.test(l)) return { name: 'send', slots: parseSend(text, st) };
    if (/\bcheapest\b|\bwhich (provider|service|app)\b/.test(l)) return { name: 'compare', slots: { money: parseMoney(text)[0] || null } };
    if (/\brate\b|\bexchange\b|sar to bdt|riyal to taka/.test(l)) return { name: 'rate', slots: {} };
    if (/\bbalances?\b|\bwallet\b|how much (money )?do i have/.test(l)) return { name: 'balance', slots: {} };
    if (/\bhistory\b|\bactivity\b|\blog\b|what did you do/.test(l)) return { name: 'activity', slots: {} };
    if (/\bcredit\b/.test(l)) return { name: 'credit', slots: {} };
    if (/\bhelp\b|what can you do|^hi\b|^hello\b|^salam/.test(l)) return { name: 'help', slots: {} };
    return { name: 'unknown', slots: {} };
  }

  function parseSend(text, st) {
    const monies = parseMoney(text);
    const m = monies.find((x) => x.currency) || monies[0] || null;
    return {
      amount: m ? m.value : null,
      currency: m && m.currency ? m.currency : 'SAR',
      recipient: findRecipient(text, st)
    };
  }

  function parseAlert(text, st) {
    const nums = parseMoney(text).map((x) => x.value).filter((v) => v >= 20 && v <= 50);
    const below = /\b(below|under|drops?|falls?|less than)\b/i.test(text);
    return { target: nums.length ? nums[nums.length - 1] : null, direction: below ? 'below' : 'above' };
  }

  function parseTuition(text, st) {
    const d = parseDate(text, st);
    const stripped = d ? text.replace(d.match, ' ') : text;
    const monies = parseMoney(stripped);
    const m = monies.find((x) => x.currency) || monies[0] || null;
    let payee = 'University tuition office';
    const pm = stripped.match(/\bto\s+(?:the\s+)?([A-Za-z][A-Za-z .'&-]{2,60}?)(?:\s+(?:on|by|in|before)\b|[,.]|$)/i);
    if (pm) payee = pm[1].trim();
    return { amount: m ? m.value : null, currency: m && m.currency ? m.currency : 'SAR', dueIndex: d ? d.index : null, payee };
  }

  function parseConvert(text) {
    const monies = parseMoney(text);
    const m = monies[0] || null;
    if (!m) return { amount: null };
    const from = m.currency === 'USD' ? 'USD' : 'SAR';
    return { amount: m.value, from, to: from === 'USD' ? 'SAR' : 'USD' };
  }

  // -------------------------------------------------------------- messages
  const say = (text, chips) => ({ type: 'text', text, chips: chips || [] });
  const card = (kind, data) => ({ type: 'card', kind, data });

  function planCard(steps) { return { type: 'plan', steps }; }

  function newPending(st, kind, tier, args, title, summary, reasons) {
    st.seq += 0;
    const id = 'ap-' + (Object.keys(st.pending).length + st.log.length + 1) + '-' + Math.floor(st.rnd() * 1e6);
    const p = { id, kind, tier, args, attempts: 0, createdDay: st.dayIndex };
    if (tier === 2) p.otp = String(100000 + Math.floor(st.rnd() * 900000));
    st.pending[id] = p;
    log(st, 'policy', 'approval_requested:' + kind, tier, tier === 2 ? 'approve+code' : 'confirm', summary);
    const msg = { type: 'approval', id, tier, tierName: TIER_NAMES[tier], title, summary, reasons: reasons || [], requiresCode: tier === 2 };
    const out = [msg];
    if (tier === 2) out.unshift({ type: 'sms', code: p.otp, text: `FinPassport code: ${p.otp}. It approves: ${summary}. Never share this code.` });
    return out;
  }

  // --------------------------------------------------------------- flows
  function rateFlow(st) {
    toolGetRate(st);
    const ins = rateInsight(st);
    const pct = Math.round(ins.percentile * 100);
    return [
      card('insight', ins),
      say(`Today 1 SAR = ${fmtRate(ins.today)} BDT. That is better than ${pct}% of the last 30 days (30-day average ${fmtRate(ins.avg)}).`,
        ['Is today a good day to send money home?', 'Compare providers for 1000 SAR'])
    ];
  }

  function bestTimeFlow(st) {
    toolGetRate(st);
    const ins = rateInsight(st);
    const pct = Math.round(ins.percentile * 100);
    const fc = toolForecast(st);
    const out = [planCard([
      { label: 'Check today\'s rate', tier: 0 },
      { label: 'Compare with the last 30 days', tier: 0 },
      { label: 'Forecast the next 7 days (AI model)', tier: 0 },
      { label: 'Recommend: send now or wait', tier: 0 }
    ]), card('insight', ins)];
    if (fc) out.push(card('forecast', fc));
    const pctText = `Today's rate (${fmtRate(ins.today)}) is better than ${pct}% of the last 30 days.`;
    if (fc && fc.decision.wait) {
      const d = fc.decision;
      out.push(say(
        `${pctText} My forecast model expects about ${fmtRate(d.expectedRate)} in ${d.h} day(s) (80% range ${fmtRate(d.low)} to ${fmtRate(d.high)}). ` +
        `Waiting could add about ${fmtBDT(d.gainBDT)} on your usual ${fmtSAR(d.usual)}, with about a ${Math.round(d.pWorse * 100)}% chance the rate ends lower. ` +
        `I can watch it and tell you when 1 SAR reaches ${fmtRate(ins.target)} BDT.`,
        [`Alert me when 1 SAR is above ${fmtRate(ins.target)} BDT`, `Send ${ins.usual} SAR to Ammu anyway`]));
    } else if (fc) {
      out.push(say(`${pctText} My forecast model does not expect a meaningful rise in the next 7 days, so sending today is reasonable.`,
        [`Send ${ins.usual} SAR to Ammu`, `Compare providers for ${ins.usual} SAR`]));
    } else if (ins.verdict === 'good') {
      out.push(say(`Yes. ${pctText} It is a good day to send.`, [`Send ${ins.usual} SAR to Ammu`, `Compare providers for ${ins.usual} SAR`]));
    } else {
      out.push(say(`${pctText} If you can wait, I can tell you when 1 SAR reaches ${fmtRate(ins.target)} BDT.`,
        [`Alert me when 1 SAR is above ${fmtRate(ins.target)} BDT`, `Send ${ins.usual} SAR to Ammu anyway`]));
    }
    return out;
  }

  function forecastFlow(st) {
    toolGetRate(st);
    const fc = toolForecast(st);
    if (!fc) return rateFlow(st);
    const d = fc.decision;
    return [card('forecast', fc), say(d.wait
      ? `Forecast: about ${fmtRate(d.expectedRate)} in ${d.h} day(s), 80% range ${fmtRate(d.low)} to ${fmtRate(d.high)}. Waiting is likely worth about ${fmtBDT(d.gainBDT)} on ${fmtSAR(d.usual)}, but there is about a ${Math.round(d.pWorse * 100)}% chance it ends lower. This is a probability, not a promise.`
      : `Forecast: no meaningful rise expected in the next 7 days. Sending today is reasonable.`,
      ['Is today a good day to send money home?'])];
  }

  function aiPickNote(cmp) {
    if (!cmp.learned || cmp.best.id === cmp.bestQuote.id) return '';
    const bq = cmp.bestQuote;
    return ` The best quote is ${bq.name}, but ${cmp.receipts.toLocaleString('en-US')} past receipts show about ${(bq.expectedCut * 100).toFixed(1)}% of it does not arrive in this situation, so ${cmp.best.name} should deliver more.`;
  }

  function compareFlow(st, slots) {
    let amount = 1000;
    if (slots.money) {
      amount = slots.money.currency === 'BDT' ? sarNeededFor(st, slots.money.value)
        : slots.money.currency === 'USD' ? slots.money.value * SAR_PER_USD : slots.money.value;
    }
    toolGetRate(st);
    const cmp = toolCompareProviders(st, amount, st.recipients[0]);
    const zeroFee = cmp.rows.find((r) => r.fee === 0);
    let note = `For ${fmtSAR(amount)}, I expect ${cmp.best.name} to deliver the most: about ${fmtBDT(cmp.best.expected)}. That is ${fmtBDT(cmp.saving)} more than ${cmp.worst.name}.`;
    note += aiPickNote(cmp);
    if (zeroFee && zeroFee.id !== cmp.best.id) note += ` ${zeroFee.name} has no fee, but its exchange rate is worse.`;
    return [card('compare', cmp), say(note, [`Send ${amount} SAR to Ammu`])];
  }

  function sendFlow(st, slots, memo) {
    if (!slots.amount) {
      return [say('How much would you like to send? For example: "Send 1000 SAR to Ammu".', ['Send 1000 SAR to Ammu', 'Send 30000 taka to Ammu'])];
    }
    if (!slots.recipient) {
      return [say(`Who should receive it? Your saved recipient is Ammu (Mother, ${st.recipients[0].account}).`, [`Send ${slots.amount} ${slots.currency} to Ammu`])];
    }
    let amountSAR = slots.amount;
    if (slots.currency === 'BDT') amountSAR = sarNeededFor(st, slots.amount);
    else if (slots.currency === 'USD') amountSAR = slots.amount * SAR_PER_USD;
    amountSAR = round(amountSAR, 2);

    const recipient = slots.recipient.existing
      ? slots.recipient.ref
      : { name: slots.recipient.name, account: 'New account (not yet verified)', trusted: false, relation: 'New recipient', wallet: /\bwallet\b/i.test(memo || '') };

    const out = [];
    out.push(planCard([
      { label: 'Check today\'s rate and forecast', tier: 0 },
      { label: 'Rank 4 providers by what should arrive (AI)', tier: 0 },
      { label: 'Risk, compliance and AI fraud checks', tier: 0 },
      { label: 'Ask for your approval', tier: null },
      { label: 'Send with the best provider', tier: null }
    ]));
    toolGetRate(st);
    const ins = rateInsight(st);
    const fc = toolForecast(st);
    const cmp = toolCompareProviders(st, amountSAR, recipient);
    out.push(card('compare', cmp));
    const pick = aiPickNote(cmp);
    if (pick) out.push(say(pick.trim()));

    const ev = evaluateTransfer(st, amountSAR, recipient, memo);
    log(st, 'policy', 'risk_check:transfer', ev.tier, ev.decision, ev.reasons.join(' '));
    if (ev.fraud) log(st, 'agent', 'model:fraud_score', 0, ev.fraud.score >= ev.fraud.thresholds.review ? 'flag' : 'clear', `score ${ev.fraud.score}${ev.fraud.reasons.length ? ': ' + ev.fraud.reasons.join(', ') : ''}`);
    out.push(card('risk', { tier: ev.tier, tierName: TIER_NAMES[ev.tier], checks: ev.checks }));

    if (ev.tier === 3) {
      out.push({ type: 'blocked', title: 'I stopped this transfer', reasons: ev.reasons.concat(['No money has moved.']) });
      return out;
    }

    if (fc && fc.decision.wait) {
      out.push(say(`Tip: my forecast expects about ${fmtRate(fc.decision.expectedRate)} in ${fc.decision.h} day(s). Waiting could add about ${fmtBDT(amountSAR * (fc.decision.expectedRate - fc.today))}, with about a ${Math.round(fc.decision.pWorse * 100)}% chance it ends lower. You can still send now.`));
    }

    const summary = `Send ${fmtSAR(amountSAR)} to ${recipient.name} via ${cmp.best.name}. They should receive about ${fmtBDT(cmp.best.expected)}.`;
    return out.concat(newPending(st, 'transfer', ev.tier,
      { amountSAR, memo: memo || '', recipient: { name: recipient.name, account: recipient.account, trusted: !!recipient.trusted, existing: !!slots.recipient.existing, wallet: !!recipient.wallet } },
      ev.tier === 2 ? 'Your approval and code are needed' : 'Please confirm',
      summary, ev.reasons));
  }

  function executeTransfer(st, args) {
    const recipient = args.recipient;
    // Re-check at execution time: balances, limits or rates may have changed since the quote.
    const ev = evaluateTransfer(st, args.amountSAR, recipient, args.memo);
    if (ev.tier === 3) {
      log(st, 'policy', 'recheck:transfer', 3, 'block', ev.reasons.join(' '));
      return [{ type: 'blocked', title: 'Stopped at the final check', reasons: ev.reasons.concat(['No money has moved.']) }];
    }
    const cmp = toolCompareProviders(st, args.amountSAR, recipient);
    const best = cmp.best;
    const bank = cmp.rows.find((r) => r.id === 'bank');
    // Simulated receipt: what actually arrives, including the provider's hidden deduction
    const c = cmp.context;
    const cut = Math.max(0, hiddenDeduction(best.id, c, rateNow(st)) + gaussian(st) * (MODELS && MODELS.provider ? MODELS.provider.noise : 0));
    const delivered = round(best.received * (1 - cut), 0);
    learnFromReceipt(st, best.id, c, cut);
    st.balances.SAR = round(st.balances.SAR - args.amountSAR, 2);
    const ref = 'FP-' + isoDate(currentDate(st)).replace(/-/g, '').slice(2) + '-' + String(st.transfers.length + 1).padStart(3, '0');
    st.transfers.push({ day: st.dayIndex, amount: args.amountSAR, recipient: recipient.name, provider: best.name, received: delivered, ref });
    if (!recipient.existing && !st.recipients.some((r) => r.name.toLowerCase() === recipient.name.toLowerCase())) {
      st.recipients.push({ id: 'r' + (st.recipients.length + 1), name: recipient.name, relation: 'Verified by you', country: 'Bangladesh', account: 'Verified account', aliases: [recipient.name.toLowerCase()], trusted: true, wallet: !!recipient.wallet });
    }
    log(st, 'agent', 'tool:send_transfer', null, 'executed', `${ref}: ${fmtSAR(args.amountSAR)} to ${recipient.name} via ${best.name}, quoted ${fmtBDT(best.received)}, arrived ${fmtBDT(delivered)}`);
    log(st, 'agent', 'model:learn_from_receipt', 0, 'updated', `${best.name}: ${(cut * 100).toFixed(2)}% hidden deduction observed; ${st.receipts} receipts learned`);
    const vsBank = bank ? round(delivered - bank.expected, 0) : 0;
    return [
      card('receipt', {
        ref, date: isoDate(currentDate(st)), recipient: recipient.name, amountSAR: args.amountSAR, provider: best.name,
        fee: best.fee, rate: best.rate, quoted: best.received, received: delivered, hidden: round(best.received - delivered, 0),
        aiExpected: best.expected, speed: best.speed, vsBank, balanceAfter: st.balances.SAR, receipts: st.receipts
      }),
      say(`Done. ${recipient.name} receives ${fmtBDT(delivered)} (${best.speed.toLowerCase()}). Compared with a bank wire, that is about ${fmtBDT(vsBank)} more. I added this receipt to what I have learned about ${best.name}.`,
        ['Show my balances', 'What did you do?'])
    ];
  }

  function alertFlow(st, slots) {
    const ins = rateInsight(st);
    const target = slots.target || ins.target;
    const direction = slots.direction || 'above';
    const a = { id: 'al' + (st.alerts.length + 1), target: round(target, 2), direction, createdDay: st.dayIndex, status: 'active' };
    st.alerts.push(a);
    log(st, 'agent', 'tool:set_rate_alert', 0, 'auto', `alert when 1 SAR ${direction} ${fmtRate(target)} BDT`);
    return [
      card('alert', a),
      say(`Alert set. I will tell you when 1 SAR goes ${direction} ${fmtRate(target)} BDT (today ${fmtRate(ins.today)}). An alert never moves money on its own; you decide when to send.`,
        ['Next day']),
    ];
  }

  function tuitionFlow(st, slots) {
    if (!slots.amount) return [say('How much is the tuition, and when is it due? For example: "Pay tuition 2500 SAR on 5 Nov".', ['Pay tuition 2500 SAR on 5 Nov'])];
    if (slots.dueIndex == null) return [say(`When is the ${fmtSAR(slots.amount)} tuition due?`, [`Pay tuition ${slots.amount} SAR on 5 Nov`])];
    let amountSAR = slots.currency === 'USD' ? slots.amount * SAR_PER_USD : slots.amount;
    amountSAR = round(amountSAR, 2);
    const due = dateForIndex(st, slots.dueIndex);
    const days = slots.dueIndex - st.dayIndex;
    const committed = st.scheduled.filter((s) => s.status === 'scheduled').reduce((a, s) => a + s.amount, 0);
    const enough = st.balances.SAR - committed >= amountSAR;
    const reasons = ['Scheduled payments to a new payee always need your approval and a code.'];
    if (!enough) reasons.push(`Heads up: your free balance (${fmtSAR(st.balances.SAR - committed)}) is below this amount today.`);
    const out = [planCard([
      { label: 'Read amount, payee and due date', tier: 0 },
      { label: 'Check your balance and other scheduled payments', tier: 0 },
      { label: 'Ask for approval of a payment mandate', tier: 2 },
      { label: 'Remind you 3 days before, then pay on the due date', tier: null }
    ])];
    log(st, 'policy', 'risk_check:schedule_payment', 2, 'approve', reasons.join(' '));
    const summary = `Pay ${fmtSAR(amountSAR)} to ${slots.payee} on ${fmtDate(due)} (in ${days} days).`;
    return out.concat(newPending(st, 'tuition', 2, { amountSAR, payee: slots.payee, dueIndex: slots.dueIndex }, 'Approve a scheduled payment', summary, reasons));
  }

  function executeTuition(st, args) {
    const s = { id: 'sp' + (st.scheduled.length + 1), payee: args.payee, amount: args.amountSAR, dueIndex: args.dueIndex, due: isoDate(dateForIndex(st, args.dueIndex)), status: 'scheduled', reminded: false };
    st.scheduled.push(s);
    log(st, 'agent', 'tool:schedule_payment', 2, 'scheduled', `${fmtSAR(s.amount)} to ${s.payee} on ${s.due}`);
    return [card('schedule', s), say(`Scheduled. I will remind you 3 days before ${fmtDate(dateForIndex(st, s.dueIndex))} and pay ${s.payee} on the due date if your balance allows.`, ['Show my balances'])];
  }

  function convertFlow(st, slots) {
    if (!slots.amount) return [say('How much should I convert? For example: "Convert 500 SAR to USD".', ['Convert 500 SAR to USD'])];
    const fromBal = st.balances[slots.from];
    const outAmt = slots.from === 'SAR' ? (slots.amount / SAR_PER_USD) * (1 - FX_SPREAD) : slots.amount * SAR_PER_USD * (1 - FX_SPREAD);
    if (slots.amount > fromBal) {
      log(st, 'policy', 'risk_check:convert', 3, 'block', 'insufficient balance');
      return [{ type: 'blocked', title: 'Not enough balance', reasons: [`You have ${slots.from === 'SAR' ? fmtSAR(fromBal) : fmtUSD(fromBal)}.`] }];
    }
    log(st, 'policy', 'risk_check:convert', 1, 'confirm', 'own-account conversion');
    const summary = `Convert ${slots.from === 'SAR' ? fmtSAR(slots.amount) : fmtUSD(slots.amount)} to ${slots.to === 'USD' ? fmtUSD(outAmt) : fmtSAR(outAmt)} (0.5% spread).`;
    return newPending(st, 'convert', 1, { amount: slots.amount, from: slots.from, to: slots.to, out: round(outAmt, 2) }, 'Please confirm', summary, ['Moving money between your own currency balances.']);
  }

  function executeConvert(st, args) {
    st.balances[args.from] = round(st.balances[args.from] - args.amount, 2);
    st.balances[args.to] = round(st.balances[args.to] + args.out, 2);
    log(st, 'agent', 'tool:convert_currency', null, 'executed', `${args.amount} ${args.from} -> ${args.out} ${args.to}`);
    return [card('balances', balancesData(st)), say(`Converted. You now have ${fmtSAR(st.balances.SAR)} and ${fmtUSD(st.balances.USD)}.`)];
  }

  function balancesData(st) {
    const committed = st.scheduled.filter((s) => s.status === 'scheduled').reduce((a, s) => a + s.amount, 0);
    return { balances: Object.assign({}, st.balances), committed, recipients: st.recipients.map((r) => ({ name: r.name, relation: r.relation, account: r.account, trusted: r.trusted })) };
  }

  function activitySummary(st) {
    const items = st.log.filter((e) => e.actor !== 'user').slice(-6).map((e) => `${e.action} (${e.decision})`);
    return [say(items.length ? 'My latest actions: ' + items.join('; ') + '. The full audit log is in the Activity tab.' : 'I have not done anything yet.')];
  }

  const HELP_CHIPS = ['Is today a good day to send money home?', 'Compare providers for 1000 SAR', 'Send 1000 SAR to Ammu', 'Pay tuition 2500 SAR on 5 Nov'];

  function handle(st, raw) {
    const text = String(raw || '').trim();
    if (!text) return [say('Type a request, or tap one of the suggestions.', HELP_CHIPS)];
    log(st, 'user', 'message', null, 'received', text.slice(0, 160));
    if (/^next day$/i.test(text)) return advanceDay(st);
    const intent = detectIntent(text, st);
    log(st, 'agent', 'understand:' + intent.name, null, 'parsed', JSON.stringify(intent.slots, (k, v) => (k === 'ref' ? v.name : v)));
    switch (intent.name) {
      case 'injection':
        log(st, 'policy', 'guard:prompt_injection', 3, 'block', 'message asked to skip safety rules');
        return [{ type: 'blocked', title: 'I can\'t do that', reasons: ['This message asks me to skip safety rules or approvals. I never do that, whoever asks.', 'No money has moved.'] }];
      case 'change_limit':
        log(st, 'policy', 'guard:change_limits', 3, 'block', 'limits can only be changed by the user in Settings');
        return [{ type: 'blocked', title: 'Limits are yours to change', reasons: ['I can\'t change your spending limits. Only you can, in Settings, after signing in again.'] }];
      case 'rate': return rateFlow(st);
      case 'best_time': return bestTimeFlow(st);
      case 'compare': return compareFlow(st, intent.slots);
      case 'send': return sendFlow(st, intent.slots, text);
      case 'forecast': return forecastFlow(st);
      case 'alert': return alertFlow(st, intent.slots);
      case 'tuition': return tuitionFlow(st, intent.slots);
      case 'convert': return convertFlow(st, intent.slots);
      case 'balance':
        log(st, 'agent', 'tool:get_balances', 0, 'auto', '');
        return [card('balances', balancesData(st))];
      case 'activity': return activitySummary(st);
      case 'credit':
        return [say('New in the country with no credit history? Start small: keep your salary or stipend in one local account, pay every bill on time, and ask your bank about a secured card with a low limit. I can remind you of due dates so nothing is late.')];
      case 'help':
        return [say('I help you move money between Saudi Arabia and Bangladesh. I can check the rate, tell you if today is a good day to send, compare providers, send money with your approval, set rate alerts, and schedule tuition payments.', HELP_CHIPS)];
      default:
        return [say('Sorry, I did not understand that. Here are things I can do:', HELP_CHIPS)];
    }
  }

  function resolve(st, id, approve, code) {
    const p = st.pending[id];
    if (!p) return [say('This request has expired or was already handled. Please ask again.')];
    if (!approve) {
      delete st.pending[id];
      log(st, 'user', 'reject:' + p.kind, p.tier, 'rejected', 'user cancelled');
      return [say('Cancelled. Nothing was sent.')];
    }
    if (p.tier === 2) {
      if (String(code || '').trim() !== p.otp) {
        p.attempts += 1;
        log(st, 'policy', 'verify_code:' + p.kind, 2, 'wrong_code', `attempt ${p.attempts}`);
        if (p.attempts >= st.settings.otpAttempts) {
          delete st.pending[id];
          log(st, 'policy', 'verify_code:' + p.kind, 2, 'cancelled', 'too many wrong codes');
          return [{ type: 'blocked', title: 'Request cancelled', reasons: ['The code was wrong 3 times, so I cancelled this request to protect you.'] }];
        }
        return [{ type: 'code_error', id, text: `That code is not right. ${st.settings.otpAttempts - p.attempts} attempt(s) left.` }];
      }
    }
    delete st.pending[id];
    log(st, 'user', 'approve:' + p.kind, p.tier, p.tier === 2 ? 'approved+code' : 'confirmed', '');
    if (p.kind === 'transfer') return executeTransfer(st, p.args);
    if (p.kind === 'tuition') return executeTuition(st, p.args);
    if (p.kind === 'convert') return executeConvert(st, p.args);
    return [];
  }

  function advanceDay(st) {
    const prev = rateNow(st);
    st.dayIndex += 1;
    if (st.dayIndex >= st.rates.length) {
      const last = st.rates[st.rates.length - 1];
      st.rates.push(round(last + (32.5 - last) * 0.15 + (st.rnd() - 0.5) * 0.16, 4));
    }
    const rate = rateNow(st);
    const out = [];
    const expired = Object.keys(st.pending);
    expired.forEach((id) => { delete st.pending[id]; });
    log(st, 'system', 'new_day', null, 'tick', `1 SAR = ${fmtRate(rate)} BDT`);
    const delta = rate - prev;
    out.push({ type: 'day', date: fmtDate(currentDate(st)), rate: round(rate, 4), delta: round(delta, 4) });
    if (expired.length) {
      log(st, 'policy', 'expire_quotes', null, 'expired', `${expired.length} pending approval(s) expired`);
      out.push(say('The rate changed, so pending approvals expired. Please ask again for a fresh quote.'));
    }

    st.alerts.filter((a) => a.status === 'active').forEach((a) => {
      const hit = a.direction === 'above' ? rate >= a.target : rate <= a.target;
      if (hit) {
        a.status = 'triggered';
        a.triggeredDay = st.dayIndex;
        const usual = usualAmount(st);
        const created = st.rates[a.createdDay];
        const extra = Math.max(0, (rate - created) * usual);
        log(st, 'agent', 'alert_triggered', 0, 'notify', `1 SAR = ${fmtRate(rate)} (target ${fmtRate(a.target)})`);
        out.push(say(`Rate alert: 1 SAR = ${fmtRate(rate)} BDT, your target was ${fmtRate(a.target)}. Sending your usual ${fmtSAR(usual)} now would deliver about ${fmtBDT(extra)} more than on the day you set the alert. Want me to prepare it?`,
          [`Send ${usual} SAR to Ammu`]));
      }
    });

    st.scheduled.filter((s) => s.status === 'scheduled').forEach((s) => {
      const left = s.dueIndex - st.dayIndex;
      if (left <= 3 && left > 0 && !s.reminded) {
        s.reminded = true;
        log(st, 'agent', 'reminder', 0, 'notify', `${s.payee} due in ${left} day(s)`);
        out.push(say(`Reminder: ${fmtSAR(s.amount)} to ${s.payee} is due in ${left} day(s).`));
      }
      if (left <= 0) {
        if (st.balances.SAR >= s.amount) {
          st.balances.SAR = round(st.balances.SAR - s.amount, 2);
          s.status = 'paid';
          log(st, 'agent', 'tool:execute_scheduled_payment', 2, 'executed', `${fmtSAR(s.amount)} to ${s.payee} under approved mandate`);
          out.push(say(`Paid ${fmtSAR(s.amount)} to ${s.payee}, as you approved earlier.`));
        } else {
          s.status = 'failed';
          log(st, 'agent', 'tool:execute_scheduled_payment', 2, 'failed', 'insufficient balance');
          out.push({ type: 'blocked', title: 'Scheduled payment not made', reasons: [`Not enough balance to pay ${s.payee}. Please top up and pay manually.`] });
        }
      }
    });
    return out;
  }

  function createAgent(opts) {
    const st = createState(opts);
    log(st, 'system', 'session_start', null, 'start', 'sandbox wallet loaded (simulated data)');
    return {
      state: st,
      handle: (t) => handle(st, t),
      resolve: (id, ok, code) => resolve(st, id, ok, code),
      advanceDay: () => advanceDay(st),
      intro: () => rateFlow(st),
      forecast: () => toolForecast(st),
      setSession: (o) => { Object.assign(st.session, o || {}); log(st, 'user', 'demo_setting', null, 'changed', JSON.stringify(st.session)); },
      models: MODELS,
      insight: () => rateInsight(st),
      balances: () => balancesData(st),
      today: () => fmtDate(currentDate(st)),
      dateFor: (idx) => fmtDate(dateForIndex(st, idx))
    };
  }

  const API = {
    createAgent, createState, detectIntent, fraudScore, fraudFeatures, toolForecast, forecastReturns, learnedCut, hiddenDeduction, MODELS, parseMoney, parseDate, findRecipient, evaluateTransfer,
    toolCompareProviders, rateInsight, sarNeededFor, isWatchlisted, percentile,
    PROVIDERS, DEFAULT_SETTINGS, WATCHLIST, TIER_NAMES, fmtSAR, fmtBDT, fmtUSD, fmtRate
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else root.FinPassport = API;
})(typeof window !== 'undefined' ? window : this);
