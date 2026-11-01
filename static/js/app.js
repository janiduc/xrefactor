/**
 * XRefactor Frontend Application (Vue 3)
 *
 * Two audiences, deliberately kept apart:
 *   RUN      - what happened on one corpus (Results view)
 *   EVIDENCE - how the models score offline (Benchmarks view)
 *
 * They must never share a figure: the offline evaluation report describes a
 * DIFFERENT repository, on a different date, from any live run. Every card
 * therefore renders a provenance strip, and `PROVENANCE_MISMATCH` fires when
 * the two are mixed.
 *
 * Design rule applied throughout: a chart with zero variance is not a chart,
 * it is a sentence. Several visuals here are deliberately statement tiles
 * rather than charts, driven by generic degenerate-data detectors so that any
 * future chart inherits the same protection (see `caveats`).
 */

const { createApp } = Vue;

/* ======================================================================== *
 * Constants
 * ======================================================================== */

/** Fallback taxonomy order, used only until /api/benchmarks responds.
 *  The authoritative order comes from the backend, which derives it from
 *  REFACTORING_TYPE_TO_ID so it cannot drift from the pipeline. */
const PATTERN_ORDER_FALLBACK = [
    'extract_method', 'move_class', 'rename_variable', 'consolidate_duplicate_code',
    'remove_dead_code', 'simplify_condition', 'split_class', 'extract_interface',
    'reduce_coupling', 'improve_naming'
];

/**
 * Call-resolution strategy groups.
 *
 * MUST mirror src/cpg/call_resolver.py: `is_resolved_strategy()` and
 * `OUT_OF_SCOPE_REASONS`. Anything not listed is an ADDRESSABLE gap, which is
 * the safe default - an unknown new strategy shows up as a resolver gap rather
 * than being silently excused as out-of-scope.
 */
const STRATEGY_RESOLVED = [
    'same_file', 'inherited', 'unique_global', 'super_chain',
    'static_class', 'static_class_inherited',
    'qualifier_type', 'qualifier_type_inherited'
];
const STRATEGY_OUT_OF_SCOPE = [
    'external_type', 'external_static_type',
    'generated_accessor', 'generated_logger_field',
    'chained_call', 'lambda_parameter', 'dotted_qualifier'
];

/** Node-type colours are resolved from CSS tokens at render time so the legend
 *  in the template and the graph cannot drift apart (they used to be two
 *  hardcoded copies of the same hex list). */
const NODE_TYPE_TOKENS = {
    class: '--c-cat-1', method: '--c-cat-2', field: '--c-cat-3',
    function: '--c-cat-5', variable: '--c-cat-4', statement: '--c-cat-6'
};

/* ======================================================================== *
 * Palette
 * ======================================================================== */

/** Chart.js draws to a canvas and cannot read CSS variables, so tokens are
 *  resolved to concrete values once per render and re-read on theme change. */
function readPalette() {
    const style = getComputedStyle(document.documentElement);
    const get = (name, fallback) => (style.getPropertyValue(name) || fallback).trim();
    return {
        ink: get('--c-ink', '#333'),
        inkDim: get('--c-ink-dim', '#666'),
        rule: get('--c-rule', '#e0e0e0'),
        brand: get('--c-brand', '#667eea'),
        pass: get('--c-pass', '#2e7d32'),
        fail: get('--c-fail', '#c62828'),
        warn: get('--c-warn', '#ef6c00'),
        na: get('--c-na', '#9e9e9e'),
        ast: get('--c-ast', '#2e7d32'),
        neural: get('--c-neural', '#6a1b9a'),
        cat: [1, 2, 3, 4, 5, 6].map(i => get(`--c-cat-${i}`, '#667eea'))
    };
}

/* ======================================================================== *
 * Chart.js setup
 * ======================================================================== */

if (typeof Chart !== 'undefined') {
    // maintainAspectRatio:false is required by the .chart-canvas-wrap contract
    // in style.css; without it a responsive canvas inside an auto-height parent
    // grows on every re-render.
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.animation.duration = 300;
    Chart.defaults.plugins.legend.display = false;
    Chart.defaults.plugins.tooltip.displayColors = false;

    /** Numeric labels on bars. Essential here: the mined-count chart spans
     *  8..1059, so the small bars are sub-pixel and unreadable without them. */
    Chart.register({
        id: 'valueLabels',
        afterDatasetsDraw(chart, _args, opts) {
            if (!opts || opts.enabled === false) return;
            const { ctx } = chart;
            ctx.save();
            ctx.font = '11px sans-serif';
            ctx.fillStyle = opts.color || '#666';
            chart.data.datasets.forEach((dataset, di) => {
                const meta = chart.getDatasetMeta(di);
                if (meta.hidden) return;
                meta.data.forEach((element, i) => {
                    const raw = dataset.data[i];
                    if (raw === null || raw === undefined) return;
                    const text = opts.format ? opts.format(raw) : String(raw);
                    const horizontal = chart.options.indexAxis === 'y';
                    ctx.textAlign = horizontal ? 'left' : 'center';
                    ctx.textBaseline = horizontal ? 'middle' : 'bottom';
                    ctx.fillText(text,
                        element.x + (horizontal ? 6 : 0),
                        element.y - (horizontal ? 0 : 4));
                });
            });
            ctx.restore();
        }
    });

    /** Dashed reference line, for macro-F1 / overall-pass-rate context. */
    Chart.register({
        id: 'referenceLine',
        afterDraw(chart, _args, opts) {
            if (!opts || !Array.isArray(opts.lines) || !opts.lines.length) return;
            const { ctx, chartArea, scales } = chart;
            const horizontal = chart.options.indexAxis === 'y';
            const scale = horizontal ? scales.x : scales.y;
            if (!scale) return;
            ctx.save();
            opts.lines.forEach(line => {
                const pos = scale.getPixelForValue(line.value);
                ctx.beginPath();
                ctx.setLineDash([4, 4]);
                ctx.strokeStyle = line.color || '#999';
                ctx.lineWidth = 1;
                if (horizontal) {
                    ctx.moveTo(pos, chartArea.top);
                    ctx.lineTo(pos, chartArea.bottom);
                } else {
                    ctx.moveTo(chartArea.left, pos);
                    ctx.lineTo(chartArea.right, pos);
                }
                ctx.stroke();
                if (line.label) {
                    ctx.setLineDash([]);
                    ctx.font = '10px sans-serif';
                    ctx.fillStyle = line.color || '#999';
                    ctx.textAlign = horizontal ? 'center' : 'right';
                    if (horizontal) ctx.fillText(line.label, pos, chartArea.top - 2);
                    else ctx.fillText(line.label, chartArea.right, pos - 3);
                }
            });
            ctx.restore();
        }
    });
}

/* ======================================================================== *
 * Diff (hand-rolled; see plan - a CDN lib is unjustified at this scale)
 * ======================================================================== */

/** Longest-common-subsequence table over two token arrays. */
function lcsOps(a, b) {
    const n = a.length, m = b.length;
    const table = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
    for (let i = n - 1; i >= 0; i--) {
        for (let j = m - 1; j >= 0; j--) {
            table[i][j] = a[i] === b[j]
                ? table[i + 1][j + 1] + 1
                : Math.max(table[i + 1][j], table[i][j + 1]);
        }
    }
    const ops = [];
    let i = 0, j = 0;
    while (i < n && j < m) {
        if (a[i] === b[j]) { ops.push({ op: 'same', value: a[i] }); i++; j++; }
        else if (table[i + 1][j] >= table[i][j + 1]) { ops.push({ op: 'del', value: a[i] }); i++; }
        else { ops.push({ op: 'add', value: b[j] }); j++; }
    }
    while (i < n) { ops.push({ op: 'del', value: a[i++] }); }
    while (j < m) { ops.push({ op: 'add', value: b[j++] }); }
    return ops;
}

/** Split a line into tokens for intra-line highlighting. */
function tokenize(line) {
    return line.split(/(\W)/).filter(t => t !== '');
}

/**
 * Build side-by-side diff rows.
 *
 * For a rename the whole change is one identifier on one line, so line-level
 * diffing alone shows a pair of highlighted lines with no indication of WHAT
 * changed - hence the second LCS pass over tokens within each changed line.
 */
function buildDiff(before, after) {
    const ops = lcsOps((before || '').split('\n'), (after || '').split('\n'));
    const left = [], right = [];
    let li = 1, ri = 1, k = 0;
    while (k < ops.length) {
        if (ops[k].op === 'same') {
            left.push({ no: li++, cls: '', parts: [{ t: ops[k].value }] });
            right.push({ no: ri++, cls: '', parts: [{ t: ops[k].value }] });
            k++;
            continue;
        }
        const dels = [], adds = [];
        while (k < ops.length && ops[k].op === 'del') dels.push(ops[k++].value);
        while (k < ops.length && ops[k].op === 'add') adds.push(ops[k++].value);
        const pairs = Math.max(dels.length, adds.length);
        for (let p = 0; p < pairs; p++) {
            const d = dels[p], a = adds[p];
            if (d !== undefined && a !== undefined) {
                const tops = lcsOps(tokenize(d), tokenize(a));
                left.push({
                    no: li++, cls: 'removed',
                    parts: tops.filter(o => o.op !== 'add')
                        .map(o => ({ t: o.value, hl: o.op === 'del' }))
                });
                right.push({
                    no: ri++, cls: 'added',
                    parts: tops.filter(o => o.op !== 'del')
                        .map(o => ({ t: o.value, hl: o.op === 'add' }))
                });
            } else if (d !== undefined) {
                left.push({ no: li++, cls: 'removed', parts: [{ t: d, hl: true }] });
                right.push({ no: null, cls: '', parts: [{ t: '' }] });
            } else {
                left.push({ no: null, cls: '', parts: [{ t: '' }] });
                right.push({ no: ri++, cls: 'added', parts: [{ t: a, hl: true }] });
            }
        }
    }
    return { left, right };
}

/* ======================================================================== *
 * Degeneracy detection (turns "the output looks bad" into a measurement)
 * ======================================================================== */

function degeneracy(text) {
    const tokens = String(text || '').trim().split(/\s+/).filter(Boolean);
    if (tokens.length < 4) return { ratio: 1, maxRun: 0, degenerate: false, tokens: tokens.length };
    const ratio = new Set(tokens).size / tokens.length;
    let maxRun = 0, run = 1;
    for (let i = 1; i < tokens.length; i++) {
        if (tokens[i] === tokens[i - 1]) { run++; maxRun = Math.max(maxRun, run); }
        else run = 1;
    }
    // Repeated bigrams catch "private final Collection" style cycles that a
    // single-token run length misses.
    let bigramRun = 1, maxBigram = 0;
    for (let i = 2; i < tokens.length - 1; i++) {
        if (tokens[i] === tokens[i - 2] && tokens[i + 1] === tokens[i - 1]) {
            bigramRun++; maxBigram = Math.max(maxBigram, bigramRun);
        } else bigramRun = 1;
    }
    const effectiveRun = Math.max(maxRun, maxBigram);
    return {
        ratio, maxRun: effectiveRun, tokens: tokens.length,
        degenerate: ratio < 0.3 || effectiveRun > 5
    };
}

/* ======================================================================== *
 * Small helpers
 * ======================================================================== */

const pct = (v, digits = 1) => (v === null || v === undefined || Number.isNaN(v))
    ? '--' : `${(v * 100).toFixed(digits)}%`;
const fx = (v, digits = 2) => (v === null || v === undefined || Number.isNaN(v))
    ? '--' : Number(v).toFixed(digits);
const basename = p => String(p || '').split(/[\\/]/).filter(Boolean).pop() || '';
const prettyPattern = p => String(p || '').replace(/_/g, ' ');

/* ======================================================================== *
 * App
 * ======================================================================== */

const app = createApp({
    data() {
        return {
            currentView: 'home',
            resultsTab: 'overview',
            benchTab: 'scorecard',
            theme: 'light',

            status: { connected: false, message: 'Connecting...' },

            pipelineRunning: false,
            pipelineElapsedSeconds: 0,
            pipelineOutput: null,
            pipelineError: null,

            runs: [],
            runId: null,
            lastResults: null,
            evidence: null,
            benchmarks: null,
            loading: { runs: false, run: false, benchmarks: false },

            selectedSuggestion: 0,
            diffUnified: false,
            // Isolated nodes are hidden by default: the graph payload is
            // node-sampled, so most nodes arrive with no surviving edge and
            // the default view would be a field of disconnected dots.
            hideIsolated: true,
            scorecardSort: { key: 'mined', dir: 'desc' },
            openCaveat: null,
            toasts: [],

            // Keyed registry so any number of charts can be tracked and torn
            // down per tab; this used to be a fixed three-key object.
            charts: {},
            network: null,

            pipelineInput: {
                dataDirectory: '../Data/ErikHage/MicroBreweryModel',
                device: 'cpu',
                outputDirectory: './outputs',
                configPath: './configs/config.yaml'
            },

            config: {
                cpg: { language: 'java', include_data_flow: true, include_control_flow: true, include_call_graph: true },
                gnn: { model_type: 'gat', hidden_dims: '256,256', output_dim: 128, num_heads: 8, num_layers: 3, dropout: 0.1 },
                transformer: { model_type: 'microsoft/codebert-base', hidden_size: 768, num_layers: 6, num_attention_heads: 12 }
            }
        };
    },

    mounted() {
        const saved = localStorage.getItem('xrefactor_theme');
        if (saved === 'dark' || saved === 'light') this.applyTheme(saved);

        this.checkHealth();
        this.checkPipelineStatus();
        this.loadRuns();
        setInterval(() => this.checkHealth(), 30000);
        setInterval(() => this.checkPipelineStatus(), 5000);
    },

    beforeUnmount() {
        Object.keys(this.charts).forEach(id => this.charts[id] && this.charts[id].destroy());
        if (this.network) this.network.destroy();
    },

    computed: {
        /* ---------------- run data ---------------- */

        /**
         * SINGLE SOURCE OF TRUTH for the suggestions path.
         *
         * The template used to reach for `raw.results.stages...`, but
         * `normalizePipelineResults` already unwraps `payload.results` before
         * assigning `raw`, so that path was always undefined and the
         * refactored-code panel had never rendered - while the confidence chart
         * beside it worked, because it used the correct path. Two copies of one
         * path is what let them disagree, so there is now only one.
         */
        suggestions() {
            return this.lastResults?.raw?.stages?.transformer?.refactoring_suggestions || [];
        },

        cpgStats() { return this.lastResults?.stats?.cpg || {}; },

        graphData() {
            return this.lastResults?.raw?.stages?.cpg?.graph
                || this.lastResults?.raw?.graph
                || { nodes: [], edges: [], truncated: false, total_nodes: 0, shown_nodes: 0 };
        },

        callCoverage() { return this.cpgStats.call_coverage || null; },

        /** Coverage segments. A labelled part-to-whole bar, not a donut: three
         *  categories where one dominates reads badly as a pie. */
        coverageSegments() {
            const c = this.callCoverage;
            if (!c || !c.invocations) return [];
            const p = readPalette();
            return [
                { key: 'resolved', label: 'Resolved', value: c.resolved, color: p.pass },
                { key: 'out_of_scope', label: 'Out of scope', value: c.out_of_scope, color: p.na },
                { key: 'addressable', label: 'Addressable gap', value: c.addressable_unresolved, color: p.warn }
            ].filter(s => s.value > 0)
                .map(s => ({ ...s, share: s.value / c.invocations }));
        },

        /** Strategy tally grouped into the three semantic buckets, with the
         *  `inherits_*` keys excluded (they are inheritance bookkeeping, not
         *  call-site outcomes, and including them breaks the partition). */
        strategyRows() {
            const raw = this.cpgStats.call_resolution_strategies || {};
            const p = readPalette();
            const rows = Object.keys(raw)
                .filter(k => !k.startsWith('inherits_'))
                .map(k => {
                    let group = 'addressable', color = p.warn;
                    if (STRATEGY_RESOLVED.includes(k)) { group = 'resolved'; color = p.pass; }
                    else if (STRATEGY_OUT_OF_SCOPE.includes(k)) { group = 'out_of_scope'; color = p.na; }
                    return { key: k, label: prettyPattern(k), value: raw[k], group, color };
                })
                .sort((a, b) => b.value - a.value);
            return rows;
        },

        inheritsBookkeeping() {
            const raw = this.cpgStats.call_resolution_strategies || {};
            return Object.keys(raw).filter(k => k.startsWith('inherits_'))
                .map(k => ({ key: k, label: prettyPattern(k), value: raw[k] }));
        },

        astNeuralSplit() {
            const ast = this.suggestions.filter(s => s.generation_path === 'ast').length;
            return { ast, neural: this.suggestions.length - ast, total: this.suggestions.length };
        },

        /** True when the confidence head gives every suggestion the same score,
         *  which is why confidence is reported as a statement, not a chart. */
        rawConfidenceUniform() {
            const values = this.suggestions.map(s => s.raw_confidence).filter(v => v !== undefined);
            return values.length > 1 && new Set(values).size === 1;
        },

        maxCardConfidence() {
            const values = this.evidenceCards.map(c => c.confidence_score).filter(v => v !== undefined);
            return values.length ? Math.max(...values) : null;
        },

        /** Metrics rows. Dispersion columns are suppressed when every metric has
         *  count===1, because min=max=mean=median and stdev=0 then describe a
         *  distribution that does not exist. */
        metricsRows() {
            const metrics = this.lastResults?.raw?.metrics || {};
            return Object.keys(metrics).map(name => ({
                name,
                count: metrics[name].count ?? 0,
                mean: metrics[name].mean ?? 0,
                median: metrics[name].median ?? 0,
                min: metrics[name].min ?? 0,
                max: metrics[name].max ?? 0,
                stdev: metrics[name].stdev ?? 0
            }));
        },

        metricsHaveDispersion() {
            return this.metricsRows.some(r => r.count > 1);
        },

        smells() { return this.lastResults?.raw?.stages?.cpg?.smells || null; },
        funnel() { return this.lastResults?.raw?.stages?.transformer?.funnel || null; },

        currentSuggestion() {
            return this.suggestions[this.selectedSuggestion] || null;
        },

        /** Change summary built from `edit`. For a rename this IS the primary
         *  artifact - the raw diff is one identifier and looks like nothing
         *  happened. */
        changeSummary() {
            const edit = this.currentSuggestion?.edit;
            if (!edit) return null;
            const bits = [];
            if (edit.old_name && edit.new_name) bits.push(`${edit.old_name} → ${edit.new_name}`);
            if (edit.target_kind) bits.push(`(${edit.target_kind})`);
            if (edit.occurrences_renamed != null) bits.push(`· ${edit.occurrences_renamed} occurrences`);
            if (edit.removed) bits.push(`· removed ${prettyPattern(edit.removed)}`);
            if (edit.statements_removed != null) bits.push(`· ${edit.statements_removed} statements`);
            if (edit.javadoc_param_updated) bits.push('· javadoc @param updated');
            if (edit.name_source) bits.push(`· name source: ${edit.name_source}`);
            return bits.join(' ');
        },

        currentDiff() {
            const s = this.currentSuggestion;
            if (!s) return { left: [], right: [] };
            return buildDiff(s.original_code, s.refactored_code);
        },

        /** The three-way pattern signal chain. `pattern_agreement` is computed
         *  backend-side as rules-vs-model ONLY, so it is labelled as such -
         *  it is true when no smell was detected, which is not "the signals
         *  agree". */
        signalChain() {
            const s = this.currentSuggestion;
            if (!s) return null;
            return {
                detected: s.detected_smell_pattern,
                learned: s.learned_pattern,
                predicted: s.predicted_pattern,
                applied: s.refactoring_type,
                source: s.pattern_source,
                rulesVsModel: s.pattern_agreement,
                appliedMatchesPredicted: s.refactoring_type === s.predicted_pattern
            };
        },

        /* ---------------- evidence ---------------- */

        evidenceCards() { return this.evidence?.evidence_cards || []; },

        riskSegments() {
            const risk = this.evidence?.risk_assessment;
            if (!risk) return [];
            const total = Object.values(risk).reduce((a, b) => a + b, 0) || 1;
            const p = readPalette();
            return [
                { key: 'low_risk', label: 'Low risk', value: risk.low_risk || 0, color: p.pass },
                { key: 'medium_risk', label: 'Medium risk', value: risk.medium_risk || 0, color: p.warn },
                { key: 'high_risk', label: 'High risk', value: risk.high_risk || 0, color: p.fail }
            ].map(s => ({ ...s, share: s.value / total }));
        },

        /* ---------------- benchmarks ---------------- */

        patternOrder() {
            return this.benchmarks?.taxonomy?.order || PATTERN_ORDER_FALLBACK;
        },

        astSupported() {
            return this.benchmarks?.taxonomy?.ast_supported || [];
        },

        classifierReport() { return this.benchmarks?.eval?.classifier || null; },
        generationReport() { return this.benchmarks?.eval?.generation || null; },
        xaiReport() { return this.benchmarks?.eval?.xai || null; },
        traceReport() { return this.benchmarks?.eval?.trace || null; },
        miningReport() { return this.benchmarks?.mining || null; },

        /**
         * Majority-class comparator, computed from class supports.
         *
         * There is NO baseline model in this repo. `outputs/baseline_prefix_*.json`
         * is byte-identical to the evaluation report and referenced by no code,
         * so quoting it as a baseline would be a fabrication. Predicting the
         * most-supported class on every example is a real, defensible floor.
         */
        majorityBaseline() {
            const perClass = this.classifierReport?.per_class;
            const testSize = this.classifierReport?.test_size;
            if (!perClass || !testSize) return null;
            const supports = Object.values(perClass).map(c => c.support || 0);
            const best = Math.max(...supports);
            const accuracy = best / testSize;
            // Macro-F1 of a constant predictor: one class gets F1 = 2p/(p+1)
            // with p = accuracy (recall 1), the other nine get 0.
            const f1 = (2 * accuracy) / (accuracy + 1);
            return { accuracy, macro_f1: f1 / supports.length, best_support: best, test_size: testSize };
        },

        /**
         * The per-pattern spine.
         *
         * A table rather than a chart because the five metrics span
         * incompatible units (counts 8..1059 against rates 0..1) and because
         * only a table can carry three DISTINCT empty states: 0.00 (measured
         * and failed), n/a (never evaluated), and no deterministic
         * implementation. Encoding three empty states in bar geometry is
         * impossible.
         */
        scorecardRows() {
            const mined = this.miningReport?.per_class_distribution || {};
            const perClass = this.classifierReport?.per_class || {};
            const perPattern = this.generationReport?.per_pattern || {};
            const minedMax = Math.max(1, ...Object.values(mined));
            const minedTotal = Object.values(mined).reduce((a, b) => a + b, 0) || 1;
            const runCounts = this.suggestions.reduce((acc, s) => {
                acc[s.refactoring_type] = (acc[s.refactoring_type] || 0) + 1;
                return acc;
            }, {});

            const rows = this.patternOrder.map((name, idx) => {
                const clf = perClass[name] || null;
                const gen = Object.prototype.hasOwnProperty.call(perPattern, name) ? perPattern[name] : null;
                const minedCount = mined[name] ?? null;
                const hasAst = this.astSupported.includes(name);

                const f1 = clf ? clf.f1 : null;
                const support = clf ? clf.support : null;
                const passRate = gen ? gen.structural_pass_rate : null;

                let verdict = 'WORKING';
                if (gen === null) verdict = 'NOT_EVALUATED';
                else if (f1 === 0 && support > 0) verdict = 'UNTRAINED';
                else if ((minedCount !== null && minedCount < 50) || (support !== null && support < 5)) verdict = 'THIN';
                else if (f1 > 0 && passRate === 0) verdict = 'WEAK';

                return {
                    id: idx, name, label: prettyPattern(name), hasAst,
                    mined: minedCount,
                    minedShare: minedCount === null ? null : minedCount / minedMax,
                    minedPct: minedCount === null ? null : minedCount / minedTotal,
                    precision: clf ? clf.precision : null,
                    recall: clf ? clf.recall : null,
                    f1, support,
                    passRate,
                    genCount: gen ? gen.count : null,
                    bleu: gen ? gen.bleu4 : null,
                    editRatio: gen ? gen.edit_distance_ratio : null,
                    verdict,
                    runCount: runCounts[name] || 0,
                    thinMined: minedCount !== null && minedCount < 50,
                    thinSupport: support !== null && support < 5,
                    thinGen: gen !== null && gen.count < 5
                };
            });

            const { key, dir } = this.scorecardSort;
            const sign = dir === 'desc' ? -1 : 1;
            return rows.slice().sort((a, b) => {
                const av = a[key], bv = b[key];
                if (av === bv) return a.id - b.id;
                if (av === null || av === undefined) return 1;
                if (bv === null || bv === undefined) return -1;
                return av > bv ? sign : -sign;
            });
        },

        neverPredicted() {
            return this.scorecardRows.filter(r => r.f1 === 0 && r.support > 0);
        },

        neverPredictedExamples() {
            return this.neverPredicted.reduce((a, r) => a + (r.support || 0), 0);
        },

        notEvaluated() {
            return this.scorecardRows.filter(r => r.passRate === null);
        },

        verdictTally() {
            return this.scorecardRows.reduce((acc, r) => {
                acc[r.verdict] = (acc[r.verdict] || 0) + 1;
                return acc;
            }, {});
        },

        generationSamples() {
            const samples = this.benchmarks?.models?.transformer?.sample_generations || [];
            return samples.map((s, i) => ({ ...s, idx: i, deg: degeneracy(s.generated) }));
        },

        /* ---------------- provenance ---------------- */

        runProvenance() {
            if (!this.lastResults) return null;
            const r = this.lastResults.raw || {};
            const stats = this.cpgStats;
            return {
                kind: 'run',
                label: r.repo_label || (r.data_directory ? basename(r.data_directory) : 'unknown corpus'),
                timestamp: this.lastResults.timestamp,
                detail: `${stats.total_files ?? '?'} files · ${stats.total_nodes ?? '?'} nodes`,
                missingLabel: !r.repo_label
            };
        },

        evalProvenance() {
            const p = this.benchmarks?.eval?.provenance;
            if (!p) return null;
            return {
                kind: 'eval',
                label: p.repo ? basename(p.repo) : 'unknown corpus',
                timestamp: p.mtime_iso,
                detail: [p.sample_size ? `n=${p.sample_size} generated` : null,
                p.test_size ? `n=${p.test_size} classified` : null]
                    .filter(Boolean).join(' · '),
                source: p.source_file
            };
        },

        /* ---------------- caveat engine ---------------- */

        /**
         * Computed limitations register.
         *
         * Deliberately GENERIC where possible (zero variance, all-zero metric,
         * single-category distribution, unplottable range) so that every future
         * chart inherits degenerate-case protection instead of needing a
         * one-off special case. Each record carries the trigger it fired on, so
         * a reader can verify the limitation is measured rather than asserted.
         */
        caveats() {
            const out = [];
            const add = (id, severity, scope, title, message, value, trigger) =>
                out.push({ id, severity, scope, title, message, value, trigger });

            const stats = this.cpgStats;
            const cards = this.evidenceCards;

            /* --- run scope --- */
            if (this.lastResults) {
                if (this.metricsRows.length && !this.metricsHaveDispersion) {
                    add('SINGLE_RUN_NO_DISPERSION', 'warning', 'run',
                        'Single run - no dispersion',
                        `All ${this.metricsRows.length} metrics have count=1, so min, max, median and stdev carry no information. Dispersion columns are hidden until multiple runs are aggregated.`,
                        `${this.metricsRows.length} metrics`,
                        'max(metrics[*].count) === 1');
                }
                if (stats.call_resolution_rate !== undefined && stats.call_resolution_rate < 0.5) {
                    const c = this.callCoverage;
                    add('LOW_CALL_RESOLUTION', 'warning', 'run',
                        'Call resolution is bounded by the corpus',
                        c ? `${pct(c.rate)} of all ${c.invocations} invocations resolved, but ${pct(c.rate_in_corpus)} of the ${c.resolved + c.addressable_unresolved} a source-only resolver could decide. The ${c.out_of_scope} out-of-scope misses are library calls, Lombok-generated members and receivers needing type inference.`
                            : `Resolution rate ${pct(stats.call_resolution_rate)}.`,
                        c ? `${pct(c.rate)} / ${pct(c.rate_in_corpus)}` : pct(stats.call_resolution_rate),
                        'call_resolution_rate < 0.5');
                }
                const strategySum = this.strategyRows.reduce((a, r) => a + r.value, 0);
                if (this.callCoverage && strategySum && strategySum !== this.callCoverage.invocations) {
                    add('STRATEGY_PARTITION_MISMATCH', 'warning', 'run',
                        'Strategy tally does not partition invocations',
                        `Call strategies sum to ${strategySum} but call_coverage.invocations is ${this.callCoverage.invocations}. A strategy key was probably added backend-side without updating the grouping map in app.js.`,
                        `${strategySum} vs ${this.callCoverage.invocations}`,
                        'sum(nonInheritsStrategies) !== call_coverage.invocations');
                }
                Object.keys(stats).filter(k => k.endsWith('_edges') && stats[k] === 0).forEach(k => {
                    add(`EMPTY_EDGE_CLASS_${k}`, 'info', 'run',
                        'Empty edge class',
                        `${prettyPattern(k.replace('_edges', ''))} produced no edges on this corpus, so any visual of it is empty by construction rather than by failure.`,
                        '0', `statistics.${k} === 0`);
                });
                if (this.graphData.truncated) {
                    const nodes = this.graphData.nodes?.length || 0;
                    const edges = this.graphData.edges?.length || 0;
                    const typesShown = new Set((this.graphData.edges || []).map(e => e.type)).size;
                    const typesTotal = Object.keys(stats.edge_types || {}).length;
                    if (typesShown < typesTotal || (nodes && edges / nodes < 0.5)) {
                        add('GRAPH_EDGE_SAMPLING_ARTIFACT', 'blocker', 'run',
                            'Graph view is a sampling artifact',
                            `Showing ${nodes} of ${stats.total_nodes} nodes but only ${edges} of ${stats.total_edges} edges, covering ${typesShown} of ${typesTotal} edge types. Nodes are selected first by type priority and edges kept only when BOTH endpoints survive, so statement and variable nodes - which carry the data and control flow - are cut first. Read the edge breakdown from the statistics, not from this graph.`,
                            `${edges}/${stats.total_edges} edges`,
                            'graph.truncated && (edgeTypesShown < edgeTypesTotal || edges/nodes < 0.5)');
                    }
                }
                if (this.suggestions.length) {
                    const unverified = this.suggestions.filter(s => !s.verification).length;
                    if (unverified) {
                        add('NEURAL_PATH_UNVERIFIED', 'warning', 'run',
                            'Neural suggestions are not compile-verified',
                            `${unverified} of ${this.suggestions.length} suggestions came from the neural path, which emits text rather than an AST edit, so javac verification does not apply. They are also not conditioned on the requested pattern.`,
                            `${unverified}/${this.suggestions.length}`,
                            'verification === null');
                    }
                    const disagree = this.suggestions.filter(s => s.pattern_agreement === false).length;
                    if (disagree) {
                        add('PATTERN_DISAGREEMENT', 'warning', 'run',
                            'Rule-based and learned signals disagree',
                            `${disagree} of ${this.suggestions.length} suggestions have the smell detector and the GNN predicting different patterns.`,
                            `${disagree}/${this.suggestions.length}`,
                            'pattern_agreement === false');
                    }
                    const opportunistic = this.suggestions.filter(s => s.pattern_source === 'ast_opportunity').length;
                    if (opportunistic) {
                        add('APPLIED_PATTERN_NOT_PREDICTED', 'blocker', 'run',
                            'Applied pattern came from neither predictor',
                            `${opportunistic} of ${this.suggestions.length} suggestions were produced by the AST engine's opportunistic scan after both the smell detector and the classifier were discarded. For those, neither ML stage contributed the decision.`,
                            `${opportunistic}/${this.suggestions.length}`,
                            "pattern_source === 'ast_opportunity'");
                    }
                    const assigned = this.suggestions.filter(s => s.confidence_kind === 'deterministic' && s.confidence === 1.0).length;
                    if (assigned) {
                        add('AST_CONFIDENCE_NOT_A_SCORE', 'info', 'run',
                            'Deterministic confidence is assigned, not predicted',
                            `${assigned} suggestions carry confidence 1.0 by construction because they are compile-verified. It is not a model score and must not be averaged with model confidences.`,
                            `${assigned} suggestions`,
                            "confidence_kind === 'deterministic' && confidence === 1.0");
                    }
                    const rawConf = this.suggestions.map(s => s.raw_confidence).filter(v => v !== undefined);
                    if (rawConf.length > 1 && new Set(rawConf).size === 1) {
                        add('ZERO_VARIANCE_SERIES_raw_confidence', 'warning', 'run',
                            'Model confidence has zero variance',
                            `raw_confidence is ${fx(rawConf[0], 4)} on all ${rawConf.length} suggestions - the confidence head is not discriminating between them, so charting it would be a row of identical bars.`,
                            fx(rawConf[0], 4),
                            'new Set(raw_confidence).size === 1');
                    }
                }
                if (this.smells && this.suggestions.length &&
                    this.suggestions.length / (this.smells.total || 1) < 0.1) {
                    add('SMELL_FUNNEL_LOSS', 'warning', 'run',
                        'Suggestion count is a configured cap, not a filter',
                        `${this.suggestions.length} suggestions were emitted from ${this.smells.total} detected smells (${pct(this.suggestions.length / this.smells.total, 1)}). That ratio is set by refactor.max_suggestions and refactor.candidate_pool, not by how many problems were judged real.`,
                        `${this.suggestions.length}/${this.smells.total}`,
                        'suggestionCount / smellTotal < 0.1');
                }
            }

            /* --- evidence scope --- */
            if (cards.length) {
                const confs = cards.map(c => c.confidence_score);
                const maxConf = Math.max(...confs);
                if (maxConf < 0.7) {
                    add('CONFIDENCE_FORMULA_CAPPED', 'blocker', 'eval',
                        'Evidence confidence cannot reach its own thresholds',
                        `Highest card confidence is ${fx(maxConf, 4)}, below the 0.70 "Recommended" threshold and the 0.90 "Highly recommended" threshold the generator itself uses. confidence = (improvement_delta + avg_solution_score)/2 with improvement_delta = mean(solution - problem); with the problem score pinned, this reduces to avg_solution_score - 0.1866. Every card is therefore "Optional" and counted high-risk by construction, not by assessment.`,
                        fx(maxConf, 4),
                        'max(cards.confidence_score) < 0.7');
                }
                if (confs.length > 1 && new Set(confs).size === 1) {
                    add('ZERO_VARIANCE_SERIES_card_confidence', 'warning', 'eval',
                        'Evidence confidence has zero variance',
                        `All ${confs.length} cards share confidence ${fx(confs[0], 4)}. The XAI problem detector returns a constant for every node of the same type, so the cards cannot be ranked.`,
                        fx(confs[0], 4),
                        'new Set(cards.confidence_score).size === 1');
                }
                if (cards.every(c => c.supporting_metrics?.attention_concentration === 1.0)) {
                    add('XAI_SOLUTION_EMBEDDING_PROXY', 'blocker', 'eval',
                        'improvement_delta is not a before/after delta',
                        'attention_concentration is exactly 1.0 on every card, the fingerprint of the pipeline passing the ORIGINAL node embedding into both the original and refactored slots of the causal module. No embedding of the generated code exists at inference time, so improvement_delta compares a vector with itself.',
                        '1.0 on all cards',
                        'cards.every(c => attention_concentration === 1.0)');
                }
                const ids = cards.map(c => c.evidence_id);
                if (new Set(ids).size !== ids.length) {
                    add('DUPLICATE_EVIDENCE_ID', 'warning', 'eval',
                        'Duplicate evidence ids',
                        `${ids.length - new Set(ids).size} card id(s) are repeated, so ids cannot key a list. Rows are keyed by index instead.`,
                        `${ids.length - new Set(ids).size} duplicate(s)`,
                        'new Set(ids).size !== ids.length');
                }
                if (cards.some(c => (c.explanation || '').trimEnd().endsWith('...'))) {
                    add('TRUNCATED_EXPLANATION', 'info', 'eval',
                        'Explanations are truncated',
                        'Card explanations are cut at the configured xai.explanation_length and end mid-sentence, so the text shown is not the full rationale.',
                        `${cards.filter(c => (c.explanation || '').trimEnd().endsWith('...')).length} cards`,
                        "explanation.endsWith('...')");
                }
                const summary = this.evidence?.summary || '';
                if (/Recommended refactorings:\s*(might|consider|Apply)\b/.test(summary)) {
                    add('SUMMARY_PARSE_BUG', 'warning', 'eval',
                        'Report summary is mis-parsed backend-side',
                        'The generated summary extracts the third whitespace-separated word of each recommendation string, which yields an adverb rather than a refactoring name. A summary rebuilt from the cards is shown instead.',
                        summary.split('\n').pop(),
                        "/Recommended refactorings: (might|consider|Apply)/");
                }
                const risk = this.evidence?.risk_assessment;
                if (risk && Object.values(risk).filter(v => v > 0).length === 1) {
                    add('SINGLE_CATEGORY_DISTRIBUTION', 'info', 'eval',
                        'Risk distribution has one non-empty bucket',
                        'Every card falls in a single risk bucket, so a part-to-whole chart would be one filled shape. Shown as a labelled bar with its cause instead.',
                        Object.entries(risk).filter(([, v]) => v > 0).map(([k, v]) => `${k}=${v}`).join(', '),
                        'values.filter(v => v > 0).length === 1');
                }
            }

            /* --- benchmark scope --- */
            if (this.classifierReport) {
                if (this.neverPredicted.length) {
                    add('CLASS_NEVER_PREDICTED', 'blocker', 'eval',
                        'Classes the classifier never predicts',
                        `${this.neverPredicted.length} of ${this.scorecardRows.length} classes score F1 = 0 with non-zero support: ${this.neverPredicted.map(r => `${r.name} (${r.support})`).join(', ')}. That is ${this.neverPredictedExamples} of ${this.classifierReport.test_size} test examples (${pct(this.neverPredictedExamples / this.classifierReport.test_size)}) the model cannot reach.`,
                        `${this.neverPredicted.length}/${this.scorecardRows.length} classes`,
                        'f1 === 0 && support > 0');
                }
                add('NO_BASELINE_MODEL', 'info', 'eval',
                    'No baseline model exists in this repository',
                    this.majorityBaseline
                        ? `The comparator shown is computed from class supports: always predicting the most-supported class gives ${pct(this.majorityBaseline.accuracy)} accuracy. outputs/baseline_prefix_*.json is byte-identical to the evaluation report and is not a baseline.`
                        : 'No trained baseline is available for comparison.',
                    this.majorityBaseline ? pct(this.majorityBaseline.accuracy) : '--',
                    'always (informational)');
            }
            if (this.generationReport) {
                const per = this.generationReport.per_pattern || {};
                const exact = Object.values(per).map(v => v.exact_match);
                if (exact.length && exact.every(v => v === 0)) {
                    add('DEGENERATE_METRIC_ALL_ZERO', 'warning', 'eval',
                        'Exact match is zero everywhere',
                        `exact_match is 0.0 across all ${exact.length} evaluated patterns, so no chart of it can carry information. Reported as a single statement instead.`,
                        `0 of ${this.generationReport.eval_sample_size} samples`,
                        'values.every(v => v === 0)');
                }
                const bleus = Object.values(per).map(v => v.bleu4).filter(v => v > 0);
                if (bleus.length > 1 && Math.max(...bleus) / Math.min(...bleus) > 1e6) {
                    add('UNPLOTTABLE_DYNAMIC_RANGE', 'info', 'eval',
                        'BLEU-4 spans too many orders of magnitude to plot',
                        `BLEU-4 ranges from ${Math.min(...bleus).toExponential(1)} to ${Math.max(...bleus).toExponential(1)}. A linear axis renders every bar invisible and a log axis cannot show the true zeros, so it is reported as a numeric column only.`,
                        `ratio ${(Math.max(...bleus) / Math.min(...bleus)).toExponential(1)}`,
                        'max(nonZero)/min(nonZero) > 1e6');
                }
                if (this.notEvaluated.length) {
                    add('PATTERN_NOT_EVALUATED', 'warning', 'eval',
                        'Patterns never evaluated for generation',
                        `${this.notEvaluated.map(r => r.name).join(', ')} are absent from the generation metrics entirely - not scored zero, never measured.`,
                        `${this.notEvaluated.length} patterns`,
                        'pattern in taxonomy && !(pattern in per_pattern)');
                }
                const thin = this.scorecardRows.filter(r => r.thinGen);
                if (thin.length) {
                    add('THIN_EVAL_SAMPLE', 'warning', 'eval',
                        'Generation sample is too thin per pattern',
                        `${thin.length} of ${Object.keys(per).length} evaluated patterns have n < 5 (${thin.map(r => `${r.name}=${r.genCount}`).join(', ')}), out of a total eval sample of ${this.generationReport.eval_sample_size}. Those rates are not measurements.`,
                        `${thin.length} patterns`,
                        'per_pattern[p].count < 5');
                }
            }
            if (this.xaiReport && cards.length) {
                const observed = cards[0]?.supporting_metrics?.avg_problem_score;
                const randomBaseline = this.xaiReport.problem_detector_mean_score_on_random_nodes;
                if (observed !== undefined && randomBaseline !== undefined &&
                    Math.abs(observed - randomBaseline) < 0.01) {
                    add('XAI_SCORE_AT_RANDOM_BASELINE', 'blocker', 'eval',
                        'Problem detector scores at its own random baseline',
                        `On the loaded run the detector scores ${fx(observed, 4)} for every method node, which is the offline evaluation's mean score on RANDOM, unflagged nodes (${fx(randomBaseline, 4)}). Its reported ${pct(this.xaiReport.problem_detector_test_accuracy)} accuracy does not transfer here: it is not discriminating on this corpus. Note this compares a baseline constant against an observed constant, across two different corpora.`,
                        `${fx(observed, 4)} vs ${fx(randomBaseline, 4)}`,
                        'abs(observed - mean_score_on_random_nodes) < 0.01');
                }
            }
            if (this.miningReport) {
                const dist = this.miningReport.per_class_distribution || {};
                const thin = Object.entries(dist).filter(([, v]) => v < 50);
                if (thin.length) {
                    add('THIN_TRAINING_CLASS', 'warning', 'dataset',
                        'Thin training classes',
                        `${thin.map(([k, v]) => `${k} (${v})`).join(', ')} have fewer than 50 mined examples, too few for the classifier to learn.`,
                        `${thin.length} classes`,
                        'per_class_distribution[p] < 50');
                }
                const repos = this.miningReport.repos || {};
                const matched = Object.entries(repos)
                    .filter(([, v]) => v && v.stage_b_matched)
                    .map(([k, v]) => [k, v.stage_b_matched]);
                const total = this.miningReport.totals?.stage_b_matched;
                if (matched.length && total) {
                    const [topRepo, topCount] = matched.reduce((a, b) => (b[1] > a[1] ? b : a));
                    if (topCount / total > 0.5) {
                        add('CORPUS_CONCENTRATION', 'warning', 'dataset',
                            'Dataset is dominated by one repository',
                            `${topRepo} contributes ${topCount} of ${total} labelled examples (${pct(topCount / total)}), so the dataset is effectively one codebase and the learned patterns may be repo-specific.`,
                            pct(topCount / total),
                            'max(repo.stage_b_matched)/totals.stage_b_matched > 0.5');
                    }
                }
            }

            /* --- cross-scope --- */
            if (this.runProvenance && this.evalProvenance) {
                const runRepo = basename(this.lastResults?.raw?.data_directory || this.runProvenance.label);
                const evalRepo = basename(this.benchmarks?.eval?.provenance?.repo || '');
                if (runRepo && evalRepo && runRepo !== evalRepo) {
                    add('PROVENANCE_MISMATCH', 'blocker', 'global',
                        'Benchmarks and the loaded run are different experiments',
                        `The offline evaluation ran on ${evalRepo}; the loaded run analysed ${runRepo}. Figures from the two must not be read as one experiment, and no chart combines them.`,
                        `${evalRepo} vs ${runRepo}`,
                        'basename(eval.repo) !== basename(run.data_directory)');
                }
            }

            const rank = { blocker: 0, warning: 1, info: 2 };
            return out.sort((a, b) => rank[a.severity] - rank[b.severity]);
        },

        caveatCount() { return this.caveats.length; },
        blockerCount() { return this.caveats.filter(c => c.severity === 'blocker').length; },

        /** A summary rebuilt from the cards, replacing the mis-parsed backend one. */
        rebuiltSummary() {
            const cards = this.evidenceCards;
            if (!cards.length) return null;
            const byProblem = cards.reduce((acc, c) => {
                acc[c.problem_type] = (acc[c.problem_type] || 0) + 1;
                return acc;
            }, {});
            const byPattern = this.suggestions.reduce((acc, s) => {
                acc[s.refactoring_type] = (acc[s.refactoring_type] || 0) + 1;
                return acc;
            }, {});
            return {
                count: cards.length,
                problems: Object.entries(byProblem).map(([k, v]) => `${prettyPattern(k)} (${v})`).join(', '),
                patterns: Object.entries(byPattern).map(([k, v]) => `${prettyPattern(k)} (${v})`).join(', ')
            };
        }
    },

    watch: {
        lastResults() { this.selectedSuggestion = 0; this.scheduleRender(); },
        evidence() { this.scheduleRender(); },
        benchmarks() { this.scheduleRender(); },
        currentView(view) {
            this.destroyAllCharts();
            if (view === 'benchmarks' && !this.benchmarks) this.loadBenchmarks();
            this.scheduleRender();
        },
        resultsTab() { this.destroyAllCharts(); this.scheduleRender(); },
        hideIsolated() { this.$nextTick(() => this.renderGraphNetwork()); },
        benchTab() { this.destroyAllCharts(); this.scheduleRender(); }
    },

    methods: {
        /* ---------------- formatting (exposed to template) ---------------- */
        pct, fx, basename, prettyPattern,

        /* ---------------- theme ---------------- */
        applyTheme(theme) {
            this.theme = theme;
            document.documentElement.setAttribute('data-theme', theme);
            try { localStorage.setItem('xrefactor_theme', theme); } catch (e) { /* private mode */ }
        },

        toggleTheme() {
            this.applyTheme(this.theme === 'dark' ? 'light' : 'dark');
            // Chart.js cannot read CSS variables, so every chart must be rebuilt
            // against the new palette.
            this.destroyAllCharts();
            this.scheduleRender();
        },

        /* ---------------- toasts ---------------- */
        notify(message, type = 'info') {
            const id = Date.now() + Math.random();
            this.toasts.push({ id, message, type });
            setTimeout(() => {
                this.toasts = this.toasts.filter(t => t.id !== id);
            }, 4500);
        },

        /* ---------------- backend ---------------- */
        async checkHealth() {
            try {
                await axios.get('/api/health', { timeout: 5000 });
                this.status.connected = true;
                this.status.message = 'Connected to backend';
            } catch (error) {
                this.status.connected = false;
                this.status.message = 'Backend unreachable';
            }
        },

        async checkPipelineStatus() {
            try {
                const { data } = await axios.get('/api/pipeline/status', { timeout: 5000 });
                if (data.running) {
                    this.pipelineRunning = true;
                    this.pipelineElapsedSeconds = data.elapsed_seconds || 0;
                } else if (this.pipelineRunning && this.pipelineElapsedSeconds > 0) {
                    this.pipelineRunning = false;
                    this.pipelineElapsedSeconds = 0;
                    this.loadRuns();
                }
            } catch (error) {
                console.debug('Pipeline status check failed:', error.message);
            }
        },

        async loadRuns() {
            this.loading.runs = true;
            try {
                const { data } = await axios.get('/api/runs');
                this.runs = data.runs || [];
                if (!this.lastResults && this.runs.length) await this.loadRun(this.runs[0].id);
            } catch (error) {
                console.error('Could not list runs:', error.message);
            } finally {
                this.loading.runs = false;
            }
        },

        async loadRun(runId) {
            if (!runId) return;
            this.loading.run = true;
            try {
                const { data } = await axios.get(`/api/runs/${runId}`);
                this.runId = data.id;
                this.lastResults = this.normalizePipelineResults({ results: data.results });
                await this.loadEvidence(data.id);
            } catch (error) {
                this.notify(`Could not load run ${runId}: ${error.message}`, 'error');
            } finally {
                this.loading.run = false;
            }
        },

        async loadEvidence(runId) {
            this.evidence = null;
            try {
                const { data } = await axios.get(`/api/runs/${runId}/evidence`);
                this.evidence = data.report;
            } catch (error) {
                // Runs predating the evidence file legitimately have none.
                console.debug(`No evidence cards for ${runId}`);
            }
        },

        async loadBenchmarks() {
            this.loading.benchmarks = true;
            try {
                const { data } = await axios.get('/api/benchmarks');
                this.benchmarks = data;
            } catch (error) {
                this.notify(`Could not load benchmarks: ${error.message}`, 'error');
            } finally {
                this.loading.benchmarks = false;
            }
        },

        async runFullPipeline() {
            if (!this.pipelineInput.dataDirectory) {
                this.notify('Please specify a data directory', 'error');
                return;
            }
            this.pipelineRunning = true;
            this.pipelineError = null;
            this.pipelineOutput = null;
            try {
                const { data } = await axios.post('/api/pipeline/run', {
                    data_directory: this.pipelineInput.dataDirectory,
                    config_path: this.pipelineInput.configPath,
                    device: this.pipelineInput.device,
                    output_directory: this.pipelineInput.outputDirectory
                }, { timeout: 600000 });
                this.pipelineOutput = data;
                this.lastResults = this.normalizePipelineResults(data);
                await this.loadRuns();
                if (data.results?.timestamp) await this.loadEvidence(data.results.timestamp);
                this.notify('Pipeline completed successfully', 'success');
                this.currentView = 'results';
            } catch (error) {
                this.pipelineError = error.response?.data?.message || error.message;
                this.notify(`Pipeline failed: ${this.pipelineError}`, 'error');
            } finally {
                this.pipelineRunning = false;
                this.pipelineElapsedSeconds = 0;
            }
        },

        async runStage(stageName) {
            if (stageName === 'cpg_construction' && !this.pipelineInput.dataDirectory) {
                this.notify('Please specify a data directory', 'error');
                return;
            }
            this.pipelineRunning = true;
            this.pipelineError = null;
            this.pipelineOutput = null;
            try {
                const { data } = await axios.post(`/api/pipeline/stage/${stageName}`, {
                    data_directory: this.pipelineInput.dataDirectory,
                    config_path: this.pipelineInput.configPath,
                    device: this.pipelineInput.device
                });
                this.pipelineOutput = data;
                if (stageName === 'cpg_construction') {
                    this.lastResults = this.normalizePipelineResults({
                        results: data, timestamp: new Date().toISOString()
                    });
                }
                this.notify(`Stage ${stageName} completed`, 'success');
            } catch (error) {
                this.pipelineError = error.response?.data?.message || error.message;
                this.notify(`Stage failed: ${this.pipelineError}`, 'error');
            } finally {
                this.pipelineRunning = false;
            }
        },

        saveConfiguration() {
            try {
                localStorage.setItem('xrefactor_config', JSON.stringify(this.config));
                this.notify('Configuration saved locally', 'success');
            } catch (error) {
                this.notify('Failed to save configuration', 'error');
            }
        },

        loadConfiguration() {
            try {
                const saved = localStorage.getItem('xrefactor_config');
                if (saved) { this.config = JSON.parse(saved); this.notify('Configuration loaded', 'success'); }
                else this.notify('No saved configuration found', 'info');
            } catch (error) {
                this.notify('Failed to load configuration', 'error');
            }
        },

        /**
         * Normalize a pipeline payload for the UI.
         *
         * `stats.cpg` SPREADS the whole statistics object before pinning the few
         * keys the view depends on. It used to hand-copy a fixed subset, which
         * silently dropped every field the backend added later - call_coverage,
         * call_resolution_strategies, cross_file_call_edges, unresolved_calls,
         * graph_nodes, inherits_edges, total_files.
         */
        normalizePipelineResults(payload) {
            if (!payload) return null;
            const results = payload.results || payload;
            const cpgStats = results?.stages?.cpg?.statistics || results?.statistics || null;
            const edgeTypes = cpgStats?.edge_types || {};
            return {
                status: results.status || payload.status || 'unknown',
                timestamp: results.timestamp || payload.timestamp || new Date().toISOString(),
                stats: {
                    cpg: {
                        ...(cpgStats || {}),
                        total_nodes: cpgStats?.total_nodes ?? 0,
                        total_edges: cpgStats?.total_edges ?? 0,
                        edge_types: edgeTypes,
                        node_types: cpgStats?.node_types ?? {},
                        data_flow_edges: cpgStats?.data_flow_edges ?? edgeTypes.data_flow ?? 0,
                        control_flow_edges: cpgStats?.control_flow_edges ?? edgeTypes.control_flow ?? 0,
                        call_graph_edges: cpgStats?.call_graph_edges ?? edgeTypes.calls ?? 0
                    }
                },
                raw: results
            };
        },

        /* ---------------- scorecard interaction ---------------- */
        sortScorecard(key) {
            if (this.scorecardSort.key === key) {
                this.scorecardSort.dir = this.scorecardSort.dir === 'desc' ? 'asc' : 'desc';
            } else {
                this.scorecardSort = { key, dir: 'desc' };
            }
        },

        scorecardMarkdown() {
            const head = '| Pattern | Engine | Mined | P | R | F1 | Support | Gen pass | n | Verdict |';
            const sep = '|---|---|---|---|---|---|---|---|---|---|';
            const rows = this.scorecardRows.map(r => [
                r.name,
                r.hasAst ? 'AST-verified' : 'neural only',
                r.mined ?? '--',
                r.precision === null ? '--' : fx(r.precision),
                r.recall === null ? '--' : fx(r.recall),
                r.f1 === null ? '--' : fx(r.f1),
                r.support ?? '--',
                r.passRate === null ? 'n/a' : fx(r.passRate),
                r.genCount ?? 'n/a',
                r.verdict
            ].join(' | ')).map(s => `| ${s} |`);
            const text = [head, sep, ...rows].join('\n');
            navigator.clipboard?.writeText(text)
                .then(() => this.notify('Scorecard copied as Markdown', 'success'))
                .catch(() => this.notify('Clipboard unavailable', 'error'));
        },

        selectSuggestion(idx) { this.selectedSuggestion = idx; },

        toggleCaveat(id) { this.openCaveat = this.openCaveat === id ? null : id; },

        caveatsFor(scope) { return this.caveats.filter(c => c.scope === scope); },

        verificationChip(suggestion) {
            const v = suggestion?.verification;
            if (!v) return { cls: 'chip-na', text: 'not verified' };
            if (v.passed) return { cls: 'chip-pass', text: v.verdict };
            if (v.broken) return { cls: 'chip-fail', text: v.verdict };
            return { cls: 'chip-warn', text: v.verdict };
        },

        /* ---------------- chart lifecycle ---------------- */
        scheduleRender() { this.$nextTick(() => this.renderAll()); },

        destroyAllCharts() {
            Object.keys(this.charts).forEach(id => {
                if (this.charts[id]) this.charts[id].destroy();
                delete this.charts[id];
            });
        },

        /**
         * Create or replace one chart.
         *
         * Tab panels use v-if rather than v-show: a display:none wrapper
         * measures 0x0 and Chart.js would size the canvas to zero and keep it
         * there. Rendering is therefore deferred to $nextTick after the panel
         * actually exists.
         */
        renderChart(id, refName, config) {
            if (typeof Chart === 'undefined') return;
            const el = this.$refs[refName];
            const canvas = Array.isArray(el) ? el[0] : el;
            if (!canvas) return;
            if (this.charts[id]) this.charts[id].destroy();
            this.charts[id] = new Chart(canvas, config);
        },

        renderAll() {
            if (this.currentView === 'results') this.renderResultsCharts();
            if (this.currentView === 'benchmarks') this.renderBenchmarkCharts();
        },

        renderResultsCharts() {
            const p = readPalette();
            if (this.resultsTab === 'overview') {
                const nodeTypes = this.cpgStats.node_types || {};
                if (Object.keys(nodeTypes).length) {
                    this.renderChart('nodeType', 'nodeTypeChart', {
                        type: 'bar',
                        data: {
                            labels: Object.keys(nodeTypes).map(prettyPattern),
                            datasets: [{
                                data: Object.values(nodeTypes),
                                backgroundColor: Object.keys(nodeTypes)
                                    .map((_, i) => p.cat[i % p.cat.length])
                            }]
                        },
                        options: {
                            indexAxis: 'y',
                            scales: { x: { beginAtZero: true, grid: { color: p.rule } }, y: { grid: { display: false } } },
                            plugins: { valueLabels: { color: p.inkDim } }
                        }
                    });
                }
                if (this.strategyRows.length) {
                    this.renderChart('strategies', 'strategyChart', {
                        type: 'bar',
                        data: {
                            labels: this.strategyRows.map(r => r.label),
                            datasets: [{
                                data: this.strategyRows.map(r => r.value),
                                backgroundColor: this.strategyRows.map(r => r.color)
                            }]
                        },
                        options: {
                            indexAxis: 'y',
                            scales: { x: { beginAtZero: true, grid: { color: p.rule } }, y: { grid: { display: false } } },
                            plugins: { valueLabels: { color: p.inkDim } }
                        }
                    });
                }
            }
            if (this.resultsTab === 'smells' && this.smells?.by_type) {
                this.renderChart('smellTypes', 'smellTypeChart', {
                    type: 'bar',
                    data: {
                        labels: Object.keys(this.smells.by_type).map(prettyPattern),
                        datasets: [{
                            data: Object.values(this.smells.by_type),
                            backgroundColor: Object.keys(this.smells.by_type).map((_, i) => p.cat[i % p.cat.length])
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        scales: { x: { beginAtZero: true, grid: { color: p.rule } }, y: { grid: { display: false } } },
                        plugins: { valueLabels: { color: p.inkDim } }
                    }
                });
            }
            if (this.resultsTab === 'graph') this.renderGraphNetwork();
        },

        renderBenchmarkCharts() {
            const p = readPalette();

            if (this.benchTab === 'scorecard' && this.scorecardRows.length) {
                // Classifier F1 and generation pass rate are the ONLY two
                // metrics here that legitimately share an axis - both are in
                // [0,1]. Mined counts (8..1059) deliberately get their own
                // chart on the Dataset tab rather than a second y-axis.
                const rows = this.scorecardRows;
                this.renderChart('patternCompare', 'patternCompareChart', {
                    type: 'bar',
                    data: {
                        labels: rows.map(r => `${r.label}${r.mined !== null ? ` (n=${r.mined})` : ''}`),
                        datasets: [
                            {
                                label: 'Classifier F1',
                                data: rows.map(r => r.f1),
                                backgroundColor: rows.map(r => r.hasAst ? p.ast : p.neural)
                            },
                            {
                                label: 'Generation structural pass',
                                data: rows.map(r => r.passRate),
                                backgroundColor: p.warn
                            }
                        ]
                    },
                    options: {
                        indexAxis: 'y',
                        scales: {
                            x: { beginAtZero: true, max: 1, grid: { color: p.rule } },
                            y: { grid: { display: false } }
                        },
                        plugins: {
                            legend: { display: true, position: 'bottom' },
                            valueLabels: { color: p.inkDim, format: v => (v === null ? '' : fx(v)) },
                            referenceLine: {
                                lines: [
                                    { value: this.classifierReport?.macro?.f1 ?? 0, color: p.neural, label: 'macro F1' },
                                    { value: this.generationReport?.overall_structural_pass_rate ?? 0, color: p.warn, label: 'overall pass' }
                                ]
                            }
                        }
                    }
                });
            }

            if (this.benchTab === 'classifier' && this.classifierReport) {
                const rows = this.scorecardRows;
                this.renderChart('clfScatter', 'clfScatterChart', {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            data: rows.filter(r => r.recall !== null).map(r => ({
                                x: r.recall, y: r.precision, label: r.label, support: r.support
                            })),
                            backgroundColor: rows.filter(r => r.recall !== null)
                                .map(r => r.hasAst ? p.ast : p.neural),
                            pointRadius: ctx => {
                                const s = ctx.raw?.support || 0;
                                const max = Math.max(...rows.map(r => r.support || 0), 1);
                                return 4 + 12 * Math.sqrt(s / max);
                            }
                        }]
                    },
                    options: {
                        scales: {
                            x: { min: 0, max: 1, title: { display: true, text: 'Recall' }, grid: { color: p.rule } },
                            y: { min: 0, max: 1, title: { display: true, text: 'Precision' }, grid: { color: p.rule } }
                        },
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    label: ctx => `${ctx.raw.label}: P=${fx(ctx.raw.y)} R=${fx(ctx.raw.x)} (n=${ctx.raw.support})`
                                }
                            }
                        }
                    }
                });
            }

            if (this.benchTab === 'generation' && this.generationReport) {
                const per = this.generationReport.per_pattern || {};
                const names = this.patternOrder.filter(n => per[n]);
                this.renderChart('genPass', 'genPassChart', {
                    type: 'bar',
                    data: {
                        labels: names.map(n => `${prettyPattern(n)} (n=${per[n].count})`),
                        datasets: [{
                            data: names.map(n => per[n].structural_pass_rate),
                            backgroundColor: names.map(n => per[n].count < 5 ? p.na : p.warn)
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        scales: { x: { beginAtZero: true, max: 1, grid: { color: p.rule } }, y: { grid: { display: false } } },
                        plugins: {
                            valueLabels: {
                                color: p.inkDim,
                                format: v => fx(v)
                            },
                            referenceLine: {
                                lines: [{
                                    value: this.generationReport.overall_structural_pass_rate,
                                    color: p.fail, label: 'overall'
                                }]
                            }
                        }
                    }
                });
                this.renderChart('genEdit', 'genEditChart', {
                    type: 'bar',
                    data: {
                        labels: names.map(prettyPattern),
                        datasets: [{
                            data: names.map(n => per[n].edit_distance_ratio),
                            backgroundColor: p.brand
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        // Pinned to 0.2 with an explicit caption that 1.0 would
                        // be an exact match, so 0.18 does not read as "good".
                        scales: { x: { beginAtZero: true, max: 0.2, grid: { color: p.rule } }, y: { grid: { display: false } } },
                        plugins: { valueLabels: { color: p.inkDim, format: v => fx(v, 3) } }
                    }
                });
            }

            if (this.benchTab === 'xai' && this.xaiReport) {
                const observed = this.evidenceCards[0]?.supporting_metrics?.avg_problem_score;
                const labels = ['Problem score: known problems', 'Problem score: random nodes'];
                const values = [
                    this.xaiReport.problem_detector_mean_score_on_known_problems,
                    this.xaiReport.problem_detector_mean_score_on_random_nodes
                ];
                const colors = [p.pass, p.na];
                if (observed !== undefined) {
                    labels.push('Problem score: observed on loaded run');
                    values.push(observed);
                    colors.push(p.fail);
                }
                labels.push('Solution evaluator accuracy');
                values.push(this.xaiReport.solution_evaluator_test_accuracy);
                colors.push(p.neural);

                this.renderChart('xaiSep', 'xaiSepChart', {
                    type: 'bar',
                    data: { labels, datasets: [{ data: values, backgroundColor: colors }] },
                    options: {
                        indexAxis: 'y',
                        scales: { x: { beginAtZero: true, max: 1, grid: { color: p.rule } }, y: { grid: { display: false } } },
                        plugins: {
                            valueLabels: { color: p.inkDim, format: v => fx(v, 3) },
                            referenceLine: { lines: [{ value: 0.5, color: p.inkDim, label: 'chance' }] }
                        }
                    }
                });
            }

            if (this.benchTab === 'dataset' && this.miningReport) {
                const dist = this.miningReport.per_class_distribution || {};
                const names = this.patternOrder.filter(n => dist[n] !== undefined)
                    .sort((a, b) => dist[b] - dist[a]);
                this.renderChart('minedDist', 'minedDistChart', {
                    type: 'bar',
                    data: {
                        labels: names.map(prettyPattern),
                        datasets: [{
                            data: names.map(n => dist[n]),
                            backgroundColor: names.map(n => this.astSupported.includes(n) ? p.ast : p.neural)
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        // Linear, NOT log: the 1059-vs-8 imbalance is the point
                        // of the figure, and a log axis would flatten it.
                        scales: { x: { beginAtZero: true, grid: { color: p.rule } }, y: { grid: { display: false } } },
                        plugins: {
                            valueLabels: { color: p.inkDim },
                            referenceLine: { lines: [{ value: 50, color: p.warn, label: 'thin-class threshold' }] }
                        }
                    }
                });
            }
        },

        renderGraphNetwork() {
            const container = this.$refs.cpgNetwork;
            if (!container || typeof vis === 'undefined') return;
            const style = getComputedStyle(document.documentElement);
            const colorFor = type => {
                const token = NODE_TYPE_TOKENS[type];
                return token ? (style.getPropertyValue(token) || '#607d8b').trim() : '#607d8b';
            };
            const targetFiles = new Set(this.suggestions.map(s => s.source_file).filter(Boolean));
            const p = readPalette();

            const raw = this.graphData.nodes || [];
            const edges = this.graphData.edges || [];
            const degree = {};
            edges.forEach(e => {
                degree[e.source] = (degree[e.source] || 0) + 1;
                degree[e.target] = (degree[e.target] || 0) + 1;
            });

            const visible = this.hideIsolated ? raw.filter(n => degree[n.id]) : raw;
            const nodes = new vis.DataSet(visible.map(n => {
                const isTarget = targetFiles.has(n.file);
                return {
                    id: n.id,
                    label: n.label,
                    title: `${n.type} · ${n.file || ''}:${n.line || ''}`,
                    color: {
                        background: colorFor(n.type),
                        border: isTarget ? p.warn : colorFor(n.type)
                    },
                    borderWidth: isTarget ? 3 : 1,
                    value: 1 + (degree[n.id] || 0)
                };
            }));
            const visibleIds = new Set(visible.map(n => n.id));
            const visEdges = new vis.DataSet(
                edges.filter(e => visibleIds.has(e.source) && visibleIds.has(e.target))
                    .map(e => ({
                        from: e.source, to: e.target, label: e.type, arrows: 'to',
                        color: { color: p.rule }, font: { size: 8, align: 'middle', color: p.inkDim }
                    }))
            );

            if (this.network) this.network.destroy();
            this.network = new vis.Network(container, { nodes, edges: visEdges }, {
                layout: { improvedLayout: true },
                physics: { stabilization: { iterations: 200 }, barnesHut: { gravitationalConstant: -3000 } },
                nodes: { shape: 'dot', scaling: { min: 6, max: 26 }, font: { size: 10, color: p.ink } },
                edges: { smooth: { type: 'continuous' } },
                interaction: { hover: true, tooltipDelay: 150 }
            });
            // With 1600-node corpora the simulation must not stay hot.
            this.network.once('stabilizationIterationsDone', () => {
                this.network.setOptions({ physics: false });
            });
        }
    },

});

app.mount('#app');
console.log('XRefactor frontend loaded');
