/* Project page scripts: theme toggle, copy buttons, two toy demos, and charts.
   Every number in DATA below is copied from the camera-ready manuscript
   (Table 3, Table 4/8, Tables 9-10, Section 4.3). No external requests. */
(function () {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent, text) {
    var n = document.createElementNS(SVGNS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  // Draw charts at their rendered width so text stays at its nominal size on phones.
  function fit(svg, maxW, h) {
    var w = Math.round(svg.getBoundingClientRect().width) || maxW;
    w = Math.max(300, Math.min(maxW, w));
    svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
    return w;
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ------------------------------------------------------------ theme */
  var themeBtn = document.getElementById('theme');
  var themeLabel = document.getElementById('theme-label');
  var themes = ['auto', 'light', 'dark'];
  function currentTheme() { return document.documentElement.getAttribute('data-theme') || 'auto'; }
  function setTheme(t) {
    if (t === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', t);
    themeLabel.textContent = t;
    themeBtn.setAttribute('aria-label', 'Colour theme: ' + (t === 'auto' ? 'follows system' : t));
    try { if (t === 'auto') localStorage.removeItem('fiducio-paper-theme'); else localStorage.setItem('fiducio-paper-theme', t); } catch (e) {}
    redrawAll();
  }
  themeLabel.textContent = currentTheme();
  themeBtn.addEventListener('click', function () {
    setTheme(themes[(themes.indexOf(currentTheme()) + 1) % themes.length]);
  });
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    (mq.addEventListener ? mq.addEventListener.bind(mq, 'change') : mq.addListener.bind(mq))(function () { redrawAll(); });
  }

  /* ------------------------------------------------------------ copy */
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise(function (res, rej) {
      var ta = document.createElement('textarea');
      ta.value = text; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy') ? res() : rej(); } catch (e) { rej(e); }
      document.body.removeChild(ta);
    });
  }
  Array.prototype.forEach.call(document.querySelectorAll('[data-copy]'), function (b) {
    b.addEventListener('click', function () {
      var src = document.getElementById(b.getAttribute('data-copy'));
      var label = b.querySelector('span');
      copyText(src.textContent).then(function () {
        b.classList.add('ok'); label.textContent = 'copied';
        setTimeout(function () { b.classList.remove('ok'); label.textContent = 'copy'; }, 1600);
      }, function () { label.textContent = 'select + copy'; });
    });
  });

  /* ------------------------------------------------------------ small math */
  function softmax(z) {
    var m = Math.max.apply(null, z), e = z.map(function (v) { return Math.exp(v - m); });
    var s = e.reduce(function (a, b) { return a + b; }, 0);
    return e.map(function (v) { return v / s; });
  }
  function matvec(W, z) { return W.map(function (r) { return r.reduce(function (a, w, j) { return a + w * z[j]; }, 0); }); }
  function add(a, b) { return a.map(function (v, i) { return v + b[i]; }); }
  function argmax(a) { var k = 0; for (var i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; }

  /* ============================================================ DATA (paper) */
  // Table 3: mean over held-out cases of Std_v(F_pool(v)), with [p25, p50, p75].
  var FREE_ENERGY = [
    { name: 'RoadSeg', mean: 1.08, q: [0.73, 1.06, 1.34] },
    { name: 'Cityscapes', mean: 2.54, q: [2.30, 2.46, 2.71] },
    { name: 'BraTS', mean: 8.77, q: [8.56, 8.69, 8.87] }
  ];
  // Section 4.3: flip audit of MS(z-bar). Remainder = flips leaving accuracy unchanged.
  var FLIPS = [
    { name: 'Cityscapes', harm: 50.4, good: 29.5 },
    { name: 'BraTS', harm: 74.8, good: 18.6 }
  ];
  // Tables 9-10: paired differences (second minus first), [mean, lo, hi, star].
  // Order of datasets: RoadSeg, BraTS, Cityscapes. DSC in fraction (converted to points on display).
  var DS = ['RoadSeg', 'BraTS', 'Cityscapes'];
  var PAIRED = {
    ti: [
      { label: 'MS → MSc', long: 'MS(z̄) → MSc(z̄)', m: {
        nll: [[-0.0000, -0.0001, 0.0000, 0], [-0.0009, -0.0029, 0.0009, 0], [-0.0763, -0.1251, -0.0278, 1]],
        dsc: [[0.0003, 0.0000, 0.0005, 1], [0.0174, -0.0186, 0.0443, 0], [-0.0068, -0.0350, 0.0103, 0]],
        ece: [[0.0000, -0.0001, 0.0001, 0], [-0.0001, -0.0003, 0.0001, 0], [-0.0105, -0.0233, -0.0026, 1]],
        ba:  [[0.0007, 0.0002, 0.0012, 1], [-0.0099, -0.0235, 0.0027, 0], [-0.0576, -0.0868, -0.0175, 1]] } },
      { label: 'MS → DC', long: 'MS(z̄) → DC(z̄)', m: {
        nll: [[-0.0004, -0.0005, -0.0002, 1], [-0.0000, -0.0026, 0.0016, 0], [-0.0805, -0.1274, -0.0354, 1]],
        dsc: [[0.0021, 0.0016, 0.0026, 1], [-0.0084, -0.0574, 0.0540, 0], [-0.0311, -0.0404, -0.0221, 1]],
        ece: [[-0.0013, -0.0014, -0.0012, 1], [-0.0000, -0.0004, 0.0003, 0], [-0.0100, -0.0214, -0.0031, 1]],
        ba:  [[-0.0088, -0.0101, -0.0077, 1], [-0.0026, -0.0189, 0.0101, 0], [-0.0591, -0.0926, -0.0151, 1]] } },
      { label: 'LTS → LTS·logS', long: 'LTS(z̄) → LTS(log S(z̄))', m: {
        nll: [[-0.0007, -0.0016, 0.0007, 0], [-0.0039, -0.0360, 0.0212, 0], [-0.0012, -0.0059, 0.0034, 0]],
        dsc: [[0.0000, -0.0000, 0.0000, 0], [0.0000, 0.0000, 0.0000, 0], [0.0000, 0.0000, 0.0001, 0]],
        ece: [[-0.0008, -0.0013, -0.0002, 1], [-0.0035, -0.0307, 0.0182, 0], [-0.0032, -0.0116, 0.0025, 0]],
        ba:  [[-0.0016, -0.0066, 0.0021, 0], [-0.0437, -0.2719, 0.1353, 0], [0.0112, -0.0083, 0.0423, 0]] } }
    ],
    dp: [
      { label: 'CDC → CMSap (p̄)', long: 'CDC(p̄) → CMSap(p̄)', m: {
        nll: [[0.0013, 0.0010, 0.0016, 1], [-0.0009, -0.0016, -0.0003, 1], [-0.0206, -0.0244, -0.0174, 1]],
        dsc: [[0.0104, 0.0086, 0.0125, 1], [0.0400, 0.0138, 0.0794, 1], [0.0877, 0.0692, 0.1004, 1]],
        ece: [[0.0015, 0.0012, 0.0017, 1], [0.0001, 0.0000, 0.0002, 1], [0.0041, 0.0004, 0.0070, 1]],
        ba:  [[0.0026, -0.0003, 0.0050, 0], [-0.0047, -0.0072, -0.0012, 1], [-0.0220, -0.0323, -0.0030, 1]] } },
      { label: 'CDC → CMSop (p̄)', long: 'CDC(p̄) → CMSop(p̄)', m: {
        nll: [[0.0013, 0.0011, 0.0016, 1], [-0.0011, -0.0018, -0.0006, 1], [-0.0083, -0.0107, -0.0059, 1]],
        dsc: [[0.0104, 0.0085, 0.0126, 1], [0.0400, 0.0137, 0.0796, 1], [0.0876, 0.0692, 0.1005, 1]],
        ece: [[0.0016, 0.0012, 0.0020, 1], [0.0001, -0.0000, 0.0002, 0], [0.0035, -0.0010, 0.0067, 0]],
        ba:  [[0.0031, -0.0001, 0.0054, 0], [-0.0052, -0.0078, -0.0018, 1], [-0.0378, -0.0492, -0.0204, 1]] } },
      { label: 'CDC → CMSap (z̄)', long: 'CDC(z̄) → CMSap(z̄)', m: {
        nll: [[0.0017, 0.0016, 0.0019, 1], [-0.0009, -0.0025, 0.0003, 0], [-0.0139, -0.0186, -0.0101, 1]],
        dsc: [[0.0102, 0.0082, 0.0128, 1], [0.0287, 0.0163, 0.0410, 1], [0.0517, 0.0473, 0.0562, 1]],
        ece: [[0.0020, 0.0018, 0.0023, 1], [0.0002, -0.0000, 0.0004, 0], [0.0042, -0.0018, 0.0087, 0]],
        ba:  [[0.0063, 0.0052, 0.0073, 1], [-0.0040, -0.0097, -0.0000, 1], [-0.0242, -0.0337, -0.0117, 1]] } },
      { label: 'CDC → CMSop (z̄)', long: 'CDC(z̄) → CMSop(z̄)', m: {
        nll: [[0.0017, 0.0015, 0.0019, 1], [-0.0011, -0.0028, 0.0002, 0], [-0.0016, -0.0042, 0.0011, 0]],
        dsc: [[0.0102, 0.0082, 0.0129, 1], [0.0287, 0.0163, 0.0412, 1], [0.0517, 0.0473, 0.0563, 1]],
        ece: [[0.0020, 0.0017, 0.0023, 1], [0.0001, -0.0001, 0.0003, 0], [0.0016, -0.0030, 0.0046, 0]],
        ba:  [[0.0063, 0.0048, 0.0076, 1], [-0.0044, -0.0099, -0.0003, 1], [-0.0366, -0.0485, -0.0277, 1]] } }
    ]
  };
  var METRIC = {
    nll: { name: 'NLL', lower: true, scale: 1, dp: 4 },
    ece: { name: 'ECE', lower: true, scale: 1, dp: 4 },
    ba:  { name: 'BA-ECE', lower: true, scale: 1, dp: 4 },
    dsc: { name: 'DSC', lower: false, scale: 100, dp: 2, unit: ' pts' }
  };
  // Table 8 (App. I): DSC (%) with 95% CI; CMS value = CMSap row (CMSop has the same point estimate).
  var DSC = [
    { ds: 'RoadSeg', pool: 'p̄', cdc: [75.7, 74.5, 76.8], cms: [76.7, 75.7, 77.7], flip: 0.5 },
    { ds: 'RoadSeg', pool: 'z̄', cdc: [75.7, 74.6, 76.8], cms: [76.7, 75.7, 77.9], flip: 0.6 },
    { ds: 'BraTS', pool: 'p̄', cdc: [72.0, 68.6, 74.9], cms: [76.0, 74.5, 77.5], flip: 0.1 },
    { ds: 'BraTS', pool: 'z̄', cdc: [74.0, 72.6, 75.5], cms: [76.9, 75.4, 78.4], flip: 0.1 },
    { ds: 'Cityscapes', pool: 'p̄', cdc: [69.3, 67.8, 71.2], cms: [78.1, 77.6, 78.6], flip: 5.1 },
    { ds: 'Cityscapes', pool: 'z̄', cdc: [74.2, 73.6, 74.8], cms: [79.4, 78.9, 79.8], flip: 5.0 }
  ];

  /* ============================================================ problem charts */
  function drawFreeEnergy() {
    var svg = document.getElementById('fe-chart'); clear(svg);
    var W = fit(svg, 520, 128), x0 = W < 420 ? 82 : 96, x1 = W - 50, max = 10, rowH = 34, top = 8;
    function X(v) { return x0 + (x1 - x0) * v / max; }
    [0, 2, 4, 6, 8, 10].forEach(function (t) {
      el('line', { x1: X(t), x2: X(t), y1: top - 2, y2: top + rowH * 3, stroke: css('--grid') }, svg);
      el('text', { x: X(t), y: top + rowH * 3 + 16, 'text-anchor': 'middle', 'font-size': 11 }, svg, t);
    });
    FREE_ENERGY.forEach(function (d, i) {
      var y = top + i * rowH + 8;
      el('text', { x: x0 - 10, y: y + 12, 'text-anchor': 'end', 'font-size': 13, class: 'lab' }, svg, d.name);
      el('rect', { x: X(0), y: y, width: X(d.mean) - X(0), height: 16, rx: 3, fill: css('--break'), opacity: 0.85 }, svg);
      el('line', { x1: X(d.q[0]), x2: X(d.q[2]), y1: y + 8, y2: y + 8, stroke: css('--ink'), 'stroke-width': 1.6 }, svg);
      el('line', { x1: X(d.q[0]), x2: X(d.q[0]), y1: y + 3, y2: y + 13, stroke: css('--ink'), 'stroke-width': 1.6 }, svg);
      el('line', { x1: X(d.q[2]), x2: X(d.q[2]), y1: y + 3, y2: y + 13, stroke: css('--ink'), 'stroke-width': 1.6 }, svg);
      el('text', { x: X(Math.max(d.mean, d.q[2])) + 8, y: y + 12, 'font-size': 12.5, fill: css('--ink') }, svg, d.mean.toFixed(2));
    });
  }

  function drawFlips() {
    var svg = document.getElementById('flip-chart'); clear(svg);
    var W = fit(svg, 520, 118), x0 = W < 420 ? 82 : 96, x1 = W - 10, rowH = 38, top = 6;
    function X(v) { return x0 + (x1 - x0) * v / 100; }
    FLIPS.forEach(function (d, i) {
      var y = top + i * rowH;
      var rest = +(100 - d.harm - d.good).toFixed(1);
      el('text', { x: x0 - 10, y: y + 17, 'text-anchor': 'end', 'font-size': 13, class: 'lab' }, svg, d.name);
      el('rect', { x: X(0), y: y + 3, width: X(d.harm) - X(0) - 1.5, height: 20, rx: 3, fill: css('--break') }, svg);
      el('rect', { x: X(d.harm), y: y + 3, width: X(d.good) - X(0) - 1.5, height: 20, rx: 3, fill: css('--hold') }, svg);
      el('rect', { x: X(d.harm + d.good), y: y + 3, width: X(rest) - X(0), height: 20, rx: 3, fill: css('--line-2') }, svg);
      el('text', { x: X(0) + 7, y: y + 17, 'font-size': 12, fill: '#fff' }, svg, d.harm + '%');
      el('text', { x: X(d.harm) + 7, y: y + 17, 'font-size': 12, fill: '#fff' }, svg, d.good + '%');
    });
    var ly = top + rowH * 2 + 14;
    [['harmful', '--break'], ['beneficial', '--hold'], ['accuracy unchanged', '--line-2']].forEach(function (l, i) {
      var lx = (W < 420 ? 4 : x0) + i * (W < 420 ? 92 : 120);
      el('rect', { x: lx, y: ly - 9, width: 10, height: 10, rx: 2, fill: css(l[1]) }, svg);
      el('text', { x: lx + 15, y: ly, 'font-size': 11.5 }, svg, l[0]);
    });
  }

  /* ============================================================ demo 1: TI */
  var TI = {
    z: [2.0, 1.2, -0.4],
    // Hand-picked MS: row sums W·1 = (1.5, 1.1, 0.7), so the output depends on c.
    Wms: [[1.2, 0.2, 0.1], [0.1, 1.0, 0.0], [0.0, 0.1, 0.6]],
    // Same matrix with the last column adjusted so every row sums to 1.1 (MSc constraint).
    Wc: [[1.2, 0.2, -0.3], [0.1, 1.0, 0.0], [0.0, 0.1, 1.0]],
    b: [0.0, 0.3, 0.2]
  };
  TI.q0ms = softmax(add(matvec(TI.Wms, TI.z), TI.b));
  TI.q0c = softmax(add(matvec(TI.Wc, TI.z), TI.b));
  var tiC = document.getElementById('ti-c'), tiOut = document.getElementById('ti-c-out');

  function drawTI() {
    var c = parseFloat(tiC.value);
    tiOut.textContent = (c >= 0 ? '+' : '−') + Math.abs(c).toFixed(1);
    var zs = TI.z.map(function (v) { return v + c; });
    var p = softmax(zs);
    var qms = softmax(add(matvec(TI.Wms, zs), TI.b));
    var qc = softmax(add(matvec(TI.Wc, zs), TI.b));
    var svg = document.getElementById('ti-svg'); clear(svg);
    var SW = fit(svg, 560, 300), pw = (SW - 12) / 2;
    var cols = [css('--cA'), css('--cB'), css('--cC')];
    var names = ['A', 'B', 'C'];
    var panels = [
      { t: SW < 420 ? 'z + c·1' : 'logits z + c·1', vals: zs, logit: true, x: 0, y: 0 },
      { t: SW < 420 ? 'softmax' : 'softmax S(z + c·1)', vals: p, x: pw + 12, y: 0, ref: softmax(TI.z) },
      { t: 'MS  (not TI)', vals: qms, x: 0, y: 150, ref: TI.q0ms, bad: true },
      { t: 'MSc  (TI)', vals: qc, x: pw + 12, y: 150, ref: TI.q0c }
    ];
    panels.forEach(function (P) {
      var g = el('g', { transform: 'translate(' + P.x + ',' + P.y + ')' }, svg);
      var w = pw - 10, h = 96, top = 26, bw = Math.min(52, w / 3 - 10), gap = (w - 3 * bw) / 4;
      el('rect', { x: 0, y: 0, width: pw, height: 140, rx: 10, fill: css('--surface-2') }, g);
      el('text', { x: 12, y: 18, 'font-size': 12.5, fill: css('--ink'), 'font-weight': 500 }, g, P.t);
      var base, Y;
      if (P.logit) { base = top + h / 2; Y = function (v) { return base - v * (h / 2) / 11; }; }
      else { base = top + h; Y = function (v) { return base - v * h; }; }
      el('line', { x1: 8, x2: w + 2, y1: base, y2: base, stroke: css('--line-2') }, g);
      var k = argmax(P.vals);
      P.vals.forEach(function (v, i) {
        var x = 5 + gap + i * (bw + gap), y = Y(v);
        if (P.ref) el('line', { x1: x - 3, x2: x + bw + 3, y1: Y(P.ref[i]), y2: Y(P.ref[i]), stroke: css('--muted'), 'stroke-dasharray': '3 3' }, g);
        el('rect', { x: x, y: Math.min(y, base), width: bw, height: Math.max(1, Math.abs(base - y)), rx: 3, fill: cols[i], opacity: i === k ? 1 : 0.55 }, g);
        var lab = P.logit ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1) : v.toFixed(2);
        var ty = P.logit ? (v >= 0 ? Math.max(y - 5, top + 2) : Math.min(y + 14, top + h + 12)) : Math.max(y - 5, top + 2);
        el('text', { x: x + bw / 2, y: ty, 'text-anchor': 'middle', 'font-size': 11.5, fill: css('--ink') }, g, lab);
        el('text', { x: x + bw / 2, y: 136, 'text-anchor': 'middle', 'font-size': 11 }, g, names[i] + (i === k ? ' ◂' : ''));
      });
    });
    function dmax(a, b) { return Math.max.apply(null, a.map(function (v, i) { return Math.abs(v - b[i]); })); }
    var dMS = dmax(qms, TI.q0ms), dC = dmax(qc, TI.q0c);
    var flipped = argmax(qms) !== argmax(TI.q0ms);
    var read = document.getElementById('ti-read');
    read.innerHTML = '';
    function badge(cls, txt) { var s = document.createElement('span'); s.className = 'badge ' + cls; s.textContent = txt; read.appendChild(s); }
    badge('hold', 'softmax: unchanged');
    badge(dMS > 0.005 ? 'break' : '', 'MS: max |Δq| = ' + dMS.toFixed(2) + (flipped ? ' · argmax flipped' : ''));
    badge('hold', 'MSc: max |Δq| = ' + dC.toFixed(2));
  }
  tiC.addEventListener('input', drawTI);

  /* ============================================================ demo 2: decision preservation */
  var GW = 48, GH = 30;
  var FIELD = (function () {
    var out = [];
    for (var j = 0; j < GH; j++) for (var i = 0; i < GW; i++) {
      var x = (i + 0.5) / GW, y = (j + 0.5) / GH, n = function (k) { return 0.45 * Math.sin(7 * x + 2.1 * k) * Math.cos(5.5 * y + 1.3 * k) + 0.25 * Math.sin(17 * x * y + k); };
      var zA = 4.2 * (y - 0.52) + 0.6 * Math.sin(5 * x) + n(0);
      var zB = 4.0 * (0.46 - y) + 0.8 * Math.sin(3.2 * x + 0.4) + n(1);
      var zC = 3.0 - 10 * (1.4 * (x - 0.68) * (x - 0.68) + 2.6 * (y - 0.5) * (y - 0.5)) + n(2);
      var off = 3 * Math.sin(2.3 * x + 1.7 * y); // arbitrary spatial offset: irrelevant to the softmax
      out.push([zA + off, zB + off, zC + off]);
    }
    return out;
  })();
  var I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  function lerpM(A, B, s) { return A.map(function (r, i) { return r.map(function (v, j) { return (1 - s) * v + s * B[i][j]; }); }); }
  // Unconstrained affine map (MS-type) and target bias, hand-picked for the toy.
  var MS_W = [[0.85, 0.35, 0.05], [0.30, 0.90, 0.10], [0.15, 0.25, 0.75]], MS_B = [-0.35, 0.55, -0.05];
  // Argmax-preserving experts, one per predicted class: W' = [[a, r],[0, A]] with A >= 0 (paper §3.2.1).
  var AP_WP = [
    [[0.70, 0.10, 0.05], [0, 0.65, 0.10], [0, 0.05, 0.80]],
    [[0.60, 0.05, 0.10], [0, 0.75, 0.05], [0, 0.10, 0.60]],
    [[0.75, 0.00, 0.10], [0, 0.55, 0.15], [0, 0.10, 0.70]]
  ];
  var AP_B = [[0, -0.30, -0.50], [0, -0.20, -0.60], [0, -0.45, -0.25]]; // b1 >= b_i: bias inside the argmax cone
  var G = [[1, 0, 0], [1, -1, 0], [1, 0, -1]]; // G^-1 = G for this G
  function applyMS(z, s) { return add(matvec(lerpM(I3, MS_W, s), z), MS_B.map(function (v) { return s * v; })); }
  function applyAP(z, s) {
    var k = argmax(z), perm = [0, 1, 2]; perm[0] = k; perm[k] = 0;      // swap argmax to position 1
    var zc = perm.map(function (p) { return z[p]; });
    var Wp = lerpM(I3, AP_WP[k], s);                                      // convex mix keeps the block structure
    var h = add(matvec(G, matvec(Wp, matvec(G, zc))), AP_B[k].map(function (v) { return s * v; }));
    var out = [0, 0, 0]; perm.forEach(function (p, i) { out[p] = h[i]; }); // inverse permutation
    return out;
  }
  var dpMode = 'ms', dpS = document.getElementById('dp-s'), dpOut = document.getElementById('dp-s-out');
  function hexToRgb(h) {
    h = h.replace('#', ''); if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join('');
    var n = parseInt(h, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function drawDP() {
    var s = parseFloat(dpS.value); dpOut.textContent = s.toFixed(2);
    var segC = document.getElementById('dp-seg'), confC = document.getElementById('dp-conf');
    var sctx = segC.getContext('2d'), cctx = confC.getContext('2d');
    var simg = sctx.createImageData(GW, GH), cimg = cctx.createImageData(GW, GH);
    var cols = [hexToRgb(css('--cA')), hexToRgb(css('--cB')), hexToRgb(css('--cC'))];
    var brk = hexToRgb(css('--break')), lo = hexToRgb(css('--surface-2')), hi = hexToRgb(css('--hold'));
    var flips = 0, c0 = 0, c1 = 0;
    FIELD.forEach(function (z, idx) {
      var p = softmax(z), k0 = argmax(p);
      var q = softmax(dpMode === 'ms' ? applyMS(z, s) : applyAP(z, s)), k1 = argmax(q);
      var flip = k0 !== k1; if (flip) flips++;
      c0 += p[k0]; c1 += q[k1];
      var col = flip ? brk : cols[k1], o = idx * 4;
      var a = flip ? 1 : 0.35 + 0.65 * (q[k1] - 1 / 3) / (2 / 3);
      simg.data[o] = col[0]; simg.data[o + 1] = col[1]; simg.data[o + 2] = col[2]; simg.data[o + 3] = Math.round(255 * a);
      var t = Math.min(1, Math.max(0, (q[k1] - 1 / 3) / (2 / 3)));
      cimg.data[o] = lo[0] + (hi[0] - lo[0]) * t; cimg.data[o + 1] = lo[1] + (hi[1] - lo[1]) * t; cimg.data[o + 2] = lo[2] + (hi[2] - lo[2]) * t; cimg.data[o + 3] = 255;
    });
    sctx.clearRect(0, 0, GW, GH); sctx.putImageData(simg, 0, 0); cctx.putImageData(cimg, 0, 0);
    var n = FIELD.length, read = document.getElementById('dp-read');
    read.innerHTML = '';
    function badge(cls, txt) { var e = document.createElement('span'); e.className = 'badge ' + cls; e.textContent = txt; read.appendChild(e); }
    badge(flips ? 'break' : 'hold', 'flipped: ' + (100 * flips / n).toFixed(1) + '% of pixels');
    badge('', 'mean confidence ' + (c0 / n).toFixed(3) + ' → ' + (c1 / n).toFixed(3));
  }
  dpS.addEventListener('input', drawDP);
  Array.prototype.forEach.call(document.querySelectorAll('[data-cal]'), function (b) {
    b.addEventListener('click', function () {
      dpMode = b.getAttribute('data-cal');
      Array.prototype.forEach.call(document.querySelectorAll('[data-cal]'), function (o) { o.setAttribute('aria-pressed', String(o === b)); });
      drawDP();
    });
  });

  /* ============================================================ forest plot */
  var fam = 'ti', met = 'nll';
  function niceMax(m) {
    if (m <= 0) return 1e-4;
    var p = Math.pow(10, Math.floor(Math.log10(m))), f = m / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
  }
  function fmt(v, d) { var s = Math.abs(v).toFixed(d); return (v < 0 && +s !== 0 ? '−' : v > 0 && +s !== 0 ? '+' : '±') + s; }
  function drawForest() {
    var M = METRIC[met], comps = PAIRED[fam], host = document.getElementById('forest-panels');
    host.innerHTML = '';
    document.getElementById('forest-cap').textContent =
      (fam === 'ti' ? 'TI counterpart minus shift-sensitive method' : 'Decision-preserving CMS minus unconstrained CDC') +
      ' · Δ' + M.name + (M.unit || '') + ' · ' + (M.lower ? 'left of 0 = better' : 'right of 0 = better');
    var read = document.getElementById('forest-read');
    DS.forEach(function (ds, di) {
      var box = document.createElement('div');
      var h = document.createElement('h4'); h.textContent = ds; box.appendChild(h);
      var rows = comps.map(function (c) { return c.m[met][di].map(function (v, i) { return i < 3 ? v * M.scale : v; }); });
      var m = niceMax(Math.max.apply(null, rows.map(function (r) { return Math.max(Math.abs(r[1]), Math.abs(r[2]), Math.abs(r[0])); })) * 1.05);
      var W = 360, L = 118, R = 14, rowH = 30, top = 6, Hh = top + rowH * rows.length + 26;
      var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + Hh, class: 'chart forest', role: 'img', 'aria-label': ds + ' paired differences for ' + M.name });
      function X(v) { return L + (W - L - R) * (v + m) / (2 * m); }
      [-m, 0, m].forEach(function (t) {
        el('line', { x1: X(t), x2: X(t), y1: top, y2: top + rowH * rows.length, stroke: t === 0 ? css('--muted') : css('--grid'), 'stroke-dasharray': t === 0 ? '' : '2 3' }, svg);
        var lab = t === 0 ? '0' : fmt(t, M.dp === 4 ? (m < 0.001 ? 4 : m < 0.01 ? 3 : 2) : 1);
        el('text', { x: X(t), y: top + rowH * rows.length + 16, 'text-anchor': t === -m ? 'start' : t === m ? 'end' : 'middle', 'font-size': 10.5 }, svg, lab);
      });
      rows.forEach(function (r, i) {
        var y = top + i * rowH + rowH / 2, sig = !!r[3];
        var better = M.lower ? r[0] < 0 : r[0] > 0; // sign of the mean; the paper's star gives significance
        var color = sig ? (better ? css('--hold') : css('--break')) : css('--ns');
        var g = el('g', { class: 'row', tabindex: 0 }, svg);
        el('rect', { class: 'bg', x: 0, y: y - rowH / 2 + 1, width: W, height: rowH - 2, rx: 6, fill: 'transparent' }, g);
        el('text', { x: 6, y: y + 4, 'font-size': 11.5, class: 'lab' }, g, comps[i].label);
        var x1 = X(r[1]), x2 = X(r[2]);
        if (x2 - x1 < 2) { x1 -= 1; x2 += 1; }
        el('line', { class: 'ci', x1: x1, x2: x2, y1: y, y2: y, stroke: color }, g);
        el('circle', { class: 'pt', cx: X(r[0]), cy: y, r: 4.6, fill: sig ? color : css('--surface'), stroke: color }, g);
        var msg = ds + ' · ' + comps[i].long + ' · Δ' + M.name + ' = ' + fmt(r[0], M.dp) + ' [' + fmt(r[1], M.dp) + ', ' + fmt(r[2], M.dp) + ']' + (M.unit || '') +
          (sig ? (better ? ' · significant, favours the constrained method' : ' · significant, against the constrained method') : ' · not significant');
        el('title', {}, g, msg);
        var show = function () { read.textContent = msg; };
        g.addEventListener('mouseenter', show); g.addEventListener('focus', show); g.addEventListener('click', show);
      });
      box.appendChild(svg); host.appendChild(box);
    });
  }
  function bindSeg(groupId, attr, set) {
    var btns = document.querySelectorAll('#' + groupId + ' button');
    Array.prototype.forEach.call(btns, function (b) {
      b.addEventListener('click', function () {
        Array.prototype.forEach.call(btns, function (o) { o.setAttribute('aria-pressed', String(o === b)); });
        set(b.getAttribute(attr)); drawForest();
        document.getElementById('forest-read').textContent = 'Hover or focus an interval for its values.';
      });
    });
  }
  bindSeg('forest-fam', 'data-fam', function (v) { fam = v; });
  bindSeg('forest-met', 'data-met', function (v) { met = v; });

  /* ============================================================ DSC dumbbell */
  function drawDSC() {
    var svg = document.getElementById('dsc-chart'); clear(svg);
    var W = fit(svg, 520, 300), L = W < 420 ? 96 : 118, R = W < 420 ? 52 : 80, lo = 66, hi = 82, rowH = 38, top = 8;
    function X(v) { return L + (W - L - R) * (v - lo) / (hi - lo); }
    for (var t = lo; t <= hi; t += 4) {
      el('line', { x1: X(t), x2: X(t), y1: top, y2: top + rowH * DSC.length, stroke: css('--grid') }, svg);
      el('text', { x: X(t), y: top + rowH * DSC.length + 16, 'text-anchor': 'middle', 'font-size': 11 }, svg, t);
    }
    el('text', { x: W - R, y: top + rowH * DSC.length + 32, 'text-anchor': 'end', 'font-size': 11 }, svg, 'DSC (%)');
    el('text', { x: W - 4, y: top - 0 + 4, 'text-anchor': 'end', 'font-size': 10.5 }, svg, W < 420 ? 'flips' : 'CDC flips');
    DSC.forEach(function (d, i) {
      var y = top + i * rowH + rowH / 2 + 6;
      if (i % 2 === 0) el('text', { x: 4, y: y + 4, 'font-size': 12.5, class: 'lab' }, svg, d.ds);
      el('text', { x: L - 10, y: y + 4, 'text-anchor': 'end', 'font-size': 11.5 }, svg, d.pool);
      el('line', { x1: X(d.cdc[0]), x2: X(d.cms[0]), y1: y, y2: y, stroke: css('--line-2'), 'stroke-width': 5, 'stroke-linecap': 'round' }, svg);
      [['cdc', '--break', -5], ['cms', '--hold', 5]].forEach(function (s) {
        var v = d[s[0]], c = css(s[1]);
        el('line', { x1: X(v[1]), x2: X(v[2]), y1: y + s[2], y2: y + s[2], stroke: c, 'stroke-width': 1.6 }, svg);
        el('circle', { cx: X(v[0]), cy: y, r: 5.5, fill: c }, svg);
        el('title', {}, svg, d.ds + ' ' + d.pool + ' · ' + s[0].toUpperCase() + ' DSC ' + v[0] + ' [' + v[1] + ', ' + v[2] + ']');
      });
      el('text', { x: W - 4, y: y + 4, 'text-anchor': 'end', 'font-size': 12, fill: css('--ink') }, svg, d.flip.toFixed(1) + '%');
      if (i % 2 === 1 && i < DSC.length - 1) el('line', { x1: 0, x2: W, y1: y + rowH / 2 + 2, y2: y + rowH / 2 + 2, stroke: css('--line') }, svg);
    });
  }

  /* ============================================================ init */
  function redrawAll() { drawFreeEnergy(); drawFlips(); drawTI(); drawDP(); drawForest(); drawDSC(); }
  redrawAll();
  var rz; window.addEventListener('resize', function () { clearTimeout(rz); rz = setTimeout(redrawAll, 150); });

  // One gentle sweep of the offset slider when the TI demo first scrolls into view.
  if (!reduceMotion && 'IntersectionObserver' in window) {
    var done = false;
    var io = new IntersectionObserver(function (es) {
      if (done || !es[0].isIntersecting) return; done = true; io.disconnect();
      var t0 = null, user = false;
      tiC.addEventListener('pointerdown', function () { user = true; }, { once: true });
      function step(ts) {
        if (user) return; if (t0 == null) t0 = ts;
        var u = (ts - t0) / 2600; if (u > 1) { tiC.value = 0; drawTI(); return; }
        tiC.value = (-6 * Math.sin(Math.PI * 2 * u)).toFixed(1); drawTI(); requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }, { threshold: 0.6 });
    io.observe(document.getElementById('ti-demo'));
  }
})();
