<!DOCTYPE html>
<html>
<head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', var(--font-sans), sans-serif;
    --red: #E24B4A;
    --amber: #BA7517;
    --teal: #0F6E56;
    --blue: #185FA5;
    --gray-dk: #2C2C2A;
    --gray-md: #5F5E5A;
    --gray-lt: #D3D1C7;
    --bg: var(--color-background-primary);
    --bg2: var(--color-background-secondary);
    --border: var(--color-border-tertiary);
    --text: var(--color-text-primary);
    --muted: var(--color-text-secondary);
  }

  body { font-family: var(--sans); color: var(--text); background: transparent; }

  .page { max-width: 680px; margin: 0 auto; padding: 2rem 0 3rem; }

  /* HERO */
  .hero { position: relative; padding: 2.5rem 2rem 2rem; border: 0.5px solid var(--border); border-radius: 12px; background: var(--bg2); overflow: hidden; margin-bottom: 2rem; }

  .hero-scan {
    position: absolute; top: 0; left: 0; width: 100%; height: 3px;
    background: linear-gradient(90deg, transparent 0%, var(--red) 50%, transparent 100%);
    animation: scan 3s ease-in-out infinite;
    opacity: 0.7;
  }
  @keyframes scan { 0%,100%{top:0;opacity:0} 10%{opacity:.7} 90%{opacity:.7} 50%{top:calc(100% - 3px)} }

  .hero-grid {
    position: absolute; inset: 0;
    background-image: linear-gradient(rgba(226,75,74,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(226,75,74,0.06) 1px, transparent 1px);
    background-size: 32px 32px;
  }

  .corner { position: absolute; width: 16px; height: 16px; }
  .corner.tl { top: 8px; left: 8px; border-top: 1.5px solid var(--red); border-left: 1.5px solid var(--red); }
  .corner.tr { top: 8px; right: 8px; border-top: 1.5px solid var(--red); border-right: 1.5px solid var(--red); }
  .corner.bl { bottom: 8px; left: 8px; border-bottom: 1.5px solid var(--red); border-left: 1.5px solid var(--red); }
  .corner.br { bottom: 8px; right: 8px; border-bottom: 1.5px solid var(--red); border-right: 1.5px solid var(--red); }

  .rec-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--red); margin-right: 6px; animation: blink 1.2s ease-in-out infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

  .hero-label { font-family: var(--mono); font-size: 11px; color: var(--red); letter-spacing: .1em; text-transform: uppercase; margin-bottom: 1rem; display: flex; align-items: center; position: relative; z-index: 1; }

  .hero h1 { font-size: 22px; font-weight: 600; line-height: 1.25; position: relative; z-index: 1; margin-bottom: .5rem; }
  .hero p { font-size: 14px; color: var(--muted); line-height: 1.6; position: relative; z-index: 1; max-width: 500px; }

  /* CLASS PILLS */
  .classes { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 1.25rem; position: relative; z-index: 1; }
  .pill {
    font-family: var(--mono); font-size: 11px; padding: 4px 10px; border-radius: 4px; letter-spacing: .03em;
    display: flex; align-items: center; gap: 6px;
  }
  .pill-dot { width: 6px; height: 6px; border-radius: 50%; }
  .pill.normal { background: rgba(15,110,86,.12); color: #0F6E56; border: 0.5px solid rgba(15,110,86,.3); }
  .pill.theft { background: rgba(186,117,23,.12); color: #854F0B; border: 0.5px solid rgba(186,117,23,.3); }
  .pill.violence { background: rgba(226,75,74,.12); color: #A32D2D; border: 0.5px solid rgba(226,75,74,.3); }
  .pill.damage { background: rgba(24,95,165,.12); color: #0C447C; border: 0.5px solid rgba(24,95,165,.3); }

  /* SECTION */
  .section { margin-bottom: 2rem; }
  .section-label { font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); margin-bottom: .75rem; display: flex; align-items: center; gap: 8px; }
  .section-label::after { content: ''; flex: 1; height: 0.5px; background: var(--border); }

  /* PIPELINE */
  .pipeline { display: flex; flex-direction: column; gap: 0; }
  .pipe-step {
    display: flex; align-items: stretch; gap: 0;
    cursor: pointer; transition: opacity .15s;
  }
  .pipe-step:hover { opacity: .8; }

  .pipe-left { width: 36px; display: flex; flex-direction: column; align-items: center; }
  .pipe-node { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; margin-top: 17px; }
  .pipe-line { flex: 1; width: 1px; background: var(--border); }

  .pipe-card {
    flex: 1; margin: 4px 0; padding: 10px 12px; border: 0.5px solid var(--border);
    border-radius: 8px; background: var(--bg);
  }
  .pipe-card-title { font-size: 13px; font-weight: 500; margin-bottom: 2px; }
  .pipe-card-desc { font-size: 12px; color: var(--muted); line-height: 1.4; }

  /* RESULTS TABLE */
  .results-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .results-table th {
    font-family: var(--mono); font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted); font-weight: 400; text-align: left; padding: 6px 10px;
    border-bottom: 0.5px solid var(--border);
  }
  .results-table td { padding: 8px 10px; border-bottom: 0.5px solid var(--border); }
  .results-table tr:last-child td { border-bottom: none; }

  .bar-cell { display: flex; align-items: center; gap: 8px; }
  .bar-bg { flex: 1; height: 5px; background: var(--bg2); border-radius: 3px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; }
  .bar-val { font-family: var(--mono); font-size: 11px; color: var(--muted); min-width: 36px; text-align: right; }

  .class-tag { font-family: var(--mono); font-size: 11px; font-weight: 700; }

  /* TECH GRID */
  .tech-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
  .tech-card {
    padding: 10px 12px; border: 0.5px solid var(--border); border-radius: 8px; background: var(--bg);
  }
  .tech-role { font-size: 11px; color: var(--muted); margin-bottom: 3px; }
  .tech-name { font-family: var(--mono); font-size: 13px; font-weight: 700; }

  /* APPS */
  .app-list { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
  .app-item { padding: 10px 12px; border: 0.5px solid var(--border); border-radius: 8px; background: var(--bg); }
  .app-title { font-size: 13px; font-weight: 500; margin-bottom: 3px; }
  .app-desc { font-size: 12px; color: var(--muted); line-height: 1.4; }

  /* DATASET */
  .dataset-card {
    padding: 14px 16px; border: 0.5px solid var(--border); border-radius: 8px; background: var(--bg);
    display: flex; gap: 16px; align-items: flex-start;
  }
  .dataset-icon { font-family: var(--mono); font-size: 20px; color: var(--muted); flex-shrink: 0; }
  .dataset-name { font-size: 14px; font-weight: 500; margin-bottom: 4px; }
  .dataset-desc { font-size: 12px; color: var(--muted); line-height: 1.5; }
  .dataset-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .badge { font-family: var(--mono); font-size: 10px; padding: 3px 7px; border-radius: 3px; background: var(--bg2); border: 0.5px solid var(--border); color: var(--muted); }

  /* FOOTER */
  .footer { font-family: var(--mono); font-size: 11px; color: var(--muted); text-align: center; margin-top: 2.5rem; letter-spacing: .05em; }
</style>
</head>
<body>
<div class="page">

  <!-- HERO -->
  <div class="hero">
    <div class="hero-grid"></div>
    <div class="hero-scan"></div>
    <div class="corner tl"></div><div class="corner tr"></div>
    <div class="corner bl"></div><div class="corner br"></div>

    <div class="hero-label"><span class="rec-dot"></span>Live threat detection</div>
    <h1>Real-Time Anomaly Detection<br>in CCTV Surveillance</h1>
    <p>A deep learning system that watches surveillance footage and automatically flags threatening activity in real time, without waiting for human review after the fact.</p>

    <div class="classes">
      <div class="pill normal"><span class="pill-dot" style="background:#0F6E56"></span>normal</div>
      <div class="pill theft"><span class="pill-dot" style="background:#854F0B"></span>theft</div>
      <div class="pill violence"><span class="pill-dot" style="background:#A32D2D"></span>violence</div>
      <div class="pill damage"><span class="pill-dot" style="background:#0C447C"></span>property_damage</div>
    </div>
  </div>

  <!-- PIPELINE -->
  <div class="section">
    <div class="section-label">Architecture</div>
    <div class="pipeline" id="pipeline"></div>
  </div>

  <!-- RESULTS -->
  <div class="section">
    <div class="section-label">Results</div>
    <div style="border: 0.5px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--bg);">
      <table class="results-table" id="results-table"></table>
    </div>
    <p style="font-size:11px;color:var(--muted);margin-top:6px;font-family:var(--mono);">Stratified test split, 15% holdout</p>
  </div>

  <!-- APPLICATIONS -->
  <div class="section">
    <div class="section-label">Applications</div>
    <div class="app-list" id="apps"></div>
  </div>

  <!-- TECH STACK -->
  <div class="section">
    <div class="section-label">Tech stack</div>
    <div class="tech-grid" id="tech-grid"></div>
  </div>

  <!-- DATASET -->
  <div class="section">
    <div class="section-label">Dataset</div>
    <div class="dataset-card">
      <div class="dataset-icon">[]</div>
      <div>
        <div class="dataset-name">UCF Crime Dataset</div>
        <div class="dataset-desc">Real-world CCTV footage across 13 anomaly categories, consolidated into 4 classes. Class imbalance handled through per-class window oversampling and weighted loss during training.</div>
        <div class="dataset-badges">
          <span class="badge">13 raw categories</span>
          <span class="badge">4 output classes</span>
          <span class="badge">oversampling</span>
          <span class="badge">weighted loss</span>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">GPU recommended (P100 / T4) &nbsp;|&nbsp; Kaggle notebook project</div>
</div>

<script>
const steps = [
  { label: 'Video frame', desc: 'Raw input from live CCTV stream or video file', color: '#888780' },
  { label: 'CLAHE contrast enhancement', desc: 'Improves visibility in dark or low-quality footage', color: '#0F6E56' },
  { label: 'YOLOv8 person detection + crop', desc: 'Focuses the model on people, not background noise', color: '#185FA5' },
  { label: 'Top-16 motion frame selection', desc: 'Picks the most action-rich frames from a 64-frame buffer', color: '#854F0B' },
  { label: 'ResNet50 feature extraction', desc: 'Pretrained CNN outputs a 2048-dim vector per frame', color: '#A32D2D' },
  { label: 'BiLSTM classifier', desc: 'Learns temporal patterns across the 16-frame sequence', color: '#533AB7' },
  { label: '4-class prediction + smoothing', desc: 'Rolling window reduces single-frame flicker', color: '#0F6E56' },
];

const pipeline = document.getElementById('pipeline');
steps.forEach((s, i) => {
  const isLast = i === steps.length - 1;
  pipeline.innerHTML += `
    <div class="pipe-step">
      <div class="pipe-left">
        <div class="pipe-node" style="background:${s.color}"></div>
        ${!isLast ? '<div class="pipe-line"></div>' : ''}
      </div>
      <div class="pipe-card">
        <div class="pipe-card-title">${s.label}</div>
        <div class="pipe-card-desc">${s.desc}</div>
      </div>
    </div>`;
});

const results = [
  { cls: 'normal',          color: '#0F6E56', prec: 0.9495, rec: 0.8710, f1: 0.9086 },
  { cls: 'theft',           color: '#854F0B', prec: 0.8350, rec: 0.8750, f1: 0.8545 },
  { cls: 'violence',        color: '#A32D2D', prec: 0.8586, rec: 0.9111, f1: 0.8841 },
  { cls: 'property_damage', color: '#0C447C', prec: 0.8703, rec: 0.9244, f1: 0.8966 },
];

const table = document.getElementById('results-table');
table.innerHTML = `<thead><tr>
  <th>Class</th>
  <th>Precision</th>
  <th>Recall</th>
  <th>F1</th>
</tr></thead><tbody>` + results.map(r => `
  <tr>
    <td><span class="class-tag" style="color:${r.color}">${r.cls}</span></td>
    <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${(r.prec*100).toFixed(1)}%;background:${r.color}"></div></div><span class="bar-val">${r.prec.toFixed(4)}</span></div></td>
    <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${(r.rec*100).toFixed(1)}%;background:${r.color}"></div></div><span class="bar-val">${r.rec.toFixed(4)}</span></div></td>
    <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${(r.f1*100).toFixed(1)}%;background:${r.color}"></div></div><span class="bar-val">${r.f1.toFixed(4)}</span></div></td>
  </tr>`).join('') + '</tbody>';

const apps = [
  { title: 'Security operations', desc: 'Automated pre-screening across multi-camera feeds' },
  { title: 'Retail loss prevention', desc: 'Real-time theft flagging at scale' },
  { title: 'Smart city CCTV', desc: 'Large-scale public incident detection' },
  { title: 'Forensic review', desc: 'Fast search through archived footage' },
];
document.getElementById('apps').innerHTML = apps.map(a =>
  `<div class="app-item"><div class="app-title">${a.title}</div><div class="app-desc">${a.desc}</div></div>`
).join('');

const tech = [
  { role: 'Person detection', name: 'YOLOv8n' },
  { role: 'Feature extraction', name: 'ResNet50' },
  { role: 'Temporal model', name: 'Bidirectional LSTM' },
  { role: 'Video decoding', name: 'Decord' },
  { role: 'Frame processing', name: 'OpenCV' },
  { role: 'Motion scoring', name: 'PyTorch' },
];
document.getElementById('tech-grid').innerHTML = tech.map(t =>
  `<div class="tech-card"><div class="tech-role">${t.role}</div><div class="tech-name">${t.name}</div></div>`
).join('');
</script>
</body>
</html>
