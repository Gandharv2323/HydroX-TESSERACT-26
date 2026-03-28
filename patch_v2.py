"""Patch script: inject all remaining v2 improvements into dashboard.html"""
import re

f = open('dashboard.html', 'rb')
content = f.read().decode('utf-8')
f.close()

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT #2 ─ Event Log
# ─────────────────────────────────────────────────────────────────────────────
event_css = """
    /* ── Event Log (#2) ───────────────────────────────────────────────── */
    .event-log-header { font-family:var(--display); font-size:10px; font-weight:700;
      letter-spacing:.18em; color:var(--text-muted); text-transform:uppercase;
      padding:14px 12px 6px; }
    .event-log { display:flex; flex-direction:column; gap:3px;
      max-height:160px; overflow-y:auto; padding:0 12px 12px; }
    .log-entry { display:flex; gap:8px; font-family:var(--mono); font-size:10px;
      padding:4px 8px; background:var(--surface-hi); border-radius:var(--r);
      border-left:2px solid var(--border); }
    .log-ts  { color:var(--text-muted); white-space:nowrap; }
    .log-msg { color:var(--text-dim); }
    .log-entry.ev-info { border-left-color:var(--cyan); }
    .log-entry.ev-info .log-msg { color:var(--cyan); }
    .log-entry.ev-warn { border-left-color:var(--amber); }
    .log-entry.ev-warn .log-msg { color:var(--amber); }
    .log-entry.ev-crit { border-left-color:var(--red); }
    .log-entry.ev-crit .log-msg { color:var(--red); }
"""
content = content.replace('  </style>', event_css + '  </style>', 1)

# Event log HTML: add at sidebar bottom before its closing </aside>
# Sidebar closing is unique: it is followed by  <!-- ── Main
event_html = """
    <div class="divider"></div>
    <div class="event-log-header">Event Log</div>
    <div class="event-log" id="eventLog"></div>
"""
content = content.replace(
    '  </aside>\r\n\r\n  <!-- ',
    event_html + '  </aside>\r\n\r\n  <!-- ', 1
)
# unix fallback
content = content.replace(
    '  </aside>\n\n  <!-- ',
    event_html + '  </aside>\n\n  <!-- ', 1
)

# Event log JS globals
event_globals = """
// ── Event Log globals (#2) ───────────────────────────────────────────────────
var _evLog = [], _prevIsAnomaly = null, _prevStatus = null;
function _addEvent(level, msg) {
  var ts = new Date().toLocaleTimeString('en-GB', {hour12:false});
  _evLog.unshift({ts: ts, level: level, msg: msg});
  if (_evLog.length > 20) _evLog.pop();
  var el = document.getElementById('eventLog');
  if (!el) return;
  el.innerHTML = _evLog.map(function(e) {
    return '<div class="log-entry ev-' + e.level + '"><span class="log-ts">' +
           e.ts + '</span><span class="log-msg">' + e.msg + '</span></div>';
  }).join('');
}

"""
content = content.replace('// exportReport (#8)', event_globals + '// exportReport (#8)', 1)

# Event detection hook inside applyState
event_hook = """
  // Event log detection (#2)
  var curMode = s.mode || 'normal';
  if (_prevIsAnomaly !== null) {
    if (anomaly.is_anomaly  && !_prevIsAnomaly) _addEvent('crit', 'ANOMALY: ' + (anomaly.failure_mode||'?').replace(/_/g,' '));
    if (!anomaly.is_anomaly &&  _prevIsAnomaly) _addEvent('info', 'Anomaly cleared \u2014 normal');
    if (health.status === 'warning'  && _prevStatus === 'healthy')  _addEvent('warn', 'Health crossed WARNING');
    if (health.status === 'critical' && _prevStatus !== 'critical') _addEvent('crit', 'Health CRITICAL \u2014 action required');
    if (health.status === 'healthy'  && _prevStatus !== 'healthy')  _addEvent('info', 'Health restored: HEALTHY');
    if (curMode !== (currentState ? currentState.mode : curMode))   _addEvent('info', 'Mode \u2192 ' + curMode);
  }
  _prevIsAnomaly = anomaly.is_anomaly;
  _prevStatus    = health.status;
"""
content = content.replace('  // Drive Three.js', event_hook + '  // Drive Three.js', 1)

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT #1 ─ Sensor History Chart (Chart.js)  
# ─────────────────────────────────────────────────────────────────────────────
chart_css = """
    /* ── Sensor Chart (#1) ─────────────────────────────────────────────── */
    .chart-wrap { padding:0 12px 12px; }
    .chart-header { display:flex; align-items:center; gap:8px; padding:10px 0 6px;
      font-family:var(--display); font-size:10px; font-weight:700;
      letter-spacing:.18em; color:var(--text-muted); text-transform:uppercase; }
    .chart-header select { background:var(--surface-hi); border:1px solid var(--border);
      color:var(--text-primary); font-family:var(--mono); font-size:10px;
      padding:2px 6px; border-radius:var(--r); cursor:pointer; flex:1; }
    .chart-canvas-wrap { position:relative; height:100px; }
"""
content = content.replace('  </style>', chart_css + '  </style>', 1)

# Chart HTML: add above sparkline-wrap (search for sparkline-label)
chart_html = """    <div class="chart-wrap">
      <div class="chart-header">
        <span>SENSOR HISTORY</span>
        <select id="chartSensorSel" onchange="switchChartSensor(this.value)">
          <option value="vibration_rms">Vibration RMS</option>
          <option value="discharge_pressure">Discharge P</option>
          <option value="suction_pressure">Suction P</option>
          <option value="flow_rate">Flow Rate</option>
          <option value="motor_current">Motor I</option>
          <option value="fluid_temp">Fluid Temp</option>
        </select>
      </div>
      <div class="chart-canvas-wrap">
        <canvas id="sensorHistChart"></canvas>
      </div>
    </div>
    <div class="divider"></div>
"""
content = content.replace(
    '<div class="sparkline-wrap">',
    chart_html + '<div class="sparkline-wrap">',
    1
)

# Add Chart.js CDN before Three.js script tag
content = content.replace(
    '<!-- Three.js CDN -->',
    '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>\r\n<!-- Three.js CDN -->'
)
content = content.replace(
    '<!-- Three.js CDN -->',
    '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>\n<!-- Three.js CDN -->'
)

# Chart.js globals & init
chart_js = """
// ── Sensor History Chart (#1) ────────────────────────────────────────────────
var _sensorHistory = {
  vibration_rms:[], discharge_pressure:[], suction_pressure:[],
  flow_rate:[], motor_current:[], fluid_temp:[]
};
var _chartSensor = 'vibration_rms';
var _sensorChart = null;

function _initSensorChart() {
  var ctx = document.getElementById('sensorHistChart');
  if (!ctx) return;
  _sensorChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        data: [],
        borderColor: '#00d4ff',
        backgroundColor: 'rgba(0,212,255,0.08)',
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 1.5,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: false,
      plugins: { legend: {display:false} },
      scales: {
        x: { display:false },
        y: { display:true, grid:{color:'rgba(30,45,61,0.5)'}, ticks:{color:'#5a7080',font:{size:9}} }
      }
    }
  });
}

function switchChartSensor(key) {
  _chartSensor = key;
  if (_sensorChart) {
    _sensorChart.data.datasets[0].data = _sensorHistory[key].slice();
    _sensorChart.data.labels = _sensorHistory[key].map(function(_,i){return i;});
    _sensorChart.update('none');
  }
}

function _updateSensorChart(sensors) {
  Object.keys(_sensorHistory).forEach(function(k) {
    if (sensors[k] !== undefined) {
      _sensorHistory[k].push(sensors[k]);
      if (_sensorHistory[k].length > 120) _sensorHistory[k].shift();
    }
  });
  if (!_sensorChart) return;
  var data = _sensorHistory[_chartSensor];
  _sensorChart.data.datasets[0].data = data.slice();
  _sensorChart.data.labels = data.map(function(_,i){return i;});
  _sensorChart.update('none');
}

"""
content = content.replace('// ── Event Log globals (#2)', chart_js + '// ── Event Log globals (#2)', 1)

# Hook chart update inside applyState (before Sparkline history)
content = content.replace(
    '  // Sparkline history',
    '  _updateSensorChart(sensors);\n\n  // Sparkline history'
)

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT #3 ─ Dual Anomaly Arc Gauges
# ─────────────────────────────────────────────────────────────────────────────
arc_css = """
    /* ── Dual Arc Gauges (#3) ──────────────────────────────────────────── */
    .dual-arc-row   { display:flex; gap:8px; padding:12px 12px 0; }
    .arc-gauge-wrap { flex:1; display:flex; flex-direction:column; align-items:center; gap:4px;
      background:var(--surface-hi); border:1px solid var(--border); border-radius:var(--r-lg); padding:10px 8px; }
    .arc-svg { width:80px; height:80px; }
    .arc-track { fill:none; stroke:var(--border); stroke-width:8; }
    .arc-fill  { fill:none; stroke-width:8; stroke-linecap:round;
      transform-origin:50px 50px; transform:rotate(-90deg);
      transition:stroke-dashoffset 0.8s ease, stroke 0.8s ease; }
    .arc-value { font-family:var(--mono); font-size:14px; font-weight:700; color:var(--text-primary); }
    .arc-label { font-family:var(--display); font-size:9px; font-weight:700;
      letter-spacing:.15em; color:var(--text-muted); text-transform:uppercase; }
"""
content = content.replace('  </style>', arc_css + '  </style>', 1)

# Arc HTML: add above the existing anomaly score bar section
arc_html = """    <!-- Dual Arc Gauges (#3) -->
    <div class="dual-arc-row">
      <div class="arc-gauge-wrap">
        <svg class="arc-svg" viewBox="0 0 100 100">
          <circle class="arc-track" cx="50" cy="50" r="38"/>
          <circle class="arc-fill" id="arcScore" cx="50" cy="50" r="38"
            stroke="#00d4ff" stroke-dasharray="239" stroke-dashoffset="239"/>
        </svg>
        <div class="arc-value" id="arcScoreVal">0.000</div>
        <div class="arc-label">Anomaly</div>
      </div>
      <div class="arc-gauge-wrap">
        <svg class="arc-svg" viewBox="0 0 100 100">
          <circle class="arc-track" cx="50" cy="50" r="38"/>
          <circle class="arc-fill" id="arcConf" cx="50" cy="50" r="38"
            stroke="#00e676" stroke-dasharray="239" stroke-dashoffset="239"/>
        </svg>
        <div class="arc-value" id="arcConfVal">0%</div>
        <div class="arc-label">Confidence</div>
      </div>
    </div>
    <div class="divider" style="margin:12px 0 0;"></div>
"""
# Insert before the existing anomaly score bar label
content = content.replace(
    '<div class="section-label">ANOMALY SCORE</div>',
    arc_html + '<div class="section-label">ANOMALY SCORE</div>',
    1
)

# Arc update JS: add inside applyState after the alert banner block
arc_js_hook = """
  // Dual arc gauge update (#3)
  var C = 239; // 2*pi*38
  var sc = anomaly.anomaly_score || 0;
  var cf = anomaly.confidence    || 0;
  var arcS = document.getElementById('arcScore');
  var arcC = document.getElementById('arcConf');
  if (arcS) { arcS.style.strokeDashoffset = (C - sc * C).toFixed(1); arcS.style.stroke = sc>0.6?'#ff3b5c':sc>0.3?'#f0a500':'#00d4ff'; }
  if (arcC) { arcC.style.strokeDashoffset = (C - cf * C).toFixed(1); arcC.style.stroke = cf>0.7?'#00e676':'#f0a500'; }
  var av = document.getElementById('arcScoreVal'); if(av) av.textContent = sc.toFixed(3);
  var cv = document.getElementById('arcConfVal');  if(cv) cv.textContent = (cf*100).toFixed(0)+'%';
"""
content = content.replace(
    '  // Alert banner (#4)',
    arc_js_hook + '  // Alert banner (#4)'
)

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT #10 ─ Maintenance Countdown
# ─────────────────────────────────────────────────────────────────────────────
maint_css = """
    /* ── Maintenance Countdown (#10) ──────────────────────────────────── */
    .maint-wrap  { padding:12px; display:flex; flex-direction:column; align-items:center; gap:6px; }
    .maint-title { font-family:var(--display); font-size:10px; font-weight:700;
      letter-spacing:.15em; color:var(--text-muted); text-transform:uppercase; }
    .maint-timer { font-family:var(--mono); font-size:20px; font-weight:700; color:var(--green); }
    .maint-date  { font-family:var(--mono); font-size:10px; color:var(--text-muted); }
    .maint-ring-wrap { position:relative; width:60px; height:60px; }
    .maint-ring  { width:60px; height:60px; transform:rotate(-90deg); }
    .maint-ring-track { fill:none; stroke:var(--border); stroke-width:6; }
    .maint-ring-fill  { fill:none; stroke:var(--green); stroke-width:6; stroke-linecap:round;
      stroke-dasharray:163; stroke-dashoffset:0;
      transition:stroke-dashoffset 1s ease, stroke 1s ease; }
"""
content = content.replace('  </style>', maint_css + '  </style>', 1)

# Maintenance HTML: add after the RUL value row
maint_html = """
    <!-- Maintenance Countdown (#10) -->
    <div class="maint-wrap">
      <div class="maint-title">Predicted Maintenance</div>
      <div class="maint-ring-wrap">
        <svg class="maint-ring" viewBox="0 0 60 60">
          <circle class="maint-ring-track" cx="30" cy="30" r="26"/>
          <circle class="maint-ring-fill" id="maintRingFill" cx="30" cy="30" r="26"/>
        </svg>
      </div>
      <div class="maint-timer" id="maintTimer">-- h</div>
      <div class="maint-date"  id="maintDate">computing&hellip;</div>
    </div>
    <div class="divider"></div>
"""
content = content.replace(
    '<div class="dual-arc-row">',
    maint_html + '<div class="dual-arc-row">',
    1
)

# Maintenance JS: add in applyState before event log detection
maint_hook = """
  // Maintenance countdown (#10)
  var rul = (health.rul_hours || 0);
  var maintRing = document.getElementById('maintRingFill');
  var maintTimer = document.getElementById('maintTimer');
  var maintDate  = document.getElementById('maintDate');
  if (maintRing && rul > 0) {
    var maxH = 720; // 30 days reference
    var ratio = Math.min(1, rul / maxH);
    var dash = 163;
    maintRing.style.strokeDashoffset = (dash - ratio * dash).toFixed(1);
    maintRing.style.stroke = rul < 24 ? '#ff3b5c' : rul < 168 ? '#f0a500' : '#00e676';
    var days = Math.floor(rul / 24);
    var hrs  = Math.floor(rul % 24);
    maintTimer.textContent = (days > 0 ? days + 'd ' : '') + hrs + 'h';
    var d = new Date(Date.now() + rul * 3600000);
    maintDate.textContent = d.toLocaleDateString('en-GB') + ' ' + d.toLocaleTimeString('en-GB', {hour:'2-digit',minute:'2-digit',hour12:false});
  }
"""
content = content.replace(
    '  // Event log detection (#2)',
    maint_hook + '  // Event log detection (#2)'
)

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT #5 ─ Settings Drawer (Hz + noise sliders → PATCH /config)
# ─────────────────────────────────────────────────────────────────────────────
settings_css = """
    /* ── Settings Drawer (#5) ─────────────────────────────────────────── */
    .settings-gear { background:none; border:none; cursor:pointer; font-size:18px;
      color:var(--text-dim); padding:4px 8px; border-radius:var(--r);
      transition:color var(--transition); line-height:1; }
    .settings-gear:hover { color:var(--cyan); }
    .settings-backdrop { position:fixed; inset:0; z-index:800; background:rgba(0,0,0,.4);
      opacity:0; pointer-events:none; transition:opacity 300ms ease; }
    .settings-backdrop.open { opacity:1; pointer-events:auto; }
    .settings-drawer { position:fixed; top:0; right:0; bottom:0; width:280px; z-index:810;
      background:var(--surface); border-left:1px solid var(--border-hi);
      transform:translateX(100%); transition:transform 350ms cubic-bezier(0.16,1,0.3,1);
      display:flex; flex-direction:column; }
    .settings-drawer.open { transform:translateX(0); }
    .settings-drawer-header { display:flex; align-items:center; justify-content:space-between;
      padding:16px 18px; border-bottom:1px solid var(--border); }
    .settings-drawer-title { font-family:var(--display); font-size:14px; font-weight:700;
      letter-spacing:.12em; color:var(--text-primary); }
    .settings-close { background:none; border:none; cursor:pointer; color:var(--text-dim);
      font-size:20px; padding:2px 6px; border-radius:var(--r); transition:color var(--transition); }
    .settings-close:hover { color:var(--red); }
    .settings-body  { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:20px; }
    .stg-group-label { font-family:var(--display); font-size:10px; font-weight:700;
      letter-spacing:.15em; color:var(--text-muted); text-transform:uppercase; margin-bottom:4px; }
    .stg-row { display:flex; align-items:center; gap:10px; }
    .stg-row label { font-family:var(--mono); font-size:11px; color:var(--text-dim); width:90px; }
    .stg-row input[type=range] { flex:1; accent-color:var(--cyan); }
    .stg-val { font-family:var(--mono); font-size:11px; color:var(--cyan); width:36px; text-align:right; }
    .stg-apply { width:100%; padding:9px; margin-top:4px;
      background:linear-gradient(135deg,#005870,#003040);
      border:1px solid var(--cyan); border-radius:var(--r);
      color:var(--cyan); font-family:var(--display); font-size:12px;
      font-weight:700; letter-spacing:.12em; cursor:pointer;
      transition:background 300ms ease; }
    .stg-apply:hover { background:linear-gradient(135deg,#007090,#004060); }
"""
content = content.replace('  </style>', settings_css + '  </style>', 1)

# Settings gear button in header (add after latency badge)
content = content.replace(
    '</div>\r\n    <div class="header-timestamp"',
    '</div>\r\n    <button class="settings-gear" onclick="openSettings()" title="Settings">&#9881;</button>\r\n    <div class="header-timestamp"',
    1
)
# Unix fallback
content = content.replace(
    '</div>\n    <div class="header-timestamp"',
    '</div>\n    <button class="settings-gear" onclick="openSettings()" title="Settings">&#9881;</button>\n    <div class="header-timestamp"',
    1
)

# Settings drawer HTML: add before closing </body>
settings_html = """
<!-- Settings Drawer (#5) -->
<div class="settings-backdrop" id="settingsBackdrop" onclick="closeSettings()"></div>
<div class="settings-drawer" id="settingsDrawer">
  <div class="settings-drawer-header">
    <div class="settings-drawer-title">&#9881; SETTINGS</div>
    <button class="settings-close" onclick="closeSettings()">&times;</button>
  </div>
  <div class="settings-body">
    <div>
      <div class="stg-group-label">Broadcast Rate</div>
      <div class="stg-row">
        <label>Hz</label>
        <input type="range" id="stgHz" min="0.5" max="10" step="0.5" value="2">
        <span class="stg-val" id="stgHzVal">2</span>
      </div>
    </div>
    <div>
      <div class="stg-group-label">Sensor Noise</div>
      <div class="stg-row">
        <label>Noise %</label>
        <input type="range" id="stgNoise" min="0.5" max="20" step="0.5" value="5">
        <span class="stg-val" id="stgNoiseVal">5</span>
      </div>
    </div>
    <button class="stg-apply" onclick="applySettings()">&#10003; APPLY</button>
  </div>
</div>

"""
content = content.replace('</body>', settings_html + '</body>', 1)

# Settings JS: add before closing </script>
settings_js = """
// ── Settings Drawer (#5) ─────────────────────────────────────────────────────
document.getElementById('stgHz').addEventListener('input', function() {
  document.getElementById('stgHzVal').textContent = this.value;
});
document.getElementById('stgNoise').addEventListener('input', function() {
  document.getElementById('stgNoiseVal').textContent = this.value;
});
function openSettings()  {
  document.getElementById('settingsDrawer').classList.add('open');
  document.getElementById('settingsBackdrop').classList.add('open');
}
function closeSettings() {
  document.getElementById('settingsDrawer').classList.remove('open');
  document.getElementById('settingsBackdrop').classList.remove('open');
}
function applySettings() {
  var hz    = parseFloat(document.getElementById('stgHz').value);
  var noise = parseFloat(document.getElementById('stgNoise').value) / 100;
  fetch(API_URL + '/config', {
    method: 'PATCH',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({broadcast_hz: hz, noise_pct: noise})
  }).then(function(r){return r.json();}).then(function(d){
    console.log('[settings] applied:', d);
    closeSettings();
  }).catch(function(e){ console.error('[settings]', e); });
}

"""
content = content.replace('</script>\r\n</body>', settings_js + '</script>\r\n</body>', 1)
content = content.replace('</script>\n</body>', settings_js + '</script>\n</body>', 1)

# ─────────────────────────────────────────────────────────────────────────────
# Init chart.js after DOM ready: add after connect() call
# ─────────────────────────────────────────────────────────────────────────────
init_chart = """
// Init Chart.js sensor chart (#1)
if (typeof Chart !== 'undefined') { _initSensorChart(); }
else { window.addEventListener('load', _initSensorChart); }
"""
content = content.replace('connect();', 'connect();\n' + init_chart, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
with open('dashboard.html', 'wb') as fout:
    fout.write(content.encode('utf-8'))

print("ALL PATCHES APPLIED OK")
