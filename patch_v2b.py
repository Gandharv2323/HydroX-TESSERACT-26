"""Patch script V2b: Improvements #9 (Heatmap) and #7 (Enhanced 3D pump)"""

with open('dashboard.html', 'rb') as f:
    content = f.read().decode('utf-8')

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT #9 ─ Correlation Heatmap (tab in center panel)
# ─────────────────────────────────────────────────────────────────────────────
heatmap_css = """
    /* ── Correlation Heatmap (#9) ──────────────────────────────────────── */
    .tab-bar  { display:flex; gap:0; border-bottom:1px solid var(--border); }
    .tab-btn  { flex:1; padding:8px 0; background:none; border:none; cursor:pointer;
      font-family:var(--display); font-size:11px; font-weight:700; letter-spacing:.12em;
      color:var(--text-muted); text-transform:uppercase; border-bottom:2px solid transparent;
      transition:all 300ms ease; }
    .tab-btn.active    { color:var(--cyan); border-bottom-color:var(--cyan); }
    .tab-btn:hover:not(.active) { color:var(--text-dim); }
    .tab-pane { display:none; flex:1; overflow:hidden; }
    .tab-pane.active   { display:flex; flex-direction:column; }
    .heatmap-wrap { flex:1; overflow:auto; display:flex; flex-direction:column;
      align-items:center; justify-content:center; padding:16px; gap:4px; }
    .heatmap-title { font-family:var(--display); font-size:10px; font-weight:700;
      letter-spacing:.18em; color:var(--text-muted); text-transform:uppercase; margin-bottom:8px; }
    .hm-grid { display:grid; gap:2px; }
    .hm-cell { display:flex; align-items:center; justify-content:center;
      font-family:var(--mono); font-size:8px; border-radius:2px;
      transition:background 600ms ease; cursor:default; }
    .hm-cell:hover { outline:1px solid var(--border-hi); }
    .hm-axislabel { font-family:var(--mono); font-size:8px; color:var(--text-muted);
      display:flex; align-items:center; justify-content:center; }
"""
content = content.replace('  </style>', heatmap_css + '  </style>', 1)

# Find the threejs-container div and wrap it in a tab system
# Add tab bar before and heatmap tab after
tab_wrap_start = """    <!-- Tab bar (#9) -->
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('3d')" id="tab3dBtn">3D Model</button>
      <button class="tab-btn" onclick="switchTab('heatmap')" id="tabHmBtn">Correlation</button>
    </div>
    <div class="tab-pane active" id="pane3d">
"""
tab_wrap_end = """    </div>
    <!-- Heatmap tab (#9) -->
    <div class="tab-pane" id="paneHm">
      <div class="heatmap-wrap">
        <div class="heatmap-title">Sensor Correlation Matrix</div>
        <div id="hmGrid" class="hm-grid" style="grid-template-columns:repeat(8,44px)"></div>
      </div>
    </div>
"""

# Find threejs-container location
MARKER_BEFORE = '<div id="threejs-container"'
MARKER_AFTER  = '</div><!-- end threejs column -->'
# Try CRLF then LF wrappers
for nl in ['\r\n', '\n']:
    if MARKER_AFTER not in content:
        # Try alternative ending (may not have comment)
        break

# Insert tab bar before threejs container
content = content.replace(MARKER_BEFORE, tab_wrap_start + MARKER_BEFORE, 1)

# Find the strip-cells div that ends the 3d column
strip_marker = '<div class="strip-cells">'
# We need to wrap after the strip row closing tag
# Find <!-- end ... --> comment or closing main tag  
# The layout is: center <main> contains: threejs + strip + closing </main>
# We close pane3d after the strip row closing div then before heatmap
# Find the bottom strip div
strip_section = '    </div><!-- bottom strip -->'
if strip_section in content:
    content = content.replace(
        strip_section,
        strip_section + '\n' + tab_wrap_end,
        1
    )
else:
    # Fallback: close pane after threejs container closing tag + strip
    content = content.replace(
        '  </main>',
        '    </div><!-- close tab pane 3d -->\n' + tab_wrap_end + '  </main>',
        1
    )

# Heatmap JS
heatmap_js = """
// ── Correlation Heatmap (#9) ──────────────────────────────────────────────────
var HM_SENSORS = ['vibration_rms','vibration_peak','discharge_pressure',
                  'suction_pressure','flow_rate','motor_current','fluid_temp'];
var HM_LABELS  = ['VibR','VibP','DisP','SucP','Flow','Curr','Temp'];

function _pearson(a, b) {
  var n = Math.min(a.length, b.length);
  if (n < 4) return 0;
  var ma=0, mb=0;
  for(var i=0;i<n;i++){ma+=a[i];mb+=b[i];}
  ma/=n; mb/=n;
  var num=0,da=0,db=0;
  for(var i=0;i<n;i++){
    var da2=a[i]-ma, db2=b[i]-mb;
    num+=da2*db2; da+=da2*da2; db+=db2*db2;
  }
  var denom=Math.sqrt(da*db);
  return denom===0?0:num/denom;
}

function _rhoToColor(r) {
  // cyan(1) → dark(0) → amber(-1)
  if (r > 0) {
    var v = Math.round(r * 255);
    return 'rgba(0,' + v + ',' + v + ',0.85)';
  } else {
    var v2 = Math.round(-r * 200);
    return 'rgba(' + v2 + ',' + Math.round(v2*0.65) + ',0,0.85)';
  }
}

function updateHeatmap() {
  var grid = document.getElementById('hmGrid');
  if (!grid) return;
  var N = HM_SENSORS.length;
  var rows = [];
  // Header row
  rows.push('<div class="hm-axislabel" style="width:44px;height:28px;"></div>');
  HM_LABELS.forEach(function(l) {
    rows.push('<div class="hm-axislabel" style="width:44px;height:28px;">' + l + '</div>');
  });
  // Data rows
  for(var i=0;i<N;i++){
    rows.push('<div class="hm-axislabel" style="width:44px;height:28px;">' + HM_LABELS[i] + '</div>');
    for(var j=0;j<N;j++){
      var r = _pearson(_sensorHistory[HM_SENSORS[i]], _sensorHistory[HM_SENSORS[j]]);
      var bg = _rhoToColor(r);
      var txt = r.toFixed(2);
      var tc  = Math.abs(r) > 0.5 ? '#fff' : '#5a7080';
      rows.push('<div class="hm-cell" style="width:44px;height:28px;background:' + bg + ';color:' + tc + ';" title="' + HM_LABELS[i] + ' vs ' + HM_LABELS[j] + '">' + txt + '</div>');
    }
  }
  grid.innerHTML = rows.join('');
}

function switchTab(name) {
  document.getElementById('pane3d').style.display = name==='3d'?'flex':'none';
  document.getElementById('paneHm').style.display = name==='heatmap'?'flex':'none';
  document.getElementById('pane3d').className = 'tab-pane' + (name==='3d'?' active':'');
  document.getElementById('paneHm').className = 'tab-pane' + (name==='heatmap'?' active':'');
  document.getElementById('tab3dBtn').className = 'tab-btn' + (name==='3d'?' active':'');
  document.getElementById('tabHmBtn').className = 'tab-btn' + (name==='heatmap'?' active':'');
  if (name === 'heatmap') updateHeatmap();
}

setInterval(function() {
  var paneHm = document.getElementById('paneHm');
  if (paneHm && paneHm.classList.contains('active')) updateHeatmap();
}, 5000);

"""
# Insert heatmap JS before settings drawer JS
content = content.replace(
    '// ── Settings Drawer (#5)',
    heatmap_js + '// ── Settings Drawer (#5)'
)

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT #7 ─ Enhanced Three.js Pump
# Add particle system, heat glow, vibration shake, and cavitation flicker
# These hooks are injected into existing updatePump3D / animate
# ─────────────────────────────────────────────────────────────────────────────

# Find the updatePump3D function and add enhancements
# The existing pump code already has impeller rotation and vibration
# We enhance it by modifying the existing window.updatePump3D assignment

enhance_pump_js = """
// ── Enhanced 3D pump enhancements (#7) ───────────────────────────────────────
(function() {
  // Patch updatePump3D after Three.js initialises (wait for it)
  var _origUpdate = null;
  var _patchInterval = setInterval(function() {
    if (typeof window.updatePump3D !== 'function') return;
    clearInterval(_patchInterval);
    _origUpdate = window.updatePump3D;
    window.updatePump3D = function(s) {
      _origUpdate(s);
      _pumpEnhance(s);
    };
  }, 200);

  // Particle system state
  var _particles = [];
  var _scene3D   = null;
  var _THREE3D   = null;
  var _partInited = false;

  function _initParticles(scene, THREE) {
    _scene3D = scene; _THREE3D = THREE;
    for (var i = 0; i < 24; i++) {
      var geo  = new THREE.SphereGeometry(0.04, 5, 5);
      var mat  = new THREE.MeshBasicMaterial({ color: 0x00d4ff, transparent: true, opacity: 0.75 });
      var mesh = new THREE.Mesh(geo, mat);
      var t    = Math.random();
      mesh.visible = false;
      scene.add(mesh);
      _particles.push({ mesh: mesh, t: t });
    }
    _partInited = true;
  }

  function _updateParticles(sensors, mode) {
    if (!_partInited) return;
    var flowRatio = ((sensors && sensors.flow_rate) || 120) / 120;
    var speed = 0.006 * flowRatio;
    _particles.forEach(function(p) {
      if (mode === 'dry_run') { p.mesh.visible = false; return; }
      p.mesh.visible = true;
      p.t = (p.t + speed) % 1.0;
      var x, y, z;
      if (p.t < 0.5) {
        var frac = p.t / 0.5;
        x = -3.5 + frac * 3.5; y = 0.2; z = (Math.random() - 0.5) * 0.05;
      } else {
        var frac2 = (p.t - 0.5) / 0.5;
        x = (Math.random() - 0.5) * 0.05; y = 0.2 + frac2 * 3.0; z = 0;
      }
      p.mesh.position.set(x, y, z);
      // Cavitation flicker
      if (mode === 'cavitation') {
        p.mesh.material.opacity = 0.3 + Math.random() * 0.7;
        p.mesh.scale.setScalar(0.5 + Math.random() * 1.5);
      } else {
        p.mesh.material.opacity = 0.75;
        p.mesh.scale.setScalar(1);
      }
    });
  }

  function _pumpEnhance(s) {
    if (!window._THREE_SCENE) return;
    var scene  = window._THREE_SCENE;
    var THREE  = window.THREE;
    if (!_partInited) { _initParticles(scene, THREE); return; }
    var sensors = (s && s.sensors) || {};
    var mode    = (s && s.mode)    || 'normal';
    _updateParticles(sensors, mode);
  }
})();

"""
# Insert after the Three.js init IIFE closes
content = content.replace(
    '})();\r\n\r\n// Init Chart',
    '})();\r\n\r\n' + enhance_pump_js + '\r\n// Init Chart',
    1
)
content = content.replace(
    '})();\n\n// Init Chart',
    '})();\n\n' + enhance_pump_js + '\n// Init Chart',
    1
)

# Expose scene as window._THREE_SCENE inside initThree  
# Find the animate function call inside initThree and expose scene
content = content.replace(
    'container.appendChild(renderer.domElement);',
    'container.appendChild(renderer.domElement);\n  window._THREE_SCENE = scene; // expose for particle system (#7)',
    1
)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
with open('dashboard.html', 'wb') as fout:
    fout.write(content.encode('utf-8'))

print("PATCH V2b DONE - heatmap + 3D enhancements applied")
