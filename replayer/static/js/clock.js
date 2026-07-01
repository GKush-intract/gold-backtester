const SPEEDS = ['1', '4', '15', '60', '240', 'MAX'];

export function mountClock(ws, chart) {
  const tfWrap = document.getElementById('tf-buttons');
  ['m1','m5','m15','m30','h1','h4','d1'].forEach((tf) => {
    const b = document.createElement('button');
    b.textContent = tf;
    if (tf === chart.tf) b.classList.add('active');
    b.onclick = () => {
      chart.setTimeframe(tf);
      [...tfWrap.children].forEach((c) => c.classList.remove('active'));
      b.classList.add('active');
      ws.send({ kind: 'tf_change', data: { timeframe: tf } });
    };
    tfWrap.appendChild(b);
  });

  const controls = document.getElementById('controls');
  let playing = false, speed = '60';
  const playBtn = document.createElement('button');
  playBtn.textContent = '▶ Play';
  playBtn.onclick = () => {
    playing = !playing;
    playBtn.textContent = playing ? '⏸ Pause' : '▶ Play';
    ws.send(playing ? { kind: 'control', action: 'play', speed: numSpeed(speed) }
                    : { kind: 'control', action: 'pause' });
  };
  const stepBtn = document.createElement('button');
  stepBtn.textContent = '⏭ Step';
  stepBtn.onclick = () => ws.send({ kind: 'control', action: 'step' });
  const skipBtn = document.createElement('button');
  skipBtn.textContent = '⏩ Skip gap';
  skipBtn.onclick = () => ws.send({ kind: 'control', action: 'skip_gap' });
  controls.append(playBtn, stepBtn, skipBtn);

  const speedSel = document.getElementById('speed');
  const sel = document.createElement('select');
  SPEEDS.forEach((s) => { const o = document.createElement('option'); o.value = s; o.textContent = s + '×'; sel.appendChild(o); });
  sel.value = speed;
  sel.onchange = () => {
    speed = sel.value;
    ws.send({ kind: 'control', action: 'speed', speed: numSpeed(speed) });
  };
  speedSel.appendChild(sel);
}

function numSpeed(s) { return s === 'MAX' ? 100000 : parseInt(s, 10); }
