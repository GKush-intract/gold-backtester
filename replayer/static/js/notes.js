export function mountNotes(ws, sessionId) {
  const el = document.getElementById('notes');
  el.innerHTML = `<h4>Notes</h4>
    <textarea id="note-text" rows="3" placeholder="Why this trade?"></textarea>
    <div class="row"><button id="note-send">Save note</button>
      <button id="rec">● Record</button><span id="rec-state"></span></div>`;

  el.querySelector('#note-send').onclick = () => {
    const t = el.querySelector('#note-text');
    if (t.value.trim()) { ws.send({ kind: 'note_text', data: { text: t.value.trim() } }); t.value = ''; }
  };

  let recorder = null, chunks = [];
  const recBtn = el.querySelector('#rec');
  const state = el.querySelector('#rec-state');
  recBtn.onclick = async () => {
    if (recorder && recorder.state === 'recording') { recorder.stop(); return; }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      const blob = new Blob(chunks, { type: 'audio/webm' });
      const fd = new FormData();
      fd.append('file', blob, 'note.webm');
      const marketTs = document.getElementById('clock-readout').dataset.ts || '';
      if (marketTs) fd.append('market_ts', marketTs);
      await fetch(`/api/sessions/${sessionId}/audio`, { method: 'POST', body: fd });
      state.textContent = 'saved';
      recBtn.textContent = '● Record';
      stream.getTracks().forEach((t) => t.stop());
    };
    recorder.start();
    recBtn.textContent = '■ Stop';
    state.textContent = 'recording…';
  };
}
