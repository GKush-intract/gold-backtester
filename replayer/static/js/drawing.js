const TOOLS = ['horizontalStraightLine', 'straightLine', 'rayLine', 'rectangle', 'fibonacciLine'];

export function mountDrawing(ws, chart) {
  const bar = document.createElement('span');
  TOOLS.forEach((tool) => {
    const b = document.createElement('button');
    b.textContent = tool.replace(/([A-Z])/g, ' $1').split(' ')[0];
    b.title = tool;
    b.onclick = () => chart.chart.createOverlay({ name: tool });
    bar.appendChild(b);
  });
  document.getElementById('controls').appendChild(bar);

  // Log overlay lifecycle. Poll the overlay snapshot when the count changes (robust across
  // klinecharts versions), and also try the onDrawEnd action hook if present.
  chart.chart.subscribeAction?.('onDrawEnd', () => flush(ws, chart));
  let lastCount = -1;
  setInterval(() => {
    const overlays = chart.chart.getOverlays ? chart.chart.getOverlays() : [];
    if (overlays.length !== lastCount) {
      lastCount = overlays.length;
      ws.send({ kind: 'draw', data: { action: 'snapshot', overlays: serialize(overlays) } });
    }
  }, 1000);
}

function serialize(overlays) {
  return (overlays || []).map((o) => ({
    id: o.id, name: o.name,
    points: (o.points || []).map((p) => ({ ts: p.timestamp, price: p.value })),
  }));
}

function flush(ws, chart) {
  const overlays = chart.chart.getOverlays ? chart.chart.getOverlays() : [];
  ws.send({ kind: 'draw', data: { action: 'draw_end', overlays: serialize(overlays) } });
}
