import { ChartView } from './chart.js';
import { WS } from './ws.js';
import { mountClock } from './clock.js';
import { mountTrade, renderAccount } from './trade.js';

async function boot() {
  const trader = prompt('Trader name?', 'anon') || 'anon';
  const res = await fetch('/api/sessions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trader, config: { balance: 10000, spread: 0.30, max_leverage: 20 } }),
  });
  const { session_id } = await res.json();
  window.__sid = session_id;

  const chart = new ChartView(document.getElementById('chart'));
  const candles = (await (await fetch(`/api/sessions/${session_id}/candles`)).json()).candles;
  window.__allCandles = candles;   // cached full history for seek re-render
  chart.setHistory([]);            // start empty; the replay streams bars in

  const ws = new WS(session_id, (msg) => onMessage(msg, chart));
  await ws.ready;
  window.__ws = ws;
  window.__chart = chart;

  mountClock(ws, chart);
  mountTrade(ws);
}

function onMessage(msg, chart) {
  switch (msg.type) {
    case 'bar': {
      chart.appendBar(msg.bar);
      const ro = document.getElementById('clock-readout');
      ro.textContent = new Date(msg.bar.t).toISOString().replace('T', ' ').slice(0, 16) + ' UTC';
      ro.dataset.ts = String(msg.bar.t);
      break;
    }
    case 'account':
      renderAccount(msg);
      break;
    case 'seeked':
      chart.setHistory(window.__allCandles.filter((c) => c.t <= msg.to));
      break;
    default:
      break;
  }
}

boot();
