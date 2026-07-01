let WSREF = null;
let seq = 0;

export function mountTrade(ws) {
  WSREF = ws;
  const ticket = document.getElementById('ticket');
  ticket.innerHTML = `
    <h4>Order ticket</h4>
    <div class="row"><select id="otype"><option value="market">Market</option>
      <option value="limit">Limit</option><option value="stop">Stop</option></select></div>
    <div class="row" id="price-row" style="display:none"><input id="oprice" type="number" step="0.01" placeholder="Price"/></div>
    <div class="row"><input id="oqty" type="number" step="0.01" value="1" placeholder="Qty (lots)"/></div>
    <div class="row"><input id="osl" type="number" step="0.01" placeholder="SL (optional)"/>
      <input id="otp" type="number" step="0.01" placeholder="TP (optional)"/></div>
    <div class="row"><button class="buy" id="buy">BUY</button><button class="sell" id="sell">SELL</button></div>`;
  const otype = ticket.querySelector('#otype');
  otype.onchange = () => {
    ticket.querySelector('#price-row').style.display = otype.value === 'market' ? 'none' : 'flex';
  };
  ticket.querySelector('#buy').onclick = () => submit('buy');
  ticket.querySelector('#sell').onclick = () => submit('sell');

  document.getElementById('orders').innerHTML = '<h4>Working orders</h4><div id="orders-body"></div>';
  document.getElementById('blotter').innerHTML = '<h4>Blotter</h4><div id="blotter-body"></div>';
}

function submit(side) {
  const t = document.getElementById('ticket');
  const order_type = t.querySelector('#otype').value;
  const data = {
    client_id: `c${Date.now()}_${seq++}`,
    side, order_type,
    qty_lots: parseFloat(t.querySelector('#oqty').value),
    price: order_type === 'market' ? null : parseFloat(t.querySelector('#oprice').value),
    sl: numOrNull(t.querySelector('#osl').value),
    tp: numOrNull(t.querySelector('#otp').value),
  };
  WSREF.send({ kind: 'order', data });
}

function numOrNull(v) { return v === '' || v == null ? null : parseFloat(v); }

export function renderAccount(msg) {
  const a = document.getElementById('account');
  const pos = msg.position;
  a.innerHTML = `<h4>Account</h4>
    <table>
      <tr><td>Balance</td><td>${fmt(msg.balance)}</td></tr>
      <tr><td>Equity</td><td>${fmt(msg.equity)}</td></tr>
    </table>` + (pos ? `
    <table>
      <tr><th>Position</th><th>${pos.direction}</th></tr>
      <tr><td>Size (lots)</td><td>${pos.size}</td></tr>
      <tr><td>Entry</td><td>${fmt(pos.entry_price)}</td></tr>
      <tr><td>SL / TP</td><td>${pos.stop_loss ?? '–'} / ${pos.take_profit ?? '–'}</td></tr>
      <tr><td>uPnL</td><td>${fmt(pos.unrealized)}</td></tr>
    </table><button id="flatten" class="sell">CLOSE POSITION</button>` : '<div>No position</div>');
  if (pos) document.getElementById('flatten').onclick = () => WSREF.send({ kind: 'close_position' });

  const ob = document.getElementById('orders-body');
  if (ob) {
    ob.innerHTML = (msg.orders || []).map((o) =>
      `<div class="row"><span>${o.side} ${o.order_type} ${o.qty_lots}@${o.price}</span>
       <button data-id="${o.client_id}">✕</button></div>`).join('') || '<div>None</div>';
    ob.querySelectorAll('button').forEach((b) =>
      (b.onclick = () => WSREF.send({ kind: 'cancel', client_id: b.dataset.id })));
  }
}

function fmt(x) { return (x ?? 0).toFixed(2); }
