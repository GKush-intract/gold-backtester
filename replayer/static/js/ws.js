export class WS {
  constructor(sid, onMessage) {
    this.sock = new WebSocket(`ws://${location.host}/ws/${sid}`);
    this.sock.onmessage = (e) => onMessage(JSON.parse(e.data));
    this.ready = new Promise((res) => (this.sock.onopen = res));
  }
  send(obj) { this.sock.send(JSON.stringify(obj)); }
}
