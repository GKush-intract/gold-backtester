import { resample, TF_SECONDS } from './resample.js';

// KLineChart is loaded via <script> (global `klinecharts`).
export class ChartView {
  constructor(el) {
    this.chart = klinecharts.init(el);
    this.m1 = [];          // full m1 history received so far (up to replay time)
    this.tf = 'm5';
    this.chart.createIndicator('MA', false, { id: 'candle_pane' });
    this.chart.createIndicator('VOL');
  }
  setTimeframe(tf) { this.tf = tf; this._render(); }
  setHistory(m1Bars) { this.m1 = m1Bars.slice(); this._render(); }
  appendBar(bar) { this.m1.push(bar); this._render(); }   // bar = {t,o,h,l,c,v}
  _render() {
    const bars = resample(this.m1, TF_SECONDS[this.tf]);
    this.chart.applyNewData(bars);
  }
  addRSI() { this.chart.createIndicator('RSI'); }
}
