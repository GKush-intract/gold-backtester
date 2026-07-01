// Pure m1 -> timeframe aggregation. Output bars use KLineChart's shape
// {timestamp, open, high, low, close, volume}. Input m1 bars: {t (epoch ms), o,h,l,c,v}.
// tfSeconds e.g. 300 for m5. The final (possibly partial) bucket is the live-forming candle.
export function resample(m1Bars, tfSeconds) {
  const bucketMs = tfSeconds * 1000;
  const out = [];
  let cur = null;
  for (const b of m1Bars) {
    const bStart = Math.floor(b.t / bucketMs) * bucketMs;
    if (!cur || cur.timestamp !== bStart) {
      if (cur) out.push(cur);
      cur = { timestamp: bStart, open: b.o, high: b.h, low: b.l, close: b.c, volume: b.v };
    } else {
      cur.high = Math.max(cur.high, b.h);
      cur.low = Math.min(cur.low, b.l);
      cur.close = b.c;
      cur.volume += b.v;
    }
  }
  if (cur) out.push(cur);
  return out;
}

export const TF_SECONDS = {
  m1: 60, m5: 300, m15: 900, m30: 1800, h1: 3600, h4: 14400, d1: 86400,
};
