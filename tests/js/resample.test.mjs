import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resample } from '../../replayer/static/js/resample.js';

const m1 = (t, o, h, l, c, v = 1) => ({ t: t * 60000, o, h, l, c, v });

test('groups five m1 bars into one m5 bucket', () => {
  const bars = [m1(0,1,2,0.5,1.5), m1(1,1.5,3,1,2), m1(2,2,2.5,1.5,2.2),
                m1(3,2.2,2.3,2,2.1), m1(4,2.1,2.6,2,2.4)];
  const out = resample(bars, 300);
  assert.equal(out.length, 1);
  assert.equal(out[0].open, 1);
  assert.equal(out[0].high, 3);
  assert.equal(out[0].low, 0.5);
  assert.equal(out[0].close, 2.4);
  assert.equal(out[0].volume, 5);
});

test('last partial bucket is the forming candle', () => {
  const bars = [m1(0,1,2,0.5,1.5), m1(1,1.5,3,1,2), m1(5,2,2.5,1.9,2.2)];
  const out = resample(bars, 300);
  assert.equal(out.length, 2);
  assert.equal(out[1].open, 2);
});

test('empty input returns empty array', () => {
  assert.deepEqual(resample([], 300), []);
});
