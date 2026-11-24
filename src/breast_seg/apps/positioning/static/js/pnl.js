// The segmentation-route measurement as the reviewer places it ("Adjust PNL"), recomputed in the page on every move
// of a handle: the nipple and the two ends of the pectoral line on an MLO, the nipple and the chest-wall edge on a CC.
//
// The same arithmetic as the server (positioning/measures.py: measure_manual, pair_verdict, compare), so the numbers
// that follow the pointer are the ones the record will hold. The server stays the one source of truth: when the
// handle is let go the placement is sent (PUT .../manual) and its answer replaces everything computed here.
// tests/test_ui_js.py runs these functions next to the Python ones on the same inputs.
//
// Plain functions of plain data, no DOM: the page imports them, and so does the test.

export const WIDE_GAP_MM = 5.0;
const FRAME = 640;

const round = (value, digits) => Math.round(value * 10 ** digits) / 10 ** digits;
/** Python's round() to a whole number: a tie goes to the even neighbour (250.5 -> 250, 251.5 -> 252), as
 *  geometry.compute_pnl rounds the foot; Math.round would take every tie up. */
export function roundHalfEven(value) {
  const nearest = Math.round(value);
  return Math.abs(value % 1) === 0.5 && nearest % 2 !== 0 ? nearest - 1 : nearest;
}
const point = (p) => [round(p[0], 1), round(p[1], 1)];
const clamp = (v) => Math.min(Math.max(Number(v), 0), FRAME);

/** A point kept to the 640 frame and to 0.01 px, as the server cleans it (labels.clean_point). */
export const cleanPoint = (p) => [round(clamp(p[0]), 2), round(clamp(p[1]), 2)];

/** Distance from `p` to the segment `a`-`b` and the foot on it: the perpendicular, its foot kept between the two
 *  ends (geometry.compute_pnl, the segmentation route's PNL; a degenerate segment gives the distance to `a`). The
 *  distance is the exact one; the foot, as compute_pnl gives it, is on the whole pixel. */
export function segmentFoot(a, b, p) {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const length2 = abx * abx + aby * aby;
  if (length2 < 1e-10) return { distance: Math.hypot(p[0] - a[0], p[1] - a[1]), foot: [a[0], a[1]] };
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / length2));
  const foot = [a[0] + t * abx, a[1] + t * aby];
  return { distance: Math.hypot(p[0] - foot[0], p[1] - foot[1]), foot: [roundHalfEven(foot[0]), roundHalfEven(foot[1])] };
}

function measured(distancePx, mmPerPx, start, end) {
  return { ok: true, px: round(distancePx, 2), mm: distancePx * mmPerPx, from: point(start), to: point(end) };
}

/** measures.measure_manual: `manual` is {nipple, line: [a, b]} on an MLO (the PNL to the segment between the two
 *  ends, as the model's fitted line is measured), {nipple, wall} on a CC (horizontal to the edge). */
export function measureManual(view, manual, mmPerPx) {
  const nipple = manual.nipple;
  let result;
  if (view === 'CC') {
    const wall = manual.wall;
    result = measured(Math.abs(nipple[0] - wall[0]), mmPerPx, nipple, [wall[0], nipple[1]]);
  } else {
    const [a, b] = manual.line;
    const { distance, foot } = segmentFoot(a, b, nipple);
    result = measured(distance, mmPerPx, nipple, foot);
    result.line = [[a[0], a[1]], [b[0], b[1]]];
  }
  result.manual = true;
  return result;
}

/** measures.pair_verdict: |PNL_MLO - D_CC| of one breast against the threshold. */
export function pairVerdict(mlo, cc, thresholdMm, severeMm) {
  if (!mlo || !cc) return { ok: false, reason: 'needs both MLO and CC' };
  if (!mlo.ok || !cc.ok) return { ok: false, reason: mlo.reason || cc.reason || 'no measurement' };
  const pnl = Number(mlo.mm);
  const wall = Number(cc.mm);
  const delta = Math.abs(pnl - wall);
  const result = {
    ok: true, pnl_mm: pnl, cc_mm: wall, delta_mm: delta,
    verdict: delta <= thresholdMm ? 'good' : 'bad',
    severe: delta > severeMm,
    threshold_mm: Number(thresholdMm),
  };
  const estimated = [['MLO', mlo], ['CC', cc]].filter(([, m]) => m.estimated).map(([view]) => view);
  if (estimated.length) result.estimated = estimated;
  const manual = [['MLO', mlo], ['CC', cc]].filter(([, m]) => m.manual).map(([view]) => view);
  if (manual.length) result.manual = manual;
  return result;
}

/** measures.compare: how far apart the two routes landed and whether they agree on the call. */
export function compare(landmark, mask) {
  if (!landmark.ok || !mask.ok) {
    const available = [['landmark', landmark], ['mask', mask]].filter(([, r]) => r.ok).map(([name]) => name);
    return {
      state: available.length ? 'partial' : 'none',
      available,
      note: available.length ? 'only one method produced a result' : 'neither method produced a result',
    };
  }
  const gap = Math.abs(landmark.delta_mm - mask.delta_mm);
  const agree = landmark.verdict === mask.verdict;
  return { state: agree ? 'agree' : 'disagree', gap_mm: gap, wide_gap: gap > WIDE_GAP_MM };
}

/**
 * Where the handles of a view start: the reviewer's own placement when there is one, else exactly what the
 * segmentation route measured from (its nipple, the two ends of its fitted line, its chest-wall edge point), so an
 * unmoved placement gives the model's number; else the pose model's points, else a neutral guess (the image centre,
 * the chest-side edge), so a view the model could not measure can still be placed by hand.
 */
export function seedManual(image) {
  const own = image.manual;
  if (own && own.nipple && (own.line || own.wall)) {
    return image.view === 'CC'
      ? { nipple: [...own.nipple], wall: [...own.wall] }
      : { nipple: [...own.nipple], line: [[...own.line[0]], [...own.line[1]]] };
  }
  const measures = image.measures || {};
  const mask = measures.mask || {};
  const landmark = measures.landmark || {};
  const points = (image.landmarks || {}).points || {};
  const nipple = (mask.ok && mask.from) || points.nipple || (landmark.ok && landmark.from) || [FRAME / 2, FRAME / 2];
  const chestX = image.chest_side === 'L' ? 0 : FRAME;
  if (image.view === 'CC') {
    const wallX = mask.ok && mask.to ? mask.to[0] : chestX;
    return { nipple: cleanPoint(nipple), wall: cleanPoint([wallX, nipple[1]]) };
  }
  let line = mask.line || landmark.line;
  if (!line && points.pectoral_top && points.pectoral_bottom) line = [points.pectoral_top, points.pectoral_bottom];
  if (!line) line = [[Math.abs(chestX - FRAME * 0.3), 0], [chestX, FRAME * 0.6]];
  return { nipple: cleanPoint(nipple), line: [cleanPoint(line[0]), cleanPoint(line[1])] };
}

/** The handles of a placement, by name, in the 640 frame. */
export function handlesOf(view, manual) {
  return view === 'CC'
    ? { nipple: manual.nipple, wall: manual.wall }
    : { nipple: manual.nipple, a: manual.line[0], b: manual.line[1] };
}

/** `manual` with one handle moved to `to` (kept inside the frame). */
export function moveHandle(view, manual, name, to) {
  const p = cleanPoint(to);
  if (view === 'CC') return name === 'nipple' ? { ...manual, nipple: p } : { ...manual, wall: p };
  if (name === 'nipple') return { ...manual, nipple: p };
  const line = [[...manual.line[0]], [...manual.line[1]]];
  line[name === 'a' ? 0 : 1] = p;
  return { ...manual, line };
}

/** True when both ends of the line are at least 2 px apart (labels.clean_manual refuses a shorter one). */
export function placeable(view, manual) {
  if (view === 'CC') return true;
  const [a, b] = manual.line;
  return Math.hypot(b[0] - a[0], b[1] - a[1]) >= 2;
}
