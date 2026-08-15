# -*- coding: utf-8 -*-
"""Accelerated kernels for Neural Workshop.

Imports the compiled ``bwcore`` C extension when it is on sys.path.
If the extension is missing, every public function falls back to a
pure-Python implementation with the same contract so the game still runs.

Public API
----------
backend()                  -> 'C' or 'Python'
compute_bt_sequence(...)    constructive Jaeggi/BT dual sequence
analyze_session(...)        rights/wrongs per modality
aggregate_day_scores(...)   graph score mean/max for one day
rounded_rect_vertices(...)  40-vertex rounded rectangle as [x,y,...]
variable_nback_list(...)    Beta(n/2, 1) variable-n sequence
sample_unique(lo, hi, k)    k distinct ints in [lo, hi]
parse_stats_text(text)      list of session dicts
is_nback_match(...)         compare current stim to history[n]
mean_tail(seq, tail=0)      mean of the last ``tail`` items (0 = all)
apply_arithmetic(op, a, b)  Decimal add/sub/mul/div (no eval)
score_arithmetic(...)       rights/wrongs for an arithmetic session
banner()                    'native: C' / 'native: Python'
grid_layout / grid_cell_count / position_col_row
ms_to_ticks / clamp_trial_interval_ms / plan_trial_phases
active_position_ids
seed(n=0)
"""
from __future__ import print_function

import math
import os
import random
import sys
from decimal import Decimal, InvalidOperation

USING_NATIVE = False
_native = None

try:
    import bwcore as _native  # type: ignore
    USING_NATIVE = True
except ImportError:
    _native = None


def backend():
    if USING_NATIVE and _native is not None:
        return 'C'
    return 'Python'


def banner():
    """Short UI/console tag, e.g. ``native: C``."""
    return 'native: %s' % backend()


def seed(n=None):
    """Seed RNGs. ``None`` (or omitted) uses entropy. ``0`` is a real seed."""
    if USING_NATIVE:
        if n is None:
            _native.seed()
        else:
            _native.seed(int(n))
        return
    if n is None:
        random.seed()
    else:
        random.seed(int(n))


def count_feedback_pixels(rgba, width, height, y0=None, y1=None):
    """Count public feedback-palette pixels in [y0, y1) of a top-down RGBA buffer."""
    width = int(width)
    height = int(height)
    if y0 is None:
        y0 = int(height * 0.75)
    if y1 is None:
        y1 = height
    y0 = max(0, int(y0))
    y1 = min(height, int(y1))
    if USING_NATIVE and hasattr(_native, 'count_feedback_pixels'):
        return _native.count_feedback_pixels(bytes(rgba), width, height, y0, y1)
    pos = neg = oops = 0
    raw = rgba
    rowb = width * 4
    for y in range(y0, y1):
        base = y * rowb
        for x in range(width):
            off = base + x * 4
            r, g, b = raw[off], raw[off + 1], raw[off + 2]
            if g >= 180 and r <= 140 and b <= 140:
                pos += 1
            elif r >= 180 and g <= 140 and b <= 140:
                neg += 1
            elif b >= 180 and r <= 140 and g <= 140:
                oops += 1
    return (pos, neg, oops)


def count_feedback_label_runs(rgba, width, height, y0=None, y1=None):
    """Count feedback *labels* in the public pixel band.

    Rule (wording- and resolution-invariant):
    1. Classify each column in ``[y0, y1)`` by majority feedback color
       (green / red / blue), requiring at least two matching pixels.
    2. Merge same-class column-runs separated by fewer than
       ``max(8, width // 40)`` empty columns. That closes glyph and
       word gaps inside one caption without joining separate labels.
    3. Each remaining run is one label. ``scalar`` later uses
       ``(n_pos - n_neg) / (n_pos + n_neg)``.
    """
    width = int(width)
    height = int(height)
    if y0 is None:
        y0 = int(height * 0.75)
    if y1 is None:
        y1 = height
    y0 = max(0, int(y0))
    y1 = min(height, int(y1))
    if USING_NATIVE and hasattr(_native, 'count_feedback_label_runs'):
        return _native.count_feedback_label_runs(bytes(rgba), width, height, y0, y1)
    raw = rgba
    rowb = width * 4
    classes = []
    for x in range(width):
        pos = neg = oops = 0
        for y in range(y0, y1):
            off = y * rowb + x * 4
            r, g, b = raw[off], raw[off + 1], raw[off + 2]
            if g >= 180 and r <= 140 and b <= 140:
                pos += 1
            elif r >= 180 and g <= 140 and b <= 140:
                neg += 1
            elif b >= 180 and r <= 140 and g <= 140:
                oops += 1
        cls = 0
        if pos >= 2 and pos >= neg and pos >= oops:
            cls = 1
        elif neg >= 2 and neg >= oops:
            cls = 2
        elif oops >= 2:
            cls = 3
        classes.append(cls)
    return _count_closed_column_runs(classes, width)


def _count_closed_column_runs(classes, width):
    gap_thresh = max(8, int(width) // 40)
    pos_runs = neg_runs = oops_runs = 0
    x = 0
    n = len(classes)
    while x < n:
        cls = classes[x]
        if cls == 0:
            x += 1
            continue
        x += 1
        while True:
            while x < n and classes[x] == cls:
                x += 1
            z = x
            while z < n and classes[z] == 0:
                z += 1
            if z < n and classes[z] == cls and (z - x) < gap_thresh:
                x = z
                continue
            break
        if cls == 1:
            pos_runs += 1
        elif cls == 2:
            neg_runs += 1
        else:
            oops_runs += 1
    return (pos_runs, neg_runs, oops_runs)


# ---------------------------------------------------------------------------
# Pure-Python fallbacks
# ---------------------------------------------------------------------------

def _nonmatch_choice(prev, hi):
    v = random.randint(1, hi)
    if v == prev:
        v = 1 if v == hi else v + 1
    return v


def _compute_bt_sequence_py(num_trials, nback, n_pos=6, n_audio=6, n_both=2,
                            pos_choices=8, audio_choices=8):
    """Construct a sequence with exact match counts in O(T)."""
    T = num_trials - nback
    if num_trials < 1 or nback < 1 or T < 1:
        raise ValueError('num_trials must be > nback and both must be positive')
    if n_both < 0 or n_pos < n_both or n_audio < n_both:
        raise ValueError('cannot realize requested match counts')
    if pos_choices < 2 or audio_choices < 2:
        raise ValueError('pos_choices and audio_choices must be >= 2')
    n_pos_only = n_pos - n_both
    n_aud_only = n_audio - n_both
    n_neither = T - n_pos_only - n_aud_only - n_both
    if n_neither < 0:
        raise ValueError('cannot realize requested match counts with this trial/n-back')

    kind = ([3] * n_both +
            [1] * n_pos_only +
            [2] * n_aud_only +
            [0] * n_neither)
    random.shuffle(kind)

    pos = [random.randint(1, pos_choices) for _ in range(nback)]
    audio = [random.randint(1, audio_choices) for _ in range(nback)]

    for k in kind:
        if k & 1:
            pos.append(pos[-nback])
        else:
            pos.append(_nonmatch_choice(pos[-nback], pos_choices))
        if k & 2:
            audio.append(audio[-nback])
        else:
            audio.append(_nonmatch_choice(audio[-nback], audio_choices))
    return [pos, audio]


def _crab_back(x, nback):
    return 1 + 2 * (x % nback)


def _resolve_back(x, nback, crab, variable_list):
    back = _crab_back(x, nback) if crab else nback
    if variable_list is not None:
        idx = x - back
        if 0 <= idx < len(variable_list):
            back = variable_list[idx]
    return max(1, back)


def _score_direct(data, inp, nback, crab, jaeggi, variable_list):
    rights = wrongs = 0
    n = len(data)
    for x in range(nback, n):
        back = _resolve_back(x, nback, crab, variable_list)
        if back > x:
            continue
        match = data[x] == data[x - back]
        inpv = bool(inp[x]) if inp is not None and x < len(inp) else False
        rights += int(match and inpv)
        wrongs += int(match ^ inpv)
        if jaeggi:
            rights += int((not match) and (not inpv))
    return rights, wrongs


def _analyze_session_py(nback, crab=False, jaeggi_scoring=False,
                        variable_list=None, modalities=None, session=None):
    if modalities is None or session is None:
        raise TypeError('modalities and session are required')
    out = {}
    for mod in modalities:
        if mod == 'arithmetic':
            out[mod] = None
            continue
        if mod in ('visvis', 'visaudio', 'audiovis'):
            now_key = 'vis' if mod.startswith('vis') else 'audio'
            then_key = 'vis' if mod.endswith('vis') else 'audio'
            now = session.get(now_key)
            then = session.get(then_key)
            inp = session.get(mod + '_input')
            if now is None or then is None:
                continue
            n = min(len(now), len(then))
            rights = wrongs = 0
            for x in range(nback, n):
                back = _resolve_back(x, nback, crab, variable_list)
                if back > x:
                    continue
                match = now[x] == then[x - back]
                inpv = bool(inp[x]) if inp is not None and x < len(inp) else False
                rights += int(match and inpv)
                wrongs += int(match ^ inpv)
                if jaeggi_scoring:
                    rights += int((not match) and (not inpv))
            out[mod] = (rights, wrongs)
        else:
            data = session.get(mod)
            inp = session.get(mod + '_input')
            if data is None:
                continue
            out[mod] = _score_direct(data, inp, nback, crab, jaeggi_scoring, variable_list)
    return out


_STYLE = {
    'N': 0, '%': 1, 'N.%': 2, 'N+2*%-1': 3, 'N+10/3+4/3': 4,
}


def _aggregate_day_scores_py(style, entries, advance=80.0, fallback=50.0):
    if not isinstance(style, int):
        style = _STYLE[style]
    if style == 4:
        den = advance - fallback or 1.0
        m = 1.0 / den
        b = -m * fallback
    scores = []
    for entry in entries:
        nback, percent = entry[0], entry[1]
        if style == 0:
            score = float(nback)
        elif style == 1:
            score = 0.01 * percent
        elif style == 2:
            score = nback + 0.01 * percent
        elif style == 3:
            score = nback - 1 + 2 * 0.01 * percent
        else:
            score = nback + b + m * percent
        scores.append(score)
    if not scores:
        return (0.0, 0.0)
    return (sum(scores) / float(len(scores)), max(scores))


def _rounded_rect_vertices_py(lx, rx, by, ty, cr):
    # Python 3: ranges must be materialised before concatenation.
    sweep_up = list(range(0, 91, 10))
    sweep_dn = list(range(90, -1, -10))
    x = ([lx + int(cr * (1 - math.cos(math.radians(i)))) for i in sweep_up] +
         [rx - int(cr * (1 - math.sin(math.radians(i)))) for i in sweep_up] +
         [rx - int(cr * (1 - math.sin(math.radians(i)))) for i in sweep_dn] +
         [lx + int(cr * (1 - math.cos(math.radians(i)))) for i in sweep_dn])
    y = ([by + int(cr * (1 - math.sin(math.radians(i)))) for i in sweep_up + sweep_dn] +
         [ty - int(cr * (1 - math.sin(math.radians(i)))) for i in sweep_up + sweep_dn])
    xy = []
    for a, b in zip(x, y):
        xy.extend((a, b))
    return xy


def _variable_nback_list_py(count, back):
    # Beta(back/2, 1) == U^(2/back)
    inv = 2.0 / float(back)
    out = []
    for _ in range(count):
        u = random.random()
        if u <= 0.0:
            u = 1e-12
        v = int(u ** inv * back + 1)
        if v < 1:
            v = 1
        if v > back:
            v = back
        out.append(v)
    return out


def _sample_unique_py(lo, hi, k):
    return random.sample(range(lo, hi + 1), k)


def _parse_stats_text_py(text):
    records = []
    for line in text.splitlines():
        if not line or line[0] not in '0123456789':
            continue
        if len(line) < 19:
            continue
        try:
            y = int(line[0:4]); mo = int(line[5:7]); d = int(line[8:10])
            H = int(line[11:13]); M = int(line[14:16]); S = int(line[17:19])
        except ValueError:
            continue
        sep = '\t' if '\t' in line else ','
        cols = line.split(sep)
        if len(cols) < 9:
            continue

        def _ival(idx, default=0):
            if idx >= len(cols):
                return default
            try:
                return int(cols[idx])
            except (TypeError, ValueError):
                return default

        cats = [_ival(9 + i, 0) for i in range(16)]
        sesstime = 0
        if len(cols) > 25:
            try:
                sesstime = int(round(float(cols[25])))
            except (TypeError, ValueError):
                sesstime = 0
        records.append({
            'year': y, 'month': mo, 'day': d,
            'hour': H, 'minute': M, 'second': S,
            'percent': _ival(2), 'mode': _ival(3), 'nback': _ival(4),
            'ticks': _ival(5), 'trials': _ival(6),
            'manual': _ival(7), 'session': _ival(8),
            'sesstime': sesstime, 'cats': cats,
        })
    return records


def _is_nback_match_py(current, history, nback_trial):
    if nback_trial < 0 or nback_trial >= len(history):
        return None
    return current == history[nback_trial]


def _mean_tail_py(seq, tail=0):
    if not seq:
        return 0.0
    chunk = seq[-tail:] if tail > 0 else seq
    if not chunk:
        return 0.0
    return sum(chunk) / float(len(chunk))


# ---------------------------------------------------------------------------
# Public wrappers — prefer C
# ---------------------------------------------------------------------------

def compute_bt_sequence(num_trials, nback, n_pos=6, n_audio=6, n_both=2,
                        pos_choices=8, audio_choices=8):
    if USING_NATIVE:
        return _native.compute_bt_sequence(
            num_trials, nback, n_pos, n_audio, n_both, pos_choices, audio_choices)
    return _compute_bt_sequence_py(
        num_trials, nback, n_pos, n_audio, n_both, pos_choices, audio_choices)


def analyze_session(nback, crab=False, jaeggi_scoring=False,
                    variable_list=None, modalities=None, session=None):
    if USING_NATIVE:
        return _native.analyze_session(
            nback, crab, jaeggi_scoring, variable_list, modalities, session)
    return _analyze_session_py(
        nback, crab, jaeggi_scoring, variable_list, modalities, session)


def aggregate_day_scores(style, entries, advance=80.0, fallback=50.0):
    if USING_NATIVE:
        return _native.aggregate_day_scores(style, entries, advance, fallback)
    return _aggregate_day_scores_py(style, entries, advance, fallback)


def rounded_rect_vertices(lx, rx, by, ty, cr):
    if USING_NATIVE:
        return _native.rounded_rect_vertices(lx, rx, by, ty, cr)
    return _rounded_rect_vertices_py(lx, rx, by, ty, cr)


def variable_nback_list(count, back):
    if USING_NATIVE:
        return _native.variable_nback_list(count, back)
    return _variable_nback_list_py(count, back)


def sample_unique(lo, hi, k):
    if USING_NATIVE:
        return _native.sample_unique(lo, hi, k)
    return _sample_unique_py(lo, hi, k)


def parse_stats_text(text):
    if USING_NATIVE:
        return _native.parse_stats_text(text)
    return _parse_stats_text_py(text)


def is_nback_match(current, history, nback_trial):
    if USING_NATIVE:
        return _native.is_nback_match(current, history, nback_trial)
    return _is_nback_match_py(current, history, nback_trial)


def mean_tail(seq, tail=0):
    if USING_NATIVE:
        return _native.mean_tail(seq, tail)
    return _mean_tail_py(seq, tail)


_ARITH_OPS = ('add', 'subtract', 'multiply', 'divide')


def apply_arithmetic(op, left, right):
    """Apply a named n-back arithmetic operation with Decimal, never eval.

    ``op`` is one of add / subtract / multiply / divide.
    Raises ValueError for an unknown op, InvalidOperation / ZeroDivisionError
    for bad operands.
    """
    if op not in _ARITH_OPS:
        raise ValueError('unknown arithmetic operation: %r' % (op,))
    a = left if isinstance(left, Decimal) else Decimal(left)
    b = right if isinstance(right, Decimal) else Decimal(right)
    if op == 'add':
        return a + b
    if op == 'subtract':
        return a - b
    if op == 'multiply':
        return a * b
    return a / b


def score_arithmetic(nback, crab=False, variable_list=None, session=None):
    """Score an arithmetic session. Returns (rights, wrongs)."""
    if session is None:
        return (0, 0)
    numbers = session.get('numbers') or []
    ops = session.get('operation') or []
    answers = session.get('arithmetic_input') or []
    n = min(len(numbers), len(ops), len(answers))
    rights = wrongs = 0
    for x in range(nback, n):
        back = _resolve_back(x, nback, crab, variable_list)
        if back > x:
            continue
        try:
            expected = apply_arithmetic(ops[x], numbers[x - back], numbers[x])
            given = answers[x]
            if not isinstance(given, Decimal):
                given = Decimal(given)
            if expected == given:
                rights += 1
            else:
                wrongs += 1
        except (InvalidOperation, ZeroDivisionError, ValueError, TypeError):
            wrongs += 1
    return (rights, wrongs)


# Classic Dual N-Back 3x3 IDs (1-8 around the center, 0/9 = center).
# col/row are 0..2 with (0,0) at the bottom-left (pyglet y-up).
_CLASSIC_3X3 = {
    0: (1, 1),
    1: (2, 1), 2: (0, 1), 3: (1, 2),
    4: (2, 2), 5: (0, 2), 6: (1, 0),
    7: (2, 0), 8: (0, 0),
    9: (1, 1),
}


def grid_layout(n, include_center=False):
    """Return [(position_id, col, row), ...] for an n x n board."""
    n = max(2, int(n))
    include_center = bool(include_center)
    if n == 3:
        ids = list(range(1, 9))
        if include_center:
            ids.append(9)
        return [(pid, _CLASSIC_3X3[pid][0], _CLASSIC_3X3[pid][1]) for pid in ids]

    skip_center = (n % 2 == 1) and not include_center
    cells = []
    pid = 1
    for row in range(n):
        for col in range(n):
            if skip_center and col == n // 2 and row == n // 2:
                continue
            cells.append((pid, col, row))
            pid += 1
    return cells


def grid_cell_count(n, include_center=False):
    return len(grid_layout(n, include_center))


def position_col_row(position, n, include_center=False):
    """Map a 1-based position id to (col, row), or None if unknown.

    position <= 0 means “field center” (no cell).
    """
    n = max(2, int(n))
    if position is None or int(position) <= 0:
        return None
    position = int(position)
    if n == 3:
        return _CLASSIC_3X3.get(position)
    for pid, col, row in grid_layout(n, include_center):
        if pid == position:
            return (col, row)
    return None


def grid_center_out_ids(n, include_center=False):
    """Position ids nearest the board center first (curriculum order)."""
    n = max(2, int(n))
    if n == 3 and not include_center:
        return [1, 2, 3, 6, 4, 5, 7, 8]
    cx = (n - 1) / 2.0
    cy = (n - 1) / 2.0
    cells = grid_layout(n, include_center)
    cells = sorted(cells, key=lambda c: ((c[1] - cx) ** 2 + (c[2] - cy) ** 2, -c[2], c[1]))
    return [c[0] for c in cells]


def ms_to_ticks(ms, tick_ms=100):
    """Convert a duration in milliseconds to an integer tick count (>= 1)."""
    tick_ms = max(1, int(tick_ms))
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        ms = tick_ms
    return max(1, int(round(ms / tick_ms)))


def clamp_trial_interval_ms(ms, tick_ms=100, min_ticks=3, max_ms=60000):
    """Keep a trial interval on the clock grid and inside [min_ticks*tick, max_ms]."""
    tick_ms = max(1, int(tick_ms))
    min_ticks = max(2, int(min_ticks))
    try:
        ms = int(ms)
    except (TypeError, ValueError):
        ms = tick_ms * min_ticks
    lo = tick_ms * min_ticks
    hi = max(lo, int(max_ms))
    if ms < lo:
        ms = lo
    if ms > hi:
        ms = hi
    # Snap to the scheduler quantum so ticks * tick_ms == interval.
    ticks = ms_to_ticks(ms, tick_ms)
    return ticks * tick_ms


def active_position_ids(n, include_center=False, limit=0):
    """IDs that may be sampled. ``limit<=0`` means the full board.

    A positive limit takes the first N cells in center-out curriculum order.
    """
    all_ids = [pid for pid, _c, _r in grid_layout(n, include_center)]
    try:
        limit = int(limit or 0)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0 or limit >= len(all_ids):
        return all_ids
    order = grid_center_out_ids(n, include_center)
    return order[:max(2, min(limit, len(order)))]


def plan_trial_phases(trial_ms, stim_ms, feedback_ms, tick_ms=1):
    """Non-overlapping stimulus / blank / feedback ticks for one trial.

    If stim + feedback would exceed the trial, both are scaled down
    proportionally so each gets at least one tick and they sum to the trial.
    """
    tick_ms = max(1, int(tick_ms))
    total = max(3, ms_to_ticks(trial_ms, tick_ms))
    stim_want = max(1, ms_to_ticks(stim_ms, tick_ms))
    fb_want = max(1, ms_to_ticks(feedback_ms, tick_ms))
    if stim_want + fb_want > total:
        stim = int(round(total * (stim_want / float(stim_want + fb_want))))
        stim = max(1, min(stim, total - 1))
        fb = total - stim
        blank = 0
    else:
        stim = stim_want
        fb = fb_want
        blank = total - stim - fb
    return {
        'total_ticks': total,
        'stimulus_ticks': int(stim),
        'blank_ticks': int(blank),
        'feedback_ticks': int(fb),
        'stimulus_ms': int(stim) * tick_ms,
        'blank_ms': int(blank) * tick_ms,
        'feedback_ms': int(fb) * tick_ms,
    }


def interval_adjust_step(ms):
    """F5/F6 step: 1 ms at high speed, larger steps when the trial is long."""
    try:
        ms = int(ms)
    except (TypeError, ValueError):
        ms = 100
    if ms <= 20:
        return 1
    if ms <= 100:
        return 5
    if ms <= 500:
        return 10
    if ms <= 2000:
        return 50
    return 100


def maybe_hint_compile():
    """Print a one-line hint when the C module is absent (debug / CLI)."""
    if USING_NATIVE:
        return
    if os.environ.get('BW_SILENT_FALLBACK'):
        return
    root = os.path.dirname(os.path.abspath(__file__))
    print('bwaccel: C extension not found, using Python fallback. '
          'Build it with:  %s "%s" build_ext --inplace'
          % (sys.executable, os.path.join(root, 'setup.py')),
          file=sys.stderr)
