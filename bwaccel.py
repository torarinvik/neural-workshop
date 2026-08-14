# -*- coding: utf-8 -*-
"""Accelerated kernels for Brain Workshop.

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


def seed(n=0):
    if USING_NATIVE:
        _native.seed(n)
        return
    if n:
        random.seed(n)
    else:
        random.seed()


# ---------------------------------------------------------------------------
# Pure-Python fallbacks
# ---------------------------------------------------------------------------

def _compute_bt_sequence_py(num_trials, nback, n_pos=6, n_audio=6, n_both=2):
    """Construct a sequence with exact match counts in O(T)."""
    T = num_trials - nback
    if num_trials < 1 or nback < 1 or T < 1:
        raise ValueError('num_trials must be > nback and both must be positive')
    if n_both < 0 or n_pos < n_both or n_audio < n_both:
        raise ValueError('cannot realize requested match counts')
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

    pos = [random.randint(1, 8) for _ in range(nback)]
    audio = [random.randint(1, 8) for _ in range(nback)]

    def _nonmatch(prev):
        v = random.randint(1, 8)
        if v == prev:
            v = 1 if v == 8 else v + 1
        return v

    for k in kind:
        if k & 1:
            pos.append(pos[-nback])
        else:
            pos.append(_nonmatch(pos[-nback]))
        if k & 2:
            audio.append(audio[-nback])
        else:
            audio.append(_nonmatch(audio[-nback]))
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

def compute_bt_sequence(num_trials, nback, n_pos=6, n_audio=6, n_both=2):
    if USING_NATIVE:
        return _native.compute_bt_sequence(num_trials, nback, n_pos, n_audio, n_both)
    return _compute_bt_sequence_py(num_trials, nback, n_pos, n_audio, n_both)


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
