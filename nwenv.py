# -*- coding: utf-8 -*-
"""Deterministic agent boundary for Neural Workshop.

Public contract (learner-facing)
--------------------------------
reset(seed) → observation
observe()   → observation
act(ports)  → receipt | rejected
advance()   → observation
step(ports) → (observation, events, done)

Observation fields: frame_seq, timestamp_ns, width, height, rgba, done,
optional outcome {scalar, evidence_digests, receipt_id}.

No cell IDs, modality names, phase names, scores, or sequences.
Actions are opaque integer port indices.

Shared-memory export (NW_SHM) is a one-way framebuffer dump, not a
complete cross-process control protocol (no seqlock, no action channel,
no reset/config, no ownership handshake).
"""
from __future__ import print_function

import hashlib
import os
import struct
import sys
import time

os.environ.setdefault('NW_HEADLESS', '1')

import pyglet
if sys.platform.startswith('linux'):
    try:
        pyglet.options['headless'] = True
    except Exception:
        pass

import bwaccel
import brainworkshop as bw

# Publicly painted feedback palette (the colors that appear on screen).
# Colors the game actually paints on feedback labels (public palette).
_FEEDBACK_POS = (64, 255, 64)    # correct
_FEEDBACK_NEG = (255, 64, 64)    # incorrect
_FEEDBACK_OOPS = (64, 64, 255)   # missed / too-early

_HEADER = struct.Struct('<4sIQQIII')  # magic, ver, seq, ts, w, h, flags
_MAGIC = b'NWFB'


def _now_ns():
    return time.monotonic_ns()


def _flip_rgba(raw, width, height):
    row = width * 4
    if row <= 0 or height <= 0:
        return raw
    out = bytearray(len(raw))
    for y in range(height):
        src = (height - 1 - y) * row
        dst = y * row
        out[dst:dst + row] = raw[src:src + row]
    return bytes(out)


def capture_rgba(window):
    from pyglet.gl import (
        GL_PACK_ALIGNMENT, GL_RGBA, GL_UNSIGNED_BYTE,
        GLubyte, glPixelStorei, glReadPixels,
    )
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    try:
        w, h = window.get_framebuffer_size()
    except Exception:
        w, h = window.width, window.height
    n = int(w) * int(h) * 4
    if n <= 0:
        return int(w), int(h), b''
    buf = (GLubyte * n)()
    glReadPixels(0, 0, int(w), int(h), GL_RGBA, GL_UNSIGNED_BYTE, buf)
    return int(w), int(h), _flip_rgba(bytes(buf), int(w), int(h))


def render_significant_frame():
    """Draw, read the *backbuffer*, then flip. Capture must precede flip."""
    window = bw.window
    window.switch_to()
    window.dispatch_events()
    bw.on_draw()
    captured = capture_rgba(window)
    window.flip()
    return captured


def digest_rgba(rgba):
    return hashlib.sha256(rgba or b'').hexdigest()


def _channel_close(pixel, target, tol):
    return all(abs(int(pixel[i]) - target[i]) <= tol for i in range(3))


def derive_public_outcome(rgba, width, height, evidence_digests, receipt_id):
    """Deterministic scalar from publicly painted feedback pixels only.

    Green (correct) → +1. Red (incorrect) and blue (oops/missed/early) → -1.
    Missing colors or a green/negative tie → None (not zero).
    Never consults game state, scores, sequences, or cell IDs.
    """
    if not rgba or width < 1 or height < 1:
        return None
    y0 = int(height * 0.75)
    pos, neg, oops = bwaccel.count_feedback_pixels(rgba, width, height, y0, height)
    negative = neg + oops
    if pos == 0 and negative == 0:
        return None
    if pos == negative:
        return None
    scalar = 1.0 if pos > negative else -1.0
    return {
        'scalar': scalar,
        'evidence_digests': list(evidence_digests),
        'receipt_id': receipt_id,
    }


def verify_public_outcome(outcome, rgba, width, height, archive=None):
    """Authenticate evidence digests against archived frames, then recompute.

    ``archive`` maps digest → immutable RGBA bytes. Every declared digest
    must be present and hash to itself. The current frame must match the
    last evidence digest.
    """
    if not outcome:
        return False
    evidence = list(outcome.get('evidence_digests') or [])
    if not evidence:
        return False
    current = digest_rgba(rgba)
    if evidence[-1] != current:
        return False
    if archive is not None:
        for digest in evidence:
            stored = archive.get(digest)
            if stored is None:
                return False
            if digest_rgba(stored) != digest:
                return False
    recomputed = derive_public_outcome(
        rgba, width, height, evidence, outcome.get('receipt_id'))
    if recomputed is None:
        return False
    return recomputed['scalar'] == outcome.get('scalar')


class FrameExport(object):
    """Optional one-way framebuffer dump. Not a control protocol."""

    def __init__(self, shm_name=None):
        self.shm_name = shm_name
        self._shm = None

    def write(self, seq, timestamp_ns, width, height, rgba, consumed):
        if not self.shm_name:
            return
        payload = rgba or b''
        header = _HEADER.pack(
            _MAGIC, 1, int(seq), int(timestamp_ns),
            int(width), int(height), 1 if consumed else 0)
        blob = header + payload
        try:
            from multiprocessing import shared_memory
        except ImportError:
            return
        size = len(blob)
        if self._shm is None or self._shm.size < size:
            self.close()
            try:
                self._shm = shared_memory.SharedMemory(
                    name=self.shm_name, create=True, size=size)
            except FileExistsError:
                old = shared_memory.SharedMemory(name=self.shm_name)
                old.close()
                old.unlink()
                self._shm = shared_memory.SharedMemory(
                    name=self.shm_name, create=True, size=size)
        self._shm.buf[:size] = blob

    def close(self):
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None


class Accounting(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.logical_trials = 0
        self.significant_frames = 0
        self.authenticated_outcomes = set()
        self.dropped_frames = 0
        self.duplicate_frames = 0
        self.action_to_outcome_ns = []
        self.t0 = time.monotonic()

    def snapshot(self):
        wall = time.monotonic() - self.t0
        trials = self.logical_trials
        return {
            'logical_trials': trials,
            'significant_frames': self.significant_frames,
            'unique_public_outcome_bits': len(self.authenticated_outcomes),
            'dropped_frames': self.dropped_frames,
            'duplicate_frames': self.duplicate_frames,
            'action_to_outcome_latency_ms': [
                ns / 1e6 for ns in self.action_to_outcome_ns],
            'wall_time_s': wall,
            'trials_per_s': (trials / wall) if wall > 0 else 0.0,
        }


class _TestProbe(object):
    """Privileged inspector. Tests only; never part of the learner API."""

    def phase(self):
        return bw.mode.phase

    def stim(self):
        return dict(bw.mode.current_stim)

    def show_missed(self):
        return bool(bw.mode.show_missed)

    def session_done(self):
        return bool(bw.mode.session_done or bw.mode.phase == 'done')

    def score_snapshot(self):
        return {
            'trial_number': bw.mode.trial_number,
            'inputs': dict(bw.mode.inputs),
            'started': bool(bw.mode.started),
        }


class NeuralWorkshopEnv(object):
    def __init__(self, seed=None, shm_name=None):
        bw.mode.step_mode = True
        try:
            pyglet.clock.unschedule(bw.update)
        except Exception:
            pass
        try:
            bw.window.set_visible(False)
        except Exception:
            pass
        bw.cfg.SHOW_FEEDBACK = True
        bw.cfg.ANIMATE_SQUARES = False
        self._export = FrameExport(shm_name=shm_name or os.environ.get('NW_SHM'))
        self.accounting = Accounting()
        self._seq = 0
        self._timestamp_ns = 0
        self._width = 0
        self._height = 0
        self._rgba = b''
        self._digest = ''
        self._pending = False
        self._consumed = True
        self._phase = None  # private
        self._trial_digests = []
        self._trial_receipt = None
        self._response_open = False
        self._receipt_seq = 0
        self._closed = False
        self._events = []
        self._archive = {}
        self._action_finalized = False
        self.probe = _TestProbe()
        self.reset(0 if seed is None else seed)

    @property
    def n_actions(self):
        return len(bw.action_button_names())

    def reset(self, seed=0):
        import random
        seed = int(seed)
        random.seed(seed)
        bwaccel.seed(seed)
        if bw.mode.started:
            bw.end_session(cancelled=True)
        bw.mode.step_mode = True
        bw.mode.session_done = False
        bw.mode.phase = None
        bw.mode.session_number = 0
        bw.mode.progress = 0
        bw.cfg.SHOW_FEEDBACK = True
        bw.cfg.ANIMATE_SQUARES = False
        bw.mode.hide_text = False
        bw.new_session()
        bw.mode.step_mode = True
        bw.mode.tick = 0
        bw.mode.phase = None
        bw.mode.session_number = 1
        random.seed(seed)
        bwaccel.seed(seed)
        self.accounting.reset()
        self._events = []
        self._trial_digests = []
        self._trial_receipt = None
        self._response_open = False
        self._receipt_seq = 0
        self._pending = False
        self._consumed = True
        self._archive = {}
        self._action_finalized = False
        phase = bw.trial_advance_significant()
        self._publish(phase)
        if phase == 'stimulus':
            self._open_trial_window()
            self.accounting.logical_trials = 1
        return self.observe()

    def observe(self):
        obs = self._observation()
        self._consumed = True
        self._pending = False
        self._export.write(self._seq, self._timestamp_ns,
                           self._width, self._height, self._rgba, True)
        return obs

    def act(self, ports=None):
        """Accept exactly one finalized action per trial (stimulus window)."""
        rejected = {
            'ok': False,
            'receipt_id': None,
            'frame_seq': self._seq,
            'timestamp_ns': _now_ns(),
            'ports': (),
        }
        if not self._response_open or self._trial_receipt is None:
            return rejected
        if self._action_finalized:
            return rejected
        indices = self._decode_ports(ports)
        names = bw.action_button_names()
        buttons = [names[i] for i in indices if 0 <= i < len(names)]
        bw.inject_match_action(buttons)
        self._action_finalized = True
        self._trial_receipt = {
            'ok': True,
            'receipt_id': self._trial_receipt['receipt_id'],
            'frame_seq': self._seq,
            'timestamp_ns': _now_ns(),
            'ports': tuple(indices),
        }
        return dict(self._trial_receipt)

    def advance(self):
        if self._pending and not self._consumed:
            self.accounting.duplicate_frames += 1
            return self.observe()
        if bw.mode.session_done or bw.mode.phase == 'done':
            self._publish('done')
            return self.observe()
        prev = bw.mode.phase
        phase = bw.trial_advance_significant()
        if prev == 'stimulus' and phase != 'stimulus':
            self._response_open = False
        if phase == 'stimulus':
            self._trial_digests = []
            self.accounting.logical_trials += 1
        self._publish(phase)
        if phase == 'stimulus':
            self._open_trial_window()
        return self.observe()

    def step(self, ports=None):
        if ports is not None:
            self.act(ports)
        obs = self.advance()
        ev = list(self._events)
        self._events = []
        return obs, ev, bool(obs.get('done'))

    def close(self):
        if self._closed:
            return
        self._closed = True
        if bw.mode.started:
            try:
                bw.end_session(cancelled=True)
            except Exception:
                pass
        self._export.close()

    def _open_trial_window(self):
        self._response_open = True
        self._action_finalized = False
        self._receipt_seq += 1
        self._trial_receipt = {
            'ok': True,
            'receipt_id': self._receipt_seq,
            'frame_seq': self._seq,
            'timestamp_ns': _now_ns(),
            'ports': (),
        }

    def _decode_ports(self, ports):
        n = self.n_actions
        if ports is None:
            return []
        if isinstance(ports, int):
            return [ports] if 0 <= ports < n else []
        if isinstance(ports, dict):
            # Refuse semantic names. Only integer keys.
            out = []
            for k, v in ports.items():
                if v and isinstance(k, int) and 0 <= k < n:
                    out.append(k)
            return out
        out = []
        for p in ports:
            if isinstance(p, int) and 0 <= p < n:
                out.append(p)
        return out

    def _observation(self):
        obs = {
            'frame_seq': self._seq,
            'timestamp_ns': self._timestamp_ns,
            'width': self._width,
            'height': self._height,
            'rgba': self._rgba,
            'done': self._phase == 'done',
        }
        if self._phase == 'feedback':
            outcome = derive_public_outcome(
                self._rgba, self._width, self._height,
                self._trial_digests,
                (self._trial_receipt or {}).get('receipt_id'),
            )
            if outcome is not None:
                obs['outcome'] = outcome
                ev_key = (
                    outcome['receipt_id'],
                    tuple(outcome['evidence_digests']),
                    outcome['scalar'],
                )
                self.accounting.authenticated_outcomes.add(ev_key)
                if self._trial_receipt:
                    lat = self._timestamp_ns - self._trial_receipt['timestamp_ns']
                    self.accounting.action_to_outcome_ns.append(lat)
                self._events.append({
                    'type': 'outcome',
                    'scalar': outcome['scalar'],
                    'evidence_digests': outcome['evidence_digests'],
                    'receipt_id': outcome['receipt_id'],
                    'frame_seq': self._seq,
                    'timestamp_ns': self._timestamp_ns,
                })
        if self._phase == 'done':
            self._events.append({
                'type': 'session_end',
                'frame_seq': self._seq,
                'timestamp_ns': self._timestamp_ns,
            })
        return obs

    def _publish(self, phase):
        if self._pending and not self._consumed:
            self.accounting.dropped_frames += 1
        w, h, rgba = render_significant_frame()
        self._seq += 1
        self._timestamp_ns = _now_ns()
        self._width = w
        self._height = h
        self._rgba = rgba
        self._digest = digest_rgba(rgba)
        self._archive[self._digest] = bytes(rgba)
        self._phase = phase
        self._pending = True
        self._consumed = False
        self._trial_digests.append(self._digest)
        self.accounting.significant_frames += 1
        self._export.write(self._seq, self._timestamp_ns, w, h, rgba, False)


def make_env(seed=0, shm_name=None):
    return NeuralWorkshopEnv(seed=seed, shm_name=shm_name)


def format_accounting(acc):
    s = acc.snapshot() if hasattr(acc, 'snapshot') else acc
    lat = s['action_to_outcome_latency_ms']
    avg_lat = (sum(lat) / len(lat)) if lat else 0.0
    return (
        'logical_trials=%i significant_frames=%i unique_public_outcome_bits=%i '
        'dropped_frames=%i duplicate_frames=%i '
        'action_to_outcome_latency_ms_avg=%.2f wall_time_s=%.3f trials/s=%.1f'
        % (s['logical_trials'], s['significant_frames'],
           s['unique_public_outcome_bits'], s['dropped_frames'],
           s['duplicate_frames'], avg_lat, s['wall_time_s'],
           s['trials_per_s'])
    )


if __name__ == '__main__':
    env = make_env(seed=1)
    done = False
    n = 0
    while not done and n < 400:
        _obs, _ev, done = env.step([])
        n += 1
    print(format_accounting(env.accounting))
    print('last_frame=%sx%s done=%s' % (
        env._width, env._height, done))
    env.close()
