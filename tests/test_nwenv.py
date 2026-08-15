#!/usr/bin/env python
"""Agent-boundary acceptance tests. Skip if a GL context cannot be created."""
from __future__ import print_function

import os
import random
import sys
import threading
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ['NW_HEADLESS'] = '1'
os.environ['NW_TICK_MS'] = '1'
os.environ['NW_TRIAL_MS'] = '10'
os.environ['NW_STIM_MS'] = '500'

warnings.filterwarnings('ignore', category=ResourceWarning)

try:
    import bwaccel
    import brainworkshop as bw
    from nwenv import (
        DiagnosticEnv, NeuralWorkshopEnv, diagnose_public_outcome,
        derive_public_outcome, digest_rgba, render_significant_frame,
        verify_public_outcome, verify_public_pixels,
    )
    _ENV_IMPORT_ERROR = None
except Exception as exc:
    DiagnosticEnv = None
    NeuralWorkshopEnv = None
    _ENV_IMPORT_ERROR = exc


def _ports_for(kind):
    """kind: 'correct' | 'incorrect' — uses discarded probe/check_match."""
    names = bw.action_button_names()
    out = []
    for i, name in enumerate(names):
        result = bw.check_match(name)
        if kind == 'correct' and result == 'correct':
            out.append(i)
        if kind == 'incorrect' and result == 'incorrect':
            out.append(i)
    return out


def _advance_to(env, phase, limit=40):
    for _ in range(limit):
        if env.probe.phase() == phase:
            return True
        if env.probe.session_done():
            return False
        env.advance()
    return env.probe.phase() == phase


def _next_scorable_stimulus(env, limit=40):
    for _ in range(limit):
        if env.probe.session_done():
            return False
        if env.probe.phase() == 'stimulus' and bw.mode.trial_number > bw.mode.back:
            return True
        env.advance()
    return False


@unittest.skipIf(DiagnosticEnv is None, 'nwenv import failed: %s' % _ENV_IMPORT_ERROR)
class CaptureAndPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = DiagnosticEnv(seed=5)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_feedback_digest_is_this_frame_not_prior_stimulus(self):
        self.env.reset(5)
        self.assertTrue(_next_scorable_stimulus(self.env))
        stim_digest = digest_rgba(self.env._rgba)
        # Force public feedback: miss a match or false-alarm.
        if not _ports_for('correct'):
            ports = _ports_for('incorrect')
            self.assertTrue(ports)
            self.env.act(ports[:1])
        obs = None
        for _ in range(8):
            obs = self.env.advance()
            if self.env.probe.phase() == 'feedback':
                break
        self.assertIsNotNone(obs)
        self.assertNotEqual(digest_rgba(obs['rgba']), stim_digest)
        self.assertIn('outcome', obs)
        self.assertEqual(obs['outcome']['evidence_digests'][-1],
                         digest_rgba(obs['rgba']))
        self.assertTrue(verify_public_outcome(
            obs['outcome'], obs['rgba'], obs['width'], obs['height'],
            archive=self.env._archive,
            receipt_ledger=self.env._receipt_ledger))
        self.assertTrue(set(obs['outcome'].keys()) <= {
            'scalar', 'evidence_digests', 'receipt_id',
            'frame_seq', 'timestamp_ns'})
        self.assertNotIn('n_pos', obs['outcome'])
        self.assertNotIn('n_neg', obs['outcome'])

    def test_verify_rejects_forged_digest(self):
        w, h = 20, 24
        row_g = bytes([64, 255, 64, 255] * w)
        row_k = bytes([10, 10, 10, 255] * w)
        rgba = row_k * 18 + row_g * 6
        real_d = digest_rgba(rgba)
        archive = {real_d: rgba}
        outcome = derive_public_outcome(rgba, w, h, [real_d], 1)
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome['scalar'], 1.0)
        self.assertTrue(verify_public_pixels(outcome, rgba, w, h, archive))
        self.assertFalse(
            verify_public_outcome(outcome, rgba, w, h, archive=archive),
            'public verifier requires a receipt ledger')
        forged = dict(outcome)
        forged['evidence_digests'] = ['forged']
        self.assertFalse(verify_public_pixels(forged, rgba, w, h, archive))
        earlier = dict(outcome)
        earlier['evidence_digests'] = ['forged-stim', real_d]
        self.assertFalse(
            verify_public_pixels(earlier, rgba, w, h, archive=None),
            'multi-frame evidence requires an archive')
        self.assertFalse(
            verify_public_pixels(earlier, rgba, w, h, archive=archive))

    def test_public_observation_schema(self):
        obs = self.env.reset(1)
        allowed = {'frame_seq', 'timestamp_ns', 'width', 'height',
                   'rgba', 'done', 'outcome',
                   'audio_pcm', 'audio_rate', 'audio_channels',
                   'audio_sample_width'}
        self.assertTrue(set(obs.keys()) <= allowed)
        for leaked in ('phase', 'position1', 'correct', 'match',
                       'bt_sequence', 'current_stim', 'feedback',
                       'n_pos', 'n_neg', 'letter', 'audio'):
            self.assertNotIn(leaked, obs)

    def test_significant_states_once_each(self):
        first = self.env.reset(6)
        seqs = [first['frame_seq']]
        phases = [self.env.probe.phase()]
        for _ in range(8):
            obs = self.env.advance()
            seqs.append(obs['frame_seq'])
            phases.append(self.env.probe.phase())
        self.assertEqual(seqs, sorted(set(seqs)))
        self.assertEqual(phases[0], 'stimulus')
        self.assertIn(phases[1], ('blank', 'feedback'))


@unittest.skipIf(DiagnosticEnv is None, 'nwenv import failed: %s' % _ENV_IMPORT_ERROR)
class ProductionEnvTests(unittest.TestCase):
    def test_production_has_no_probe(self):
        os.environ.pop('NW_DIAGNOSTICS', None)
        env = NeuralWorkshopEnv(seed=1)
        try:
            self.assertFalse(hasattr(env, 'probe'))
            with self.assertRaises(RuntimeError):
                NeuralWorkshopEnv(seed=1, diagnostics=True)
        finally:
            env.close()

    def test_observe_emits_outcome_once(self):
        env = DiagnosticEnv(seed=7)
        try:
            self.assertTrue(_next_scorable_stimulus(env))
            if not _ports_for('correct'):
                ports = _ports_for('incorrect')
                if ports:
                    env.act(ports[:1])
            first = None
            while env.probe.phase() != 'feedback':
                first = env.advance()
            n_out = lambda ev: sum(1 for e in ev if e.get('type') == 'outcome')
            ev1 = n_out(env._events)
            lat1 = len(env.accounting.action_to_outcome_ns)
            self.assertEqual(ev1, 1)
            self.assertEqual(lat1, 1)
            self.assertIn('outcome', first)
            pixels = first['rgba']
            for _ in range(3):
                again = env.observe()
                self.assertNotIn('outcome', again)
                self.assertEqual(again['rgba'], pixels)
            self.assertEqual(n_out(env._events), 1)
            self.assertEqual(len(env.accounting.action_to_outcome_ns), 1)
            self.assertEqual(len(env._delivered), len(set(env._delivered)))
        finally:
            env.close()

    def test_session_end_emits_once(self):
        env = DiagnosticEnv(seed=8, num_trials=4)
        try:
            done = False
            guard = 0
            while not done and guard < 80:
                obs = env.advance()
                done = bool(obs.get('done'))
                guard += 1
            self.assertTrue(done)
            n_end = sum(1 for e in env._events if e.get('type') == 'session_end')
            self.assertEqual(n_end, 1)
            seq = env._seq
            for _ in range(4):
                obs = env.advance()
                self.assertTrue(obs.get('done'))
                self.assertEqual(obs['frame_seq'], seq)
            self.assertEqual(
                sum(1 for e in env._events if e.get('type') == 'session_end'), 1)
        finally:
            env.close()

    def test_headless_terminates_without_audio_thread_exceptions(self):
        caught = []

        def hook(args):
            caught.append(args)

        prev = threading.excepthook
        threading.excepthook = hook
        env = DiagnosticEnv(seed=4, game_mode=2, num_trials=4)
        try:
            done = False
            for _ in range(40):
                _obs, _ev, done = env.step([])
                if done:
                    break
            self.assertTrue(done)
            driver = __import__('pyglet').media.get_audio_driver()
            self.assertEqual(driver.__class__.__name__, 'SilentDriver')
            self.assertTrue(isinstance(bw.player, bw.CapturePlayer))
            self.assertEqual(caught, [], 'OpenAL/audio worker raised: %s' % caught)
        finally:
            threading.excepthook = prev
            env.close()

    def test_act_stores_logp_on_receipt(self):
        env = DiagnosticEnv(seed=3)
        try:
            rec = env.act(0, logp=-1.25)
            self.assertTrue(rec.get('ok'))
            self.assertEqual(rec.get('logp'), -1.25)
            self.assertEqual(env._trial_receipt['logp'], -1.25)
            self.assertEqual(
                env._receipt_ledger[rec['receipt_id']]['logp'], -1.25)
        finally:
            env.close()


@unittest.skipIf(DiagnosticEnv is None, 'nwenv import failed: %s' % _ENV_IMPORT_ERROR)
class ReceiptAndOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = DiagnosticEnv(seed=2, num_trials=12)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_positive_outcome_required(self):
        self.env.reset(20)
        found = None
        for _ in range(50):
            if not _next_scorable_stimulus(self.env):
                break
            ports = _ports_for('correct')
            if not ports:
                self.env.advance()
                continue
            self.env.act(ports)
            while self.env.probe.phase() != 'feedback':
                obs = self.env.advance()
                if self.env.probe.session_done():
                    break
            else:
                obs = self.env.observe() if False else {
                    'outcome': None, 'rgba': self.env._rgba,
                    'width': self.env._width, 'height': self.env._height,
                }
                # last advance already observed; grab from last publish via step back
            # Re-read current published observation
            cur = {
                'rgba': self.env._rgba,
                'width': self.env._width,
                'height': self.env._height,
            }
            out = derive_public_outcome(
                cur['rgba'], cur['width'], cur['height'],
                self.env._trial_digests,
                self.env._trial_receipt['receipt_id'])
            if out and out['scalar'] == 1.0:
                found = out
                self.assertTrue(verify_public_outcome(
                    out, cur['rgba'], cur['width'], cur['height'],
                    archive=self.env._archive,
                    receipt_ledger=self.env._receipt_ledger))
                break
            if self.env.probe.session_done():
                break
        self.assertIsNotNone(found, 'required a +1 public outcome')

    def test_negative_false_alarm_required(self):
        self.env.reset(21)
        found = None
        for _ in range(50):
            if not _next_scorable_stimulus(self.env):
                break
            ports = _ports_for('incorrect')
            if not ports:
                self.env.advance()
                continue
            self.env.act(ports[:1])
            while self.env.probe.phase() != 'feedback':
                self.env.advance()
                if self.env.probe.session_done():
                    break
            out = derive_public_outcome(
                self.env._rgba, self.env._width, self.env._height,
                self.env._trial_digests,
                self.env._trial_receipt['receipt_id'])
            if out and out['scalar'] == -1.0:
                found = out
                break
            if self.env.probe.session_done():
                break
        self.assertIsNotNone(found, 'required a -1 false-alarm outcome')

    def test_missed_match_is_negative(self):
        self.env.reset(22)
        found = None
        for _ in range(50):
            if not _next_scorable_stimulus(self.env):
                break
            if not _ports_for('correct'):
                self.env.advance()
                continue
            # Deliberate no-action on a real match → blue oops → -1
            while self.env.probe.phase() != 'feedback':
                self.env.advance()
                if self.env.probe.session_done():
                    break
            out = derive_public_outcome(
                self.env._rgba, self.env._width, self.env._height,
                self.env._trial_digests,
                self.env._trial_receipt['receipt_id'])
            if out and out['scalar'] == -1.0:
                found = out
                break
            if self.env.probe.session_done():
                break
        self.assertIsNotNone(found, 'required a -1 missed-match outcome')

    def test_no_action_still_gets_a_receipt(self):
        self.env.reset(2)
        self.assertIsNotNone(self.env._trial_receipt)
        self.assertEqual(self.env._trial_receipt['ports'], ())
        rid = self.env._trial_receipt['receipt_id']
        _advance_to(self.env, 'feedback')
        self.assertEqual(self.env._trial_receipt['receipt_id'], rid)

    def test_second_act_fails_closed(self):
        self.env.reset(3)
        _advance_to(self.env, 'stimulus')
        first = self.env.act(0)
        self.assertTrue(first.get('ok'))
        second = self.env.act(0)
        self.assertFalse(second.get('ok'))
        self.assertEqual(self.env._trial_receipt['ports'], first['ports'])

    def test_late_action_fails_closed(self):
        self.env.reset(3)
        while self.env.probe.phase() == 'stimulus':
            self.env.advance()
        self.assertFalse(self.env._response_open)
        held = dict(self.env._trial_receipt)
        rejected = self.env.act(0)
        self.assertFalse(rejected.get('ok'))
        self.assertIsNone(rejected.get('receipt_id'))
        self.assertEqual(self.env._trial_receipt, held)

    def test_missing_feedback_yields_no_outcome(self):
        self.env.reset(9)
        bw.cfg.SHOW_FEEDBACK = False
        try:
            saw = False
            for _ in range(12):
                obs = self.env.advance()
                if self.env.probe.phase() == 'feedback':
                    saw = True
                    self.assertNotIn('outcome', obs)
                    self.assertIsNone(derive_public_outcome(
                        obs['rgba'], obs['width'], obs['height'], [], 1))
                    break
            self.assertTrue(saw)
        finally:
            bw.cfg.SHOW_FEEDBACK = True

    def test_duplicate_frame_does_not_advance(self):
        self.env.reset(10)
        first = self.env.advance()
        seq = first['frame_seq']
        self.env._pending = True
        self.env._consumed = False
        again = self.env.advance()
        self.assertEqual(again['frame_seq'], seq)
        self.assertGreaterEqual(self.env.accounting.duplicate_frames, 1)

    def test_reward_shuffle_fails_verify(self):
        # Bottom-quarter green band so the ROI scanner sees it
        row = bytes([64, 255, 64, 255] * 20)
        top = bytes([10, 10, 10, 255] * 20) * 18
        bot = row * 6
        rgba = top + bot
        w, h = 20, 24
        d = digest_rgba(rgba)
        real = derive_public_outcome(rgba, w, h, [d], 1)
        self.assertIsNotNone(real)
        self.assertEqual(real['scalar'], 1.0)
        shuffled = dict(real)
        shuffled['scalar'] = -1.0
        archive = {d: rgba}
        self.assertFalse(verify_public_pixels(shuffled, rgba, w, h, archive))

    def test_action_shuffled_receipt_fails_verify(self):
        self.env.reset(15)
        self.assertTrue(_next_scorable_stimulus(self.env))
        self.env.act(0)
        while self.env.probe.phase() != 'feedback':
            self.env.advance()
        obs = {
            'rgba': self.env._rgba, 'width': self.env._width,
            'height': self.env._height,
        }
        out = derive_public_outcome(
            obs['rgba'], obs['width'], obs['height'],
            self.env._trial_digests,
            self.env._trial_receipt['receipt_id'])
        self.assertIsNotNone(out)
        shuffled = dict(out)
        shuffled['receipt_id'] = out['receipt_id'] + 999
        self.assertFalse(verify_public_outcome(
            shuffled, obs['rgba'], obs['width'], obs['height'],
            archive=self.env._archive,
            receipt_ledger=self.env._receipt_ledger))

    def test_foreign_valid_receipt_fails_verify(self):
        """Another ledger receipt from a different trial must not verify."""
        env = self.env
        env.reset(18)
        collected = []
        for _ in range(40):
            if len(collected) >= 2:
                break
            if not _next_scorable_stimulus(env):
                break
            env.act(_ports_for('incorrect')[:1] or [0])
            obs = None
            while env.probe.phase() != 'feedback':
                obs = env.advance()
                if env.probe.session_done():
                    break
            if not obs or 'outcome' not in obs:
                continue
            collected.append({
                'outcome': dict(obs['outcome']),
                'rgba': obs['rgba'],
                'width': obs['width'],
                'height': obs['height'],
            })
        self.assertGreaterEqual(len(collected), 2, 'need two scored trials')
        a, b = collected[0], collected[1]
        self.assertNotEqual(a['outcome']['receipt_id'],
                            b['outcome']['receipt_id'])
        self.assertTrue(verify_public_outcome(
            b['outcome'], b['rgba'], b['width'], b['height'],
            archive=env._archive, receipt_ledger=env._receipt_ledger))
        swapped = dict(b['outcome'])
        swapped['receipt_id'] = a['outcome']['receipt_id']
        self.assertTrue(
            verify_public_pixels(
                swapped, b['rgba'], b['width'], b['height'],
                archive=env._archive),
            'pixel-only diagnostic still accepts a swapped receipt')
        self.assertFalse(
            verify_public_outcome(
                swapped, b['rgba'], b['width'], b['height'],
                archive=env._archive),
            'public verifier fails closed without a ledger')
        self.assertFalse(verify_public_outcome(
            swapped, b['rgba'], b['width'], b['height'],
            archive=env._archive, receipt_ledger=env._receipt_ledger),
            'valid receipt from another trial must not bind')

    def test_public_verify_requires_ledger(self):
        env = self.env
        env.reset(19)
        self.assertTrue(_next_scorable_stimulus(env))
        env.act(_ports_for('incorrect')[:1] or [0])
        obs = None
        while env.probe.phase() != 'feedback':
            obs = env.advance()
            if env.probe.session_done():
                break
        self.assertIsNotNone(obs)
        self.assertIn('outcome', obs)
        self.assertFalse(
            verify_public_outcome(
                obs['outcome'], obs['rgba'], obs['width'], obs['height'],
                archive=env._archive),
            'archive without ledger must fail closed')
        self.assertFalse(
            verify_public_outcome(
                obs['outcome'], obs['rgba'], obs['width'], obs['height'],
                receipt_ledger=env._receipt_ledger),
            'ledger without archive must fail closed')
        self.assertTrue(verify_public_outcome(
            obs['outcome'], obs['rgba'], obs['width'], obs['height'],
            archive=env._archive, receipt_ledger=env._receipt_ledger))

    def test_count_fields_fail_verify(self):
        w, h = 20, 24
        row = bytes([64, 255, 64, 255] * w)
        rgba = bytes([10, 10, 10, 255] * w) * 18 + row * 6
        d = digest_rgba(rgba)
        out = derive_public_outcome(rgba, w, h, [d], 1)
        self.assertNotIn('n_pos', out)
        self.assertNotIn('n_neg', out)
        leaked = dict(out)
        leaked['n_pos'] = 1
        leaked['n_neg'] = 0
        ledger = {1: {
            'receipt_id': 1, 'trial_seq': 1, 'stimulus_digest': d,
            'evidence_digests': [d], 'feedback_digest': d,
        }}
        self.assertFalse(verify_public_outcome(
            leaked, rgba, w, h, archive={d: rgba}, receipt_ledger=ledger))

    def test_delayed_resolution_keeps_stimulus_receipt(self):
        """Action during stimulus is resolved only at later feedback."""
        self.env.reset(16)
        self.assertTrue(_next_scorable_stimulus(self.env))
        stim = dict(self.env.probe.stim())
        rec = self.env.act(_ports_for('incorrect')[:1] or [0])
        self.assertTrue(rec.get('ok'))
        rid = rec['receipt_id']
        self.assertNotIn('outcome', self.env.observe())
        while self.env.probe.phase() == 'stimulus':
            obs = self.env.advance()
            if self.env.probe.phase() != 'feedback':
                self.assertNotIn('outcome', obs)
        self.assertFalse(self.env._response_open)
        while self.env.probe.phase() != 'feedback':
            obs = self.env.advance()
            if self.env.probe.phase() != 'feedback':
                self.assertNotIn('outcome', obs)
        self.assertEqual(self.env._trial_receipt['receipt_id'], rid)
        out = derive_public_outcome(
            self.env._rgba, self.env._width, self.env._height,
            self.env._trial_digests, rid)
        self.assertIsNotNone(out)
        self.assertEqual(out['receipt_id'], rid)
        acted_scalar = out['scalar']

        # Same seed, no action: delayed resolution is causal, not a stimulus tag.
        self.env.reset(16)
        self.assertTrue(_next_scorable_stimulus(self.env))
        self.assertEqual(self.env.probe.stim(), stim)
        while self.env.probe.phase() != 'feedback':
            self.env.advance()
        idle = derive_public_outcome(
            self.env._rgba, self.env._width, self.env._height,
            self.env._trial_digests,
            self.env._trial_receipt['receipt_id'])
        idle_scalar = None if idle is None else idle['scalar']
        self.assertNotEqual(acted_scalar, idle_scalar)

    def test_action_shuffled_control_changes_outcomes(self):
        """Same stimuli, permuted trial-actions: outcome sequence must move."""
        env = self.env
        actions = [[0], [1] if env.n_actions > 1 else [0], [], [0],
                   [1] if env.n_actions > 1 else [], [], [0],
                   [1] if env.n_actions > 1 else [0]]

        def collect(act_list):
            env.reset(17)
            stims, outcomes, receipts = [], [], []
            box = [0]

            def maybe_act():
                if env.probe.phase() != 'stimulus':
                    return
                if bw.mode.trial_number <= bw.mode.back:
                    return
                if box[0] >= len(act_list):
                    return
                rec = env.act(act_list[box[0]])
                box[0] += 1
                receipts.append((rec.get('receipt_id'), tuple(rec.get('ports') or ())))
                stims.append((bw.mode.trial_number, tuple(sorted(
                    env.probe.stim().items()))))

            maybe_act()
            for _ in range(80):
                if env.probe.session_done():
                    break
                env.advance()
                if env.probe.phase() == 'feedback':
                    out = derive_public_outcome(
                        env._rgba, env._width, env._height,
                        env._trial_digests,
                        (env._trial_receipt or {}).get('receipt_id'))
                    outcomes.append(None if out is None else out['scalar'])
                maybe_act()
            return stims, outcomes, receipts

        a = collect(actions)
        shuffled = list(actions)
        rng = random.Random(0)
        rng.shuffle(shuffled)
        if shuffled == actions:
            shuffled = list(reversed(actions))
        b = collect(shuffled)
        self.assertEqual(a[0], b[0], 'stimuli must be seed-identical')
        self.assertNotEqual(a[1], b[1], 'shuffled actions must move outcomes')
        self.assertNotEqual(a[2], b[2], 'receipts must follow the permuted acts')


@unittest.skipIf(DiagnosticEnv is None, 'nwenv import failed: %s' % _ENV_IMPORT_ERROR)
class DeterminismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = DiagnosticEnv(seed=0)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def _trace(self, seed, n=8):
        obs = self.env.reset(seed)
        frames = [digest_rgba(obs['rgba'])]
        receipts = [tuple(self.env._trial_receipt['ports'])]
        outcomes = []
        stims = [dict(self.env.probe.stim())]
        for _ in range(n):
            obs, ev, done = self.env.step(None)
            frames.append(digest_rgba(obs['rgba']))
            receipts.append(tuple(self.env._trial_receipt['ports'])
                            if self.env._trial_receipt else None)
            if self.env.probe.phase() == 'stimulus':
                stims.append(dict(self.env.probe.stim()))
            for e in ev:
                if e.get('type') == 'outcome':
                    outcomes.append(e['scalar'])
            if done:
                break
        return frames, receipts, outcomes, stims

    def test_seed_zero_and_nonzero_repeat(self):
        for seed in (0, 1, 42):
            a = self._trace(seed, 6)
            b = self._trace(seed, 6)
            self.assertEqual(a[0], b[0], 'frame digests seed=%s' % seed)
            self.assertEqual(a[1], b[1], 'receipts seed=%s' % seed)
            self.assertEqual(a[2], b[2], 'outcomes seed=%s' % seed)
            self.assertEqual(a[3], b[3], 'stims seed=%s' % seed)


@unittest.skipIf(DiagnosticEnv is None, 'nwenv import failed: %s' % _ENV_IMPORT_ERROR)
class ParityTests(unittest.TestCase):
    """Stepped vs scheduled ``update()`` parity with the window hidden.

    Limitation: this is not literal visible-window parity. The scheduled
    path calls ``window.set_visible(False)``. Pixel, action, input,
    outcome, and termination checks still run on the full session.
    """
    @classmethod
    def setUpClass(cls):
        cls.env = DiagnosticEnv(seed=13, game_mode=2, num_trials=6)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    @staticmethod
    def _policy(trial_number):
        if trial_number <= 2:
            return []
        return [0] if trial_number % 2 == 0 else []

    def _bootstrap_clock_session(self, seed):
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
        bw.mode.mode = 2
        bw.cfg.GAME_MODE = 2
        bw.mode.num_trials = 6
        bw.mode.num_trials_factor = 0
        bw.mode.num_trials_total = 6
        bw.new_session()
        bw.mode.tick = 0
        bw.mode.phase = None
        bw.mode.session_number = 1
        bw.mode.num_trials = 6
        bw.mode.num_trials_factor = 0
        bw.mode.num_trials_total = 6
        random.seed(seed)
        bwaccel.seed(seed)
        bw.mode.step_mode = False

    def test_step_vs_scheduled_update_parity(self):
        """Full-session step vs scheduled update(); window stays hidden."""
        seed = 13
        env = self.env
        first = env.reset(seed)
        self.assertEqual(env.probe.phase(), 'stimulus')
        rec0 = env.act(self._policy(bw.mode.trial_number), logp=-0.5)
        step_digests = [digest_rgba(first['rgba'])]
        step_stims = [dict(env.probe.stim())]
        step_receipts = [(rec0.get('receipt_id'), tuple(rec0.get('ports') or ()),
                          rec0.get('logp'))]
        step_outcomes = []
        step_inputs = []
        step_term = False
        while True:
            obs, ev, done = env.step(None)
            step_digests.append(digest_rgba(obs['rgba']))
            phase = env.probe.phase()
            if phase == 'stimulus':
                step_stims.append(dict(env.probe.stim()))
                rec = env.act(self._policy(bw.mode.trial_number),
                              logp=-0.5 if bw.mode.trial_number % 2 == 0 else None)
                step_receipts.append((
                    rec.get('receipt_id'), tuple(rec.get('ports') or ()),
                    rec.get('logp')))
            if phase == 'feedback':
                step_inputs.append(dict(bw.mode.inputs))
            for e in ev:
                if e.get('type') == 'outcome':
                    step_outcomes.append(e['scalar'])
            if done:
                step_term = True
                break
            if len(step_digests) > 80:
                self.fail('step path did not terminate')
        step_trial = bw.mode.trial_number
        step_session = {
            'position1': list(bw.stats.session.get('position1', [])),
            'audio': list(bw.stats.session.get('audio', [])),
            'position1_input': list(bw.stats.session.get('position1_input', [])),
            'audio_input': list(bw.stats.session.get('audio_input', [])),
        }
        step_stats = env.accounting.snapshot()
        sessions_today = bw.stats.sessions_today

        self._bootstrap_clock_session(seed)
        bw.stats.sessions_today = sessions_today
        try:
            bw.window.set_visible(False)
        except Exception:
            pass
        clock_digests = []
        clock_stims = []
        clock_receipts = []
        clock_outcomes = []
        clock_inputs = []
        last_phase = None
        guard = 0
        clock_term = False
        while guard < 20000:
            bw.update(0.001)
            ph = bw.mode.phase
            if ph and ph != last_phase:
                # Capture *before* injecting, matching step: publish then act.
                w, h, rgba = render_significant_frame()
                clock_digests.append(digest_rgba(rgba))
                if ph == 'stimulus':
                    clock_stims.append(dict(bw.mode.current_stim))
                    names = bw.action_button_names()
                    ports = self._policy(bw.mode.trial_number)
                    buttons = [names[i] for i in ports if 0 <= i < len(names)]
                    bw.inject_match_action(buttons)
                    clock_receipts.append((
                        bw.mode.trial_number, tuple(ports),
                        -0.5 if bw.mode.trial_number % 2 == 0 else None))
                if ph == 'feedback':
                    clock_inputs.append(dict(bw.mode.inputs))
                    out = derive_public_outcome(
                        rgba, w, h, [digest_rgba(rgba)], bw.mode.trial_number)
                    if out is not None:
                        clock_outcomes.append(out['scalar'])
                last_phase = ph
            if bw.mode.phase == 'done' or not bw.mode.started:
                clock_term = True
                break
            guard += 1
        try:
            bw.window.set_visible(False)
        except Exception:
            pass
        bw.mode.step_mode = True
        clock_session = {
            'position1': list(bw.stats.session.get('position1', [])),
            'audio': list(bw.stats.session.get('audio', [])),
            'position1_input': list(bw.stats.session.get('position1_input', [])),
            'audio_input': list(bw.stats.session.get('audio_input', [])),
        }

        self.assertTrue(step_term)
        self.assertTrue(clock_term)
        self.assertEqual(step_stims, clock_stims)
        # In-session frames must match. The post-session analysis overlay
        # includes session counters / timestamps, so the terminal digest is
        # not part of the trial protocol.
        self.assertGreaterEqual(len(step_digests), 2)
        self.assertEqual(len(step_digests), len(clock_digests))
        self.assertEqual(step_digests[:-1], clock_digests[:-1])
        self.assertEqual([r[1] for r in step_receipts],
                         [r[1] for r in clock_receipts])
        self.assertEqual(step_inputs, clock_inputs)
        self.assertEqual(step_outcomes, clock_outcomes)
        self.assertTrue(any(s == 1.0 for s in step_outcomes)
                        or any(s == -1.0 for s in step_outcomes)
                        or any(s == 0.0 for s in step_outcomes),
                        'parity session produced no public outcomes')
        self.assertEqual(step_trial, bw.mode.trial_number)
        self.assertEqual(step_session, clock_session)
        self.assertGreaterEqual(step_stats['logical_trials'], 6)
        self.assertTrue(bw.mode.phase == 'done' or not bw.mode.started)


class LabelAggregationTests(unittest.TestCase):
    def _frame(self, specs):
        """Build a 80x24 frame with colored column bands in the bottom quarter.

        specs: list of (x0, x1, rgb)
        """
        w, h = 80, 24
        pix = bytearray([10, 10, 10, 255] * (w * h))
        for x0, x1, rgb in specs:
            for y in range(18, 24):
                for x in range(x0, x1):
                    off = (y * w + x) * 4
                    pix[off:off + 3] = bytes(rgb)
        return bytes(pix), w, h

    def test_run_count_invariant_to_band_width(self):
        narrow, w, h = self._frame([(5, 10, (64, 255, 64)), (40, 45, (255, 64, 64))])
        wide, _, _ = self._frame([(5, 25, (64, 255, 64)), (40, 70, (255, 64, 64))])
        a = bwaccel.count_feedback_label_runs(narrow, w, h, 18, 24)
        b = bwaccel.count_feedback_label_runs(wide, w, h, 18, 24)
        self.assertEqual(a, (1, 1, 0))
        self.assertEqual(b, (1, 1, 0))
        oa = derive_public_outcome(narrow, w, h, ['d'], 1)
        ob = derive_public_outcome(wide, w, h, ['d'], 1)
        self.assertEqual(oa['scalar'], 0.0)
        self.assertEqual(ob['scalar'], 0.0)

    def test_two_correct_two_incorrect(self):
        two_g, w, h = self._frame([(4, 12, (64, 255, 64)), (30, 38, (64, 255, 64))])
        two_r, _, _ = self._frame([(4, 12, (255, 64, 64)), (30, 38, (255, 64, 64))])
        self.assertEqual(derive_public_outcome(two_g, w, h, ['d'], 1)['scalar'], 1.0)
        self.assertEqual(derive_public_outcome(two_r, w, h, ['d'], 1)['scalar'], -1.0)

    def test_miss_plus_correct(self):
        mix, w, h = self._frame([(4, 12, (64, 64, 255)), (30, 38, (64, 255, 64))])
        self.assertEqual(derive_public_outcome(mix, w, h, ['d'], 1)['scalar'], 0.0)


@unittest.skipIf(DiagnosticEnv is None, 'nwenv import failed: %s' % _ENV_IMPORT_ERROR)
class DualModalityLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = DiagnosticEnv(seed=30, game_mode=2, num_trials=24)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def _seek_ports(self, want_correct, want_incorrect, limit=80):
        for _ in range(limit):
            if not _next_scorable_stimulus(self.env):
                return None, None
            c = _ports_for('correct')
            i = _ports_for('incorrect')
            if len(c) >= want_correct and len(i) >= want_incorrect:
                return c, i
            self.env.advance()
        return None, None

    def _feedback_scalar(self):
        while self.env.probe.phase() != 'feedback':
            self.env.advance()
            if self.env.probe.session_done():
                return None
        return diagnose_public_outcome(
            self.env._rgba, self.env._width, self.env._height,
            self.env._trial_digests,
            self.env._trial_receipt['receipt_id'])

    def test_dual_stimulus_publishes_waveform_not_letter_id(self):
        self.env.reset(30)
        found = None
        for _ in range(40):
            obs = self.env.observe()
            if self.env.probe.phase() == 'stimulus' and obs.get('audio_pcm'):
                found = obs
                break
            self.env.advance()
        self.assertIsNotNone(found)
        self.assertIsInstance(found['audio_pcm'], (bytes, bytearray))
        self.assertGreater(len(found['audio_pcm']), 0)
        self.assertGreater(int(found['audio_rate']), 0)
        self.assertNotIn('audio', found)
        self.assertNotIn('current_stim', found)
        self.assertNotIn('letter', found)

    def test_dual_one_correct_one_incorrect(self):
        self.env.reset(30)
        self.assertGreaterEqual(self.env.n_actions, 2)
        c, i = self._seek_ports(1, 1)
        self.assertIsNotNone(c)
        self.env.act([c[0], i[0]])
        out = self._feedback_scalar()
        self.assertIsNotNone(out)
        self.assertEqual(out['n_pos'], 1)
        self.assertEqual(out['n_neg'], 1)
        self.assertEqual(out['scalar'], 0.0)
        public = derive_public_outcome(
            self.env._rgba, self.env._width, self.env._height,
            self.env._trial_digests,
            self.env._trial_receipt['receipt_id'])
        self.assertNotIn('n_pos', public)
        self.assertNotIn('n_neg', public)

    def test_dual_two_correct(self):
        found = None
        for seed in range(31, 80):
            self.env.reset(seed)
            c, i = self._seek_ports(2, 0)
            if c and len(c) >= 2:
                self.env.act(c[:2])
                found = self._feedback_scalar()
                break
        self.assertIsNotNone(found, 'needed a two-match trial')
        self.assertEqual(found['n_pos'], 2)
        self.assertEqual(found['n_neg'], 0)
        self.assertEqual(found['scalar'], 1.0)

    def test_dual_two_incorrect(self):
        found = None
        for seed in range(32, 80):
            self.env.reset(seed)
            c, i = self._seek_ports(0, 2)
            if i and len(i) >= 2:
                self.env.act(i[:2])
                found = self._feedback_scalar()
                break
        self.assertIsNotNone(found, 'needed a two-nonmatch trial')
        self.assertEqual(found['n_pos'], 0)
        self.assertEqual(found['n_neg'], 2)
        self.assertEqual(found['scalar'], -1.0)

    def test_dual_one_missed_one_correct(self):
        found = None
        for seed in range(33, 80):
            self.env.reset(seed)
            c, i = self._seek_ports(2, 0)
            if c and len(c) >= 2:
                self.env.act(c[:1])  # press one match, miss the other
                found = self._feedback_scalar()
                break
        self.assertIsNotNone(found, 'needed a two-match trial to miss one')
        self.assertEqual(found['n_pos'], 1)
        self.assertEqual(found['n_neg'], 1)
        self.assertEqual(found['scalar'], 0.0)


@unittest.skipIf(DiagnosticEnv is None, 'nwenv import failed: %s' % _ENV_IMPORT_ERROR)
class GymConfigTests(unittest.TestCase):
    """Constructor-owned session knobs. Training must not poke bw.cfg."""

    def test_constructor_applies_and_keeps_session_knobs(self):
        env = DiagnosticEnv(
            seed=8,
            game_mode=10,
            num_trials=12,
            n_back=3,
            grid_size=3,
            active_cells=2,
        )
        try:
            self.assertEqual(bw.mode.mode, 10)
            self.assertEqual(bw.mode.back, 3)
            self.assertEqual(bw.mode.num_trials_total, 12)
            self.assertEqual(bw.cfg.GRID_SIZE, 3)
            self.assertEqual(bw.cfg.ACTIVE_POSITION_CELLS, 2)
            self.assertFalse(bw.cfg.USE_MUSIC)
            self.assertTrue(bw.mode.manual)
            env.reset(9)
            self.assertEqual(bw.mode.back, 3)
            self.assertEqual(bw.mode.num_trials_total, 12)
            self.assertEqual(len(bw.current_active_position_ids()), 2)
        finally:
            env.close()
            bw.cfg.GRID_SIZE = 3
            bw.cfg.ACTIVE_POSITION_CELLS = 0
            bw.cfg.POSITION_CELL_COUNT = 0
            bw.cfg.GAME_MODE = 2
            bw.mode.mode = 2
            bw.mode.back = 2

    def test_dual_constructor_sets_two_ports_and_depth(self):
        env = DiagnosticEnv(
            seed=11, game_mode=2, num_trials=8, n_back=1,
        )
        try:
            self.assertEqual(bw.mode.mode, 2)
            self.assertEqual(bw.mode.back, 1)
            self.assertEqual(env.n_actions, 2)
        finally:
            env.close()


class CurriculumTests(unittest.TestCase):
    def test_grid_coverage_full_and_subset(self):
        for n in (2, 3, 4, 5, 8, 16, 32):
            total = bwaccel.grid_cell_count(n, False)
            full = bwaccel.active_position_ids(n, False, 0)
            self.assertEqual(len(full), total, 'full %sx%s' % (n, n))
            if total >= 4:
                subset = bwaccel.active_position_ids(n, False, 4)
                self.assertEqual(len(subset), 4)
                self.assertTrue(set(subset).issubset(set(full)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
