#!/usr/bin/env python
"""Agent-boundary acceptance tests. Skip if a GL context cannot be created."""
from __future__ import print_function

import os
import random
import sys
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
        NeuralWorkshopEnv, derive_public_outcome, digest_rgba,
        render_significant_frame, verify_public_outcome,
    )
    _ENV_IMPORT_ERROR = None
except Exception as exc:
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


@unittest.skipIf(NeuralWorkshopEnv is None, 'nwenv import failed: %s' % _ENV_IMPORT_ERROR)
class CaptureAndPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = NeuralWorkshopEnv(seed=5)

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
            archive=self.env._archive))

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
        self.assertTrue(verify_public_outcome(outcome, rgba, w, h, archive))
        forged = dict(outcome)
        forged['evidence_digests'] = ['forged']
        self.assertFalse(verify_public_outcome(forged, rgba, w, h, archive))

    def test_public_observation_schema(self):
        obs = self.env.reset(1)
        allowed = {'frame_seq', 'timestamp_ns', 'width', 'height',
                   'rgba', 'done', 'outcome'}
        self.assertTrue(set(obs.keys()) <= allowed)
        for leaked in ('phase', 'position1', 'correct', 'match',
                       'bt_sequence', 'current_stim', 'feedback'):
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


@unittest.skipIf(NeuralWorkshopEnv is None, 'nwenv import failed: %s' % _ENV_IMPORT_ERROR)
class ReceiptAndOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = NeuralWorkshopEnv(seed=2)

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
                    self.env._archive))
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
        self.assertFalse(verify_public_outcome(shuffled, rgba, w, h, archive))


@unittest.skipIf(NeuralWorkshopEnv is None, 'nwenv import failed: %s' % _ENV_IMPORT_ERROR)
class DeterminismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = NeuralWorkshopEnv(seed=0)

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


@unittest.skipIf(NeuralWorkshopEnv is None, 'nwenv import failed: %s' % _ENV_IMPORT_ERROR)
class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = NeuralWorkshopEnv(seed=13)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_step_vs_scheduled_update_parity(self):
        """Headless step path vs the scheduled update() callback (human clock)."""
        seed = 13
        n_frames = 8
        first = self.env.reset(seed)
        step_digests = [digest_rgba(first['rgba'])]
        step_stims = [dict(self.env.probe.stim())]
        step_term = False
        for _ in range(n_frames - 1):
            obs = self.env.advance()
            step_digests.append(digest_rgba(obs['rgba']))
            if self.env.probe.phase() == 'stimulus':
                step_stims.append(dict(self.env.probe.stim()))
            if obs.get('done'):
                step_term = True
                break
        step_trial = bw.mode.trial_number
        step_inputs = dict(bw.mode.inputs)

        random.seed(seed)
        bwaccel.seed(seed)
        if bw.mode.started:
            bw.end_session(cancelled=True)
        bw.mode.step_mode = True
        bw.mode.session_number = 0
        bw.cfg.SHOW_FEEDBACK = True
        bw.cfg.ANIMATE_SQUARES = False
        bw.mode.hide_text = False
        bw.new_session()
        bw.mode.tick = 0
        bw.mode.phase = None
        bw.mode.session_number = 1
        random.seed(seed)
        bwaccel.seed(seed)
        # Human scheduled path: the same update() pyglet would call.
        bw.mode.step_mode = False
        try:
            bw.window.set_visible(True)
        except Exception:
            pass
        clock_digests = []
        clock_stims = []
        last_phase = None
        guard = 0
        while len(clock_digests) < len(step_digests) and guard < 20000:
            bw.update(0.001)
            ph = bw.mode.phase
            if ph and ph != last_phase:
                if ph == 'stimulus':
                    clock_stims.append(dict(bw.mode.current_stim))
                w, h, rgba = render_significant_frame()
                clock_digests.append(digest_rgba(rgba))
                last_phase = ph
            if bw.mode.phase == 'done' or not bw.mode.started:
                break
            guard += 1
        try:
            bw.window.set_visible(False)
        except Exception:
            pass
        bw.mode.step_mode = True
        self.assertEqual(step_stims[:1], clock_stims[:1])
        self.assertEqual(step_digests, clock_digests[:len(step_digests)])
        self.assertEqual(step_trial, bw.mode.trial_number)
        if step_term:
            self.assertTrue(bw.mode.phase == 'done' or not bw.mode.started)


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
