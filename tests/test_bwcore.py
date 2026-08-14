#!/usr/bin/env python
"""Correctness tests for bwaccel (C extension + Python fallback).

Run from the project root:

    python tests/test_bwcore.py
"""
from __future__ import print_function

import os
import sys
import time
import unittest
from decimal import Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bwaccel


def _count_matches(seq, nback):
    return sum(1 for i in range(nback, len(seq)) if seq[i] == seq[i - nback])


SAMPLE_STATS = """\
# comment line
2010-08-17 02:45:38,2xD3B,61,258,3,35,24,0,1,33,75,0,0,0,0,0,0,0,75,0,0,0,0,0,0,12.4,0
2010-08-18 14:00:00,D2B,85,2,2,30,24,0,2,90,80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,0
2010-08-18 15:00:00,D2B,40,2,2,30,24,1,0,40,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,0
not a record
"""


class SequenceTests(unittest.TestCase):
    def test_exact_match_counts(self):
        for nback in (2, 3, 5, 8, 12):
            trials = 20 + nback * nback
            pos, audio = bwaccel.compute_bt_sequence(trials, nback, 6, 6, 2)
            self.assertEqual(len(pos), trials)
            self.assertEqual(len(audio), trials)
            self.assertTrue(all(1 <= v <= 8 for v in pos + audio))
            self.assertEqual(_count_matches(pos, nback), 6)
            self.assertEqual(_count_matches(audio, nback), 6)
            both = sum(
                1 for i in range(nback, trials)
                if pos[i] == pos[i - nback] and audio[i] == audio[i - nback]
            )
            self.assertEqual(both, 2)

    def test_python_fallback_also_exact(self):
        pos, audio = bwaccel._compute_bt_sequence_py(24, 2, 6, 6, 2)
        self.assertEqual(_count_matches(pos, 2), 6)
        self.assertEqual(_count_matches(audio, 2), 6)

    def test_impossible_counts_raise(self):
        with self.assertRaises(ValueError):
            bwaccel.compute_bt_sequence(8, 2, 20, 20, 10)

    def test_custom_position_range(self):
        pos, audio = bwaccel.compute_bt_sequence(40, 2, 6, 6, 2, 16, 8)
        self.assertTrue(all(1 <= v <= 16 for v in pos))
        self.assertTrue(all(1 <= v <= 8 for v in audio))
        self.assertEqual(_count_matches(pos, 2), 6)


class GridLayoutTests(unittest.TestCase):
    def test_classic_3x3_has_eight_cells(self):
        cells = bwaccel.grid_layout(3, include_center=False)
        self.assertEqual(len(cells), 8)
        self.assertEqual(bwaccel.grid_cell_count(3), 8)
        # IDs 1-8 and no center (1,1) except via id 0/9
        coords = {(c, r) for _, c, r in cells}
        self.assertNotIn((1, 1), coords)
        self.assertEqual(bwaccel.position_col_row(1, 3), (2, 1))
        self.assertEqual(bwaccel.position_col_row(8, 3), (0, 0))
        self.assertIsNone(bwaccel.position_col_row(0, 3))

    def test_4x4_uses_every_cell(self):
        cells = bwaccel.grid_layout(4)
        self.assertEqual(len(cells), 16)
        self.assertEqual(bwaccel.grid_cell_count(4), 16)
        self.assertEqual(bwaccel.position_col_row(1, 4), (0, 0))
        self.assertEqual(bwaccel.position_col_row(16, 4), (3, 3))

    def test_large_grids(self):
        self.assertEqual(bwaccel.grid_cell_count(10), 100)
        self.assertEqual(bwaccel.grid_cell_count(16), 256)
        self.assertEqual(bwaccel.position_col_row(256, 16), (15, 15))
        self.assertEqual(bwaccel.grid_cell_count(32), 1024)
        self.assertEqual(bwaccel.position_col_row(1024, 32), (31, 31))

    def test_5x5_skips_center_unless_asked(self):
        self.assertEqual(bwaccel.grid_cell_count(5, False), 24)
        self.assertEqual(bwaccel.grid_cell_count(5, True), 25)
        coords = {(c, r) for _, c, r in bwaccel.grid_layout(5, False)}
        self.assertNotIn((2, 2), coords)

    def test_center_out_3x3_curriculum(self):
        self.assertEqual(
            bwaccel.grid_center_out_ids(3),
            [1, 2, 3, 6, 4, 5, 7, 8])


class AnalyzeTests(unittest.TestCase):
    def _session(self):
        # 2-back, 8 trials. Position matches at indices 2 and 5.
        pos = [1, 2, 1, 3, 4, 3, 7, 8]
        inp = [0, 0, 1, 0, 0, 0, 0, 0]  # hit the first match, miss the second
        audio = [5, 6, 5, 1, 2, 9, 2, 3]  # matches at 2 and 6
        a_in = [0, 0, 0, 0, 0, 0, 1, 0]
        vis = [1, 1, 2, 2, 3, 3, 4, 4]
        return {
            'position1': pos, 'position1_input': inp,
            'audio': audio, 'audio_input': a_in,
            'vis': vis,
            'visvis_input': [0, 0, 0, 1, 0, 0, 0, 0],
        }

    def test_direct_and_combo(self):
        session = self._session()
        result = bwaccel.analyze_session(
            2, False, False, None,
            ['position1', 'audio', 'visvis', 'arithmetic'],
            session,
        )
        # position: match@2 hit, match@5 miss => 1 right, 1 wrong
        self.assertEqual(result['position1'], (1, 1))
        # audio: match@2 miss, match@6 hit => 1 right, 1 wrong
        self.assertEqual(result['audio'], (1, 1))
        # visvis compares vis[x] vs vis[x-2]. vis = [1,1,2,2,3,3,4,4]
        # no matches; input at x=3 is a false alarm => 0 rights, 1 wrong
        self.assertEqual(result['visvis'], (0, 1))
        self.assertIsNone(result['arithmetic'])

        py = bwaccel._analyze_session_py(
            2, False, False, None,
            ['position1', 'audio', 'visvis', 'arithmetic'],
            session,
        )
        self.assertEqual(result['position1'], py['position1'])
        self.assertEqual(result['audio'], py['audio'])
        self.assertEqual(result['visvis'], py['visvis'])

    def test_jaeggi_scoring_counts_correct_rejections(self):
        session = self._session()
        r = bwaccel.analyze_session(
            2, False, True, None, ['position1'], session)
        # non-matches at x=3,4,6,7 with no input => 4 extra rights
        # plus the 1 hit => 5 rights, 1 wrong
        self.assertEqual(r['position1'], (5, 1))

    def test_crab_back(self):
        # crab back at x uses 1+2*(x % nback)
        data = [1, 2, 3, 1, 5, 6, 7, 8]
        # nback=2, x=3: back=1+2*(3%2)=3, data[3]==data[0] => match
        session = {'position1': data, 'position1_input': [0, 0, 0, 1, 0, 0, 0, 0]}
        r = bwaccel.analyze_session(2, True, False, None, ['position1'], session)
        self.assertEqual(r['position1'][0], 1)


class GraphAndStatsTests(unittest.TestCase):
    def test_aggregate_styles(self):
        entries = [[3, 80], [4, 50], [2, 100]]
        mean, mx = bwaccel.aggregate_day_scores('N', entries)
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(mx, 4.0)

        mean, mx = bwaccel.aggregate_day_scores('%', entries)
        self.assertAlmostEqual(mean, (0.80 + 0.50 + 1.00) / 3.0)
        self.assertAlmostEqual(mx, 1.0)

        mean, mx = bwaccel.aggregate_day_scores('N.%', entries)
        self.assertAlmostEqual(mean, (3.80 + 4.50 + 3.00) / 3.0)

        mean, mx = bwaccel.aggregate_day_scores('N+2*%-1', entries)
        expected = [3 - 1 + 1.6, 4 - 1 + 1.0, 2 - 1 + 2.0]
        self.assertAlmostEqual(mean, sum(expected) / 3.0)
        self.assertAlmostEqual(mx, max(expected))

        py = bwaccel._aggregate_day_scores_py('N+10/3+4/3', entries, 80, 50)
        c = bwaccel.aggregate_day_scores('N+10/3+4/3', entries, 80, 50)
        self.assertAlmostEqual(py[0], c[0])
        self.assertAlmostEqual(py[1], c[1])

    def test_parse_stats_text(self):
        recs = bwaccel.parse_stats_text(SAMPLE_STATS)
        self.assertEqual(len(recs), 3)
        self.assertEqual(recs[0]['year'], 2010)
        self.assertEqual(recs[0]['month'], 8)
        self.assertEqual(recs[0]['day'], 17)
        self.assertEqual(recs[0]['hour'], 2)
        self.assertEqual(recs[0]['mode'], 258)
        self.assertEqual(recs[0]['nback'], 3)
        self.assertEqual(recs[0]['percent'], 61)
        self.assertEqual(recs[0]['manual'], 0)
        self.assertEqual(recs[0]['cats'][0], 33)
        self.assertEqual(recs[0]['cats'][1], 75)
        self.assertEqual(recs[0]['sesstime'], 12)
        self.assertEqual(recs[1]['mode'], 2)
        self.assertEqual(recs[2]['manual'], 1)

        py = bwaccel._parse_stats_text_py(SAMPLE_STATS)
        self.assertEqual(len(py), 3)
        for a, b in zip(recs, py):
            for k in ('year', 'month', 'day', 'hour', 'percent', 'mode',
                      'nback', 'manual', 'session', 'sesstime'):
                self.assertEqual(a[k], b[k], k)
            self.assertEqual(list(a['cats']), list(b['cats']))


class GeometryAndMiscTests(unittest.TestCase):
    def test_rounded_rect_matches_python(self):
        c = list(bwaccel.rounded_rect_vertices(10, 90, 20, 80, 8))
        p = bwaccel._rounded_rect_vertices_py(10, 90, 20, 80, 8)
        self.assertEqual(len(c), 80)
        self.assertEqual(len(p), 80)
        self.assertEqual(c, p)

    def test_sample_unique(self):
        s = bwaccel.sample_unique(1, 8, 4)
        self.assertEqual(len(s), 4)
        self.assertEqual(len(set(s)), 4)
        self.assertTrue(all(1 <= v <= 8 for v in s))

    def test_variable_nback_range(self):
        seq = bwaccel.variable_nback_list(200, 5)
        self.assertEqual(len(seq), 200)
        self.assertTrue(all(1 <= v <= 5 for v in seq))

    def test_is_nback_match(self):
        hist = [1, 2, 3, 4]
        self.assertTrue(bwaccel.is_nback_match(3, hist, 2))
        self.assertFalse(bwaccel.is_nback_match(9, hist, 2))
        self.assertIsNone(bwaccel.is_nback_match(1, hist, 99))

    def test_mean_tail(self):
        self.assertAlmostEqual(bwaccel.mean_tail([10, 20, 30, 40], 2), 35.0)
        self.assertAlmostEqual(bwaccel.mean_tail([10, 20, 30], 0), 20.0)
        self.assertAlmostEqual(bwaccel.mean_tail([], 5), 0.0)


class ArithmeticTests(unittest.TestCase):
    def test_apply_ops(self):
        self.assertEqual(bwaccel.apply_arithmetic('add', 3, 4), Decimal(7))
        self.assertEqual(bwaccel.apply_arithmetic('subtract', 10, 4), Decimal(6))
        self.assertEqual(bwaccel.apply_arithmetic('multiply', 3, 5), Decimal(15))
        self.assertEqual(bwaccel.apply_arithmetic('divide', 9, 2), Decimal(9) / Decimal(2))
        self.assertEqual(
            bwaccel.apply_arithmetic('divide', Decimal(-4), Decimal(8)),
            Decimal('-0.5'))

    def test_unknown_op_raises(self):
        with self.assertRaises(ValueError):
            bwaccel.apply_arithmetic('__import__', 1, 2)
        with self.assertRaises(ValueError):
            bwaccel.apply_arithmetic('add; print(1)', 1, 2)

    def test_divide_by_zero(self):
        with self.assertRaises(Exception):
            bwaccel.apply_arithmetic('divide', 1, 0)

    def test_score_arithmetic(self):
        session = {
            'numbers': [10, 4, 2, 5, 3],
            'operation': ['none', 'none', 'add', 'subtract', 'multiply'],
            # trial 2: 10+2=12 (correct), trial 3: 4-5=-1 (user said 0),
            # trial 4: 2*3=6 (correct)
            'arithmetic_input': [Decimal(0), Decimal(0), Decimal(12),
                                 Decimal(0), Decimal(6)],
        }
        rights, wrongs = bwaccel.score_arithmetic(2, False, None, session)
        self.assertEqual((rights, wrongs), (2, 1))

    def test_score_bad_op_counts_wrong(self):
        session = {
            'numbers': [1, 1, 1],
            'operation': ['none', 'none', 'not-an-op'],
            'arithmetic_input': [0, 0, 0],
        }
        rights, wrongs = bwaccel.score_arithmetic(2, False, None, session)
        self.assertEqual((rights, wrongs), (0, 1))


class BackendTests(unittest.TestCase):
    def test_backend_string(self):
        self.assertIn(bwaccel.backend(), ('C', 'Python'))

    def test_banner(self):
        self.assertEqual(bwaccel.banner(), 'native: %s' % bwaccel.backend())


def _bench():
    """Rough timing of sequence generation (C vs Python fallback)."""
    trials, nback = 20 + 12 * 12, 12  # high n, where rejection-sampling died
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        bwaccel.compute_bt_sequence(trials, nback, 6, 6, 2)
    t_live = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n):
        bwaccel._compute_bt_sequence_py(trials, nback, 6, 6, 2)
    t_py = time.perf_counter() - t0
    print('backend:          %s' % bwaccel.backend())
    print('C/live  %d seqs:  %.3f ms' % (n, t_live * 1000.0))
    print('Python  %d seqs:  %.3f ms' % (n, t_py * 1000.0))
    if t_live > 0:
        print('speedup:          %.1fx' % (t_py / t_live))


if __name__ == '__main__':
    print('bwaccel backend: %s' % bwaccel.backend())
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    _bench()
    sys.exit(0 if result.wasSuccessful() else 1)
