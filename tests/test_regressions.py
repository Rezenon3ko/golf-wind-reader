from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

import obs_live_wind_reader as O
import wind_reader as W
from overlay import Overlay


def _recognizer_state() -> O.Recognizer:
    """构造只用于状态机单测的 Recognizer，不加载字形库、不连接 OBS。"""
    r = object.__new__(O.Recognizer)
    r._choice = None
    r.locked = None
    r.votes = {}
    r.vote_span = {}
    r.vote_band_ok = {}
    r.measure_t0 = None
    r.dial_t = None
    r.choice_options = None
    r.choice_hint = ""
    r.ask_since = None
    r.choice_all = False
    r.band_asked = False
    r.break_run = 0
    r.value_run = 0
    r.value_seen = None
    r.lock_event = ""
    r.lock_mismatch = ""
    r.last_theta = None
    r.flip_run = 0
    r.turn_run = 0
    r.ema = None
    r.shown = None
    r.last_dir = {}
    return r


class RecognizerRegressionTests(unittest.TestCase):
    def test_choice_popup_is_not_bypassed_by_later_votes(self):
        r = _recognizer_state()
        now = time.time()
        r.votes = {7: [2, 2.0], 8: [2, 2.0]}
        r.vote_span = {7: [now - 2, now], 8: [now - 2, now]}
        r.vote_band_ok = {7: 2, 8: 2}
        r.measure_t0 = now - O.WIND_MEASURE_S - 1
        band = (50.0, 68.0, 0.70, 0.68)

        def frame(speed):
            return SimpleNamespace(
                speed=float(speed), speed_range=None,
                ocr={"conf": 1.0}, arrow_band=band)

        r._update_lock(frame(7))
        self.assertIsNone(r.locked)
        self.assertIsNotNone(r.choice_options)

        r._update_lock(frame(7))
        self.assertIsNone(r.locked)
        self.assertIsNotNone(r.choice_options)

    def test_visible_dial_refreshes_dial_timer_without_arrow(self):
        r = _recognizer_state()
        r.args = SimpleNamespace(verbose=False)
        r.dial = W.Dial(W.KIND_GAUGE, 1690, 140, 115, 0.82)
        r.shape = (1080, 1920)
        r.last_locate_t = 0.0
        r.side = "右"
        r.bad = 0
        r.dial_t = time.time() - 10
        r.ocr = W.OcrState()
        r.votes = {7: [2, 2.0]}
        r.vote_span = {7: [time.time() - 1, time.time()]}
        r.vote_band_ok = {7: 2}
        r.measure_t0 = time.time() - 1
        before = r.dial_t

        with mock.patch.object(W, "gauge_verified", return_value=True), \
                mock.patch.object(W, "read_wind", return_value=W.FrameReading(
                    ok=False, speed=7.0, message="no arrow")):
            result = r.recognize(np.zeros((1080, 1920, 3), dtype=np.uint8))

        self.assertFalse(result["ok"])
        self.assertGreater(r.dial_t, before)
        self.assertTrue(r.votes)

    def test_reset_clears_angle_state(self):
        r = _recognizer_state()
        r.dial = "old"
        r.shape = (1080, 1920)
        r.bad = 3
        r.last_theta = 10
        r.flip_run = 2
        r.turn_run = 1
        r.ema = r.shown = 10
        r.last_dir = {"lateral_dir": "向右"}

        r.reset()

        self.assertIsNone(r.dial)
        self.assertIsNone(r.last_theta)
        self.assertEqual(r.flip_run, 0)
        self.assertEqual(r.turn_run, 0)
        self.assertEqual(r.last_dir, {})
        self.assertIsNone(r.ema)
        self.assertIsNone(r.shown)


class GeometryRegressionTests(unittest.TestCase):
    def test_empty_local_fit_returns_none(self):
        rgb = np.zeros((300, 300, 3), dtype=np.uint8)
        self.assertIsNone(W._fit_disc_local(rgb, 150, 150, 100))

    def test_disc_route_returns_gauge_center(self):
        rgb, cx, cy = self._synthetic_gauge()
        arrow = W.arrow_mask(W.rgb_to_hsv(rgb))
        dial = W._find_dial_by_disc(rgb, arrow)

        self.assertIsNotNone(dial)
        self.assertLessEqual(abs(dial.cx - cx), 6)
        self.assertLessEqual(abs(dial.cy - cy), 6)

    def test_fallback_runs_after_white_ring_candidate_fails_gate(self):
        rgb, cx, cy = self._synthetic_gauge()
        hsv = W.rgb_to_hsv(rgb)
        angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        ring_pixels = np.stack([
            cx + 115 * np.cos(angles),
            cy + 115 * np.sin(angles),
        ], axis=1)
        ring = W.Component(
            area=1000, x0=1600, y0=100, x1=1750, y1=250,
            cx=cx, cy=cy, pixels=ring_pixels)
        fallback_dial = W.Dial(
            W.KIND_GAUGE, cx, cy, 115, 0.82, anchored_by="fallback-test")
        verify_calls = {"n": 0}

        def verify(*_args, **_kwargs):
            verify_calls["n"] += 1
            return verify_calls["n"] >= 2

        with mock.patch.object(W, "_find_dial_by_prior", return_value=None), \
                mock.patch.object(W, "_mask_fragmented", return_value=False), \
                mock.patch.object(W, "label_components", return_value=[ring]), \
                mock.patch.object(W, "fit_circle_ls", return_value=(cx, cy, 115)), \
                mock.patch.object(W, "_ring_coverage", return_value=1.0), \
                mock.patch.object(W, "_grass_mask", return_value=np.ones(
                    (1080, 1920), dtype=bool)), \
                mock.patch.object(W, "_fit_disc_local", return_value=None), \
                mock.patch.object(W, "gauge_verified", side_effect=verify), \
                mock.patch.object(W, "_find_dial_by_arrow_color",
                                  return_value=fallback_dial), \
                mock.patch.object(W, "_find_dial_by_disc", return_value=None), \
                mock.patch.object(W, "_find_dial_by_arrow", return_value=None), \
                mock.patch.object(W, "_calibrate_disc"):
            good = W.find_dials(rgb, hsv)

        self.assertEqual(len(good), 1)
        self.assertEqual(good[0].anchored_by, "fallback-test")
        self.assertEqual(verify_calls["n"], 2)

    @staticmethod
    def _synthetic_gauge():
        rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
        cx, cy = 1690, 140
        ratio = W.CONFIG.disc_panel_ratio
        disc_cy = cy - 0.30 * 115 * ratio
        a = ratio * 115
        b = a * 0.82
        draw.ellipse((cx - a, disc_cy - b, cx + a, disc_cy + b), fill=(60, 200, 60))
        draw.polygon([
            (cx - 50, disc_cy - 8), (cx + 35, disc_cy - 8),
            (cx + 35, disc_cy - 22), (cx + 62, disc_cy),
            (cx + 35, disc_cy + 22), (cx + 35, disc_cy + 8),
            (cx - 50, disc_cy + 8),
        ], fill=(220, 50, 160))
        return np.asarray(im).copy(), cx, cy


class InputRegressionTests(unittest.TestCase):
    def test_cli_accepts_obs_source(self):
        args = O.build_parser().parse_args(["--obs-source", "视频采集设备", "-t", "-v"])
        self.assertEqual(args.obs_source, "视频采集设备")
        self.assertTrue(args.test)
        self.assertTrue(args.verbose)

    def test_coefficient_parsing_and_validation(self):
        class Var:
            def __init__(self, s):
                self.s = s

            def get(self):
                return self.s

        class FakeOverlay:
            COEF_DIGITS = 4

            def __init__(self, s):
                self.coef = Var(s)

        self.assertAlmostEqual(Overlay._coef_value(FakeOverlay("115")), 1.15)
        self.assertAlmostEqual(Overlay._coef_value(FakeOverlay("085")), 0.85)
        self.assertAlmostEqual(Overlay._coef_value(FakeOverlay("85")), 8.5)
        self.assertTrue(Overlay._coef_ok(FakeOverlay("85"), "85"))
        self.assertFalse(Overlay._coef_ok(FakeOverlay("¹"), "¹"))


if __name__ == "__main__":
    unittest.main()
