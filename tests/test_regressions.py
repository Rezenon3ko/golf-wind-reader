from __future__ import annotations

import time
import unittest
from pathlib import Path
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

    def test_single_low_confidence_candidate_requests_choice(self):
        r = _recognizer_state()
        now = time.time()
        r.votes = {7: [2, 0.90]}
        r.vote_span = {7: [now - 2, now]}
        r.vote_band_ok = {7: 2}
        r.measure_t0 = now - O.WIND_MEASURE_S - 1
        frame = SimpleNamespace(
            speed=7.0, speed_range=None,
            ocr={"conf": 0.45}, arrow_band=(50.0, 68.0, 0.70, 0.68))

        r._update_lock(frame)

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


class OcrTopologyRegressionTests(unittest.TestCase):
    @staticmethod
    def _template(ch: str):
        path = Path(__file__).resolve().parents[1] / "glyph_bank.npz"
        with np.load(path, allow_pickle=False) as bank:
            return bank[ch][0].copy()

    @staticmethod
    def _render(glyph: np.ndarray) -> Image.Image:
        canvas = np.zeros((80, 100), dtype=np.uint8)
        canvas[16:64, 34:66] = (glyph * 255).astype(np.uint8)
        return Image.fromarray(canvas).convert("RGB")

    def test_significant_holes_distinguish_5_6_8(self):
        self.assertEqual(
            [W._hole_count(self._template(ch) > 0.5) for ch in "568"],
            [0, 1, 2])

    def test_tiny_noise_hole_does_not_break_5(self):
        glyph = self._template("5")
        for y in range(22, 24):
            for x in range(12, 14):
                glyph[y, x] = 0

        self.assertEqual(W._hole_count(glyph > 0.5), 0)
        result = W.match_glyphs(self._render(glyph), state=self._state())
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "5")

    def test_multiple_templates_are_aggregated_per_digit(self):
        bank = {ch: [self._template(ch)] for ch in "0123456789"}
        bank["5"].append(np.roll(bank["5"][0], 1, axis=1))
        state = W.OcrState()
        state.bank = bank

        result = W.match_glyphs(
            self._render(self._template("5")), state=state)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "5")

    def test_formal_bank_has_no_duplicate_templates(self):
        path = Path(__file__).resolve().parents[1] / "glyph_bank.npz"
        with np.load(path, allow_pickle=False) as bank:
            for ch in bank.files:
                templates = bank[ch]
                for i in range(len(templates)):
                    for j in range(i + 1, len(templates)):
                        self.assertFalse(
                            np.array_equal(templates[i], templates[j]),
                            f"{ch} templates {i} and {j} are identical")

    def test_edge_trim_removes_single_pixel_protrusion(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        mask[10, 15] = True                 # 右侧 1 像素凸起

        trimmed = W._trim_glyph_edges(mask)
        self.assertEqual(trimmed.shape[1], 10)

    def test_soft_similarity_tolerates_one_pixel_shift(self):
        original = self._template("2")
        shifted = np.zeros_like(original)
        shifted[:, 1:] = original[:, :-1]

        exact = W._glyph_sim(original, shifted)
        soft = W._glyph_sim_soft(original, shifted)
        self.assertGreater(soft, exact)
        self.assertGreater(soft, 0.90)

    def test_edge_protrusion_does_not_break_2(self):
        canvas = np.asarray(self._render(self._template("2"))).copy()
        canvas[40:42, 66:68] = 255          # 右侧 2×2 亮块
        result = W.match_glyphs(
            Image.fromarray(canvas).convert("RGB"), state=self._state())
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "2")

    def test_relaxed_color_band_does_not_hide_zero_wind(self):
        dial = W.Dial(W.KIND_GAUGE, 1690, 140, 115, 0.82)

        def fake_detect(*_args, **kwargs):
            return None if kwargs.get("relax", 0) == 0 else (
                178.0, 194.0, 0.70, 0.72)

        def fake_ocr(*_args, **kwargs):
            allow = kwargs.get("allow")
            return dict(value=0 if allow is None else 1, raw="",
                        conf=0.99, box=None, ok=True, reject=None)

        with mock.patch.object(W, "_find_text_lines", return_value=[(0, 0, 10, 10)]), \
                mock.patch.object(W, "detect_arrow_band", side_effect=fake_detect), \
                mock.patch.object(W, "ocr_speed", side_effect=fake_ocr):
            result = W.read_wind(
                np.zeros((1080, 1920, 3), dtype=np.uint8), dial)

        self.assertTrue(result.ok)
        self.assertTrue(result.no_wind)
        self.assertEqual(result.speed, 0.0)

    def test_missing_upper_hole_never_locks_8_as_6(self):
        glyph = self._template("8")
        mask = glyph > 0.5
        holes = W.label_components(~mask, keep_pixels=True, min_area=1)
        h, w = mask.shape
        filled = mask.copy()
        for hole in holes:
            if (hole.x0 > 0 and hole.y0 > 0 and hole.x1 < w - 1
                    and hole.y1 < h - 1 and hole.cy < h * 0.5):
                pixels = np.asarray(hole.pixels, dtype=int)
                filled[pixels[:, 1], pixels[:, 0]] = True

        result = W.match_glyphs(
            self._render(filled.astype(np.float32)), state=self._state())
        self.assertTrue(result is None or result[0] == "8")

    @staticmethod
    def _state():
        state = W.OcrState()
        W.load_glyph_bank(
            str(Path(__file__).resolve().parents[1] / "glyph_bank.npz"),
            state=state)
        return state


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

    def test_obs_non_string_image_data_is_handled(self):
        class FakeCli:
            def get_source_screenshot(self, *_args):
                return SimpleNamespace(image_data=b"not-a-string")

        src = object.__new__(O.ObsWebSocketSource)
        src.cli = FakeCli()
        src.source = "test"
        src.fmt = "jpg"
        src.width, src.height, src.quality = 1920, 1080, 90
        src.fails, src.dead, src.last_error = 0, False, ""

        def fail(why):
            src.last_error = why
            src.fails += 1

        src._fail = fail
        self.assertIsNone(src.read())
        self.assertEqual(src.fails, 1)
        self.assertIn("解码", src.last_error)


if __name__ == "__main__":
    unittest.main()
