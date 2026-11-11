"""Fast checks of the logic that decides what silent chat does with what it hears and sees. No camera, mic, model
downloads or Claude calls. Run:  .venv/bin/python -m unittest discover tests
With SLOW=1 it also runs the small local model (needs it downloaded; ~10 s)."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gaze_click  # noqa: E402
import local_commands  # noqa: E402
import presence  # noqa: E402
import privacy  # noqa: E402
import shortcuts  # noqa: E402
import speech  # noqa: E402
import improve  # noqa: E402
import local_router  # noqa: E402
import numpy as np  # noqa: E402
import gaze  # noqa: E402
import gaze_passive  # noqa: E402


class InstantCommands(unittest.TestCase):
    def cmd(self, text, front=""):
        c = local_commands.parse(text, front)
        return c.args if c else None

    def test_music(self):
        self.assertEqual(self.cmd("pause"), ["spotify", "pause"])
        self.assertEqual(self.cmd("please pause"), ["spotify", "pause"])  # (was struck through once)
        self.assertEqual(self.cmd("next song"), ["spotify", "next"])
        self.assertEqual(self.cmd("hey claude, play the music please"), ["spotify", "play"])

    def test_web_search_only_when_asked(self):
        self.assertEqual(self.cmd("search for flights to tokyo")[0], "url")
        self.assertIsNone(self.cmd("search my emails for invoices"))
        self.assertIsNone(self.cmd("search in discord for alex"))

    def test_not_a_command(self):
        self.assertIsNone(self.cmd("play rap on spotify"))  # (needs a search: Claude)
        self.assertIsNone(self.cmd("so I'm going to talk a little bit about this"))

    def test_lip_reading_needs_agreement(self):
        part = {"hyps": [("PLEASE PAUSE", -1), ("PLEASE PAUSE", -2)]}
        self.assertEqual(local_commands.parse_part(part).args, ["spotify", "pause"])


class DictateOrRequest(unittest.TestCase):
    """Holding Option types what you say, unless it's a request for Claude."""

    def test_requests(self):
        for t in ["This does work very well. Message Marianne on Discord that this works very well.",
                  "open Spotify", "hey Claude, what is the weather", "can you play some rap",
                  "text Alex that I am running late"]:
            self.assertTrue(local_commands.is_request(t), t)

    def test_dictation(self):
        for t in ["I will send you the file tomorrow", "sounds good, see you at 4", "Tell me when you get there",
                  "what time works for you?", "Let me know if you can message me later",
                  "I told Sam about it on Discord"]:
            self.assertFalse(local_commands.is_request(t), t)


class GazeClick(unittest.TestCase):
    """Tap Right Command to click what you're looking at."""

    def test_tap(self):
        d = gaze_click.TapDetector()
        d.flags(54, True, 1.0)
        d.flags(54, False, 1.2)
        self.assertEqual(d.take(), [1.2])

    def test_not_a_tap(self):
        d = gaze_click.TapDetector()
        d.flags(54, True, 1.0)
        d.other()  # Right Cmd + C
        d.flags(54, False, 1.1)
        d.flags(54, True, 2.0)
        d.flags(54, False, 2.6)  # held too long
        d.flags(54, True, 3.0)
        d.flags(56, True, 3.05)  # Shift joined in
        d.flags(54, False, 3.1)
        d.flags(55, True, 4.0)  # left Command: not ours
        d.flags(55, False, 4.1)
        self.assertEqual(d.take(), [])

    def test_snaps_to_nearest_clickable(self):
        cands = [{"role": "AXButton", "label": "Far", "rect": (300, 100, 60, 24)},
                 {"role": "AXLink", "label": "Near", "rect": (130, 100, 60, 20)},
                 {"role": "AXStaticText", "label": "text under the gaze", "rect": (90, 95, 30, 30)},
                 {"role": "AXGroup", "label": "", "rect": (100, 100, 10, 10)}]
        self.assertEqual(gaze_click.choose(100, 110, 80, cands)["label"], "Near")
        self.assertIsNone(gaze_click.choose(100, 110, 80, cands[:1]))  # (out of reach)

    def test_risky(self):
        self.assertTrue(gaze_click.risky({"label": "Send"}))
        self.assertTrue(gaze_click.risky({"label": "Delete conversation"}))
        self.assertFalse(gaze_click.risky({"label": "Sender details"}))
        self.assertFalse(gaze_click.risky({"label": "Open"}))


class HandsFreeFilter(unittest.TestCase):
    apps = ["Spotify", "Discord", "Google Chrome"]

    def test_requests_pass(self):
        for t in ["PLAY RAP ON SPOTIFY", "PLEASE WRITE UP ON SPOTIFY", "PLEASE PAUSE", "SEARCH FOR NORTH KOREAN NEWS",
                  "CLICK ON THE FIRST RESULT"]:
            self.assertTrue(presence.plausible_request([(t, 0)], self.apps), t)

    def test_chatter_is_dropped(self):
        for t in ["OKAY", "SO I'M GOING TO", "A LITTLE BIT", "NOW", "I TOOK A LOT OF", ""]:
            self.assertFalse(presence.plausible_request([(t, 0)], self.apps), t)


class LocalRouterGuards(unittest.TestCase):
    """The checks applied to the small model's answer (no model needed)."""

    def test_play_with_a_genre_goes_to_claude(self):
        d = local_router.parse('{"kind": "act", "text": "play rap on Spotify", "cmd": "spotify play"}', [])
        self.assertEqual(d["kind"], "claude")

    def test_web_search_needs_search_words(self):
        d = local_router.parse('{"kind": "act", "text": "click on the first result", "cmd": "web first result"}', [])
        self.assertEqual(d["kind"], "claude")
        d = local_router.parse('{"kind": "act", "text": "search cats", "cmd": "web cats"}', [])
        self.assertEqual(d["cmd"][0], "url")

    def test_no_new_tab_or_closing(self):
        for c in ("key cmd+t", "key cmd+w", "close Safari"):
            d = local_router.parse('{"kind": "act", "text": "x", "cmd": "%s"}' % c, [])
            self.assertEqual(d["kind"], "claude", c)

    def test_garbage_goes_to_claude(self):
        self.assertEqual(local_router.parse("not json", [("HELLO", 0)])["kind"], "claude")


class WakeWord(unittest.TestCase):
    def test_hallucinations_are_dropped(self):
        for t in ["Closed Captioning by www.cdc.gov", "Thank you.", "So, let's go.", "Vocabulary, voiceover, Claude",
                  "Um...", "Yeah."]:
            self.assertTrue(speech.HALLUCINATION.match(t), t)

    def test_real_speech_kept(self):
        for t in ["Hey Claude, play rap", "Thank you for that, play music"]:
            self.assertFalse(speech.HALLUCINATION.match(t), t)

    def test_wake_phrase(self):
        self.assertTrue(speech.WAKE_PHRASE.search("Hey Claude, what's the weather"))
        self.assertTrue(speech.WAKE_PHRASE.search("hey cloud play music"))
        self.assertFalse(speech.WAKE_PHRASE.search("I told Claude about it"))


class Privacy(unittest.TestCase):
    def test_secrets_are_blanked(self):
        out = privacy.redact("card 4111 1111 1111 1111, ssn 123-45-6789, key sk-ant-abcdefghijklmnopqrstuv, "
                             "your code is 482913, password: hunter2")
        for secret in ("4111", "123-45-6789", "sk-ant-abc", "482913", "hunter2"):
            self.assertNotIn(secret, out)

    def test_ordinary_text_untouched(self):
        t = "Front: Discord - '#fishing'\nButtons: Send, Gift"
        self.assertEqual(privacy.redact(t), t)

    def test_private_by_rule(self):
        self.assertEqual(privacy.rule_verdict("1Password", "Vault"), "credentials")
        self.assertEqual(privacy.rule_verdict("Google Chrome", "Chase - Account Summary"), "financial")
        self.assertEqual(privacy.rule_verdict("Safari", "MyChart - Lab Results"), "medical")
        self.assertIsNone(privacy.rule_verdict("Spotify", "Spotify Premium"))

    def test_filter_without_model_still_redacts(self):
        g = privacy.Guard(None)
        g._save = lambda: None
        self.assertNotIn("4111", g.filter("Front: Notes - 'x'\nSelected: '4111 1111 1111 1111'"))
        self.assertIn("private: financial",
                      g.filter("Front: Google Chrome - 'Wells Fargo'\nHeadings: Balance"))


class SelfImprovement(unittest.TestCase):
    def test_sending_steps_must_ask(self):
        steps = improve._safe_steps(["focus Messages", "type hi", "key return"], [])
        self.assertEqual(steps[-1], "ask:key return")
        self.assertEqual(improve._safe_steps(['press "Send"'], []), ['ask:press "Send"'])

    def test_return_in_a_switcher_is_fine(self):
        steps = improve._safe_steps(["focus Discord", "key cmd+k", "type {name}", "wait 1", "key return"], ["name"])
        self.assertEqual(steps[-1], "key return")

    def test_only_known_steps(self):
        self.assertIsNone(improve._safe_steps(["rm -rf /"], []))
        self.assertIsNone(improve._safe_steps(["type {who}"], []))  # (a blank the phrase doesn't have)


class Shortcuts(unittest.TestCase):
    def test_match_fills_blanks(self):
        orig = shortcuts.load
        shortcuts.load = lambda: [{"phrase": "open my dms with {name}", "steps": [], "uses": 0, "values": {}}]
        try:
            s, values = shortcuts.match("Open my DMs with Alex")
            self.assertEqual(values, {"name": "Alex"})
        finally:
            shortcuts.load = orig


GEOM = {"screen": [0.0, 0.0, 1512.0, 982.0], "display": 1, "mm_per_pt": 0.2, "cam": [756.0, 16.0], "builtin": True}


def eyes_for(w, target, pose=(0.0, -11.0, -45.0, 0.0, -3.0), roll=0.0):
    """The gaze features a person with weights w would show looking at screen point `target` with their head at
    pose (tx, ty, tz, yaw, pitch): the model, solved for the two iris offsets (lids follow the gaze down)."""
    cal = gaze.SelfCalibration(GEOM)
    x, y = cal._cm(target)
    tx, ty, tz, yaw, pitch = pose
    D, ps, ph = -tz, np.radians(yaw), np.radians(pitch)
    op = -0.004 * y  # (cm down the screen -> lids a little more closed)
    ax = (x - w[0] - w[1] * tx - D * (w[2] * ps + w[3] * ph)) / D - w[4] - w[8] * op
    by = (y - w[11] - w[12] * ty - D * (w[13] * ph + w[14] * ps)) / D - w[15] - w[17] * op
    eh, ev = np.linalg.solve([[w[5], w[7]], [w[18], w[16]]], [ax, by])
    c = gaze.CENTRE
    return np.array([eh + c[0], ev + c[1], op + c[2], c[0] - eh, -(ev + c[1]), op + c[2], yaw, pitch, tx, ty, tz, roll])


def person(offsets=(1.5, 0.03, -2.0, 0.05), gain=1.0):
    """Someone else's weights: their own "straight ahead", and eyes that move more or less per unit of iris."""
    w = gaze.PRIOR_MEAN.copy()
    w[gaze.OFFSETS] += offsets
    w[[5, 16, 17]] *= gain
    return w


def clean_error(cal, w, rng, n=60):
    """Median error on looks at random points, in random postures, that nothing has learned from."""
    errs = []
    for _ in range(n):
        t = (rng.uniform(60, 1450), rng.uniform(40, 940))
        pose = (rng.uniform(-4, 8), rng.uniform(-15, -9), rng.uniform(-52, -40), rng.uniform(-6, 8), rng.uniform(-9, 3))
        errs.append(np.linalg.norm(cal.predict(eyes_for(w, t, pose))[0] - t))
    return float(np.median(errs))


def work(cal, w, rng, n, junk=0.4, src="click", bias=(0.0, 0.0), t0=0.0):
    """n hints from someone working: in `junk` of them the eyes are somewhere else entirely."""
    for i in range(n):
        t = (rng.uniform(60, 1450), rng.uniform(40, 940))
        looked = (rng.uniform(60, 1450), rng.uniform(40, 940)) if rng.random() < junk else \
            (t[0] + rng.normal(0, 30), t[1] + rng.normal(0, 30))
        pose = (rng.uniform(-4, 8), rng.uniform(-15, -9), rng.uniform(-52, -40), rng.uniform(-6, 8), rng.uniform(-9, 3))
        cal.learn(eyes_for(w, looked, pose), (t[0] + bias[0], t[1] + bias[1]), src, t=t0 + 20.0 * i)


class GazeSelfCalibration(unittest.TestCase):
    """gaze.py: no calibration step; it learns "straight ahead" from noisy hints and guards against bad ones."""

    def test_the_model_inverts(self):
        w, t = person(), (300.0, 700.0)
        cal = gaze.SelfCalibration(GEOM)
        cal.fast.w = w
        self.assertLess(np.linalg.norm(cal.predict(eyes_for(w, t))[0] - t), 1e-6)

    def test_new_profile_finds_straight_ahead_from_noisy_clicks(self):
        rng, w = np.random.default_rng(1), person()
        cal = gaze.SelfCalibration(GEOM)
        self.assertFalse(cal.warm)
        before = clean_error(cal, w, rng)
        work(cal, w, rng, 10)
        early = clean_error(cal, w, rng)
        work(cal, w, rng, 50, t0=200.0)
        after = clean_error(cal, w, rng)
        self.assertGreater(before, 120)  # (their offsets are ~2 cm and ~3 degrees from the starting ones)
        self.assertLess(early, before * 0.6)  # the first ten clicks already help
        self.assertLess(after, 60)  # ... although 4 in 10 were made looking elsewhere
        self.assertTrue(cal.warm)

    def test_clicks_never_touch_the_gains(self):
        rng, w = np.random.default_rng(2), person(gain=0.8)
        cal = gaze.SelfCalibration(GEOM)
        work(cal, w, rng, 80, junk=0.5)
        gains = [i for i in range(gaze.N) if i not in gaze.OFFSETS]
        np.testing.assert_array_equal(cal.fast.w[gains], gaze.PRIOR_MEAN[gains])

    def test_recalibrate_measures_the_gains(self):
        rng, w = np.random.default_rng(3), person(gain=1.3)
        cal = gaze.SelfCalibration(GEOM)
        work(cal, w, rng, 60, junk=0.3)
        without = clean_error(cal, w, rng)  # offsets learned, but these eyes move 30 % more than assumed
        dots = [(60 + 1390 * i / 4, 80 + 820 * j / 3) for i in range(5) for j in range(4)]
        fresh = [(250 + 500 * i + rng.uniform(-80, 80), 200 + 290 * j + rng.uniform(-50, 50)) for i in range(3) for j in range(3)]
        look = lambda t: (np.array([eyes_for(w, t) + rng.normal(0, 1e-3, 12) for _ in range(20)]), t[0], t[1])
        r = cal.calibrate([look(t) for t in dots], [look(t) for t in fresh])
        self.assertLess(r["look_error_pt"], 40)  # measured on the 9 dots it hadn't fitted
        self.assertLess(clean_error(cal, w, rng), without * 0.6)
        self.assertEqual(len(r["looks"]), 9)

    def test_a_misleading_kind_of_hint_is_caught(self):
        rng, w = np.random.default_rng(4), person()
        cal = gaze.SelfCalibration(GEOM)
        work(cal, w, rng, 40, junk=0.3)
        good = clean_error(cal, w, rng)
        for k in range(8):  # cursor landings that are consistently 160 pt off, between honest clicks
            work(cal, w, rng, 12, junk=0.0, src="land", bias=(160.0, 110.0), t0=1000.0 + 600 * k)
            work(cal, w, rng, 10, junk=0.3, t0=1300.0 + 600 * k)
        self.assertLess(clean_error(cal, w, rng), max(70.0, good * 1.5))  # the live model didn't follow them
        self.assertGreater(cal.trust["land"], 1.2)  # ... and that kind of hint now counts for less
        self.assertTrue(any(n.get("fast_reset") or n.get("live") == "safe" for n in cal.notes))

    def test_far_off_hints_are_ignored_until_it_is_clearly_lost(self):
        rng, w = np.random.default_rng(5), person()
        cal = gaze.SelfCalibration(GEOM)
        work(cal, w, rng, 60, junk=0.2)
        t = (400.0, 300.0)
        err, used = cal.learn(eyes_for(w, (1300.0, 800.0)), t, "click", t=5000.0)  # eyes 1000 pt from the click
        self.assertFalse(used)
        moved = person(offsets=(1.5 - 6.0, 0.03, -2.0 + 5.0, 0.05))  # a new chair: everything shifted 5-6 cm
        work(cal, moved, rng, 40, junk=0.2, t0=6000.0)
        self.assertTrue(any(n.get("reopened") for n in cal.notes))
        self.assertLess(clean_error(cal, moved, rng), 80)

    def test_saved_and_loaded(self):
        rng, w = np.random.default_rng(6), person()
        cal = gaze.SelfCalibration(GEOM)
        work(cal, w, rng, 35)
        again = gaze.SelfCalibration.from_dict(__import__("json").loads(__import__("json").dumps(cal.to_dict())))
        x = eyes_for(w, (900.0, 500.0))
        np.testing.assert_allclose(again.predict(x), cal.predict(x), atol=0.5)
        self.assertEqual(again.n["click"], cal.n["click"])
        self.assertAlmostEqual(again.radius, cal.radius, delta=1.0)

    def test_older_rows_without_roll_still_work(self):
        x = eyes_for(gaze.PRIOR_MEAN, (700.0, 400.0))
        cal = gaze.SelfCalibration(GEOM)
        np.testing.assert_allclose(cal.predict(x[:11]), cal.predict(x))

    def test_a_tilted_head_turns_what_the_eyes_say(self):
        w = gaze.PRIOR_MEAN.copy()
        w[[0, 1, 2, 3, 11, 12, 13, 14]] = 0.0  # (only the eyes: the head terms are already in the camera's frame)
        x = eyes_for(w, (300.0, 700.0), pose=(0.0, 0.0, -45.0, 0.0, 0.0))
        flat = (gaze._rows(x) @ w)[0]
        x[11] = 20.0
        tilted = (gaze._rows(x) @ w)[0]
        a = np.radians(20.0)
        np.testing.assert_allclose(tilted, [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]] @ flat, atol=1e-9)

    def test_turning_the_picture_turns_the_answer(self):
        w = gaze.PRIOR_MEAN.copy()
        w[[0, 1, 11, 12]] = 0.0  # (the camera's own position and the head's: not part of this)
        x = eyes_for(w, (300.0, 700.0), pose=(0.0, 0.0, -45.0, 6.0, -12.0))
        flat = (gaze._rows(x) @ w)[0]
        a = np.radians(15.0)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        x[11] = 15.0
        x[6], x[7] = R @ x[6:8]  # the head's direction turns with the picture, like everything in it
        np.testing.assert_allclose((gaze._rows(x) @ w)[0], R @ flat, atol=1e-9)

    def test_head_fusion_follows_the_camera_without_airpods(self):
        h = gaze.HeadFusion()
        for i in range(60):
            out = h.update(i / 30, (10.0, -5.0))
        self.assertAlmostEqual(out[0], 10.0, delta=0.5)
        self.assertAlmostEqual(out[1], -5.0, delta=0.5)


def feats(k):
    return np.full(len(gaze.NAMES), float(k))


def play(teacher, t0, seconds, gaze_at, cursor, tag=1.0, fps=30, noise=0.0, rng=None):
    """Feed frames: gaze_at / cursor are points or functions of time. -> the time it ended at."""
    n = int(seconds * fps)
    for i in range(n):
        t = t0 + i / fps
        g = gaze_at(t) if callable(gaze_at) else gaze_at
        c = cursor(t) if callable(cursor) else cursor
        if g is not None and noise:
            g = (g[0] + rng.normal(0, noise), g[1] + rng.normal(0, noise))
        teacher.frame(t, None if g is None else feats(tag), g, c)
    return t0 + n / fps


class GazeHints(unittest.TestCase):
    """gaze_passive.py: which moments count as knowing where you looked."""

    def test_click_takes_the_look_that_was_on_it(self):
        t = gaze_passive.Teacher()
        end = play(t, 0.0, 0.5, (200.0, 200.0), (900.0, 500.0), tag=1)   # reading something else
        end = play(t, end, 0.4, (880.0, 510.0), (900.0, 500.0), tag=2)   # then the eyes land on the button
        t.click(end, 900.0, 500.0)
        play(t, end, 0.3, (880.0, 510.0), (900.0, 500.0), tag=2)
        (h,) = t.hints()
        self.assertEqual((h["src"], h["y"]), ("click", (900.0, 500.0)))
        self.assertEqual(h["x"][0], 2.0)
        self.assertGreater(len(h["trace"]), 10)

    def test_no_look_no_hint(self):
        t = gaze_passive.Teacher()
        end = play(t, 0.0, 0.6, None, (900.0, 500.0))  # eyes closed / no face
        t.click(end, 900.0, 500.0)
        play(t, end, 0.3, None, (900.0, 500.0))
        self.assertEqual(t.hints(), [])

    def test_double_click_counts_once(self):
        t = gaze_passive.Teacher()
        end = play(t, 0.0, 0.6, (300.0, 300.0), (300.0, 300.0))
        t.click(end, 300.0, 300.0)
        end = play(t, end, 0.25, (300.0, 300.0), (300.0, 300.0))
        t.click(end, 302.0, 300.0)
        play(t, end, 0.3, (300.0, 300.0), (300.0, 300.0))
        self.assertEqual(len(t.hints()), 1)

    def cursor_trip(self, a, b, t0, t1):
        return lambda t: a if t < t0 else b if t > t1 else (a[0] + (b[0] - a[0]) * (t - t0) / (t1 - t0),
                                                               a[1] + (b[1] - a[1]) * (t - t0) / (t1 - t0))

    def test_cursor_coming_to_rest_with_the_eyes_on_it(self):
        t = gaze_passive.Teacher()
        t.radius, t.cold = 100.0, False
        trip = self.cursor_trip((200.0, 200.0), (800.0, 600.0), 0.5, 0.9)
        play(t, 0.0, 2.6, lambda tt: (200.0, 200.0) if tt < 0.4 else (810.0, 590.0), trip)
        (h,) = t.hints()
        self.assertEqual(h["src"], "land")
        self.assertLess(abs(h["y"][0] - 800.0) + abs(h["y"][1] - 600.0), 5)

    def test_cursor_parked_while_looking_elsewhere_says_nothing(self):
        t = gaze_passive.Teacher()
        t.radius, t.cold = 100.0, False
        play(t, 0.0, 2.6, (300.0, 250.0), self.cursor_trip((200.0, 200.0), (1200.0, 800.0), 0.5, 0.9))
        self.assertEqual(t.hints(), [])

    def test_a_click_where_it_landed_is_one_hint_not_two(self):
        t = gaze_passive.Teacher()
        t.radius, t.cold = 100.0, False
        trip = self.cursor_trip((200.0, 200.0), (800.0, 600.0), 0.5, 0.9)
        end = play(t, 0.0, 1.3, lambda tt: (200.0, 200.0) if tt < 0.4 else (810.0, 590.0), trip)
        t.click(end, 800.0, 600.0)
        play(t, end, 1.5, (810.0, 590.0), (800.0, 600.0))
        self.assertEqual([h["src"] for h in t.hints()], ["click"])

    def test_pop_up_counts_when_the_eyes_jump_to_it(self):
        banner = (1140.0, 36.0, 356.0, 84.0)
        t = gaze_passive.Teacher()
        t.radius, t.cold = 100.0, False
        play(t, 0.0, 0.8, (500.0, 600.0), (500.0, 600.0), tag=1)
        t.appeared(0.8, banner)
        end = play(t, 0.8, 0.35, (500.0, 600.0), (500.0, 600.0), tag=1)
        play(t, end, 1.5, (1290.0, 90.0), (500.0, 600.0), tag=2)
        (h,) = t.hints()
        self.assertEqual((h["src"], h["x"][0]), ("appear", 2.0))
        self.assertGreaterEqual(h["sig"][0], 100.0)
        t = gaze_passive.Teacher()  # ... and not when they stay where they were
        t.radius, t.cold = 100.0, False
        play(t, 0.0, 0.8, (500.0, 600.0), (500.0, 600.0))
        t.appeared(0.8, banner)
        play(t, 0.8, 2.0, (500.0, 600.0), (500.0, 600.0))
        self.assertEqual(t.hints(), [])

    def test_eyes_following_a_drag(self):
        rng = np.random.default_rng(0)
        path = lambda tt: (300.0 + 350.0 * tt, 400.0 + 40.0 * tt)
        t = gaze_passive.Teacher()
        t.radius, t.cold = 100.0, False
        play(t, 0.0, 1.6, lambda tt: (path(tt)[0] - 30.0, path(tt)[1] + 20.0), path, noise=12.0, rng=rng)
        got = [h for h in t.hints() if h["src"] == "pursuit"]
        self.assertTrue(got)
        self.assertLess(abs(got[0]["y"][1] - path(0.0)[1]), 80)
        t = gaze_passive.Teacher()  # eyes jump ahead to the target and wait there: that isn't following
        t.radius, t.cold = 100.0, False
        play(t, 0.0, 1.6, lambda tt: (300.0, 400.0) if tt < 0.3 else (860.0, 464.0), path, noise=12.0, rng=rng)
        self.assertEqual([h for h in t.hints() if h["src"] == "pursuit"], [])

    def test_caret_only_while_typing_and_only_near_the_gaze(self):
        asked = []

        def caret():
            asked.append(1)
            return (640.0, 400.0, 2.0, 18.0)
        t = gaze_passive.Teacher(caret=caret)
        t.radius, t.cold = 100.0, False
        end = play(t, 0.0, 0.5, (650.0, 420.0), (100.0, 100.0))
        t.key(end)  # one key isn't typing
        self.assertEqual(asked, [])
        for i in range(4):
            end = play(t, end, 0.2, (650.0, 420.0), (100.0, 100.0))
            t.key(end)
        for _ in range(50):  # (the answer comes from another thread)
            if not t.results.empty():
                break
            __import__("time").sleep(0.01)
        play(t, end, 0.3, (650.0, 420.0), (100.0, 100.0))
        hints = t.hints()
        self.assertEqual([h["src"] for h in hints], ["caret"])
        self.assertEqual(hints[0]["y"], (641.0, 409.0))
        self.assertEqual(len(asked), 1)  # (asked once, not per key)

    def test_new_profile_starts_around_the_middle(self):
        t = gaze_passive.Teacher()
        t.screen = GEOM["screen"]
        play(t, 0.0, 6.0, (700.0, 400.0), lambda tt: (300.0 + 100.0 * tt, 300.0))
        hints = t.hints()
        self.assertTrue(hints and all(h["src"] == "centre" for h in hints if h["src"] != "pursuit"))
        cal = gaze.SelfCalibration(GEOM)
        w = person(offsets=(0.0, 0.0, 0.0, 0.2))  # someone whose gaze reads ~450 pt too high with the starting weights
        mid = (756.0, 442.0)
        before = np.linalg.norm(cal.predict(eyes_for(w, mid))[0] - mid)
        for h in [h for h in hints if h["src"] == "centre"][:5]:
            cal.learn(eyes_for(w, (600.0, 500.0)), h["y"], "centre", sig=h["sig"])
        self.assertLess(np.linalg.norm(cal.predict(eyes_for(w, mid))[0] - mid), before * 0.75)
        self.assertFalse(cal.warm)  # (these don't count as having learned anything about you)


@unittest.skipUnless(os.environ.get("SLOW"), "set SLOW=1 to run the local model")
class LocalModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = local_router.Router()

    def test_routes(self):
        self.assertEqual(self.r.decide([("I'M GOING TO SHOW YOU", 0)])["kind"], "ignore")
        self.assertEqual(self.r.decide([("PAUSE THE MUSIC PLEASE", 0)])["cmd"], ["spotify", "pause"])
        d = self.r.decide([("PLEASE PLAY RAP ON SPOTIFY", 0)])
        self.assertEqual(d["kind"], "claude")

    def test_privacy_judge(self):
        g = privacy.Guard(self.r)
        g._save = lambda: None
        self.assertIn("Discord", g.filter("Front: Discord - '#fishing'\nButtons: Send"))
        self.assertIn("private", g.filter("Front: Google Chrome - 'Portfolio'\nHeadings: Holdings, AAPL $27,400"))


if __name__ == "__main__":
    unittest.main()
