"""Fast checks of the logic that decides what silent chat does with what it hears and sees. No camera, mic, model
downloads or Claude calls. Run:  .venv/bin/python -m unittest discover tests
With SLOW=1 it also runs the small local model (needs it downloaded; ~10 s)."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_commands  # noqa: E402
import presence  # noqa: E402
import privacy  # noqa: E402
import shortcuts  # noqa: E402
import speech  # noqa: E402
import improve  # noqa: E402
import local_router  # noqa: E402


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
