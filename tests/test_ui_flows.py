"""The review screen end to end: the real app (fake models, phantom DICOMs) served on a local port, driven in headless
Chromium over the DevTools protocol.

These are the flows a unit test cannot reach, because they live in the page's state: stepping through a dataset past
studies that fail, opening a second folder while one is open, the two-click "clear" (also with a dataset open), an
analysis that is still running when the store is cleared or the dataset left, a folder opened a second time (its
stored studies are shown, not uploaded again), that leaving a dataset removes nothing and a study opened or corrected
from another tab stays, a single study uploaded while a dataset is open, a study with more files than the server
takes, the keyboard and the study list while something else is going on, a study opened from the recent list or by
its address while a dataset is open, one measurement read in every place that shows it, and the server's refusals in
the reader's language. Also: a one-breast or single-view study fills the grid and is drawn at once (with the list's
marks explained in the bar), pictures / empty files / files the browser cannot read are named, upload notes in the
reader's language, a fresh start in English and the dark theme, views that zoom on their own, the editor's right-drag
pan, "Hide contours" and "Smooth / clean labels", and a view without a nipple measured at the contour and marked.

Skipped without uvicorn, websocket-client or a headless Chromium (``$BREAST_SEG_CHROMIUM`` or a Playwright download).
"""
from __future__ import annotations

import copy
import io
import json
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")
websocket = pytest.importorskip("websocket")
pydicom = pytest.importorskip("pydicom")

from breast_seg.apps.positioning import app_config_from_dict, create_app  # noqa: E402
from breast_seg.imaging.phantom import phantom_dicom  # noqa: E402
from breast_seg.inference.testing import FakeModel, fake_loader, placeholder_weights  # noqa: E402

from .test_dataset_js import _chromium, _free_port  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS_YAML = REPO / "configs" / "models.yaml"


class Page:
    """One Chromium tab: evaluate JavaScript, hand files to an input, wait for a condition."""

    def __init__(self, chrome, url, profile):
        port = _free_port()
        self.process = subprocess.Popen(
            [chrome, "--headless=new", f"--remote-debugging-port={port}", "--remote-allow-origins=*", "--no-sandbox",
             "--disable-gpu", "--window-size=1400,900", f"--user-data-dir={profile}", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.socket = None
        try:
            target = None
            for _ in range(150):
                try:
                    tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2))
                    target = next(tab for tab in tabs if tab["type"] == "page")
                    break
                except Exception:  # noqa: BLE001 - not up yet
                    time.sleep(0.1)
            assert target, "headless Chromium did not start"
            self.socket = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=60)
            self.count = 0
            self.call("DOM.enable")
            self.wait("document.readyState === 'complete' && !!document.querySelector('#dataset-input')")
        except BaseException:
            self.close()  # a page that never came up must not leave its browser running
            raise

    def call(self, method, **params):
        self.count += 1
        self.socket.send(json.dumps({"id": self.count, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == self.count:
                assert "error" not in message, message["error"]
                return message.get("result", {})

    def js(self, expression):
        result = self.call("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        assert "exceptionDetails" not in result, result["exceptionDetails"]
        return result["result"].get("value")

    def wait(self, expression, timeout=30.0):
        """Poll until ``expression`` is truthy; its last value otherwise fails the test."""
        deadline = time.time() + timeout
        value = None
        while time.time() < deadline:
            try:
                value = self.js(expression)
            except AssertionError:  # the page is still loading: its context can be replaced under us
                value = None
            if value:
                return value
            time.sleep(0.1)
        raise AssertionError(f"timed out waiting for: {expression} (last: {value!r}; state: {self.state()!r})")

    def click(self, selector):
        self.js(f"document.querySelector({json.dumps(selector)}).click()")

    def choose(self, selector, paths):
        """Give ``paths`` to a file input and fire its change event, as a pick in the file dialog would."""
        root = self.call("DOM.getDocument", depth=-1, pierce=True)["root"]["nodeId"]
        node = self.call("DOM.querySelector", nodeId=root, selector=selector)["nodeId"]
        self.call("DOM.setFileInputFiles", files=[str(path) for path in paths], nodeId=node)
        self.js(f"document.querySelector({json.dumps(selector)}).dispatchEvent(new Event('change'))")

    def open_dataset(self, paths):
        """Give ``paths`` to the dataset input as a folder pick would (CDP cannot pick a directory itself)."""
        self.js("(() => { const i = document.querySelector('#dataset-input');"
                " i.removeAttribute('webkitdirectory'); i.removeAttribute('directory'); })()")
        self.choose("#dataset-input", paths)

    def state(self):
        return self.js("""(() => {
          const q = (selector) => document.querySelector(selector);
          const pick = q('#dataset-pick');
          return {
            screen: ['landing', 'processing', 'review'].find((name) => !q('#screen-' + name).hidden),
            bar: !q('#dataset-nav').hidden,
            options: [...pick.options].map((option) => option.textContent),
            at: pick.value, moving: pick.disabled,
            next: q('#dataset-next-label').textContent, status: q('#dataset-state').textContent,
            banner: q('#banner').hidden ? '' : q('#banner-text').textContent,
            clear: q('#clear-history').textContent,
          };
        })()""")

    def settled(self, timeout=30.0):
        """The state once nothing is being opened or read any more."""
        self.wait("(() => !document.querySelector('#dataset-pick').disabled"
                  " && document.querySelector('#screen-processing').hidden)()", timeout)
        return self.state()

    def close(self):
        try:
            if self.socket is not None:
                self.socket.close()
        finally:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)


@pytest.fixture()
def app(tmp_path, request):
    """The app with fake models; ``app.state.slow`` makes analyses from the n-th on take that many seconds.
    Parametrised indirectly, the parameter is extra config sections (``{"upload": {"max_files": 2}}``)."""
    config = app_config_from_dict({
        "models": {"registry": str(MODELS_YAML), "weights_root": str(placeholder_weights(MODELS_YAML, tmp_path / "w"))},
        "storage": {"data_dir": str(tmp_path / "data")},
        "display": {"preview_size": 400},
        **getattr(request, "param", {}),
    }, base_dir=tmp_path)
    application = create_app(config, loader=fake_loader(), load_in_background=False)
    application.state.slow = {"from_call": 0, "seconds": 0.0, "calls": 0}
    return application


@pytest.fixture()
def served(app):
    """``(base url, app)`` with the app running under uvicorn in a thread."""
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "the app did not start"
    engine = app.state.service.engine
    analyse, slow = engine.analyse, app.state.slow

    def paced(pixels, meta):
        slow["calls"] += 1
        if slow["from_call"] and slow["calls"] >= slow["from_call"]:
            time.sleep(slow["seconds"])
        return analyse(pixels, meta)

    engine.analyse = paced
    yield f"http://127.0.0.1:{port}", app
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture()
def page(served, tmp_path):
    chrome = _chromium()
    if not chrome:
        pytest.skip("needs a headless Chromium to drive the page")
    tab = Page(chrome, served[0] + "/", tmp_path / "chrome")
    yield tab
    tab.close()


@pytest.fixture()
def second_tab(served, tmp_path):
    """``open(study id)``: another browser on the same server, opened on that study (the address route)."""
    tabs = []

    def open_study(study_id):
        tab = Page(_chromium(), f"{served[0]}/#{study_id}", tmp_path / f"chrome-{len(tabs) + 2}")
        tabs.append(tab)
        tab.wait("!document.querySelector('#screen-review').hidden")
        return tab
    yield open_study
    for tab in tabs:
        tab.close()


def _study_file(folder, name, uid, readable=True, view="MLO", side="L"):
    """One phantom view on disk as a one-file study; ``readable=False`` leaves out the view, so the server
    cannot analyse it (422) although it is a proper DICOM image."""
    data = phantom_dicom(view, side, width=300, height=400, seed=len(name),
                         include_view=readable, include_laterality=readable)
    dataset = pydicom.dcmread(io.BytesIO(data))
    dataset.StudyInstanceUID = uid
    path = folder / name
    dataset.save_as(path)
    return path


def _total(base):
    return json.load(urllib.request.urlopen(base + "/api/studies", timeout=10))["total"]


def _api(base, path, method="GET", body=None):
    request = urllib.request.Request(base + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=10))


def _upload(base, *paths):
    """Store the files as one study on its own, as another reader's "Choose files" would; returns the record."""
    boundary = uuid.uuid4().hex
    parts = [b"--%s\r\nContent-Disposition: form-data; name=\"files\"; filename=\"%s\"\r\n"
             b"Content-Type: application/dicom\r\n\r\n%s\r\n" % (boundary.encode(), path.name.encode(), path.read_bytes())
             for path in paths]
    request = urllib.request.Request(
        base + "/api/studies", method="POST", data=b"".join(parts) + b"--%s--\r\n" % boundary.encode(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(request, timeout=30))


def _wait_total(base, count, timeout=10.0):
    deadline = time.time() + timeout
    while _total(base) != count and time.time() < deadline:
        time.sleep(0.05)
    assert _total(base) == count


def _shown(page):
    """The id of the study on screen (the page keeps it in the address)."""
    return page.js("location.hash.replace('#', '')")


def _listing(base):
    """What the server lists as recent (ids), how many studies it stores and how many of them nobody opened."""
    listing = _api(base, "/api/studies")
    return sorted(row["id"] for row in listing["studies"]), listing["total"], listing["unopened"]


def _correct(base, study_id):
    """What the editor's Save sends: the first view with its nipple landmark moved."""
    view = _api(base, f"/api/studies/{study_id}")["images"][0]
    nipple = view["landmarks"]["points"]["nipple"]
    moved = [nipple[0] - 30.0, nipple[1]]
    _api(base, f"/api/studies/{study_id}/labels", "PUT", {"images": [{
        "view_key": view["view_key"], "polygons": view["polygons"], "lesions": view["lesions"],
        "landmarks": {"points": {"nipple": moved}}}]})
    return moved


def test_a_study_that_fails_is_stepped_over_and_the_dataset_can_be_finished(page, tmp_path):
    files = [_study_file(tmp_path, "a_good.dcm", "1.2.3.1"), _study_file(tmp_path, "b_bad.dcm", "1.2.3.2", False),
             _study_file(tmp_path, "c_bad.dcm", "1.2.3.3", False)]
    page.open_dataset(files)
    page.wait("!document.querySelector('#screen-review').hidden")
    assert page.settled()["at"] == "0"

    page.click("#dataset-next")                       # study 2 cannot be analysed
    page.wait("document.querySelector('#dataset-pick').value === '1'")
    state = page.settled()
    assert state["options"][1].endswith("✕") and "b_bad.dcm" in state["banner"] and state["screen"] == "review"

    page.click("#dataset-next")                       # ... and Next moves on to study 3 instead of trying 2 again
    page.wait("document.querySelector('#dataset-pick').value === '2'")
    state = page.settled()
    assert state["options"][2].endswith("✕") and "c_bad.dcm" in state["banner"]
    assert state["next"] == "Finish"                  # the last study failed: the dataset can still be finished

    page.click("#dataset-next")
    state = page.wait("(() => document.querySelector('#banner-text').textContent.startsWith('Dataset finished'))()") \
        and page.state()
    assert "1 opened ✓" in state["banner"] and "2 could not be analysed ✕" in state["banner"]
    assert page.js("document.querySelector('#dataset-next').disabled")      # finished: nothing further to step to

    page.click("#dataset-prev")                       # and back over the failed one, to the study that works
    page.wait("document.querySelector('#dataset-pick').value === '1'")
    page.settled()
    page.click("#dataset-prev")
    page.wait("document.querySelector('#dataset-pick').value === '0'")
    assert page.settled()["options"][0].endswith("✓")


def test_a_folder_that_cannot_be_opened_leaves_the_open_dataset_alone(page, tmp_path):
    first = [_study_file(tmp_path, "a_good.dcm", "1.2.3.1"), _study_file(tmp_path, "b_good.dcm", "1.2.3.2")]
    page.open_dataset(first)
    page.wait("!document.querySelector('#screen-review').hidden")
    before = page.settled()
    assert before["bar"] and len(before["options"]) == 2

    notes = tmp_path / "notes.txt"                    # a folder with no DICOM study in it
    notes.write_text("hello")
    page.open_dataset([notes])
    page.wait("(() => document.querySelector('#banner-text').textContent.startsWith('No DICOM study'))()")
    state = page.settled()
    assert state["bar"] and len(state["options"]) == 2 and state["screen"] == "review"

    # reading the folder fails outright: a visible error, and the screen does not stay on "processing"
    page.js("(() => { window.__compare = Object.getOwnPropertyDescriptor(Intl.Collator.prototype, 'compare');"
            " Object.defineProperty(Intl.Collator.prototype, 'compare',"
            " { configurable: true, get() { throw new Error('boom'); } }); })()")
    page.open_dataset([_study_file(tmp_path, "c_good.dcm", "1.2.3.5"), _study_file(tmp_path, "d_good.dcm", "1.2.3.6")])
    page.wait("(() => document.querySelector('#banner-text').textContent.includes('could not be read: boom'))()")
    state = page.settled()
    assert state["bar"] and len(state["options"]) == 2 and state["screen"] == "review"
    page.js("Object.defineProperty(Intl.Collator.prototype, 'compare', window.__compare)")

    page.click("#dataset-next")                       # the dataset that was open still works
    page.wait("document.querySelector('#dataset-pick').value === '1'")
    assert page.settled()["options"][1].endswith("✓")


def test_the_clear_question_takes_down_only_itself(page, served, tmp_path):
    base, _ = served
    page.open_dataset([_study_file(tmp_path, "a_good.dcm", "1.2.3.1")])
    page.wait("!document.querySelector('#screen-review').hidden")
    page.settled()
    page.click("#new-study")                          # back to the landing, where the recent list is
    page.wait("!document.querySelector('#screen-landing').hidden")
    assert _total(base) == 1

    page.click("#clear-history")
    page.click("#clear-history")                      # a double click is one question, and deletes nothing
    armed = page.wait("document.querySelector('#clear-history').classList.contains('armed')") and page.state()
    assert armed["clear"] == "delete all 1?" and armed["banner"].startswith("All 1 stored studies will be deleted")
    page.click("[data-lang='tr']")                    # the language changes while it is armed: still armed, in TR
    turkish = page.state()
    assert turkish["clear"] == "1 çalışma silinsin mi?" and turkish["banner"].startswith("1 çalışmanın tümü")
    page.click("[data-lang='en']")
    assert page.state()["clear"] == "delete all 1?" and page.js(
        "document.querySelector('#clear-history').classList.contains('armed')")

    notes = tmp_path / "notes.txt"                    # another notice takes the banner while the question is open
    notes.write_text("hello")
    page.open_dataset([notes])
    page.wait("(() => document.querySelector('#banner-text').textContent.startsWith('No DICOM study'))()")
    page.wait("!document.querySelector('#clear-history').classList.contains('armed')", timeout=12)
    state = page.state()
    assert state["clear"] == "clear" and state["banner"].startswith("No DICOM study")   # the notice is still there
    assert _total(base) == 1                          # nothing was deleted without the second click


def _wait_for_calls(app, count, timeout=15.0):
    deadline = time.time() + timeout
    while app.state.slow["calls"] < count and time.time() < deadline:
        time.sleep(0.05)
    assert app.state.slow["calls"] >= count, "the upload did not reach the server"


def test_an_analysis_still_running_when_the_store_is_cleared_or_the_dataset_left_is_kept(page, served, tmp_path):
    base, app = served
    files = [_study_file(tmp_path, "a_good.dcm", "1.2.3.1"), _study_file(tmp_path, "b_good.dcm", "1.2.3.2"),
             _study_file(tmp_path, "c_good.dcm", "1.2.3.3")]
    app.state.slow.update(from_call=2, seconds=3.0)   # every analysis after the first takes 3 s
    page.open_dataset(files)
    page.wait("!document.querySelector('#screen-review').hidden")
    assert page.settled()["options"][0].endswith("✓")
    _wait_for_calls(app, 2)                           # study 1 is on screen, study 2 is being analysed behind it
    assert _total(base) == 1

    # the store is cleared while that analysis runs (two clicks, as the button asks)
    page.click("#clear-history")
    page.wait("document.querySelector('#clear-history').classList.contains('armed')")
    time.sleep(0.7)
    page.click("#clear-history")
    page.wait("!document.querySelector('#screen-landing').hidden")
    assert _total(base) == 0
    # the open dataset is as it was when the folder was opened: nothing is opened or stored any more
    state = page.state()
    assert state["bar"] and not any(option.endswith(("✓", "•", "–")) for option in state["options"])
    _wait_total(base, 1, timeout=6)                   # the analysis ends after the clear: its record is stored,
    assert _listing(base) == ([], 1, 1)               # not opened yet, so not among the recent studies,
    page.wait("document.querySelector('#dataset-pick').options[1].textContent.endsWith('•')")   # and the page knows it

    # reaching study 2 shows that record: it is not analysed a second time
    page.js("(() => { const p = document.querySelector('#dataset-pick'); p.value = '1';"
            " p.dispatchEvent(new Event('change')); })()")
    page.wait("!document.querySelector('#screen-review').hidden")
    assert page.settled()["options"][1].endswith("✓")
    second = _shown(page)
    _wait_for_calls(app, 3)                           # only the preload of study 3 was added
    assert app.state.slow["calls"] == 3 and _listing(base) == ([second], 1, 0)

    # the dataset is left while study 3 is being analysed: the analysis finishes and its record is kept, unopened
    page.click("#new-study")
    page.wait("!document.querySelector('#screen-landing').hidden")
    _wait_total(base, 2, timeout=6)
    time.sleep(0.5)
    assert _listing(base) == ([second], 2, 1)         # nothing was deleted by leaving


def test_a_folder_opened_again_shows_its_stored_studies_and_leaving_removes_nothing(page, served, tmp_path):
    base, app = served
    files = [_study_file(tmp_path, "a_good.dcm", "1.2.3.1"), _study_file(tmp_path, "b_good.dcm", "1.2.3.2"),
             _study_file(tmp_path, "c_good.dcm", "1.2.3.3"), _study_file(tmp_path, "d_good.dcm", "1.2.3.4")]
    page.open_dataset(files)
    page.wait("!document.querySelector('#screen-review').hidden")
    page.settled()
    first = _shown(page)
    page.click("#dataset-next")
    page.wait("document.querySelector('#dataset-pick').value === '1'")
    page.settled()
    second = _shown(page)
    _wait_for_calls(app, 3)                           # studies 1 and 2 were opened, study 3 was analysed ahead
    _wait_total(base, 3)
    _correct(base, first)                             # a reviewer corrects study 1

    # leaving the dataset removes nothing; the recent list shows what was opened, not what was analysed ahead
    page.click("#dataset-close")
    page.wait("!document.querySelector('#screen-landing').hidden")
    page.wait("document.querySelectorAll('#history .history-item').length === 2")
    listed = page.js("[...document.querySelectorAll('#history .history-item')].map((item) => item.dataset.id)")
    assert sorted(listed) == sorted([first, second]) and _listing(base) == (sorted([first, second]), 3, 1)

    # the same folder again: all three come from their stored records, nothing is uploaded for them
    page.open_dataset(files)
    page.wait("!document.querySelector('#screen-review').hidden")
    state = page.settled()
    assert _shown(page) == first
    assert state["options"][0].endswith("✓") and state["options"][1].endswith("•") and state["options"][2].endswith("•")
    assert not state["options"][3].endswith(("✓", "•"))
    assert page.wait("[...document.querySelectorAll('.tile-badge')].some((badge) => badge.textContent === 'edited')")
    time.sleep(0.5)
    assert app.state.slow["calls"] == 3 and _total(base) == 3

    page.click("#dataset-next")                       # study 2 from its record
    page.wait("document.querySelector('#dataset-pick').value === '1'")
    assert page.settled()["options"][1].endswith("✓") and _shown(page) == second
    page.click("#dataset-next")                       # study 3 from the record that was analysed ahead last time:
    page.wait("document.querySelector('#dataset-pick').value === '2'")
    assert page.settled()["options"][2].endswith("✓")
    third = _shown(page)
    _wait_for_calls(app, 4)                           # only study 4 is analysed now, ahead
    _wait_total(base, 4)
    assert app.state.slow["calls"] == 4
    assert _listing(base) == (sorted([first, second, third]), 4, 1)     # opened now, so listed

    # the page is left with the dataset open: nothing changes on the server
    page.js("(() => { location.href = 'about:blank'; })()")
    time.sleep(1.0)
    assert _listing(base) == (sorted([first, second, third]), 4, 1)


def test_a_record_opened_and_corrected_from_another_tab_survives_the_dataset_being_left(page, served, second_tab,
                                                                                        tmp_path):
    """The page that analysed a study ahead never opened it; another tab did, and corrected it. Whatever the first
    page does next (leave the dataset, leave the page), that record and its correction stay."""
    base, app = served
    files = [_study_file(tmp_path, "a_good.dcm", "1.2.3.1"), _study_file(tmp_path, "b_good.dcm", "1.2.3.2")]
    page.open_dataset(files)
    page.wait("!document.querySelector('#screen-review').hidden")
    page.settled()
    first = _shown(page)
    _wait_for_calls(app, 2)
    _wait_total(base, 2)
    ahead = next(path.name for path in (tmp_path / "data" / "studies").iterdir() if path.name != first)
    assert _listing(base) == ([first], 2, 1)          # analysed ahead: stored, not listed

    other = second_tab(ahead)                         # another reviewer opens it by its address ...
    assert _shown(other) == ahead
    deadline = time.time() + 5
    while _listing(base)[2] and time.time() < deadline:
        time.sleep(0.1)
    assert _listing(base) == (sorted([first, ahead]), 2, 0)             # ... which lists it,
    moved = _correct(base, ahead)                     # and corrects it

    page.click("#dataset-close")                      # the first page never opened it, and leaves the dataset
    page.wait("!document.querySelector('#screen-landing').hidden")
    page.wait("document.querySelectorAll('#history .history-item').length === 2")
    page.js("(() => { location.href = 'about:blank'; })()")
    time.sleep(1.0)
    kept = _api(base, f"/api/studies/{ahead}")
    assert kept["edited"] and kept["images"][0]["landmarks"]["points"]["nipple"] == moved
    assert _listing(base) == (sorted([first, ahead]), 2, 0)


def test_a_single_study_uploaded_while_a_dataset_is_open_leaves_the_dataset(page, served, tmp_path):
    base, _ = served
    files = [_study_file(tmp_path, "a_bad.dcm", "1.2.3.1", False), _study_file(tmp_path, "b_good.dcm", "1.2.3.2")]
    page.open_dataset(files)                          # study 1 cannot be analysed: the landing stays, with the bar
    page.wait("(() => document.querySelector('#banner-text').textContent.includes('a_bad.dcm'))()")
    state = page.settled()
    assert state["bar"] and state["screen"] == "landing"

    single = _study_file(tmp_path, "single.dcm", "1.2.3.9")
    page.choose("#file-input", [single])              # a study on its own, as "Choose files" gives it
    page.wait("!document.querySelector('#screen-review').hidden")
    state = page.state()
    assert not state["bar"]                           # the bar of the dataset does not stay over a study it does not hold
    uploaded = _shown(page)
    page.js("document.dispatchEvent(new KeyboardEvent('keydown', { key: 'PageDown', bubbles: true, cancelable: true }))")
    time.sleep(0.5)
    assert _shown(page) == uploaded and page.state()["screen"] == "review"          # and its keys move nothing
    assert uploaded in _listing(base)[0]              # a study uploaded on its own is opened: it is listed


@pytest.mark.parametrize("app", [{"upload": {"max_files": 2}}], indirect=True)
def test_a_study_with_more_files_than_the_server_takes_sends_its_views_and_says_so(page, served, tmp_path):
    base, app = served
    files = [_study_file(tmp_path, "a_extra.dcm", "1.2.3.1", view="MLO", side="R"),
             _study_file(tmp_path, "b_LMLO.dcm", "1.2.3.1", view="MLO", side="L"),
             _study_file(tmp_path, "c_LCC.dcm", "1.2.3.1", view="CC", side="L"),
             _study_file(tmp_path, "d_other.dcm", "1.2.3.2")]
    page.open_dataset(files)
    page.wait("!document.querySelector('#screen-review').hidden")
    state = page.settled()
    # three files, two allowed: the two whose name says which view they are, not the first two in path order
    views = sorted(image["view_key"] for image in _api(base, f"/api/studies/{_shown(page)}")["images"])
    assert views == ["L-CC", "L-MLO"]
    assert state["banner"].startswith("This study has 3 files but the server takes 2 per study; not sent: 1.")
    assert page.js("document.querySelector('#dataset-skipped').textContent") == "1 of 3 files of this study not sent"

    page.click("#dataset-next")                       # a study within the limit says nothing
    page.wait("document.querySelector('#dataset-pick').value === '1'")
    state = page.settled()
    assert state["banner"] == "" and page.js("document.querySelector('#dataset-skipped').textContent") == ""
    whole = _shown(page)

    # the folder again: both studies come from their records. The limit leaves out the same file every time, so
    # analysing the cut study again could only give the same record: it is complete as stored, found by its key
    cut = sorted(path.name for path in (tmp_path / "data" / "studies").iterdir() if path.name != whole)
    calls = app.state.slow["calls"]
    page.open_dataset(files)
    page.wait("!document.querySelector('#screen-review').hidden")
    state = page.settled()
    time.sleep(0.5)
    again = sorted(path.name for path in (tmp_path / "data" / "studies").iterdir() if path.name != whole)
    assert len(cut) == 1 and again == cut and _shown(page) == cut[0]
    assert app.state.slow["calls"] == calls                            # nothing analysed again


def test_the_criteria_drawer_does_not_stop_the_keys_and_a_refused_pick_goes_back(page, served, tmp_path):
    base, app = served
    files = [_study_file(tmp_path, "a_good.dcm", "1.2.3.1"), _study_file(tmp_path, "b_good.dcm", "1.2.3.2"),
             _study_file(tmp_path, "c_good.dcm", "1.2.3.3")]
    page.open_dataset(files)
    page.wait("!document.querySelector('#screen-review').hidden")
    page.settled()
    first = _shown(page)
    key = ("(key) => document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }))")

    # the report covers the review: PageDown stays quiet under it
    page.click("#report-toggle")
    page.wait("!document.querySelector('#report-panel').hidden")
    page.js(f"({key})('PageDown')")
    time.sleep(0.4)
    assert page.settled()["at"] == "0"
    page.click("#report-close")

    # the criteria drawer sits next to the images: PageDown and PageUp go on working while it is open
    page.click("#criteria-toggle")
    page.wait("!document.querySelector('#criteria-drawer').hidden")
    page.js(f"({key})('PageDown')")
    page.wait("document.querySelector('#dataset-pick').value === '1'")
    assert page.settled()["options"][1].endswith("✓")
    assert not page.js("document.querySelector('#criteria-drawer').hidden")       # and it stays open
    page.js(f"({key})('PageUp')")
    page.wait("document.querySelector('#dataset-pick').value === '0'")
    page.settled()
    assert _shown(page) == first

    # a study picked from the list while the dataset is on its way to another one is refused, and the list says
    # where it is going (study 2, asked for by its address: a study of the dataset is gone to through the dataset)
    _wait_for_calls(app, 3)
    _wait_total(base, 3)
    other = next(row["id"] for row in _api(base, "/api/studies")["studies"] if row["id"] != first)
    pick = (" const pick = document.querySelector('#dataset-pick'); pick.value = '2';"
            " pick.dispatchEvent(new Event('change'));"
            " return [pick.value, document.querySelector('#banner-text').textContent]; })()")
    picked = page.js("(() => { history.replaceState(null, '', '#" + other + "');"
                     " window.dispatchEvent(new HashChangeEvent('hashchange'));" + pick)
    assert picked == ["1", ""]
    page.wait("document.querySelector('#dataset-pick').value === '1' && !document.querySelector('#dataset-pick').disabled")
    assert page.settled()["bar"] and _shown(page) == other

    # while another folder is being read the page is busy: a pick is refused with the reason, and the list goes back
    # (the folder is handed over and the pick made in one step, before the page has read a single file)
    picked = page.js("(() => { const input = document.querySelector('#dataset-input'), folder = new DataTransfer();"
                     " folder.items.add(new File(['hello'], 'notes.txt')); input.files = folder.files;"
                     " input.dispatchEvent(new Event('change'));" + pick)
    assert picked == ["1", "Still working on the previous step — try again in a moment."]
    page.wait("(() => document.querySelector('#banner-text').textContent.startsWith('No DICOM study'))()")
    assert page.settled()["bar"] and _shown(page) == other             # the folder held no study: the dataset stays


def test_the_same_folder_opened_over_itself_uploads_nothing_again(page, served, tmp_path):
    base, app = served
    files = [_study_file(tmp_path, "a_good.dcm", "1.2.3.1"), _study_file(tmp_path, "b_good.dcm", "1.2.3.2"),
             _study_file(tmp_path, "c_good.dcm", "1.2.3.3")]
    page.open_dataset(files)
    page.wait("!document.querySelector('#screen-review').hidden")
    page.settled()
    first = _shown(page)
    _wait_for_calls(app, 2)                           # study 1 is on screen, study 2 was analysed ahead
    _wait_total(base, 2)
    stored = sorted(path.name for path in (tmp_path / "data" / "studies").iterdir())

    page.open_dataset(files)                          # the folder is picked again while it is open
    page.wait("!document.querySelector('#screen-review').hidden")
    state = page.settled()
    time.sleep(0.6)
    assert sorted(path.name for path in (tmp_path / "data" / "studies").iterdir()) == stored
    assert app.state.slow["calls"] == 2 and state["options"][1].endswith("•")

    page.click("#dataset-close")                      # leaving it removes nothing either
    page.wait("!document.querySelector('#screen-landing').hidden")
    time.sleep(0.6)
    assert _listing(base) == ([first], 2, 1)


# ── a study opened on its own while a dataset is open ──────────────────────
def _a_dataset_whose_first_study_fails(page, served, tmp_path):
    """The landing with a dataset open (its first study cannot be analysed) and the recent list under it. Two studies
    of the dataset are stored from an earlier session, one opened then and one only analysed ahead; a third study was
    stored on its own. Returns ``(the dataset's files, opened, ahead, own)``."""
    base, app = served
    good = [_study_file(tmp_path, "b_good.dcm", "1.2.3.2"), _study_file(tmp_path, "c_good.dcm", "1.2.3.3")]
    page.open_dataset(good)                           # the earlier session
    page.wait("!document.querySelector('#screen-review').hidden")
    page.settled()
    opened = _shown(page)
    _wait_for_calls(app, 2)
    _wait_total(base, 2)
    ahead = next(path.name for path in (tmp_path / "data" / "studies").iterdir() if path.name != opened)
    own = _upload(base, _study_file(tmp_path, "x_own.dcm", "1.2.3.9"))["id"]
    page.click("#new-study")
    page.wait("!document.querySelector('#screen-landing').hidden")
    page.wait("document.querySelectorAll('#history .history-item').length === 2")       # opened and own, not ahead

    files = [_study_file(tmp_path, "a_bad.dcm", "1.2.3.1", False), *good]
    page.open_dataset(files)                          # study 1 cannot be analysed: the landing stays, with the bar
    page.wait("(() => document.querySelector('#banner-text').textContent.includes('a_bad.dcm'))()")
    state = page.settled()
    assert state["bar"] and state["screen"] == "landing"
    assert state["options"][1].endswith("•") and state["options"][2].endswith("•")      # found by their keys
    return files, opened, ahead, own


def test_a_study_of_the_open_dataset_opened_on_its_own_moves_the_bar_to_it(page, served, tmp_path):
    """The recent list and the address are other ways to the same study: the dataset stays open and its bar goes to
    the study, instead of staying on another one while this one is on screen."""
    base, app = served
    _, opened, ahead, own = _a_dataset_whose_first_study_fails(page, served, tmp_path)
    calls = app.state.slow["calls"]

    page.click(f'#history .history-item[data-id="{opened}"]')          # from the recent list
    page.wait("!document.querySelector('#screen-review').hidden")
    state = page.settled()
    assert state["bar"] and state["at"] == "1" and state["options"][1].endswith("✓") and _shown(page) == opened

    page.js(f"(() => {{ location.hash = '#{ahead}'; }})()")              # by its address
    page.wait("document.querySelector('#dataset-pick').value === '2'")
    state = page.settled()
    assert state["bar"] and state["options"][2].endswith("✓") and _shown(page) == ahead
    assert state["next"] == "Finish"                  # the bar knows it is on the last study

    page.js("document.dispatchEvent(new KeyboardEvent('keydown', { key: 'PageUp', bubbles: true, cancelable: true }))")
    page.wait("document.querySelector('#dataset-pick').value === '1'")  # and the keys step from there
    assert page.settled()["bar"] and _shown(page) == opened
    assert app.state.slow["calls"] == calls           # shown from their records: nothing was analysed again
    assert _listing(base) == (sorted([opened, ahead, own]), 3, 0)       # the one analysed ahead is opened now


def test_a_study_outside_the_open_dataset_opened_on_its_own_leaves_the_dataset(page, served, tmp_path):
    """As a single upload and "New study" do: the bar of a dataset does not stay over a study it does not hold."""
    base, _ = served
    files, opened, ahead, own = _a_dataset_whose_first_study_fails(page, served, tmp_path)

    page.click(f'#history .history-item[data-id="{own}"]')             # from the recent list
    page.wait("!document.querySelector('#screen-review').hidden")
    state = page.settled()
    assert not state["bar"] and _shown(page) == own
    page.js("document.dispatchEvent(new KeyboardEvent('keydown', { key: 'PageDown', bubbles: true, cancelable: true }))")
    time.sleep(0.5)
    assert _shown(page) == own and page.state()["screen"] == "review"   # the dataset's keys move nothing any more
    assert _listing(base) == (sorted([opened, own]), 3, 1)              # leaving removed nothing

    other = _upload(base, _study_file(tmp_path, "y_own.dcm", "1.2.3.8"))["id"]
    page.open_dataset(files)                          # the dataset again; its first study still fails
    page.wait("(() => document.querySelector('#banner-text').textContent.includes('a_bad.dcm'))()")
    assert page.settled()["bar"]
    page.js(f"(() => {{ location.hash = '#{other}'; }})()")              # by its address
    page.wait(f"location.hash === '#{other}' && !document.querySelector('#screen-review').hidden"
              " && document.querySelector('#dataset-nav').hidden")
    assert _shown(page) == other and not page.settled()["bar"]
    assert _listing(base) == (sorted([opened, own, other]), 4, 1)


# ── one number, wherever it is shown ───────────────────────────────────────
def test_a_measurement_reads_the_same_under_its_image_on_its_card_and_in_the_report(page, served, tmp_path):
    """A nipple 47.949 mm from the chest wall is 47.9 mm under the CC image, on the card and in the report. Rounded
    to two decimals on its way to the card (47.95), it read 48.0 mm there."""
    base, app = served
    service = app.state.service
    analyse = service.engine.analyse

    def scaled(pixels, meta):                         # the pixel size at which the CC landmark distance is 47.949377 mm
        analysis = analyse(pixels, meta)
        if meta["view"] == "CC":
            probe = service.assessor.build_image(meta["view_key"], meta, copy.deepcopy(analysis))
            analysis["mm_per_px"] *= 47.949377 / probe["measures"]["landmark"]["mm"]
        return analysis

    service.engine.analyse = scaled
    page.choose("#file-input", [_study_file(tmp_path, "L_MLO.dcm", "1.2.3.1"),
                                _study_file(tmp_path, "L_CC.dcm", "1.2.3.1", view="CC")])
    page.wait("!document.querySelector('#screen-review').hidden")
    record = _api(base, f"/api/studies/{_shown(page)}")
    cc = next(image for image in record["images"] if image["view"] == "CC")
    assert round(cc["measures"]["landmark"]["mm"], 2) == 47.95         # what the summary used to keep

    shown = page.js("""(() => {
      const text = (nodes) => [...nodes].map((node) => node.textContent.trim());
      const tiles = {};
      for (const tile of document.querySelectorAll('#quad .view-tile')) {
        const foot = tile.querySelectorAll('.tile-foot .route-value');
        if (foot.length) tiles[tile.querySelector('.tile-key').textContent.trim()] = text(foot);
      }
      const cards = [...document.querySelectorAll('.verdict-card')].map((card) => text(card.querySelectorAll('dl.pair dd')));
      return { tiles, cards };
    })()""")
    landmark, mask = shown["cards"]                   # the left breast's two cards: [PNL, nipple to wall, threshold]
    assert shown["tiles"]["L-CC"] == [landmark[1], mask[1]] and landmark[1] == "47.9 mm"
    assert shown["tiles"]["L-MLO"] == [landmark[0], mask[0]]
    report = _api(base, f"/api/studies/{record['id']}/report?lang=en")["text"]
    assert "(CC) 47.9 mm" in report and "(CC) 48.0 mm" not in report


# ── the server's refusals, in the reader's language ────────────────────────
@pytest.mark.parametrize("app", [{"language": "tr"}], indirect=True)
def test_the_refusals_of_the_server_are_shown_in_the_readers_language(page, served, tmp_path):
    base, _ = served
    banner = "document.querySelector('#banner-text').textContent"
    page.wait("document.documentElement.lang === 'tr'")

    page.js("(() => { location.hash = '#doesnotexist'; })()")          # a study that is not there
    page.wait(f"{banner}.includes('Çalışma bulunamadı.')")

    page.choose("#file-input", [_study_file(tmp_path, "a_bad.dcm", "1.2.3.1", False)])   # nothing can be analysed
    page.wait(f"{banner}.includes('Hiçbir görüntü işlenemedi.')")
    text = page.state()["banner"]
    assert "a_bad.dcm: CC mi MLO mu anlaşılamadı" in text and "could not tell" not in text

    page.choose("#file-input", [_study_file(tmp_path, "b_good.dcm", "1.2.3.2")])
    page.wait("!document.querySelector('#screen-review').hidden")
    page.click("#new-study")
    page.wait("!document.querySelector('#screen-landing').hidden")
    page.click("#clear-history")                      # "clear", confirmed for the one study that is stored ...
    page.wait("document.querySelector('#clear-history').classList.contains('armed')")
    _upload(base, _study_file(tmp_path, "c_good.dcm", "1.2.3.3"))      # ... while another reader stores one more
    time.sleep(0.8)
    page.click("#clear-history")
    page.wait(f"{banner}.includes('Şu anda 2 çalışma saklanıyor, 1 değil; hiçbir şey silinmedi.')")
    assert _total(base) == 2


# ── the grid shows what the study holds ────────────────────────────────────
TILES = """(() => [...document.querySelectorAll('#quad .view-tile')].map((tile) => {
  const canvas = tile.querySelector('canvas');
  let drawn = false;
  if (canvas && canvas.width) {
    const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 0; i < data.length; i += 4 * 97) if (data[i] > 12) { drawn = true; break; }
  }
  return [tile.querySelector('.tile-key').textContent.trim(), tile.classList.contains('missing'), drawn];
}))()"""


def _layout(page, cols, rows, pairing=None):
    page.click("#layout-toggle")
    page.click(f".cell[data-cols='{cols}'][data-rows='{rows}']")
    if pairing:
        page.js(f"[...document.querySelectorAll('#pairing button')][{0 if pairing == 'view' else 1}].click()")


def test_a_study_of_one_breast_fills_the_grid_and_is_drawn_at_once(page, tmp_path):
    """A dataset of one-breast studies (as in CMMD): the first study is drawn as soon as it is opened, and only its
    views take a pane, in every layout; a single view fills the grid alone. The list's marks are explained in the bar."""
    right = [_study_file(tmp_path, "s1_RMLO.dcm", "1.2.3.1", view="MLO", side="R"),
             _study_file(tmp_path, "s1_RCC.dcm", "1.2.3.1", view="CC", side="R")]
    left = [_study_file(tmp_path, "s2_LMLO.dcm", "1.2.3.2", view="MLO", side="L"),
            _study_file(tmp_path, "s2_LCC.dcm", "1.2.3.2", view="CC", side="L")]
    single = [_study_file(tmp_path, "s3_LCC.dcm", "1.2.3.3", view="CC", side="L")]
    page.open_dataset(right + left + single)
    page.wait("!document.querySelector('#screen-review').hidden")
    page.settled()
    tiles = page.wait(f"(() => {{ const t = {TILES}; return t.length && t.every((x) => x[2]) && t; }})()")
    assert tiles == [["R-MLO", False, True], ["R-CC", False, True]]   # both drawn, no empty pane for the left breast
    legend = page.js("(() => { const n = document.querySelector('#dataset-legend');"
                     " return n.offsetParent !== null && n.textContent; })()")
    assert legend.startswith("✓ opened · • analysed")                 # the marks of the list, without hovering

    _layout(page, 2, 1, pairing="breast")                             # by breast: the left breast's page is left out
    assert page.js("document.querySelector('#pager').hidden") and [t[0] for t in page.js(TILES)] == ["R-MLO", "R-CC"]
    _layout(page, 1, 1)                                               # one pane: the right breast's two views only
    assert [t[0] for t in page.js(TILES)] == ["R-MLO"] and page.js("document.querySelector('#page-label').textContent") \
        == "1 / 2"
    _layout(page, 2, 2)
    page.click("#dataset-next")
    page.wait("document.querySelector('#dataset-pick').value === '1'")
    page.settled()
    assert page.wait(f"(() => {{ const t = {TILES}; return t.every((x) => x[2]) && t.map((x) => x[0]); }})()") \
        == ["L-MLO", "L-CC"]
    page.click("#dataset-next")
    page.wait("document.querySelector('#dataset-pick').value === '2'")
    page.settled()
    assert page.wait(f"(() => {{ const t = {TILES}; return t.length === 1 && t[0][2] && t; }})()") == [["L-CC", False, True]]
    assert page.js("getComputedStyle(document.querySelector('#quad')).gridTemplateColumns.split(' ').length") == 1


# ── pictures, empty files and files the browser cannot read ─────────────────
def test_pictures_empty_files_and_unreadable_files_are_named(page, tmp_path):
    banner = "document.querySelector('#banner-text').textContent"
    pictures = [tmp_path / "L-CC.png", tmp_path / "R-CC.jpg"]
    for path in pictures:
        path.write_bytes(b"\x89PNG not really")
    page.choose("#file-input", pictures)                              # the server says it, by code
    page.wait(f"{banner}.startsWith('This application reads DICOM (.dcm) only')")
    assert page.state()["banner"].endswith("Not used: L-CC.png, R-CC.jpg.")
    page.js("(() => { const i = document.querySelector('#folder-input');"
            " i.removeAttribute('webkitdirectory'); i.removeAttribute('directory'); })()")
    page.choose("#folder-input", pictures)                            # a folder of pictures: the page says it itself
    page.wait(f"{banner}.startsWith('This application reads DICOM (.dcm) only') && {banner}.includes('L-CC.png')")

    empty = tmp_path / "empty.dcm"
    empty.write_bytes(b"")
    page.choose("#file-input", [empty])
    page.wait(f"{banner} === 'The selected file is empty (0 bytes): empty.dcm.'")
    good = _study_file(tmp_path, "L_MLO.dcm", "1.2.3.1")
    page.choose("#file-input", [good, empty])                         # the study goes through, the empty file is named
    page.wait("!document.querySelector('#screen-review').hidden")
    page.wait(f"{banner} === 'Empty files (0 bytes) were not sent: empty.dcm.'")

    page.click("#new-study")
    page.wait("!document.querySelector('#screen-landing').hidden")
    # A file the browser can no longer read once it was picked (here: its read permission is taken away; on Windows a
    # moved file does the same): the request fails before it leaves, and the browser says only "network error".
    # (Setting the files of the page's own input starts the upload at once, so the file is picked on a spare input and
    # dropped on the drop zone afterwards, as a plain file: an entry would be opened afresh.)
    changed = _study_file(tmp_path, "changed.dcm", "1.2.3.2")
    page.js("(() => { const i = document.createElement('input'); i.type = 'file'; i.id = 'spare';"
            " document.body.appendChild(i); })()")
    root = page.call("DOM.getDocument", depth=-1, pierce=True)["root"]["nodeId"]
    node = page.call("DOM.querySelector", nodeId=root, selector="#spare")["nodeId"]
    page.call("DOM.setFileInputFiles", files=[str(changed)], nodeId=node)
    changed.chmod(0)
    page.js("(() => { DataTransferItem.prototype.webkitGetAsEntry = () => null;"
            " const transfer = new DataTransfer(); transfer.items.add(document.querySelector('#spare').files[0]);"
            " document.querySelector('#dropzone').dispatchEvent(new DragEvent('drop', { dataTransfer: transfer,"
            " bubbles: true, cancelable: true })); })()")
    try:
        page.wait(f"{banner} === 'The browser could not read changed.dcm; it may have been moved or changed.'")
    finally:
        changed.chmod(0o600)


@pytest.mark.parametrize("app", [{"language": "tr"}], indirect=True)
def test_the_upload_notes_of_a_stored_study_are_in_the_readers_language(page, served, tmp_path):
    page.wait("document.documentElement.lang === 'tr'")
    picture = tmp_path / "L-CC.png"
    picture.write_bytes(b"\x89PNG not really")
    page.choose("#file-input", [_study_file(tmp_path, "L_MLO.dcm", "1.2.3.1"), picture])
    page.wait("!document.querySelector('#screen-review').hidden")
    notes = page.js("[...document.querySelectorAll('#notices li')].map((li) => li.textContent)")
    assert "L-CC.png: atlandı — bu uygulama yalnız DICOM (.dcm) okur; PNG/JPG’de piksel aralığı ve görünüm bilgisi " \
           "olmadığı için mm ölçülemez." in notes
    assert not any("skipped" in note for note in notes)


# ── fresh start: English and dark; light is remembered, the language only for the session ─
def test_every_start_is_in_english_and_dark_until_light_is_chosen(served, tmp_path):
    chrome = _chromium()
    if not chrome:
        pytest.skip("needs a headless Chromium to drive the page")
    profile = tmp_path / "chrome-start"
    page = Page(chrome, served[0] + "/", profile)
    try:
        page.wait("document.documentElement.lang === 'en'")
        assert page.js("document.documentElement.dataset.bsTheme") == "dark"
        page.click("[data-lang='tr']")
        page.wait("document.documentElement.lang === 'tr'")
        page.click("#theme-toggle")
        assert page.js("document.documentElement.dataset.bsTheme") == "light"
        page.js("location.reload()")                                  # the same session: still Turkish
        page.wait("document.readyState === 'complete' && document.querySelector('#splash').hidden")
        assert page.js("document.documentElement.lang") == "tr"
        assert page.js("Object.keys(localStorage).some((key) => key.includes('lang'))") is False
    finally:
        page.close()
    page = Page(chrome, served[0] + "/", profile)                      # the next start, same browser profile
    try:
        page.wait("document.querySelector('#splash').hidden")
        assert page.js("document.documentElement.lang") == "en"       # the TR pick held for that session only
        assert page.js("document.documentElement.dataset.bsTheme") == "light"   # the theme pick is remembered
    finally:
        page.close()


# ── zoom and pan per view ──────────────────────────────────────────────────
def test_the_views_zoom_on_their_own_unless_synchronised(page, tmp_path):
    page.choose("#file-input", [_study_file(tmp_path, "L_MLO.dcm", "1.2.3.1"),
                                _study_file(tmp_path, "L_CC.dcm", "1.2.3.1", view="CC")])
    page.wait("!document.querySelector('#screen-review').hidden")
    pixels = """(() => [...document.querySelectorAll('#quad canvas')].map((canvas) => {
      const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let sum = 0; for (let i = 0; i < data.length; i += 4 * 31) sum += data[i]; return sum; }))()"""
    wheel = """(() => { const wrap = document.querySelectorAll('#quad .tile-canvas-wrap')[0];
      const r = wrap.getBoundingClientRect();
      for (let i = 0; i < 4; i += 1) wrap.dispatchEvent(new WheelEvent('wheel', { deltaY: -100, bubbles: true,
        cancelable: true, clientX: r.left + r.width / 2, clientY: r.top + r.height / 3 })); })()"""
    page.wait(f"{pixels}.every((sum) => sum > 0)")
    assert not page.js("document.querySelector('#sync-toggle').checked")
    before = page.js(pixels)
    page.js(wheel)
    after = page.js(pixels)
    assert after[0] != before[0] and after[1] == before[1]            # only the view under the cursor zoomed
    page.click("#layout-toggle")
    page.click("#sync-toggle")                                        # synchronised: one zoom for every view
    page.js(wheel)
    synced = page.js(pixels)
    assert synced[1] != after[1]


# ── the editor: right-drag pans, contours hide, labels can be cleaned ──────
def _drag(page, button, start, end):
    for kind, (x, y) in (("mousePressed", start), ("mouseMoved", end), ("mouseReleased", end)):
        page.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button=button,
                  buttons={"left": 1, "right": 2, "middle": 4}[button] if kind != "mouseReleased" else 0,
                  clickCount=1)


def test_the_editor_pans_with_the_right_button_and_hides_or_cleans_contours(page, served, tmp_path):
    page.choose("#file-input", [_study_file(tmp_path, "L_MLO.dcm", "1.2.3.1"),
                                _study_file(tmp_path, "L_CC.dcm", "1.2.3.1", view="CC")])
    page.wait("!document.querySelector('#screen-review').hidden")
    page.click("#open-editor")
    page.wait("!document.querySelector('#editor').hidden")
    transform = "document.querySelector('#editor-canvas').style.transform"
    dirty = "!document.querySelector('#editor-dirty').hidden"
    centre = page.js("(() => { const r = document.querySelector('#editor-stage').getBoundingClientRect();"
                     " return [Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2)]; })()")
    before = page.js(transform)
    _drag(page, "right", centre, (centre[0] + 60, centre[1] + 25))     # right-drag pans
    moved = page.js(transform)
    assert moved != before and not page.js(dirty)
    _drag(page, "middle", centre, (centre[0] - 30, centre[1]))         # the middle button still pans too
    assert page.js(transform) != moved and not page.js(dirty)
    _drag(page, "right", centre, centre)                               # a right click that does not move: nothing
    assert not page.js(dirty) and page.js("document.querySelector('#editor').hidden") is False

    chips = "[...document.querySelectorAll('#layerbar input')].slice(0, 3).map((input) => input.checked)"
    assert page.js(chips) == [True, True, True]
    page.js("document.dispatchEvent(new KeyboardEvent('keydown', { key: 'h', bubbles: true }))")
    assert page.js(chips) == [False, False, False]                     # the review's own layer toggles follow
    assert page.js("document.querySelector('#editor-hide').getAttribute('aria-pressed')") == "true"
    assert not page.js(dirty) and "hidden" in page.js("document.querySelector('#tool-hint').textContent")
    page.click("#editor-hide")
    assert page.js(chips) == [True, True, True] and not page.js(dirty)

    hint = "document.querySelector('#tool-hint').textContent"
    idle = "!document.querySelector('#editor-clean').disabled"
    undo = "document.dispatchEvent(new KeyboardEvent('keydown', { key: 'z', ctrlKey: true, bubbles: true }))"
    page.click("#editor-clean")
    page.wait(idle)
    assert page.js(hint).startswith("Cleaned:") and page.js(dirty)    # an unsaved edit
    page.click("#editor-clean")                                        # a second click changes nothing
    page.wait(idle)
    assert page.js(hint).startswith("Nothing to clean on this view") and page.js(dirty)
    page.js(undo)
    assert not page.js(dirty)                                          # Undo takes the first one back

    # the answer goes to the view it was asked for, also when the reader has moved on meanwhile
    service = served[1].state.service
    clean = service.clean_view
    service.clean_view = lambda study_id, payload: time.sleep(1.5) or clean(study_id, payload)
    name = "document.querySelector('#editor-view-name').textContent"
    asked = page.js(name)
    page.click("#editor-clean")
    page.js("[...document.querySelectorAll('#editor-views .view-btn')].find((b) => !b.classList.contains('on')).click()")
    assert page.js(name) != asked
    page.wait(idle, timeout=10)
    assert page.js(hint).startswith(f"{asked} cleaned:") and page.js(name) != asked
    marks = dict(page.js("[...document.querySelectorAll('#editor-views .view-btn')]"
                         ".map((b) => [b.firstChild.textContent, !!b.querySelector('.edited')])"))
    assert marks[asked] and not any(mark for key, mark in marks.items() if key != asked)
    page.js(undo)                                                      # Undo goes back to that view and reverts it
    assert page.js(name) == asked and not page.js(dirty)


@pytest.mark.parametrize("network", ["instance", "semantic"])
def test_a_view_without_a_nipple_is_measured_as_an_estimate_and_marked_by_its_kind(page, served, tmp_path, network):
    """An instance model (seg_yolo26m) that every nipple search misses: measured at the contour point. A semantic network
    (SegFormer) where no nipple probability reaches 0.02: the paper's rule places it at the most probable pixel."""
    _, app = served
    models = app.state.service.engine._models
    if network == "instance":
        models["seg_segformer_b3"] = FakeModel("segmentation", names=models["seg_segformer_b3"].names,
                                               nipple_conf=0.001)
    else:
        models["seg_segformer_b3"].nipple_conf = 0.001
    page.choose("#file-input", [_study_file(tmp_path, "L_MLO.dcm", "1.2.3.1"),
                                _study_file(tmp_path, "L_CC.dcm", "1.2.3.1", view="CC")])
    page.wait("!document.querySelector('#screen-review').hidden")
    shown = page.js("""(() => ({
      badges: [...document.querySelectorAll('.tile-badge')].map((node) => node.textContent),
      marks: [...document.querySelectorAll('.tile-foot .est-mark')].length,
      card: [...document.querySelectorAll('.card-estimated')].map((node) => node.textContent),
      notes: [...document.querySelectorAll('#notices li')].map((li) => li.textContent),
    }))()""")
    assert shown["marks"] == 2 and "nipple rescued" not in shown["badges"]
    if network == "instance":
        assert shown["badges"].count("nipple estimated (contour)") == 2
        assert shown["card"] == ["Estimated — nipple not found on MLO + CC; measured at the farthest breast-tissue "
                                 "contour point."]
        assert any(note.startswith("L-MLO: nipple not found — the segmentation PNL is estimated")
                   for note in shown["notes"])
    else:
        assert shown["badges"].count("nipple estimated (most probable pixel)") == 2
        assert shown["card"] == ["Estimated — on MLO + CC no nipple probability reached 0.02; measured from the most "
                                 "probable pixel."]
        assert any(note.startswith("L-MLO: no nipple probability reached 0.02") for note in shown["notes"])


_SHOWN_VALUES = """(() => {
  const text = (nodes) => [...nodes].map((node) => node.textContent.trim());
  const tiles = {};
  for (const tile of document.querySelectorAll('#quad .view-tile')) {
    const foot = tile.querySelectorAll('.tile-foot .route-value');
    if (foot.length) tiles[tile.querySelector('.tile-key').textContent.trim()] = text(foot);
  }
  const cards = [...document.querySelectorAll('.verdict-card')].map((card) => ({
    pairs: text(card.querySelectorAll('dl.pair dd')), delta: card.querySelector('.delta-value')?.textContent,
    verdict: card.querySelector('.verdict-pill')?.textContent }));
  return { tiles, cards, badges: text(document.querySelectorAll('.tile-badge')),
           reset: !document.querySelector('#adjust-reset').hidden };
})()"""


def _screen_point(page, view_key, point):
    """Where a point of the 640 frame is on screen, on a tile shown whole (no crop, no zoom)."""
    return page.js(f"""(() => {{
      const tile = [...document.querySelectorAll('#quad .view-tile')]
        .find((node) => node.querySelector('.tile-key').textContent.trim() === {json.dumps(view_key)});
      const r = tile.querySelector('canvas').getBoundingClientRect();
      return [r.left + {point[0]} * r.width / 640, r.top + {point[1]} * r.height / 640];
    }})()""")


def _mouse(page, kind, at, buttons):
    page.call("Input.dispatchMouseEvent", type=kind, x=at[0], y=at[1], button="left", buttons=buttons, clickCount=1)


def test_adjust_pnl_follows_the_pointer_is_saved_on_release_and_reset_to_the_model(page, served, tmp_path):
    """The nipple handle of the segmentation measurement dragged on the L-MLO tile: the millimetres under the tile and
    on the card follow while the button is down, letting go stores the placement as manual (and the server's numbers
    are exactly the ones the page showed), a reload keeps it, and "Reset to model" brings the model's back."""
    base, _ = served
    page.choose("#file-input", [_study_file(tmp_path, "L_MLO.dcm", "1.2.3.1"),
                                _study_file(tmp_path, "L_CC.dcm", "1.2.3.1", view="CC")])
    page.wait("!document.querySelector('#screen-review').hidden")
    page.js("(() => { const c = document.querySelector('#crop-toggle'); c.checked = false;"
            " c.dispatchEvent(new Event('change')); })()")
    study_id = _shown(page)
    record = _api(base, f"/api/studies/{study_id}")
    mlo = next(image for image in record["images"] if image["view_key"] == "L-MLO")
    nipple = mlo["measures"]["mask"]["from"]
    original = page.js(_SHOWN_VALUES)
    assert not original["reset"]

    # the handles start on the model's own points: measured as they start, every view gives the model's number
    seeded = page.js("""import('/static/js/pnl.js').then((pnl) => fetch('/api/studies/' + location.hash.slice(1))
      .then((r) => r.json()).then((s) => s.images.map((i) => [i.measures.mask.mm,
        pnl.measureManual(i.view, pnl.seedManual(i), i.mm_per_px).mm])))""")
    assert len(seeded) == 2 and all(model == pytest.approx(page_mm, abs=1e-9) for model, page_mm in seeded)

    page.click("#adjust-toggle")
    assert page.js("document.querySelector('#adjust-toggle').getAttribute('aria-pressed')") == "true"
    start = _screen_point(page, "L-MLO", nipple)
    _mouse(page, "mousePressed", start, 1)            # a press that does not move the handle places nothing
    _mouse(page, "mouseMoved", (start[0] + 2, start[1]), 1)
    _mouse(page, "mouseReleased", (start[0] + 2, start[1]), 0)
    time.sleep(0.3)
    assert _api(base, f"/api/studies/{study_id}") == record and page.js(_SHOWN_VALUES)["tiles"] == original["tiles"]
    _mouse(page, "mousePressed", start, 1)
    _mouse(page, "mouseMoved", (start[0] - 12, start[1] + 4), 1)
    first = page.js(_SHOWN_VALUES)
    end = (start[0] - 30, start[1] + 10)
    _mouse(page, "mouseMoved", end, 1)
    live = page.js(_SHOWN_VALUES)
    assert first["tiles"]["L-MLO"][1] != original["tiles"]["L-MLO"][1] != live["tiles"]["L-MLO"][1]
    assert live["tiles"]["L-MLO"][1].endswith(" manual") and live["tiles"]["L-MLO"][0] == original["tiles"]["L-MLO"][0]
    assert live["cards"][1]["pairs"][0] == live["tiles"]["L-MLO"][1]   # the segmentation card follows the tile
    assert live["cards"][1]["delta"] != original["cards"][1]["delta"] and live["cards"][0] == original["cards"][0]
    assert _api(base, f"/api/studies/{study_id}") == record           # nothing is sent while the button is down
    _mouse(page, "mouseReleased", end, 0)

    page.wait(f"""fetch('/api/studies/{study_id}').then((r) => r.json())
      .then((s) => s.images.some((i) => i.view_key === 'L-MLO' && i.manual))""")
    page.wait("[...document.querySelectorAll('.tile-badge')].some((node) => node.textContent === 'PNL manual')")
    saved = page.js(_SHOWN_VALUES)
    assert saved["tiles"] == live["tiles"] and saved["cards"] == live["cards"] and saved["reset"]
    stored = _api(base, f"/api/studies/{study_id}")
    placed = next(image for image in stored["images"] if image["view_key"] == "L-MLO")
    assert placed["measures"]["mask"]["manual"]
    assert placed["measures"]["mask"]["from"] == pytest.approx(placed["manual"]["nipple"], abs=0.051)
    assert placed["manual"]["line"] == mlo["measures"]["mask"]["line"]  # only the nipple was moved
    assert abs(placed["manual"]["nipple"][0] - (nipple[0] - 30 * 640 / page.js(
        "document.querySelector('#quad canvas').getBoundingClientRect().width"))) < 2

    page.js("location.reload()")
    page.wait("!document.querySelector('#screen-review').hidden"
              " && [...document.querySelectorAll('.tile-badge')].some((node) => node.textContent === 'PNL manual')")
    reloaded = page.js(_SHOWN_VALUES)
    assert reloaded["tiles"] == saved["tiles"] and reloaded["cards"] == saved["cards"]

    page.click("#adjust-reset")
    page.wait(f"""fetch('/api/studies/{study_id}').then((r) => r.json())
      .then((s) => !s.images.some((i) => i.manual))""")
    page.wait("document.querySelector('#adjust-reset').hidden")
    back = page.js(_SHOWN_VALUES)
    assert back["tiles"] == original["tiles"] and back["cards"] == original["cards"]
    assert "PNL manual" not in back["badges"]


def _double_click(page, at):
    """Two left clicks at one spot, as fast as a reader's double-click: the browser fires ``dblclick`` on the second."""
    for count in (1, 2):
        for kind in ("mousePressed", "mouseReleased"):
            page.call("Input.dispatchMouseEvent", type=kind, x=at[0], y=at[1], button="left",
                      buttons=1 if kind == "mousePressed" else 0, clickCount=count)


def test_a_double_click_that_places_and_grabs_a_point_keeps_the_zoom(page, tmp_path):
    page.choose("#file-input", [_study_file(tmp_path, "L_MLO.dcm", "1.2.3.1"),
                                _study_file(tmp_path, "L_CC.dcm", "1.2.3.1", view="CC")])
    page.wait("!document.querySelector('#screen-review').hidden")
    page.click("#open-editor")
    page.wait("!document.querySelector('#editor').hidden")
    level = "document.querySelector('#editor-zoom-level').textContent"
    dirty = "!document.querySelector('#editor-dirty').hidden"
    fitted = page.js(level)
    for _ in range(3):
        page.click("#editor-zoom-in")
    zoomed = page.js(level)
    assert zoomed != fitted
    centre = page.js("(() => { const r = document.querySelector('#editor-stage').getBoundingClientRect();"
                     " return [Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2)]; })()")
    page.js("document.dispatchEvent(new KeyboardEvent('keydown', { key: 'n', bubbles: true }))")   # the point tool
    _double_click(page, centre)                       # place the point, then grab it again at once to move it
    assert page.js(level) == zoomed and page.js(dirty)
    page.js("document.dispatchEvent(new KeyboardEvent('keydown', { key: 'v', bubbles: true }))")
    _double_click(page, centre)                       # grab the point twice with the select tool: still no re-fit
    assert page.js(level) == zoomed
    _double_click(page, (centre[0] + 150, centre[1]))  # a double-click away from the point edits nothing: it fits
    assert page.js(level) == fitted


def test_a_vertex_dragged_in_the_editor_is_measured_live_and_leaving_unsaved_stores_nothing(page, served, tmp_path):
    """Dragging a pectoral vertex sends the editor's labels to POST .../preview while nothing is saved; going back
    without saving keeps the record."""
    base, app = served
    service = app.state.service
    previews, preview_labels = [], service.preview_labels

    def recorded(study_id, payload):
        previews.append(payload)
        return preview_labels(study_id, payload)

    service.preview_labels = recorded
    page.choose("#file-input", [_study_file(tmp_path, "L_MLO.dcm", "1.2.3.1"),
                                _study_file(tmp_path, "L_CC.dcm", "1.2.3.1", view="CC")])
    page.wait("!document.querySelector('#screen-review').hidden")
    study_id = _shown(page)
    record = _api(base, f"/api/studies/{study_id}")
    mlo = next(image for image in record["images"] if image["view"] == "MLO")
    points = mlo["polygons"]["pectoral"]["points"]
    index, vertex = max(enumerate(points), key=lambda item: min(item[1][0], item[1][1], 640 - item[1][0],
                                                               640 - item[1][1]))
    page.click("#open-editor")
    page.wait("!document.querySelector('#editor').hidden")
    page.js(f"[...document.querySelectorAll('#editor-views button')]"
            f".find((b) => b.textContent.includes({json.dumps(mlo['view_key'])})).click()")
    pectoral = page.js("""(() => {
      const node = [...document.querySelectorAll('#editor-classes *')]
        .find((n) => n.children.length === 0 && /^Pectoral muscle/.test(n.textContent.trim()));
      const r = node.getBoundingClientRect();
      return [r.left + r.width / 2, r.top + r.height / 2];
    })()""")
    _mouse(page, "mousePressed", pectoral, 1)
    _mouse(page, "mouseReleased", pectoral, 0)
    scale = page.js("document.querySelector('#editor-canvas').getBoundingClientRect().width / 640")
    origin = page.js("(() => { const r = document.querySelector('#editor-canvas').getBoundingClientRect();"
                     " return [r.left, r.top]; })()")
    start = (origin[0] + vertex[0] * scale, origin[1] + vertex[1] * scale)
    _mouse(page, "mousePressed", start, 1)
    for step in range(1, 5):
        _mouse(page, "mouseMoved", (start[0] + 5 * step, start[1]), 1)
        time.sleep(0.05)
    _mouse(page, "mouseReleased", (start[0] + 20, start[1]), 0)
    deadline = time.time() + 15
    while time.time() < deadline and not any(
            entry["images"][0]["polygons"]["pectoral"]["points"][index][0] > vertex[0] + 10 for entry in previews):
        time.sleep(0.1)
    sent = [entry["images"][0] for entry in previews]
    assert sent and all(entry["view_key"] == mlo["view_key"] for entry in sent)
    moved = sent[-1]["polygons"]["pectoral"]["points"][index]
    assert moved[0] == pytest.approx(vertex[0] + 20 / scale, abs=1.0) and moved[1] == pytest.approx(vertex[1], abs=1.0)
    assert page.js("!document.querySelector('#editor-dirty').hidden")
    assert _api(base, f"/api/studies/{study_id}") == record          # a preview stores nothing

    page.js("window.confirm = () => true")                           # leave and discard the unsaved edit
    page.click("#editor-back")
    page.wait("document.querySelector('#editor').hidden")
    assert _api(base, f"/api/studies/{study_id}") == record


def test_a_dataset_whose_study_uids_cannot_be_read_opens_and_is_analysed(page, tmp_path):
    uid = "2d8d39b8b1de7d9701b4e2d024395fb5"           # VinDr-Mammo's study UIDs are hex strings, not DICOM UIDs
    page.open_dataset([_study_file(tmp_path, "L_MLO.dcm", uid), _study_file(tmp_path, "L_CC.dcm", uid, view="CC")])
    state = page.settled()
    assert state["screen"] == "review" and state["banner"] == "" and len(state["options"]) == 1


def test_all_the_labels_of_a_dataset_are_downloaded_in_one_file(page, tmp_path):
    files = [_study_file(tmp_path, f"{name}.dcm", f"1.2.3.{index}")
             for index, name in enumerate(["a_one", "b_two", "c_three"], start=1)]
    page.open_dataset(files)
    page.settled()
    page.js("""(() => { window.__saved = null;
      HTMLAnchorElement.prototype.click = function () {    // read the file before saveBlob revokes its URL
        const name = this.download;
        fetch(this.href).then((r) => r.text()).then((text) => { window.__saved = { name, body: JSON.parse(text) }; });
      };
    })()""")
    page.click("#dataset-download")
    saved = page.wait("window.__saved")
    body = saved["body"]
    assert page.wait("!document.querySelector('#dataset-download').disabled")
    assert page.js("document.querySelector('#dataset-download').textContent").startswith("All labels")
    assert saved["name"].startswith("labels-") and saved["name"].endswith(".json")
    assert [entry["index"] for entry in body["studies"]] == [1, 2, 3]
    names = page.js("[...document.querySelectorAll('#dataset-pick option')].map((o) => o.textContent)")
    assert all(entry["name"] in names[entry["index"] - 1] for entry in body["studies"])
    first = body["studies"][0]
    assert first["id"] and [image["view_key"] for image in first["labels"]["images"]] == ["L-MLO"]
    assert all((entry["labels"] is None) == (entry["id"] is None) for entry in body["studies"])


def test_a_placement_let_go_while_another_study_is_saved_is_sent_too(page, served, tmp_path):
    """A placement on one study whose answer is still on its way when a second study is opened and adjusted: both are
    stored (the queue of the second is not dropped when the first answer comes back for a study no longer shown)."""
    base, app = served
    first = _upload(base, _study_file(tmp_path, "A_MLO.dcm", "1.2.3.1"), _study_file(tmp_path, "A_CC.dcm", "1.2.3.1",
                                                                                     view="CC"))
    second = _upload(base, _study_file(tmp_path, "B_MLO.dcm", "1.2.3.2"), _study_file(tmp_path, "B_CC.dcm", "1.2.3.2",
                                                                                      view="CC"))
    service = app.state.service
    put_manual = service.put_manual
    service.put_manual = lambda study_id, payload: (time.sleep(1.5) if study_id == first["id"] else None) or put_manual(
        study_id, payload)

    def drag_nipple(record):
        page.js(f"location.hash = '#{record['id']}'")
        page.wait(f"location.hash === '#{record['id']}' && !document.querySelector('#screen-review').hidden"
                  " && document.querySelectorAll('#quad canvas').length > 0")
        time.sleep(0.3)
        mlo = next(image for image in record["images"] if image["view"] == "MLO")
        start = _screen_point(page, mlo["view_key"], mlo["measures"]["mask"]["from"])
        _mouse(page, "mousePressed", start, 1)
        _mouse(page, "mouseMoved", (start[0] - 20, start[1] + 6), 1)
        _mouse(page, "mouseReleased", (start[0] - 20, start[1] + 6), 0)

    page.js(f"location.hash = '#{first['id']}'")
    page.wait("!document.querySelector('#screen-review').hidden")
    page.js("(() => { const c = document.querySelector('#crop-toggle'); c.checked = false;"
            " c.dispatchEvent(new Event('change')); })()")
    page.click("#adjust-toggle")
    drag_nipple(first)                                 # its answer takes 1.5 s
    drag_nipple(second)                                # queued while the first is in flight
    for record in (first, second):
        page.wait(f"""fetch('/api/studies/{record['id']}').then((r) => r.json())
          .then((s) => s.images.some((i) => i.view === 'MLO' && i.manual))""", timeout=15)
    page.wait("[...document.querySelectorAll('.tile-badge')].some((node) => node.textContent === 'PNL manual')")
    assert _shown(page) == second["id"]


def _adjusting(page, files):
    """``files`` opened as one study, shown whole (no crop), in "Adjust PNL"; returns the stored record."""
    page.choose("#file-input", files)
    page.wait("!document.querySelector('#screen-review').hidden && document.querySelectorAll('#quad canvas').length > 0")
    page.js("(() => { const c = document.querySelector('#crop-toggle'); c.checked = false;"
            " c.dispatchEvent(new Event('change')); })()")
    page.click("#adjust-toggle")
    page.js("window.__errors = []; window.addEventListener('error', (e) => window.__errors.push(String(e.message)));")
    return _shown(page)


def test_a_placement_saved_while_another_handle_is_held_stays_where_it_was_put(page, served, tmp_path):
    """The L-MLO nipple is let go (its answer is slow) and an L-CC handle is grabbed and moved while the answer comes
    in: the L-MLO number stays where it was put, not the model's, until the record is shown on release."""
    base, app = served
    service = app.state.service
    put_manual = service.put_manual
    calls = []

    def slow(study_id, payload):
        calls.append(payload)
        if len(calls) == 1:
            time.sleep(1.0)
        return put_manual(study_id, payload)

    service.put_manual = slow
    study_id = _adjusting(page, [_study_file(tmp_path, "L_MLO.dcm", "1.2.3.1"),
                                 _study_file(tmp_path, "L_CC.dcm", "1.2.3.1", view="CC")])
    record = _api(base, f"/api/studies/{study_id}")
    mlo = next(image for image in record["images"] if image["view_key"] == "L-MLO")
    cc = next(image for image in record["images"] if image["view_key"] == "L-CC")
    original = page.js(_SHOWN_VALUES)

    start = _screen_point(page, "L-MLO", mlo["measures"]["mask"]["from"])
    _mouse(page, "mousePressed", start, 1)
    _mouse(page, "mouseMoved", (start[0] - 30, start[1] + 10), 1)
    _mouse(page, "mouseReleased", (start[0] - 30, start[1] + 10), 0)
    dragged = page.js(_SHOWN_VALUES)["tiles"]["L-MLO"]
    assert dragged != original["tiles"]["L-MLO"]

    held = _screen_point(page, "L-CC", cc["measures"]["mask"]["from"])
    _mouse(page, "mousePressed", held, 1)              # an L-CC handle held while the L-MLO answer comes in
    _mouse(page, "mouseMoved", (held[0] + 1, held[1]), 1)
    page.wait(f"""fetch('/api/studies/{study_id}').then((r) => r.json())
      .then((s) => s.images.some((i) => i.view_key === 'L-MLO' && i.manual))""", timeout=15)
    time.sleep(0.5)                                    # the answer reaches the page while the handle is still held
    _mouse(page, "mouseMoved", (held[0] + 2, held[1]), 1)
    assert page.js(_SHOWN_VALUES)["tiles"]["L-MLO"] == dragged
    _mouse(page, "mouseReleased", (held[0] + 2, held[1]), 0)   # 2 px: a click, nothing placed on L-CC

    page.wait("[...document.querySelectorAll('.tile-badge')].some((node) => node.textContent === 'PNL manual')")
    shown = page.js(_SHOWN_VALUES)
    assert shown["tiles"]["L-MLO"] == dragged and shown["tiles"]["L-CC"] == original["tiles"]["L-CC"]
    assert len(calls) == 1 and page.js("window.__errors") == []
    stored = _api(base, f"/api/studies/{study_id}")
    assert not next(image for image in stored["images"] if image["view_key"] == "L-CC").get("manual")


def test_a_handle_held_when_the_study_changes_places_nothing_on_the_next_one(page, served, tmp_path):
    base, _ = served
    first = _upload(base, _study_file(tmp_path, "A_MLO.dcm", "1.2.3.1"), _study_file(tmp_path, "A_CC.dcm", "1.2.3.1",
                                                                                     view="CC"))
    second = _upload(base, _study_file(tmp_path, "B_MLO.dcm", "1.2.3.2"), _study_file(tmp_path, "B_CC.dcm", "1.2.3.2",
                                                                                      view="CC"))

    def show(record):
        page.js(f"location.hash = '#{record['id']}'")
        page.wait(f"location.hash === '#{record['id']}' && !document.querySelector('#screen-review').hidden"
                  " && document.querySelectorAll('#quad canvas').length > 0")
        time.sleep(0.3)

    def nipple_of(record):
        mlo = next(image for image in record["images"] if image["view"] == "MLO")
        return _screen_point(page, mlo["view_key"], mlo["measures"]["mask"]["from"])

    show(first)
    page.js("(() => { const c = document.querySelector('#crop-toggle'); c.checked = false;"
            " c.dispatchEvent(new Event('change')); })()")
    page.click("#adjust-toggle")
    page.js("window.__errors = []; window.addEventListener('error', (e) => window.__errors.push(String(e.message)));")
    start = nipple_of(first)
    _mouse(page, "mousePressed", start, 1)
    _mouse(page, "mouseMoved", (start[0] - 10, start[1]), 1)
    show(second)                                       # the study changes under the held handle
    _mouse(page, "mouseMoved", (start[0] - 30, start[1] + 5), 1)
    _mouse(page, "mouseReleased", (start[0] - 30, start[1] + 5), 0)
    time.sleep(0.8)
    for record in (first, second):
        assert not any(image.get("manual") for image in _api(base, f"/api/studies/{record['id']}")["images"])
    assert page.js("window.__errors") == []

    start = nipple_of(second)                          # and a handle of the study now shown is placed as usual
    _mouse(page, "mousePressed", start, 1)
    _mouse(page, "mouseMoved", (start[0] - 20, start[1] + 6), 1)
    _mouse(page, "mouseReleased", (start[0] - 20, start[1] + 6), 0)
    page.wait(f"""fetch('/api/studies/{second['id']}').then((r) => r.json())
      .then((s) => s.images.some((i) => i.view === 'MLO' && i.manual))""", timeout=15)
    page.wait("[...document.querySelectorAll('.tile-badge')].some((node) => node.textContent === 'PNL manual')")


def test_a_handle_held_when_the_page_turns_places_nothing(page, served, tmp_path):
    base, _ = served
    study_id = _adjusting(page, [_study_file(tmp_path, "L_MLO.dcm", "1.2.3.1"),
                                 _study_file(tmp_path, "L_CC.dcm", "1.2.3.1", view="CC")])
    page.js("document.querySelector('.cell[data-cols=\"1\"][data-rows=\"1\"]').click()")   # one view per page
    page.wait("document.querySelectorAll('#quad canvas').length === 1")
    time.sleep(0.3)
    key = page.js("document.querySelector('#quad .tile-key').textContent.trim()")
    record = _api(base, f"/api/studies/{study_id}")
    image = next(image for image in record["images"] if image["view_key"] == key)
    start = _screen_point(page, key, image["measures"]["mask"]["from"])
    _mouse(page, "mousePressed", start, 1)
    _mouse(page, "mouseMoved", (start[0] - 10, start[1]), 1)
    turn = "ArrowRight" if page.js("document.querySelector('#page-prev').disabled") else "ArrowLeft"
    page.js(f"document.dispatchEvent(new KeyboardEvent('keydown', {{ key: '{turn}', bubbles: true }}))")
    page.wait(f"document.querySelector('#quad .tile-key').textContent.trim() !== {json.dumps(key)}")
    _mouse(page, "mouseMoved", (start[0] - 30, start[1] + 5), 1)
    _mouse(page, "mouseReleased", (start[0] - 30, start[1] + 5), 0)
    time.sleep(0.8)
    assert not any(image.get("manual") for image in _api(base, f"/api/studies/{study_id}")["images"])
    assert page.js("window.__errors") == []
    assert "PNL manual" not in page.js(_SHOWN_VALUES)["badges"]


def test_a_label_export_stops_fetching_when_its_dataset_is_left(page, served, tmp_path):
    base, app = served
    service = app.state.service
    labels = service.labels
    calls = []

    def slow(study_id):
        calls.append(study_id)
        time.sleep(0.8)
        return labels(study_id)

    files = [_study_file(tmp_path, f"s{index:02d}.dcm", f"1.2.3.{index}") for index in range(1, 11)]
    page.open_dataset(files)
    page.settled()
    for index in range(1, 10):                        # every study opened once, so every one has labels to fetch
        page.click("#dataset-next")
        page.wait(f"document.querySelector('#dataset-pick').value === '{index}'")
        page.settled()
    _wait_total(base, 10, timeout=30)
    service.labels = slow
    page.click("#dataset-download")
    deadline = time.time() + 10
    while not calls and time.time() < deadline:
        time.sleep(0.05)
    assert calls
    page.choose("#file-input", [_study_file(tmp_path, "single.dcm", "1.2.3.99")])   # leaves the dataset
    page.wait("!document.querySelector('#screen-review').hidden && document.querySelector('#dataset-nav').hidden")
    time.sleep(0.2)
    fetched = len(calls)
    time.sleep(2.5)
    assert len(calls) == fetched < 10
    assert page.js("document.querySelector('#dataset-download').textContent").startswith("All labels")
