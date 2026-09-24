#!/usr/bin/env python3
"""
mutants.py -- the mutation runner every campaign shares (tools/mutants_p15.py, tools/mutants_p21.py, ...).

A campaign is a list of mutants, each one deliberate defect:
    (name, source file under src/, old text, new text [, old text 2, new text 2 ...])
The tree is copied to build/mutants/, each mutant applied in turn (every `old` must occur exactly once --
otherwise the mutant is reported INVALID, not silently skipped), the campaign's suites built and run in
QUICK mode; the mutant is KILLED if a suite fails. A survivor is a test gap -- closed, and the mutant run
again -- or EQUIVALENT, with the reason written in COMMANDMENTS.md. Some defects can only show on
Windows (an allocation Linux never makes): those mutants are named in `wine_only` and, if Linux does not
kill them, run under Wine.

Usage (from a campaign file):  run(M, suites, by_file={...}, wine_only={...})   and on the command line
    tools/mutants_pNN.py [name-prefix ...]
"""
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(REPO, "build", "mutants")


def sh(cmd, cwd, timeout=600):
    try:
        p = subprocess.run(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def first_fail(out):
    lines = [l.strip() for l in out.replace("\r", "").splitlines() if "FAIL [" in l and "planted failure" not in l]
    return lines[0][:110] if lines else None


def run(M, suites, by_file=None, wine_only=None, argv=None):
    """M: the mutants; suites: the tests every mutant runs; by_file: {source file: extra suites};
    wine_only: {mutant name prefix: suite to run under Wine if Linux does not kill it}"""
    by_file = by_file or {}
    wine_only = wine_only or {}
    only = sys.argv[1:] if argv is None else argv
    every = sorted(set(suites) | {t for ts in by_file.values() for t in ts})
    if os.path.exists(W): shutil.rmtree(W)
    os.makedirs(W)
    for d in ("src", "test"):
        shutil.copytree(os.path.join(REPO, d), os.path.join(W, d), ignore=shutil.ignore_patterns("data"))
    os.symlink(os.path.join(REPO, "test", "data"), os.path.join(W, "test", "data"))
    shutil.copy(os.path.join(REPO, "Makefile"), W)
    targets = " ".join("build/linux/" + t for t in every)
    rc, out = sh("make -s -j8 " + targets, W)
    assert rc == 0, out
    killed = survived = invalid = 0
    for m in M:
        name, f, edits = m[0], m[1], list(zip(m[2::2], m[3::2]))
        tag = name.split()[0]
        if only and not any(tag == o or name.startswith(o + " ") for o in only): continue
        path = os.path.join(W, "src", f)
        orig = open(path).read()
        text, bad = orig, None
        for old, new in edits:
            if text.count(old) != 1: bad = text.count(old); break
            text = text.replace(old, new)
        if bad is not None:
            print("%-60s PATTERN FOUND %d TIMES" % (name, bad)); invalid += 1; continue
        open(path, "w").write(text)
        rc, out = sh("make -s " + targets, W)
        if rc != 0:
            print("%-60s DID NOT COMPILE" % name); print(out[-800:]); invalid += 1
        else:
            fails = []
            for t in list(suites) + by_file.get(f, []):
                r, o = sh("ENGRAM_TEST_QUICK=1 ./" + t, os.path.join(W, "build/linux"), timeout=600)
                if r != 0: fails.append("%s(%s)" % (t, first_fail(o) or "exit %d" % r))
            if not fails and tag in wine_only:
                t = wine_only[tag]
                if sh("make -s build/win/%s.exe" % t, W)[0] == 0:
                    r, o = sh("WINEPREFIX=%s WINEDEBUG=-all WINEDLLOVERRIDES='mscoree,mshtml=' ENGRAM_TEST_QUICK=1 wine ./%s.exe"
                              % (os.path.join(REPO, "build", "wineprefix"), t), os.path.join(W, "build/win"), timeout=900)
                    if r != 0: fails.append("wine %s(%s)" % (t, first_fail(o) or "exit %d" % r))
            if fails: killed += 1; print("%-60s KILLED   %s" % (name, fails[0]))
            else: survived += 1; print("%-60s SURVIVED" % name)
        open(path, "w").write(orig)
        sys.stdout.flush()
    print("\n%d killed, %d survived, %d invalid, of %d" % (killed, survived, invalid, len(M)))
