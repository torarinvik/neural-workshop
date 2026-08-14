#!/usr/bin/env python
"""Build bwcore if needed and refuse to continue without it.

Used by freeze.bat / freeze.sh so a packaged build never silently
ships the Python fallback. Source players can still run without compiling.
"""
from __future__ import print_function

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _has_bwcore():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    sys.modules.pop('bwcore', None)
    try:
        import bwcore  # noqa: F401
        return True
    except ImportError:
        return False


def _built_artifacts():
    patterns = [
        os.path.join(ROOT, 'bwcore*.pyd'),
        os.path.join(ROOT, 'bwcore*.so'),
        os.path.join(ROOT, 'bwcore*.dll'),
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat))
    return found


def main():
    os.chdir(ROOT)
    if not _has_bwcore():
        print('bwcore missing; compiling with setup.py build_ext --inplace')
        rc = subprocess.call(
            [sys.executable, 'setup.py', 'build_ext', '--inplace'],
            cwd=ROOT)
        if rc != 0:
            print('ERROR: native bwcore failed to compile', file=sys.stderr)
            return rc
    if not _has_bwcore():
        print('ERROR: bwcore still not importable after compile', file=sys.stderr)
        return 1
    import bwcore
    print('bwcore ready (%s) artifacts=%s' % (
        getattr(bwcore, 'backend', lambda: 'C')(),
        _built_artifacts()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
