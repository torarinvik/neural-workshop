#!/bin/sh
# Compile C kernels and freeze, failing if bwcore is not in the dist.
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python}

echo "=== Building native bwcore ==="
"$PYTHON" tools/ensure_native.py

echo "=== Freezing Brain Workshop ==="
cxfreeze brainworkshop.py --target-dir dist --include-files=res --include-modules=pyglet.resource,pyglet.clock,pyglet.graphics,pyglet.sprite,bwcore,bwaccel

echo "=== Copying bwcore into dist ==="
for f in bwcore*.so bwcore*.pyd; do
    [ -f "$f" ] && cp -f "$f" dist/
done

"$PYTHON" -c "
import glob, os, sys
hits = [os.path.join(r, f)
        for r, ds, fs in os.walk('dist')
        for f in fs if f.startswith('bwcore')]
print('bwcore in dist:', hits)
sys.exit(0 if hits else 1)
"

echo "Freeze complete, native module bundled."
