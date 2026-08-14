@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

echo === Building native bwcore ===
python tools\ensure_native.py
if errorlevel 1 exit /b 1

echo === Freezing Brain Workshop ===
cxfreeze brainworkshop.py --target-dir dist --include-files=res --include-modules=pyglet.resource,pyglet.clock,pyglet.graphics,pyglet.sprite,bwcore,bwaccel
if errorlevel 1 exit /b 1

echo === Copying bwcore into dist ===
for %%F in (bwcore*.pyd bwcore*.so) do (
    if exist "%%F" copy /y "%%F" dist\ >nul
)

python -c "import glob,os,sys; hits=[os.path.join(r,f) for r,ds,fs in os.walk('dist') for f in fs if f.startswith('bwcore')]; print('bwcore in dist:', hits); sys.exit(0 if hits else 1)"
if errorlevel 1 (
    echo ERROR: frozen dist is missing bwcore — players would need a compiler.
    exit /b 1
)

echo Freeze complete, native module bundled.
exit /b 0
