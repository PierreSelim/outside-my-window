# PyInstaller spec for the Dash sidecar shipped inside the Electron app.
# Build with:  npm run build:server   (output: sidecar/omw-server/)

from PyInstaller.utils.hooks import collect_all

datas = [
    ("data/stations.json", "data"),
    ("data/resources.json", "data"),
    ("assets", "assets"),
]
binaries = []
hiddenimports = []

# Dash and Plotly ship templates, JS bundles and package metadata that no import graph reveals.
for package in ("dash", "plotly", "polars", "narwhals", "waitress"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    ["server.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["mypy", "pytest", "ruff", "IPython", "tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="omw-server",
    console=False,
    strip=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="omw-server",
)
