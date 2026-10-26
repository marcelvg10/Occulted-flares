"""Install a reversible Windows DLL-path workaround in stix-research only."""
import json
import sys
from pathlib import Path

root = Path(sys.prefix)
if root.name != "stix-research" or sys.platform != "win32":
    raise SystemExit("Run with the Windows stix-research interpreter")
site = root / "Lib" / "site-packages"
helper = site / "_stix_conda_dll_paths.py"
helper.write_text('''# VS Code may launch Conda Python without activating the environment.
import os
import sys
from pathlib import Path

root = Path(sys.prefix)
directories = [root, root / "Library" / "mingw-w64" / "bin",
               root / "Library" / "usr" / "bin", root / "Library" / "bin",
               root / "Scripts", root / "bin"]
directories = [str(p) for p in directories if p.is_dir()]
existing = os.environ.get("PATH", "").split(os.pathsep)
os.environ["PATH"] = os.pathsep.join(directories + [
    p for p in existing if os.path.normcase(p) not in {
        os.path.normcase(d) for d in directories}])
# Retain handles: closing them removes the DLL search directories.
_dll_handles = [os.add_dll_directory(p) for p in directories]
''', encoding="utf-8")
(site / "stix_conda_dll_paths.pth").write_text(
    "import _stix_conda_dll_paths\n", encoding="utf-8")

# Remove the captured Codex process PATH from the earlier kernelspec.
from jupyter_core.paths import jupyter_data_dir
spec = Path(jupyter_data_dir()) / "kernels" / "stix-research" / "kernel.json"
if spec.is_file():
    data = json.loads(spec.read_text(encoding="utf-8"))
    data.pop("env", None)
    spec.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"Installed DLL-path startup helper in {site}")
print("To undo: remove stix_conda_dll_paths.pth and _stix_conda_dll_paths.py there.")
