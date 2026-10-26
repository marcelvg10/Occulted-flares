"""Execute every event notebook in a fresh local kernel and retain outputs.

Run with Python from the stix-research Conda environment. This local kernel specification is confined to
the workspace cache; no system/user kernel registration is changed. Default
notebook cells use local data and keep network/selection/export switches off.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--name", help="Validate one exact notebook filename; write a separate validation record")
    parser.add_argument("--timeout", type=int, default=240, help="Seconds per cell; increase for multistart spectral fits")
    args = parser.parse_args()
    work = Path(__file__).resolve().parent.parent
    directory = args.directory or work/"SOL2026-07-07/Geometry/notebooks"
    directory = directory.resolve()
    # Default notebook execution must leave original inputs and existing
    # exports unchanged. Notebook files and small runtime caches are excluded.
    event = directory.parent.parent
    if directory.name != "notebooks" or not (event/"data_manifest.json").is_file():
        raise ValueError("Expected an event stage/notebooks directory with an event data manifest")
    geometry = event/"Geometry"
    def snapshot():
        paths = list(geometry.glob("*.json"))
        for folder in ("Stereo/fits", "selections", "ephemeris", "outputs"):
            paths.extend(p for p in (geometry/folder).rglob("*") if p.is_file())
        manifest = event/"data_manifest.json"
        paths.append(manifest)
        for record in json.loads(manifest.read_text())["files"]:
            target = (event/record["path"]).resolve()
            if not target.is_relative_to(event.resolve()):
                raise ValueError("Manifest path escapes event")
            paths.append(target)
        paths.extend(directory.parent.glob("*.json"))
        # Protect this stage's saved reference exports and IDL drivers too.
        # Light-curve notebooks also read the earlier Instrument SAV files.
        for stage in {directory.parent, event/"Instrument"}:
            for folder in ("outputs", "idl"):
                paths.extend(p for p in (stage/folder).rglob("*") if p.is_file())
        checksums = {}
        for p in set(paths):
            with p.open("rb") as stream:
                checksums[str(p)] = hashlib.file_digest(stream, "sha256").hexdigest()
        return checksums
    original_files = snapshot()
    cache = work/".cache-geometry/jupyter"
    for name in ("config", "runtime", "data"):
        (cache/name).mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_CONFIG_DIR"] = str(cache/"config")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(cache/"runtime")
    os.environ["JUPYTER_DATA_DIR"] = str(cache/"data")
    os.environ["IPYTHONDIR"] = str(cache/"ipython")
    from occulted_flares.geometry.geometry_runtime import configure_cache
    configure_cache()
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client import KernelManager

    kernel_root = cache/"kernels"
    kernel = kernel_root/"stix-research-local"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel/"kernel.json").write_text(json.dumps(dict(
        argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        display_name="Python (stix-research)", language="python")))
    records = []
    if args.name and (Path(args.name).name != args.name or not args.name.endswith(".ipynb")):
        raise ValueError("Use one notebook basename ending in .ipynb")
    paths = [directory/args.name] if args.name else sorted(directory.glob("*.ipynb"))
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="stix-research-local",
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
        print("Executing from fresh kernel:", path.name, flush=True)
        client = NotebookClient(notebook, km=manager, timeout=args.timeout,
                                resources={"metadata": {"path": str(directory.resolve())}},
                                allow_errors=False)
        try:
            client.execute()
        finally:
            # Supplying our own manager transfers cleanup responsibility here.
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in code_cells)
        assert not any(o.output_type == "error" for cell in code_cells for o in cell.outputs)
        figures = sum("image/png" in o.get("data", {}) for cell in code_cells for o in cell.outputs)
        if not path.name.startswith("00_") and not figures:
            raise AssertionError(f"No inline figures rendered in {path.name}")
        # Persist the registered research kernel rather than the temporary validation kernel.
        notebook.metadata.kernelspec = dict(name="stix-research",
            display_name="Python (stix-research)", language="python")
        nbformat.write(notebook, path)
        records.append(dict(notebook=path.name, code_cells=len(code_cells), inline_figures=figures, status="passed"))
        print(f"Passed: {len(code_cells)} code cells; {figures} inline figures", flush=True)
    if not records:
        raise ValueError("No notebooks found")
    if snapshot() != original_files:
        raise AssertionError("Default notebook execution changed inputs or existing exports")
    record_name = "validation_"+Path(args.name).stem+".json" if args.name else "validation.json"
    record_directory = work/"docs/validation/notebooks"/event.name/directory.parent.name
    record_directory.mkdir(parents=True, exist_ok=True)
    (record_directory/record_name).write_text(json.dumps(dict(
        interpreter=sys.executable, method="Fresh kernel per notebook; all default cells; allow_errors=False",
        scope="Selected notebook only" if args.name else "All notebooks",
        protected_input_and_export_files_unchanged=True,
        excluded="Live network query/download, human widget interaction, accepting points and optional exports are off by default",
        notebooks=records), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
