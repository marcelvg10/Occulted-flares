"""Shared runtime configuration for geometry modules."""
import os
from pathlib import Path
import sys
import json


def default_geometry_config():
    """Event defaults for VS Code Run Python File; calculations remain generic."""
    settings = workspace_root() / "config/default_event.json"
    if not settings.is_file():
        raise FileNotFoundError(
            "No private event configuration is bundled. Supply --config where "
            "supported, or create config/default_event.json with a geometry_config "
            "path pointing to your own selection_groups.json."
        )
    data = json.loads(settings.read_text(encoding="utf-8"))
    return (settings.parent / data["geometry_config"]).resolve()


def default_output(config_path, product):
    """One stable output folder per product; generated files are replaced."""
    return config_path.resolve().parent / "outputs" / product


def configure_gui():
    """Use Tcl/Tk scripts belonging to the active Windows Python runtime.

    A venv shares its base interpreter's DLLs. Inherited TCL_LIBRARY settings
    from another Python can load incompatible scripts. Change only this process.
    """
    if sys.platform != "win32":
        return
    import _tkinter
    root = Path(sys.base_prefix) / "tcl"
    for variable, directory, filename in (
        ("TCL_LIBRARY", f"tcl{_tkinter.TCL_VERSION}", "init.tcl"),
        ("TK_LIBRARY", f"tk{_tkinter.TK_VERSION}", "tk.tcl"),
    ):
        library = root / directory
        if not (library / filename).is_file():
            raise RuntimeError(f"Active Python lacks {library / filename}; install matching Tcl/Tk or use --points.")
        os.environ[variable] = str(library)


def configure_cache():
    """Keep library configuration and caches in Work, not the user profile."""
    cache = workspace_root() / ".cache-geometry"
    for variable, name in (("SUNPY_CONFIGDIR", "sunpy"), ("MPLCONFIGDIR", "matplotlib")):
        os.environ.setdefault(variable, str(cache / name))
    import sunpy
    import sunpy.util.config as config_module
    sunpy.config.set("downloads", "remote_data_manager_dir", str(cache / "data_manager"))
    config_module.CACHE_DIR = str(cache / "downloads")


def workspace_root():
    """Find a source checkout, without embedding one user's Windows path."""
    override = os.environ.get("OCCULTED_FLARES_WORK")
    candidates = [Path(override)] if override else []
    candidates += list(Path(__file__).resolve().parents)
    candidates += [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/occulted_flares").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Run inside the pipeline checkout or set OCCULTED_FLARES_WORK to its root")
