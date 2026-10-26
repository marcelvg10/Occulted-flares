"""Restore notebook atomic inputs from SolarSoft, checking retained SHA256s."""
import hashlib
import json
import urllib.request
from pathlib import Path

work = Path(__file__).resolve().parents[1]
provenance = json.loads((work / 'SOL2026-07-07/Spectroscopy/outputs/reference_inputs.json').read_text())
base = 'https://soho.nascom.nasa.gov/solarsoft/packages/xray/dbase/chianti/'
for record in provenance['atomic_files'].values():
    target = work / record['path']
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == record['sha256']:
        print(f'Already verified: {target.name}', flush=True)
        continue
    if target.exists():
        raise RuntimeError(f'Refusing to overwrite changed input: {target}')
    print(f'Downloading {target.name}', flush=True)
    with urllib.request.urlopen(base + target.name, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise RuntimeError(f'Download hash mismatch: {target.name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f'Verified and restored {target.name}: {len(data)} bytes', flush=True)
