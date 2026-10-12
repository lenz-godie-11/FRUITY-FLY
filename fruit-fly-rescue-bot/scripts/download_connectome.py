"""Download official MaleCNS v1.0 files; verify the GCS object's MD5 hash.

Run with venv/bin/python scripts/download_connectome.py. Interrupted downloads
are resumed by curl. No credentials, global packages or morphology volumes.
"""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = {
    'annotations': 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'neurotransmitters': 'body-neurotransmitters-male-cns-v1.0.feather',
    'connections': 'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
}


def main():
    dest = ROOT / 'data/raw/malecns_v1'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset': 'MaleCNS v1.0', 'license': 'CC-BY-4.0',
                'source': 'https://male-cns.janelia.org/download/', 'files': {}}
    lock_path = ROOT / 'data/source-lock.json'
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else None
    for key, name in FILES.items():
        url = BASE + name
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
            headers = response.headers
        expected_md5 = next((x.strip()[4:] for x in ','.join(headers.get_all('x-goog-hash', [])).split(',')
                             if x.strip().startswith('md5=')), None)
        if expected_md5 is None:
            raise RuntimeError(f'No server checksum for {name}; refusing unverified download')
        path = dest / name
        partial = path.with_suffix('.feather.part')
        if not path.exists():
            print(f'Downloading {name}', flush=True)
            subprocess.run(['curl', '--fail', '--location', '--retry', '4', '--continue-at', '-',
                            '--output', str(partial), url], check=True)
            candidate = partial
        else:
            candidate = path
        sha, md5 = hashlib.sha256(), hashlib.md5()
        with candidate.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                sha.update(block)
                md5.update(block)
        if base64.b64encode(md5.digest()).decode() != expected_md5:
            raise RuntimeError(f'Checksum mismatch: {candidate}; remove this file and retry')
        if lock and sha.hexdigest() != lock['files'][key]['sha256']:
            raise RuntimeError(f'{name} differs from the pinned dataset; refusing silently changed data')
        if candidate == partial:
            partial.replace(path)
        manifest['files'][key] = {'filename': name, 'url': url, 'bytes': path.stat().st_size,
                                  'sha256': sha.hexdigest(), 'gcs_md5_base64': expected_md5,
                                  'gcs_generation': headers.get('x-goog-generation')}
        print(f'Verified {name}: {path.stat().st_size:,} bytes', flush=True)
        manifest['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
        (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
