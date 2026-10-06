"""Read frozen COCO inputs; expose only fitting and development observations."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / '.deps'), str(ROOT / 'vendor/gbvs-master')]
FROZEN = ROOT / 'reports/frozen'
SPLIT_PATH = FROZEN / 'split.json'
MAP_PATH = ROOT / 'data/processed/factor-maps-full.npz'
ARCHIVE_PATH = ROOT / 'data/COCOSearch18-fixations-TP.zip'
ALLOWED_SPLITS = ('fit', 'dev')
FROZEN_HASHES = {
    'split.json': '481a6fe60d481b6b627c5e61c1261919826ccf0c41c203d5e5d73447c76f6fb0',
    'pilot-split.json': '384e7a27252777c9cf546f388b88c4437e3b6c40206376133827d8826b716134',
    'feature-audit.json': 'b7cd7a314a04c51dd99d2ca00ddfe6fc89de0ae7d41a40c7eb8b14246695f2ab',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_frozen(name):
    """Reject edited manifests rather than silently creating a new split."""
    path = FROZEN / name
    if sha256(path) != FROZEN_HASHES[name]:
        raise ValueError(f'Frozen manifest changed: {name}')
    return json.loads(path.read_text())


def saliency_version():
    source = ROOT / 'vendor/gbvs-master/saliency_models'
    digest = hashlib.sha256(b'BGR-420x263-GBVS-defaults-PIL-BOX-32x20-v1')
    for path in sorted(source.rglob('*.py')):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_inputs():
    """Check immutable identities and near-duplicate grouping without scoring gaze."""
    split = read_frozen('split.json')['images']
    pilot = read_frozen('pilot-split.json')['images']
    manifest = read_frozen('feature-audit.json')
    if manifest['saliency_version'] != saliency_version():
        raise ValueError('Feature maps are stale: saliency source changed')
    if manifest['factor_maps_sha256'] != sha256(MAP_PATH):
        raise ValueError('Feature map checksum does not match its frozen manifest')
    groups = {}
    for meta in split.values():
        groups.setdefault(meta['group'], set()).add(meta['split'])
    if any(len(values) != 1 for values in groups.values()):
        raise ValueError('Image groups overlap across frozen splits')
    old_test_groups = {m['group'] for m in pilot.values() if m['split'] == 'test'}
    if any(m['split'] != 'previous_test' for m in split.values() if m['group'] in old_test_groups):
        raise ValueError('Previously evaluated image groups are no longer excluded')
    if any(name in pilot for name, m in split.items() if m['split'] == 'test'):
        raise ValueError('Full test overlaps with pilot images')
    return {
        'frozen_manifests_verified': len(FROZEN_HASHES),
        'factor_maps_sha256': manifest['factor_maps_sha256'],
        'saliency_version': manifest['saliency_version'],
        'images_by_split': {s: sum(m['split'] == s for m in split.values())
                            for s in ('fit', 'dev', 'test', 'previous_test')},
        'accessible_gaze_splits': list(ALLOWED_SPLITS),
        'test_scored': False,
    }


def load_rows():
    """Filter by frozen image identity before returning any trial records."""
    split = read_frozen('split.json')['images']
    rows = []
    with zipfile.ZipFile(ARCHIVE_PATH) as source:
        for release in ('train', 'validation'):
            records = json.loads(source.read(f'coco_search18_fixations_TP_{release}_split1.json'))
            rows.extend(r for r in records
                        if split.get(r['name'], {}).get('split') in ALLOWED_SPLITS)
    return rows
