"""Check the provenance records of the raw results and compare the recorded source fingerprints with the archive.

For every file under results/raw the script looks for a provenance record (time stamp, package versions, platform,
seed, configuration)
  * in the file itself: top-level keys or the key 'meta' / 'provenance' of a JSON file, '# key: value' header lines
    of a CSV file, the array 'meta' of an NPZ file;
  * else in a JSON record of the same folder: <name>.provenance.json, the result file of the same run, prepare.json
    or timing.json (reference arrays and tables written by the stage that wrote the record);
  * else in the folder-level record provenance.json of the benchmark (compiled after the runs; B5 only).
It then compares every source fingerprint recorded in a result file with the corresponding file of the archive and
lists the result files that were produced with an earlier state of a source file. Earlier states that differ from
the file of the archive only in a docstring are listed in EQUIVALENT and counted separately.

Usage: python scripts/check_provenance.py [--list]      (--list prints every file without a record)
Exit status 1 if a result file has no record at all.
"""
import collections, hashlib, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'results', 'raw')
NEED = ('timestamp_utc', 'python', 'numpy', 'platform')
VERSIONS = ('python', 'numpy', 'scipy', 'torch', 'mpmath', 'platform')
KINDS = ('in file', 'record of the same folder', 'folder-level record', 'log', 'NO RECORD')
# Fingerprints of earlier states of a source file that differ from the file of the archive only in a docstring
# (no statement changed): file name or path as recorded -> {fingerprint: what was changed}.
_INIT = {'567de9449b4747da': 'docstring of the package (list of modules) rewritten on 12 October 2026; the file consists of this '
                             'docstring and the version string, which is unchanged (tests/test_all.py, test_package_init_has_no_code)'}
EQUIVALENT = {'__init__.py': _INIT, 'code/fracsp/__init__.py': _INIT}


def record_of_json(d):
    """Provenance record of a parsed JSON result, or None."""
    if isinstance(d, dict):
        for r in (d, d.get('meta'), d.get('provenance')):
            if isinstance(r, dict) and all(k in r for k in NEED):
                return r
    return None


def record_of_file(path):
    """Provenance record stored in the file itself, or None."""
    if path.endswith('.json'):
        return record_of_json(json.load(open(path, encoding='utf-8')))
    if path.endswith('.csv'):
        rec = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                if not line.startswith('# '):
                    break
                k, _, v = line[2:].rstrip('\n').partition(': ')
                try:
                    rec[k] = json.loads(v)
                except ValueError:
                    rec[k] = v
        return rec if all(k in rec for k in NEED) else None
    if path.endswith('.npz'):
        z = np.load(path, allow_pickle=False)
        if 'meta' in z.files:
            rec = json.loads(str(z['meta']))
            return rec if all(k in rec for k in NEED) else None
    return None


_FOLDER = {}


def folder_records(folder):
    """Names of the JSON files of a folder that hold a provenance record."""
    if folder not in _FOLDER:
        _FOLDER[folder] = {f for f in os.listdir(folder) if f.endswith('.json')
                           and record_of_json(json.load(open(os.path.join(folder, f), encoding='utf-8'))) is not None}
    return _FOLDER[folder]


def covering_record(path):
    """Kind of record that covers a file without a record of its own ('' if none)."""
    folder, name = os.path.split(path); stem = os.path.splitext(name)[0]; recs = folder_records(folder)
    if any(c in recs for c in (stem + '.provenance.json', stem + '.json', stem.replace('_test_rho', '') + '.json', 'prepare.json', 'timing.json')):
        return 'record of the same folder'
    p = os.path.join(os.path.dirname(folder), 'provenance.json')
    if os.path.isfile(p) and os.path.relpath(path, os.path.dirname(p)).replace(os.sep, '/') in json.load(open(p, encoding='utf-8')).get('files', {}):
        return 'folder-level record'
    return ''


def current_fingerprints():
    """sha256 (16 hex digits) of the source files of the archive: raw bytes by path, normalised line endings by name."""
    raw, norm = {}, {}
    for sub in ('code/fracsp', 'code/experiments', 'code/configs', 'scripts'):
        d = os.path.join(ROOT, sub)
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                b = open(p, 'rb').read()
                raw[f'{sub}/{f}'] = hashlib.sha256(b).hexdigest()[:16]
                norm[f] = hashlib.sha256(b.replace(b'\r\n', b'\n')).hexdigest()[:16]
    return raw, norm


def main():
    raw_fp, norm_fp = current_fingerprints()
    stat = collections.defaultdict(collections.Counter); missing = []; fp = collections.defaultdict(collections.Counter)
    versions = collections.Counter(); stamps = []; older = collections.defaultdict(list)
    for folder, _, files in sorted(os.walk(RAW)):
        rel = os.path.relpath(folder, RAW).replace(os.sep, '/')
        for name in sorted(files):
            path = os.path.join(folder, name)
            if name.endswith('.log') or (name == 'provenance.json' and rel.count('/') == 0):
                stat[rel]['log'] += name.endswith('.log'); continue
            rec = record_of_file(path)
            kind = 'in file' if rec is not None else (covering_record(path) or 'NO RECORD')
            stat[rel][kind] += 1
            if kind == 'NO RECORD':
                missing.append(f'{rel}/{name}'); continue
            if rec is None:
                continue
            versions[tuple(str(rec.get(k, '-')) for k in VERSIONS)] += 1; stamps.append(str(rec['timestamp_utc'])[:19])
            diff = []
            for key, h in (rec.get('code_sha256') or {}).items():                # line endings normalised, base names
                state = 'identical' if norm_fp.get(key) == h else 'docstring' if h in EQUIVALENT.get(key, ()) else 'different'
                fp[key][state] += 1
                if state == 'different':
                    diff.append(key)
            for key, h in (rec.get('source_sha256_16') or {}).items():           # raw bytes, paths relative to the root
                state = 'identical' if raw_fp.get(key) == h else 'docstring' if h in EQUIVALENT.get(key, ()) else 'different'
                fp[key][state] += 1
                if state == 'different':
                    diff.append(key.split('/')[-1])
            if diff:
                older[', '.join(sorted(diff))].append(f"{rel}/{name} ({str(rec['timestamp_utc'])[:19]})")
    print('Provenance records under results/raw (number of files)')
    print(f"{'folder':30s}" + ''.join(f'{k:>27s}' for k in KINDS))
    tot = collections.Counter()
    for rel in sorted(stat):
        print(f'{rel:30s}' + ''.join(f'{stat[rel][k]:27d}' for k in KINDS)); tot.update(stat[rel])
    print(f"{'total':30s}" + ''.join(f'{tot[k]:27d}' for k in KINDS))
    print('\nEnvironments recorded in the files (' + ', '.join(VERSIONS) + '): number of files')
    for v, n in versions.most_common():
        print(f'  {n:6d}  ' + ', '.join(v))
    if stamps:
        print(f'\nTime stamps of the records (UTC): {min(stamps)} ... {max(stamps)}')
    print('\nSource fingerprints recorded in the result files against the files of the archive (number of result files)')
    for key in sorted(fp):
        print(f"  {key:44s} identical {fp[key]['identical']:5d}   docstring changed {fp[key]['docstring']:5d}   different {fp[key]['different']:3d}")
    for key in sorted(EQUIVALENT):
        for h, why in EQUIVALENT[key].items():
            if fp[key]['docstring']:
                print(f'  docstring changed: {key} {h}: {why}')
    if older:
        print('\nResult files produced with an earlier state of a source file (source files: result files)')
        for key in sorted(older):
            print(f'  {key}:')
            print('\n'.join('      ' + x for x in older[key]))
    if missing:
        print(f'\n{len(missing)} result files without any provenance record')
        if '--list' in sys.argv:
            print('\n'.join('  ' + m for m in missing))
    print(f'\nsummary: {sum(tot[k] for k in KINDS if k not in ("log", "NO RECORD"))} result files with a record, {tot["NO RECORD"]} without; '
          f'{sum(len(v) for v in older.values())} result files produced with an earlier state of at least one source file')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
