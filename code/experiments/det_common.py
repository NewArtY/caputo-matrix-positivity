"""Shared helpers of the deterministic experiment drivers (E1-E5, Dirac B3/B4/tail): provenance records,
single-thread BLAS, result writers and a small process pool. No numerical content.

Every raw result file written through this module carries a provenance record (package versions, platform,
CPU, UTC time stamp, command line, SHA-256 of the source files that produced it):
  * CSV : leading comment lines '# key: value'  (read with pandas.read_csv(path, comment='#') or read_csv below);
  * JSON: top-level key 'meta';
  * NPZ : array 'meta' holding the JSON text.
No host name, user name or absolute path is recorded.
"""
import os, sys
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')          # must precede the first import of numpy in this process
import datetime, hashlib, io, json, platform, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, '..'))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
RAW = os.path.join(ROOT, 'results', 'raw')
if CODE not in sys.path:
    sys.path.insert(0, CODE)

MAX_WORKERS = min(6, max(1, int(os.environ.get('FRACSP_MAX_WORKERS', '6'))))   # upper bound on worker processes of every driver (never above 6)


def rel(path):
    """Path relative to the archive root, with forward slashes (never an absolute path in a result file)."""
    return os.path.relpath(os.path.abspath(path), ROOT).replace(os.sep, '/')


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def disable_power_throttling():
    """Windows 11 on hybrid CPUs confines background console processes to the efficiency cores (EcoQoS).
    Opt this process out of execution-speed throttling; no effect on other systems. Returns True if applied."""
    if sys.platform != 'win32':
        return False
    try:
        import ctypes
        import ctypes.wintypes as wt

        class _PPTS(ctypes.Structure):
            _fields_ = [('Version', wt.ULONG), ('ControlMask', wt.ULONG), ('StateMask', wt.ULONG)]
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetCurrentProcess.restype = wt.HANDLE
        k32.SetProcessInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        st = _PPTS(1, 0x1, 0x0)
        return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st)))
    except Exception:
        return False


def cpu_name():
    try:
        if sys.platform == 'win32':
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
                return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
        if os.path.exists('/proc/cpuinfo'):
            for line in open('/proc/cpuinfo'):
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def total_ram_gb():
    try:
        if sys.platform == 'win32':
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong)] + \
                           [(n, ctypes.c_ulonglong) for n in ('ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile', 'ullAvailPageFile',
                                                              'ullTotalVirtual', 'ullAvailVirtual', 'ullAvailExtendedVirtual')]
            ms = MS(); ms.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return round(ms.ullTotalPhys / 2 ** 30, 1)
        return round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30, 1)
    except Exception:
        return None


def system_load_percent():
    """Total CPU utilisation in per cent at the time of the call (Windows: Win32_Processor.LoadPercentage;
    elsewhere: 1-minute load average / logical CPUs). None if it cannot be determined."""
    try:
        if sys.platform == 'win32':
            out = subprocess.run(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).LoadPercentage'],
                                 capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
            return float(out[0]) if out else None
        return round(100.0 * os.getloadavg()[0] / os.cpu_count(), 1)
    except Exception:
        return None


def blas_info():
    try:
        import numpy as np
        cfg = np.show_config(mode='dicts')
        b = cfg.get('Build Dependencies', {}).get('blas', {})
        return f"{b.get('name', '?')} {b.get('version', '?')}"
    except Exception:
        return 'unknown'


def provenance(script=None, config=None, extra=None):
    """Provenance record (dict of JSON-serialisable values) for a result file."""
    import numpy, scipy, mpmath
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = 'not imported'
    import fracsp
    src = {}
    fdir = os.path.join(CODE, 'fracsp')
    for fn in sorted(os.listdir(fdir)):
        if fn.endswith('.py'):
            src['code/fracsp/' + fn] = _sha256(os.path.join(fdir, fn))[:16]
    for p in ([script] if script else []) + ([config] if config else []) + [os.path.abspath(__file__)]:
        if p and os.path.exists(p):
            src[rel(p)] = _sha256(p)[:16]
    rec = dict(
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        command=' '.join([os.path.basename(sys.argv[0])] + sys.argv[1:]) if sys.argv and sys.argv[0] else '',
        python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__, torch=torch_v,
        mpmath=mpmath.__version__, fracsp=fracsp.__version__, blas=blas_info(),
        platform=platform.platform(), machine=platform.machine(), cpu=cpu_name(), logical_cpus=os.cpu_count(),
        ram_gb=total_ram_gb(), blas_threads=os.environ.get('OPENBLAS_NUM_THREADS', '?'),
        source_sha256_16=src)
    if extra:
        rec.update(extra)
    return rec


def _meta_lines(meta):
    out = []
    for k, v in meta.items():
        out.append(f'# {k}: {json.dumps(v, ensure_ascii=True) if isinstance(v, (dict, list)) else v}')
    return out


def write_csv(path, header, rows, meta):
    """CSV with '# key: value' provenance lines, then the header line, then the rows (iterables or strings)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = _meta_lines(meta) + [header if isinstance(header, str) else ','.join(header)]
    for r in rows:
        lines.append(r if isinstance(r, str) else ','.join(str(x) for x in r))
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path, comment='#')


def read_meta(path):
    """Provenance record of a CSV written by write_csv (dict of strings; JSON values are decoded)."""
    meta = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.startswith('# '):
                break
            k, _, v = line[2:].rstrip('\n').partition(': ')
            try:
                meta[k] = json.loads(v)
            except Exception:
                meta[k] = v
    return meta


def write_json(path, data, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(dict(meta=meta, **data), f, indent=1, default=_json_default)
        f.write('\n')


def _json_default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def write_npz(path, meta, **arrays):
    import numpy as np
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta, default=_json_default)), **arrays)


class Log:
    """Line-buffered log to stdout and to a file under results/raw (time stamps relative to the start)."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, 'w', encoding='utf-8', newline='\n'); self.t0 = time.perf_counter()

    def __call__(self, *a):
        s = f'[{time.perf_counter() - self.t0:8.1f}s] ' + ' '.join(str(x) for x in a)
        print(s, flush=True); self.f.write(s + '\n'); self.f.flush()

    def close(self):
        self.f.close()


def _worker_init():
    disable_power_throttling()


def pool_map(func, tasks, workers=MAX_WORKERS, log=None, label=''):
    """Ordered parallel map with at most MAX_WORKERS single-threaded processes (results in task order)."""
    workers = max(1, min(workers, MAX_WORKERS, len(tasks)))
    if workers == 1:
        out = []
        for i, t in enumerate(tasks):
            out.append(func(t))
            if log and (i + 1) % max(1, len(tasks) // 10) == 0:
                log(f'{label} {i + 1}/{len(tasks)}')
        return out
    import multiprocessing as mp
    out = [None] * len(tasks)
    with mp.get_context('spawn').Pool(workers, initializer=_worker_init) as pool:
        its = [pool.apply_async(func, (t,)) for t in tasks]
        step = max(1, len(tasks) // 10)
        for i, it in enumerate(its):
            out[i] = it.get()
            if log and ((i + 1) % step == 0 or i + 1 == len(tasks)):
                log(f'{label} {i + 1}/{len(tasks)} tasks')
    return out


def fmt_unc(value, err, max_sig=6):
    """Value with the uncertainty in the last printed digit(s), e.g. 0.31841(3): only digits supported by the error
    estimate are printed (the error is rounded up to one significant digit, two if that digit is 1).
    err <= 0 or non-finite: the value is printed with max_sig significant digits and no bracket."""
    import math
    if value is None or not math.isfinite(value):
        return 'nan'
    if err is None or not math.isfinite(err) or err <= 0:
        return f'{value:.{max_sig}g}'
    e = math.floor(math.log10(err))
    lead = err / 10 ** e
    nd = 2 if lead < 1.95 else 1                         # digits of the bracket
    dec = -(e - (nd - 1))                                # decimals shown
    u = math.ceil(err * 10 ** dec - 1e-9)
    if u >= 10 ** nd:                                    # rounding overflow (e.g. 9.6 -> 10)
        dec -= 1; u = math.ceil(err * 10 ** dec - 1e-9)
    if dec <= 0:
        return f'{round(value, dec):.0f}({u * 10 ** (-dec):.0f})'
    if abs(value) > 0 and abs(value) < 1e-3:
        ex = math.floor(math.log10(abs(value)))
        return fmt_unc(value / 10 ** ex, err / 10 ** ex, max_sig) + f'e{ex:d}'
    return f'{value:.{dec}f}({u})'
