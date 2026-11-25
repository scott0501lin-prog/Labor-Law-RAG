"""Offline only; never presents these timings as user-facing API latency."""
import sys, statistics, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Fronted import verified_rag as v

def bench(label, fn, n=120):
    results=[]
    for _ in range(n):
        t=time.perf_counter();fn();results.append((time.perf_counter()-t)*1000)
    print(f'{label}: median {statistics.median(results):.3f} ms; p95 {sorted(results)[int(.95*n)-1]:.3f} ms')

if __name__=='__main__':
    bench('offline exact lookup (includes rendering)',lambda:v.answer('勞基法第24條',online=False))
    bench('local BM25 candidate retrieval',lambda:v.retrieve('公司要求加班沒給加班費'))
    print('Excludes API, official web, database, network, UI. Not an end-to-end benchmark.')
