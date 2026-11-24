"""Check that (a) every quoted source sentence occurs verbatim in the fetched source of its paper,
(b) every 'mine' sentence occurs verbatim in method.tex, (c) nothing in method.tex prose is left unmapped."""
import re, sys, glob, os, json
sys.path.insert(0, os.path.dirname(__file__))
from prov import E, PAPERS
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = sys.argv[1]
ws = lambda s: re.sub(r'\s+', ' ', s).strip()

def corpus(key):
    fs = sorted(glob.glob(f'{S}/src/{key}/x/**/*.tex', recursive=True))
    if fs:
        t = '\n'.join(re.sub(r'(?<!\\)%.*', '', open(f, encoding='utf-8', errors='ignore').read()) for f in fs)
    else:
        t = open(f'{S}/src/{key}/text.txt', encoding='utf-8').read()
    return ws(t)

C = {k: corpus(k) for k in PAPERS}
bad = 0
for e in E:
    if ws(e['src']) not in C[e['key']]:
        bad += 1; print('SOURCE NOT FOUND', e['id'], e['key'], '|', e['src'][:90])
method = ws(re.sub(r'(?<!\\)%.*', '', open(TEX, encoding='utf-8').read()))
rest = method
for e in E:
    m = ws(e['mine'])
    if m not in method:
        bad += 1; print('MINE NOT IN TEX ', e['id'], '|', m[:90])
    rest = rest.replace(m, ' <<%s>> ' % e['id'], 1)
ids = [e['id'] for e in E]; assert len(ids) == len(set(ids))
print(f'{len(E)} sentences, {bad} problems')
if '--rest' in sys.argv:
    rest = re.sub(r'\\begin\{(equation|align)\}.*?\\end\{\1\}', ' [EQ] ', rest)
    print(re.sub(r'(\s*<<\w+>>\s*)+', ' # ', rest))
from collections import Counter
print(Counter(e['key'] for e in E).most_common())
json.dump(dict(papers=PAPERS, entries=E), open(f'{S}/prov.json', 'w'), ensure_ascii=False, indent=1)
