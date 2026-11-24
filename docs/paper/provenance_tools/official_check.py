import re, sys, os, json, unicodedata, warnings
warnings.filterwarnings('ignore'); import pymupdf
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(f'{S}/prov.json'))
alpha = lambda s: re.sub(r'[^a-z]', '', unicodedata.normalize('NFKD', s).lower())
T = {}
for k in d['papers']:
    p = f'{S}/official/{k}.pdf'
    if os.path.exists(p): T[k] = alpha(' '.join(pg.get_text() for pg in pymupdf.open(p)))
T['nature_drone'] = alpha(open(f'{S}/src/nature_drone/text.txt').read())
def runs(src):
    s = re.sub(r'~?\\(cite|citep|citet|ref|cref|Cref|eqref|label)\{[^}]*\}', ' @ ', src)
    s = re.sub(r'\$[^$]*\$', ' @ ', s)
    s = re.sub(r'\\(emph|textbf|textit|text)\{([^}]*)\}', r'\2', s)
    s = re.sub(r'\\[a-zA-Z]+\*?', ' ', s).replace('~', ' ')
    s = re.sub(r'\d\)', ' @ ', s)
    return [r.strip() for r in s.split('@') if len(alpha(r)) >= 8]
res = {}
for e in d['entries']:
    rs = runs(e['src']); ok = [alpha(r) in T[e['key']] for r in rs]
    res[e['id']] = (sum(ok), len(ok))
    if not all(ok) or not rs:
        print(e['id'], e['key'], f'{sum(ok)}/{len(ok)}', '| missing:', [r for r, o in zip(rs, ok) if not o])
print('fully confirmed in official version:', sum(a == b and b > 0 for a, b in res.values()), '/', len(res))
json.dump(res, open(f'{S}/official_check.json', 'w'))
