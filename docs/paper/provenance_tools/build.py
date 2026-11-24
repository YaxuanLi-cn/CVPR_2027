import re, sys, os
sys.path.insert(0, os.path.dirname(__file__)); from prov import E
d = {e['id']: e['mine'] for e in E}
t = open(os.path.join(os.path.dirname(__file__), 'experiment_template.tex'), encoding='utf-8').read()
used = set()
def rep(m): used.add(m.group(1)); return d[m.group(1)]
out = re.sub(r'<<(\w+)>>', rep, t)
assert used == set(d), (set(d) - used, used - set(d))
open(sys.argv[1], 'w', encoding='utf-8').write(out); print('wrote', sys.argv[1], len(out))
