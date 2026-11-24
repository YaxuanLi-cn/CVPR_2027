import re, sys, os, json
from collections import Counter, OrderedDict
sys.path.insert(0, os.path.dirname(__file__)); from prov import E, PAPERS
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
chk = json.load(open(f'{S}/official_check.json'))
SHORT = {'superglue': 'SuperGlue', 'loftr': 'LoFTR', 'pixloc': 'PixLoc', 'orienternet': 'OrienterNet', 'pixsfm': 'Pixel-Perfect SfM', 'dust3r': 'DUSt3R',
         'nwm': 'Navigation World Models', 'kendall_unc': 'Kendall & Gal', 'lightglue': 'LightGlue', 'reloc3r': 'Reloc3r', 'nature_drone': 'Kaufmann et al. (Swift)'}
def clean(s):
    s = s.replace(r'\duster{}', 'DUSt3R').replace(r'\supp', 'supplemental').replace(r'\degree', '°').replace(r'\dag', '†').replace(r'\ie', 'i.e.')
    s = re.sub(r'\\SI\{(\d+)\}\{\\meter\}', r'\1m', s); s = s.replace(r'{\times}', '×')
    s = s.replace(r'\*', '')
    s = re.sub(r'~?\\(cite|citep|citet)\{[^}]*\}', ' [cite]', s)
    s = re.sub(r'~?\\(ref|cref|Cref|eqref)\{[^}]*\}', ' [ref]', s)
    s = re.sub(r'\\(emph|textbf|textit|underline|red|green|blue|b)\{([^}]*)\}', r'\2', s)
    s = s.replace('$,\\, $', ', ').replace('``', '“').replace("''", '”')
    return re.sub(r'\s+', ' ', s.replace('~', ' ')).strip()
OFFICIAL = {'T8': 'We emphasize that all results are obtained with the same DUSt3R model (our default model is denoted as ‘DUSt3R 512’, other DUSt3R models serve for the ablations in Sec. F of the suppl.), i.e. we never finetune our model on a particular downstream task (zero-shot settings).'}
cnt = Counter(e['key'] for e in E); o = []
ncap = sum(e['para'] == 'Captions' for e in E); npend = sum(e['pending'] for e in E)
o.append('# Experiment 逐句结构来源对照表\n')
o.append(f'对应文件：`experiment.tex`（同目录）。共 **{len(E)} 句**（正文 {len(E) - ncap} 句，图表标题 {ncap} 句），每一句的句式骨架都取自下表 {len(cnt)} 篇论文中的一句原文；内容（数字、设置、结论）来自仓库里的实验记录和本次重新统计的结果，出处见“数字来源”。\n')
o.append('## 核验方式\n')
o.append('- **原句逐字核验**：每条“原”都由脚本在该论文的 arXiv LaTeX 源码（Nature 论文为 nature.com 正文）中做了逐字匹配，全部命中；`experiment.tex` 里除表格和小标题外没有未登记的句子。下面显示时只做了排版清理：引用写成 `[cite]`、交叉引用写成 `[ref]`、去掉 `\\emph` 等格式命令。')
n_ok = sum(a == b and b > 0 for a, b in chk.values())
o.append(f'- **会议正式版交叉核对**：又把每条原句的文字部分与会议官方 PDF（CVF Open Access / NeurIPS Proceedings）做了自动比对，{n_ok}/{len(E)} 句完全一致。两处例外都已人工确认：**I1**（Reloc3r）在正式版里文字相同，只是 PDF 中这句话被一张表格隔成两段，自动比对没接上；**T8**（DUSt3R）正式版的括号内措辞与 arXiv 版略有不同（“serve … in Sec. F of the suppl.”，句末多了 “(zero-shot settings)”），表中 T8 已直接采用正式版措辞，句式骨架不受影响。')
o.append('- 闭环一节的句子（S7–S18、C20、C21）是仿真跑完后补写的：来自 Nature 论文的句子比对的就是 nature.com 正式版正文，其余来自 PixLoc、OrienterNet、LoFTR、NWM 的句子已与 CVF 正式版 PDF 比对一致。')
o.append('- **只用正文**：所有原句都出自论文正文或正文图表的标题，没有用补充材料里的句子。')
o.append('- **句子拆分口径**：以句号为界；带编号的列举（E3、B1）整体算一句；图表标题里加粗的小标题与紧随其后的第一句算一句。段落小标题、表格内容不计。\n')
o.append('## 来源论文\n')
o.append('| 简称 | 论文 | 发表 | 取用句数 |\n|---|---|---|---:|')
for k, n in cnt.most_common():
    t, a, v, ref = PAPERS[k]; o.append(f'| {SHORT[k]} | {a}, *{t}* | {v} | {n} |')
o.append('\n没有使用列表之外的来源；ECCV、CoRL、ICRA、TPAMI、arXiv-only 等不在你指定范围内的论文一律未用作句式来源（ViNT、NoMaD、MASt3R 等只作为被引用的方法出现）。\n')
o.append(open(os.path.join(os.path.dirname(__file__), 'numbers_section.md'), encoding='utf-8').read())
o.append('## 逐句对照\n')
o.append('每条三行：**新** = `experiment.tex` 中的句子；**原** = 来源原句；**构** = 两句共用的句式骨架。\n')
groups = OrderedDict()
for e in E: groups.setdefault(e['para'], []).append(e)
for para, es in groups.items():
    o.append(f'### {para}\n')
    for e in es:
        t, a, v, ref = PAPERS[e['key']]
        o.append(f'**{e["id"]}** · {SHORT[e["key"]]}（{v}）' + (' ⏳' if e['pending'] else ''))
        o.append('```text')
        o.append('新: ' + re.sub(r'\s+', ' ', e['mine']))
        o.append('原: ' + (OFFICIAL.get(e['id']) or clean(e['src'])))
        o.append('构: ' + e['skel'])
        o.append('```')
    o.append('')
open(sys.argv[1], 'w', encoding='utf-8').write('\n'.join(o)); print('wrote', sys.argv[1], len(E), 'entries')
