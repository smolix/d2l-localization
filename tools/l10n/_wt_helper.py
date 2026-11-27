"""Run inside a build worktree of d2l-neu (cwd = the worktree), by scope.py.

It uses the tag's own tools, so numbers and words match what the tag's build
renders:

    python3 _wt_helper.py request.json answer.json

request: {"labels": {file: [[label, "label"|"eqlabel"], ...]}}
answer:  {"terms": {"Chapter": ..., "Section": ..., "Figure": ..., "Table": ...,
                    "Equation": ...},          # Quarto's words in the build language
          "pdf_terms": {"Chapter": ..., "Section": ...},  # gen_pdf.py's (pdf.chapter/section)
          "numbering": {file: [N, k] | null},  # CHAPTER_NUMBERING, every key
          "labels": {label: {"file", "id", "kind", "number", "title"}}}

`labels` numbers labels that are not a page's H1 label (subsections,
figures, tables, equations) the way Quarto numbers them in the tag's build:
the page's logical prefix, then headings by level (outside divs; `.unnumbered`
skipped), figures/tables/equations by their order in the page.
"""

import json
import re
import sys
from pathlib import Path

WT = Path.cwd()
sys.path.insert(0, str(WT / 'tools'))

import i18n  # noqa: E402
from d2l_preprocess import CHAPTER_NUMBERING, convert_file, convert_label_id  # noqa: E402

TERMS = {'Chapter': 'crossref-ch-prefix', 'Section': 'crossref-sec-prefix',
         'Figure': 'crossref-fig-prefix', 'Table': 'crossref-tbl-prefix',
         'Equation': 'crossref-eq-prefix'}

FENCE = re.compile(r'^\s*(`{3,}|~{3,})')
CODE_LABEL = re.compile(r'^\s*#\|\s*label:\s*((?:fig|tbl)-[\w-]+)')
ID_ATTR = re.compile(r'\{#((?:fig|tbl|eq)-[^\s}]+)')
HEADING = re.compile(r'^(#{1,6})\s+(.*?)\s*$')


def eq_id(label):
    q = convert_label_id(label)
    for pfx in ('fig-', 'tbl-', 'sec-'):
        if q.startswith(pfx):
            return 'eq-' + q[len(pfx):]
    return q if q.startswith('eq-') else 'eq-' + q


def numbers(rel):
    """{quarto id: (kind, number or None, title)} for one page."""
    nums = CHAPTER_NUMBERING.get(rel)
    prefix = '.'.join(str(n) for n in nums) if nums else None
    qmd = convert_file(WT / rel, 'pytorch', chapter_number=nums)
    out, count, sub = {}, {'fig': 0, 'tbl': 0, 'eq': 0}, {}
    fence, depth = None, 0

    def add(qid):
        kind = qid.split('-', 1)[0]
        count[kind] += 1
        if qid not in out:
            out[qid] = (kind, f'{prefix}.{count[kind]}' if prefix else None, '')

    for line in qmd.split('\n'):
        fm = FENCE.match(line)
        if fence:
            if fm and fm.group(1)[0] == fence[0] and len(fm.group(1)) >= len(fence) \
                    and not line.strip()[len(fm.group(1)):].strip():
                fence = None
            else:
                m = CODE_LABEL.match(line)
                if m:
                    add(m.group(1))
            continue
        if fm:
            fence = fm.group(1)
            continue
        s = line.strip()
        if re.match(r'^:{3,}\s*$', s):
            depth = max(0, depth - 1)
        elif re.match(r'^:{3,}\s*\S', s):
            depth += 1
        m = HEADING.match(line)
        if m and depth == 0:
            level, title = len(m.group(1)), m.group(2)
            am = re.search(r'\s*\{([^}]*)\}\s*$', title)
            attrs = am.group(1) if am else ''
            title = title[:am.start()].strip() if am else title.strip()
            im = re.search(r'#([\w:.\-]+)', attrs)
            unnumbered = '.unnumbered' in attrs or attrs.strip() == '-'
            num = None
            if not unnumbered:
                if level == 1:
                    sub = {}
                    num = prefix
                else:
                    sub[level - 1] = sub.get(level - 1, 0) + 1
                    for k in list(sub):
                        if k > level - 1:
                            del sub[k]
                    num = '.'.join([prefix] + [str(sub.get(i, 0)) for i in range(1, level)]) \
                        if prefix else None
            if im:
                out.setdefault(im.group(1), ('section', num, title))
            continue
        for im in ID_ATTR.finditer(line):
            add(im.group(1))
    return out


def main():
    req = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    labels = {}
    for rel, wanted in sorted(req.get('labels', {}).items()):
        table = numbers(rel)
        for label, how in wanted:
            qid = eq_id(label) if how == 'eqlabel' else convert_label_id(label)
            kind, num, title = table.get(qid, ('section', None, ''))
            labels[label] = {'file': rel, 'id': qid, 'kind': kind, 'number': num,
                             'title': title}
    ans = {'terms': {k: i18n.quarto_term(v) for k, v in TERMS.items()},
           'pdf_terms': {'Chapter': i18n.t('pdf.chapter'), 'Section': i18n.t('pdf.section')},
           'numbering': {k: v for k, v in CHAPTER_NUMBERING.items()},
           'labels': labels}
    Path(sys.argv[2]).write_text(json.dumps(ans, ensure_ascii=False, indent=1),
                                 encoding='utf-8')


if __name__ == '__main__':
    main()
