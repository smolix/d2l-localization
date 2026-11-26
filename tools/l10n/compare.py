"""Side-by-side comparison of several translations of the same work files.

    l10n compare de evals/de-conv-layer [--out compare.html]

The directory holds the work files (`*.md`, as `l10n emit` wrote them) and one
`answers-<agent>/` directory per translator with that agent's answer files.
Each agent's answers are checked like `l10n apply --dry-run`, and the page shows
every English block beside each agent's translation, with the check counts,
the DECISIONS and NOTES each agent recorded, and the check output. The page is
one self-contained HTML file (Markdown and math are rendered in the browser).
"""
from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path

from .answers import parse_answer

TEMPLATE = Path(__file__).with_name('compare.html')


def _blocks(work_files):
    blocks, order, rev, page = {}, [], None, None
    for wf in work_files:
        for a in parse_answer(wf.read_text(encoding='utf-8'), str(wf)):
            rev = rev or a.attrs.get('rev')
            page = page or a.path
            for anc in a.order:
                if anc in a.english and anc not in blocks:
                    attrs, text = a.english[anc]
                    blocks[anc] = {'anchor': anc, 'kind': attrs.get('kind', ''),
                                   'unit': a.uid.split('#')[-1], 'en': text, 'tr': {}}
                    order.append(anc)
    return blocks, order, rev, page


def _check(lang_code, files, rev):
    from .cli import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            main(['apply', lang_code, *map(str, files), '--rev', rev, '--dry-run'])
        except SystemExit:
            pass
    out = buf.getvalue().strip()
    errors = sum(int(x) for x in re.findall(r'(\d+) error\(s\)', out))
    warnings = sum(int(x) for x in re.findall(r'(\d+) warning\(s\)', out))
    return out, errors, warnings


def collect(lang_code: str, evaldir: Path) -> dict:
    work = sorted(p for p in evaldir.glob('*.md') if p.is_file())
    if not work:
        raise SystemExit(f'{evaldir}: no work files (*.md)')
    blocks, order, rev, page = _blocks(work)
    agents = []
    for d in sorted(evaldir.glob('answers-*')):
        if not d.is_dir():
            continue
        name = d.name.removeprefix('answers-')
        files = sorted(d.glob('*.md'))
        dec, notes = [], []
        for f in files:
            for a in parse_answer(f.read_text(encoding='utf-8'), str(f)):
                unit = a.uid.split('#')[-1]
                for anc, (_, text) in a.tr.items():
                    if anc in blocks:
                        blocks[anc]['tr'][name] = text
                if a.decisions.strip():
                    dec.append(f'**{unit}**\n\n' + a.decisions.strip())
                if a.notes.strip():
                    notes.append(f'**{unit}**\n\n' + a.notes.strip())
        out, errors, warnings = _check(lang_code, files, rev)
        words = sum(len(blocks[a]['tr'].get(name, '').split()) for a in order)
        missing = sum(1 for a in order if name not in blocks[a]['tr'])
        agents.append({'name': name, 'errors': errors, 'warnings': warnings, 'words': words,
                       'missing': missing, 'check': out,
                       'decisions': '\n\n'.join(dec), 'notes': '\n\n'.join(notes)})
    if not agents:
        raise SystemExit(f'{evaldir}: no answers-<agent>/ directories')
    return {'lang': lang_code, 'page': page, 'rev': rev,
            'en_words': sum(len(blocks[a]['en'].split()) for a in order),
            'agents': agents, 'blocks': [blocks[a] for a in order]}


def write_page(data: dict, out: Path) -> Path:
    payload = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    html = TEMPLATE.read_text(encoding='utf-8').replace('/*DATA*/null', payload)
    out.write_text(html, encoding='utf-8')
    return out


def compare(lang_code: str, evaldir: str, out: str | None = None) -> Path:
    d = Path(evaldir)
    data = collect(lang_code, d)
    path = write_page(data, Path(out) if out else d / 'compare.html')
    for a in data['agents']:
        print(f"{a['name']:>12}: {a['errors']} error(s), {a['warnings']} warning(s), "
              f"{a['words']} words ({a['words'] / max(data['en_words'], 1):.2f}x English)"
              + (f", {a['missing']} block(s) missing" if a['missing'] else ''))
    print(f'wrote {path}')
    return path
