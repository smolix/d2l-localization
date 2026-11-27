"""Side-by-side comparison of several translations of the same work files.

    l10n compare de evals/de-conv-layer [--out compare.html]

The directory holds the work files (`*.md`, as `l10n emit` wrote them) and one
`answers-<agent>/` directory per translator with that agent's answer files.
Each agent's answers are checked like `l10n apply --fresh` (a dry run against an untranslated page), and the page shows
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

from . import config
from .answers import parse_answer

TEMPLATE = Path(__file__).with_name('compare.html')
INDEX = Path(__file__).with_name('compare-index.html')
NO_SPACES = {'ja', 'zh-Hans', 'zh-Hant', 'zh', 'th'}  # length in characters, not words


def _length(text, lang_code):
    if lang_code in NO_SPACES:
        return len(re.sub(r'\s', '', text))
    return len(text.split())


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
            main(['apply', lang_code, *map(str, files), '--rev', rev, '--fresh'])
        except SystemExit:
            pass
    out = buf.getvalue().strip()
    errors = sum(int(x) for x in re.findall(r'(\d+) error\(s\)', out))
    warnings = sum(int(x) for x in re.findall(r'(\d+) warning\(s\)', out))
    return out, errors, warnings


def collect(lang_code: str, evaldir: Path) -> dict:
    work = sorted(p for p in evaldir.glob('*.md')   # README, PROMPT, RESULTS are not work files
                  if p.is_file() and '<!--unit:' in p.read_text(encoding='utf-8'))
    if not work:
        raise SystemExit(f'{evaldir}: no work files (*.md)')
    blocks, order, rev, page = _blocks(work)
    agents = []
    for d in sorted(evaldir.glob('answers-*')):
        if not d.is_dir():
            continue
        name = d.name.removeprefix('answers-')
        files = sorted(f for f in d.glob('*.md') if '<!--unit:' in f.read_text(encoding='utf-8'))
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
        words = sum(_length(blocks[a]['tr'].get(name, ''), lang_code) for a in order)
        missing = sum(1 for a in order if name not in blocks[a]['tr'])
        agents.append({'name': name, 'errors': errors, 'warnings': warnings, 'words': words,
                       'missing': missing, 'check': out,
                       'decisions': '\n\n'.join(dec), 'notes': '\n\n'.join(notes)})
    if not agents:
        raise SystemExit(f'{evaldir}: no answers-<agent>/ directories')
    try:
        from .config import get_language
        L = get_language(lang_code)
        name, direction = L.name, L.dir
    except Exception:
        name, direction = lang_code, 'ltr'
    return {'lang': lang_code, 'name': name, 'dir': direction, 'page': page, 'rev': rev,
            'unit': 'characters' if lang_code in NO_SPACES else 'words',
            'en_words': sum(_length(blocks[a]['en'], lang_code) for a in order),
            'agents': agents, 'blocks': [blocks[a] for a in order]}


def write_page(data: dict, out: Path, back: str | None = None) -> Path:
    payload = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    html = TEMPLATE.read_text(encoding='utf-8').replace('/*DATA*/null', payload)
    html = html.replace('<title>Translation comparison</title>',
                        f'<title>Translations: {data.get("name", data["lang"])}</title>')
    if back:
        html = html.replace('  <header>\n', f'  <header>\n    <p class="back"><a href="{back}">← All languages</a></p>\n', 1)
    out.write_text(html, encoding='utf-8')
    return out


def compare(lang_code: str, evaldir: str, out: str | None = None) -> Path:
    d = Path(evaldir)
    data = collect(lang_code, d)
    path = write_page(data, Path(out) if out else d / 'compare.html')
    for a in data['agents']:
        print(f"{a['name']:>12}: {a['errors']} error(s), {a['warnings']} warning(s), "
              f"{a['words']} {data['unit']} ({a['words'] / max(data['en_words'], 1):.2f}x English)"
              + (f", {a['missing']} block(s) missing" if a['missing'] else ''))
    print(f'wrote {path}')
    return path


def style_status(lang) -> str:
    """'reviewed', 'draft' or '' (no guide), from the head of <lang>/style.md."""
    p = Path(__file__).resolve().parents[2] / lang / 'style.md'
    if not p.exists():
        return ''
    head = ' '.join(p.read_text(encoding='utf-8').splitlines()[:6])
    return 'draft' if re.search(r'\bdraft\b', head, re.I) else 'reviewed'


def compare_all(evals: str, out: str) -> Path:
    """One page per test directory (<lang>-<name>/) plus an index that summarises them."""
    ev, od = Path(evals), Path(out)
    od.mkdir(parents=True, exist_ok=True)
    tests = []
    for d in sorted(p for p in ev.iterdir() if p.is_dir() and '-' in p.name):
        lang = d.name.split('-', 1)[0]
        if not any(d.glob('answers-*')):
            continue
        data = collect(lang, d)
        page = f'{d.name}.html'
        write_page(data, od / page, back='index.html')
        results = d / 'RESULTS.md'
        verdict = ''
        if results.exists():
            # the last verdict paragraph (later runs append theirs), unwrapped
            found = re.findall(r'\*\*Verdict[^*]*\*\*\s*(.+?)(?:\n\s*\n|\Z)',
                               results.read_text(encoding='utf-8'), re.S)
            verdict = ' '.join(found[-1].split()) if found else ''
        try:
            translator = config.get_language(lang).raw.get('translator') or ''
        except config.L10nError:
            translator = ''
        tests.append({'test': d.name, 'page': page, 'lang': lang, 'name': data['name'],
                      'unit': data['unit'], 'en': data['en_words'], 'verdict': verdict,
                      'translator': translator,
                      'chosen': next((a['name'] for a in data['agents']
                                      if re.search(rf'\b{re.escape(a["name"])}\b', translator)), ''),
                      'style': style_status(lang),
                      'agents': [{k: a[k] for k in ('name', 'errors', 'warnings', 'words', 'missing')}
                                 for a in data['agents']]})
        print(f'{d.name}: ' + ', '.join(f"{a['name']} {a['errors']}/{a['warnings']}" for a in data['agents']))
    payload = json.dumps(tests, ensure_ascii=False).replace('</', '<\\/')
    html = INDEX.read_text(encoding='utf-8').replace('/*DATA*/null', payload)
    (od / 'index.html').write_text(html, encoding='utf-8')
    print(f'wrote {od / "index.html"} and {len(tests)} page(s)')
    return od / 'index.html'

