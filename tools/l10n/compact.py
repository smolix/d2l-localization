"""Compacting a decisions log into a lexicon (DESIGN.md section 5).

A running log (`<lang>/decisions.md`, one `## <unit>` section per unit) is replaced
by a sorted lexicon, one line per English term and sense, between the markers of
units.LEXICON_BEGIN/END; its convention rules move into `<lang>/style.md`. The
judgement (which entries to keep, which rendering wins) is done by agents; this
module does the mechanical steps around them:

  split     the log -> chunkNN.md slices of whole sections (chunks.json)
  merge     agents' extNN.jsonl -> conflicts.md / conflicts.json (same English key,
            different target)
  assemble  ext + resolved*.jsonl -> the new decisions.md; style-section.md -> the
            next numbered section of style.md

extNN.jsonl lines: {"en", "sense", "ar"|"tr", "note", "src"} (the target-language
term may be under "tr" or under the language code). resolved.jsonl lines add
"key", the conflict key they settle.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from .config import L10nError
from .units import LEXICON_BEGIN, LEXICON_END, LEXICON_LINE

CHUNK_BYTES = 70_000


def split(text: str, out: Path, size: int = CHUNK_BYTES) -> list:
    """Write chunkNN.md files of whole `## ` sections; returns the chunk metadata."""
    if LEXICON_BEGIN in text:
        raise L10nError('decisions.md already has a lexicon; fold new sections with l10n fold')
    secs = re.split(r'(?m)^(?=## )', text)[1:]
    if not secs:
        raise L10nError('decisions.md has no "## " sections')
    chunks, cur, n = [], [], 0
    for s in secs:
        b = len(s.encode())
        if cur and n + b > size:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(s)
        n += b
    chunks.append(cur)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, c in enumerate(chunks):
        p = out / f'chunk{i:02d}.md'
        p.write_text(''.join(c), encoding='utf-8')
        meta.append({'chunk': i, 'path': str(p), 'sections': len(c),
                     'bytes': sum(len(x.encode()) for x in c),
                     'first': c[0].splitlines()[0], 'last': c[-1].splitlines()[0]})
    (out / 'chunks.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                     encoding='utf-8')
    return meta


def group_key(en: str) -> str:
    en = re.sub(r'[\s-]+', ' ', en.lower().strip())
    return re.sub(r'(?<=\w)(es|s)\b', '', en) if len(en) > 4 else en


def _target(e: dict, code: str) -> str:
    return e.get(code) or e.get('tr') or e.get('ar') or ''


def load_extracts(d: Path, code: str) -> list:
    files = sorted(d.glob('ext[0-9][0-9].jsonl'))
    if not files:
        raise L10nError(f'{d}: no extNN.jsonl files')
    out = []
    for f in files:
        i = int(f.stem[3:])
        for n, line in enumerate(f.read_text(encoding='utf-8').splitlines()):
            if line.strip():
                e = json.loads(line)
                e['_ord'] = (i, n)
                e['_tr'] = _target(e, code)
                out.append(e)
    return out


def merge(d: Path, code: str) -> dict:
    """Group the extracts by English key; write conflicts.md/json. Returns counts."""
    E = load_extracts(d, code)
    G = collections.defaultdict(list)
    for e in E:
        G[group_key(e['en'])].append(e)
    conf = {k: v for k, v in G.items() if len({x['_tr'].strip() for x in v}) > 1}
    with (d / 'conflicts.md').open('w', encoding='utf-8') as f:
        for i, (k, v) in enumerate(sorted(conf.items())):
            f.write(f'### C{i:03d} {k}\n')
            for x in v:
                f.write(f"- [{x['_ord'][0]:02d}.{x['_ord'][1]:03d}] en={x['en']!r} "
                        f"sense={x.get('sense', '')!r} {code}={x['_tr']!r} "
                        f"note={x.get('note', '')!r} src={x.get('src', '')!r}\n")
            f.write('\n')
    (d / 'conflicts.json').write_text(json.dumps(sorted(conf), ensure_ascii=False),
                                      encoding='utf-8')
    return {'entries': len(E), 'keys': len(G), 'conflicts': len(conf)}


def _clean(x) -> str:
    return ' '.join((x or '').replace('→', '->').split())


def lexicon_line(e: dict, tr: str) -> tuple:
    en = _clean(e['en']).replace('**', '')
    sense = _clean(e.get('sense'))
    head = en + (f' ({sense})' if sense else '')
    note = _clean(e.get('note'))
    line = f'- {head} → {_clean(tr)}' + (f' — {note}' if note else '')
    return ' '.join(head.lower().split()), line


HEADER = """# Decisions log: {name} ({code})

Binding like a verified glossary entry (TRANSLATING.md §7). Compacted on {date}
from {sections} unit and chapter-check sections ({kb} KB): the convention rules
moved to {code}/style.md §{sec}, the term choices are the lexicon below, one line
per English term and sense, `- english / variant (sense) → term — note`. A work
file shows only the lexicon lines whose English occurs in its unit. `l10n apply`
appends a section per unit after the lexicon; each chapter's editorial pass folds
them in with `bin/l10n fold {code} <lines>` (DESIGN.md §5). The log before
compaction is in git history.

## Lexicon

"""


def next_section(style: str) -> int:
    nums = [int(m) for m in re.findall(r'(?m)^## (\d+)\.', style)]
    return max(nums, default=0) + 1


def assemble(d: Path, lang, date: str) -> dict:
    """Write the new <lang>/decisions.md and append style-section.md to style.md."""
    code = lang.code
    log_path, style_path = lang.root / 'decisions.md', lang.root / 'style.md'
    old = log_path.read_text(encoding='utf-8')
    if LEXICON_BEGIN in old:
        raise L10nError(f'{log_path} already has a lexicon')
    E = load_extracts(d, code)
    conf = set(json.loads((d / 'conflicts.json').read_text(encoding='utf-8')))
    res = collections.defaultdict(list)
    rfiles = sorted(d.glob('resolved*.jsonl'))
    if conf and not rfiles:
        raise L10nError(f'{d}: no resolved*.jsonl ({len(conf)} conflicts)')
    for rp in rfiles:
        for line in rp.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                res[r['key']].append(r)
    missing = conf - set(res)
    if missing:
        raise L10nError(f'resolved*.jsonl lack {len(missing)} conflict keys, e.g. '
                        + ', '.join(sorted(missing)[:5]))
    final, done = {}, set()
    for e in sorted(E, key=lambda e: e['_ord']):          # later entries win
        k = group_key(e['en'])
        if k in conf:
            if k not in done:
                done.add(k)
                for r in res[k]:
                    kk, line = lexicon_line(r, _target(r, code))
                    final[kk] = line
            continue
        kk, line = lexicon_line(e, e['_tr'])
        final[kk] = line
    lines = [final[k] for k in sorted(final)]
    bad = [l for l in lines if not LEXICON_LINE.match(l)]
    if bad:
        raise L10nError(f'{len(bad)} malformed lexicon lines, e.g. {bad[0]!r}')
    style = style_path.read_text(encoding='utf-8')
    sec = next_section(style)
    head = HEADER.format(name=lang.name, code=code, date=date, sec=sec,
                         sections=len(re.findall(r'(?m)^## ', old)),
                         kb=round(len(old.encode()) / 1000))
    log_path.write_text(head + LEXICON_BEGIN + '\n' + '\n'.join(lines) + '\n'
                        + LEXICON_END + '\n', encoding='utf-8')
    sp = d / 'style-section.md'
    rules = 0
    if sp.exists() and sp.read_text(encoding='utf-8').strip():
        s = sp.read_text(encoding='utf-8').strip('\n')
        s = re.sub(r'(?m)^## (\d+|X|N)\. ', f'## {sec}. ', s, count=1)
        s = re.sub(r'(?m)^### (\d+|X|N)\.(\d+)', rf'### {sec}.\2', s)
        s = re.sub(r'§(X|N)\.(\d+)', rf'§{sec}.\2', s)
        style_path.write_text(style.rstrip('\n') + '\n\n' + s + '\n', encoding='utf-8')
        rules = s.count('\n')
    return {'lexicon_lines': len(lines), 'bytes': len(log_path.read_bytes()),
            'style_section': sec, 'style_lines_added': rules}
