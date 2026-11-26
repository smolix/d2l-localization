"""Parse translator answers, check them, and merge them into <lang>/book/."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass, field

from . import config, tfile
from .glossary import load as load_glossary
from .protect import CheckOptions, check_block
from .units import english_blocks, units_for

UNIT_RE = re.compile(r'^<!--unit:(?P<path>[^#\s]+)#(?P<uid>u\d+)(?P<attrs>(?: [a-z]+=\S+?)*)-->$')
MARK_RE = re.compile(
    r'^<!--(?P<side>en|tr):(?P<anchor>[0-9a-f]{8}(?:-\d+)?)(?P<attrs>(?: [a-z]+(?:=[^\s>]+)?)*)-->$')
SECTION_RE = re.compile(r'^=== (DECISIONS|NOTES|UNIT|BRIEF|CONTEXT)\b.*===\s*$')


@dataclass
class Answer:
    path: str
    uid: str
    attrs: dict
    english: dict = field(default_factory=dict)   # anchor -> (attrs, text) for en markers
    tr: dict = field(default_factory=dict)        # anchor -> (attrs, text)
    order: list = field(default_factory=list)
    decisions: str = ''
    notes: str = ''
    source: str = ''


class AnswerError(ValueError):
    pass


def _attrs(s: str) -> dict:
    out = {}
    for a in s.split():
        k, _, v = a.partition('=')
        out[k] = v if _ else True
    return out


def parse_answer(text: str, source: str = '') -> list:
    """Answers in a text (one or more units). Work-file BRIEF/CONTEXT are skipped."""
    lines = text.split('\n')
    # drop everything up to the last `=== UNIT ===` header before each unit marker
    answers, cur, side = [], None, None
    buf: list = []
    section = None

    def flush_block():
        nonlocal buf
        if cur is None or side is None:
            buf = []
            return
        kind, anchor, attrs = side
        body = '\n'.join(buf).strip('\n')
        if kind == 'tr':
            cur.tr[anchor] = (attrs, body)
            cur.order.append(anchor)
        else:
            cur.english.setdefault(anchor, (attrs, body))
        buf = []

    for line in lines:
        sm = SECTION_RE.match(line)
        if sm:
            flush_block()
            side = None
            section = sm.group(1)
            continue
        if section in ('BRIEF', 'CONTEXT'):
            continue
        um = UNIT_RE.match(line)
        if um:
            flush_block()
            side = None
            cur = Answer(um.group('path'), um.group('uid'), _attrs(um.group('attrs')), source=source)
            answers.append(cur)
            section = 'UNIT'
            continue
        if section in ('DECISIONS', 'NOTES'):
            if cur is not None:
                key = 'decisions' if section == 'DECISIONS' else 'notes'
                setattr(cur, key, (getattr(cur, key) + '\n' + line))
            continue
        mm = MARK_RE.match(line)
        if mm and cur is not None:
            flush_block()
            side = (mm.group('side'), mm.group('anchor'), _attrs(mm.group('attrs')))
            continue
        buf.append(line)
    flush_block()
    for a in answers:
        a.decisions = a.decisions.strip()
        a.notes = a.notes.strip()
    if not answers:
        raise AnswerError(f'{source}: no <!--unit:...--> marker found')
    return answers


@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    locked_edits: list = field(default_factory=list)

    def ok(self) -> bool:
        return not self.errors


def check_options(lang) -> CheckOptions:
    return CheckOptions.from_dict(lang.raw.get('check'))


def check_translation(lang, kind: str, english: str, tr: str, where: str, rep: Report,
                      glossary=None, acknowledged: bool = False):
    e, w = check_block(kind, english, tr, check_options(lang))
    if acknowledged:
        w = [x for x in w if 'reordered' not in x]
    rep.errors += [f'{where}: {x}' for x in e]
    rep.warnings += [f'{where}: {x}' for x in w]
    if glossary is not None:
        for v in glossary.violations(english, tr):
            rep.warnings.append(f'{where}: glossary term not used: {v}')


def apply_answer(lang, ans: Answer, agent: str | None, model: str | None,
                 force: bool = False, dry_run: bool = False) -> Report:
    rep = Report()
    rev = lang.aligned
    if not rev:
        raise config.L10nError(f'{lang.code} has no pinned revision (l10n pin {lang.code} <rev>)')
    if ans.attrs.get('lang') not in (None, True, lang.code):
        rep.errors.append(f'{ans.path}#{ans.uid}: unit is for language {ans.attrs.get("lang")}')
    if ans.attrs.get('rev') not in (None, True, rev):
        rep.warnings.append(f'{ans.path}#{ans.uid}: emitted at {ans.attrs.get("rev")}, '
                            f'book now aligned with {rev}')
    blocks = english_blocks(rev, ans.path)
    units = {u.id: u for u in units_for(ans.path, blocks)}
    if ans.uid not in units:
        rep.errors.append(f'{ans.path}: no unit {ans.uid} at {rev}')
        return rep
    unit = units[ans.uid]
    expected = {blocks[k].anchor: blocks[k] for k in unit.blocks}
    entries = tfile.load(lang.book, ans.path)
    where0 = f'{ans.path}#{ans.uid}'
    missing = [a for a in expected if a not in ans.tr]
    extra = [a for a in ans.tr if a not in expected]
    if missing:
        rep.errors.append(f'{where0}: missing translation markers: {" ".join(missing[:8])}')
    if extra:
        rep.errors.append(f'{where0}: unknown markers: {" ".join(extra[:8])}')
    for a, (attrs, en_text) in ans.english.items():
        if a in expected and 'old' not in attrs and en_text.strip() and \
                ' '.join(en_text.split()) != ' '.join(expected[a].content.split()):
            rep.warnings.append(f'{where0}/{a}: English text under the marker was edited (ignored)')
    g = load_glossary(lang.locale)
    notes = ans.notes
    new_entries = dict(entries)
    for a, b in expected.items():
        if a not in ans.tr:
            continue
        attrs, text = ans.tr[a]
        where = f'{where0}/{a}'
        old = entries.get(a)
        locked = old is not None and not old.stale
        if locked and text == old.text:
            continue
        if locked:
            rep.locked_edits.append(a)
            if a not in notes:
                rep.warnings.append(f'{where}: locked block edited without a NOTE')
        check_translation(lang, b.kind, b.content, text, where, rep, g, acknowledged=a in notes)
        new_entries[a] = tfile.Entry(text)
    if rep.errors and not force:
        return rep
    if dry_run:
        return rep
    order = [b.anchor for b in blocks if b.translatable]
    tfile.save(lang.book, ans.path, new_entries, order)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    if ans.decisions:
        p = lang.root / 'decisions.md'
        head = '' if p.exists() else f'# Decisions log: {lang.name} ({lang.code})\n'
        with p.open('a', encoding='utf-8') as f:
            f.write(f'{head}\n## {ans.path}#{ans.uid} ({agent or "unknown agent"}, {stamp[:10]})\n\n'
                    f'{ans.decisions}\n')
    if ans.notes:
        d = lang.root / 'notes'
        d.mkdir(parents=True, exist_ok=True)
        (d / (ans.path[:-3].replace('/', '__') + f'__{ans.uid}.md')).write_text(
            f'# Notes on {ans.path}#{ans.uid} ({agent or "unknown agent"}, {stamp})\n\n{ans.notes}\n',
            encoding='utf-8')
    log = {
        'time': stamp, 'unit': f'{ans.path}#{ans.uid}', 'rev': rev, 'agent': agent,
        'model': model, 'blocks': len(ans.tr), 'locked_edits': rep.locked_edits,
        'errors': len(rep.errors), 'warnings': len(rep.warnings), 'source': ans.source,
        'answer_sha256': hashlib.sha256('\n'.join(t for _, t in ans.tr.values()).encode()).hexdigest()[:16],
    }
    with (lang.root / 'units.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(log, ensure_ascii=False) + '\n')
    return rep


# ── whole-language check and status ──────────────────────────────────

def file_status(lang, rev, path):
    blocks = english_blocks(rev, path)
    entries = tfile.load(lang.book, path)
    anchors = {b.anchor: b for b in blocks if b.translatable}
    st = {'blocks': len(anchors), 'done': 0, 'stale': 0, 'missing': 0, 'orphan': 0,
          'words': 0, 'words_done': 0}
    for a, b in anchors.items():
        w = len(b.content.split())
        st['words'] += w
        e = entries.get(a)
        if e is None:
            st['missing'] += 1
        elif e.stale:
            st['stale'] += 1
        else:
            st['done'] += 1
            st['words_done'] += w
    st['orphan'] = sum(1 for a in entries if a not in anchors)
    return st


def check_language(lang, files=None, complete=False) -> Report:
    rep = Report()
    rev = lang.aligned
    if not rev:
        raise config.L10nError(f'{lang.code} has no pinned revision')
    have = sorted(str(p.relative_to(lang.book)) for p in lang.book.rglob('*.md')) \
        if lang.book.exists() else []
    book = set(config.book_files(rev))
    for p in have:
        if p not in book:
            rep.errors.append(f'{p}: no such English file at {rev}')
    targets = config.select_files(rev, files) if files else [p for p in have if p in book]
    if complete:
        targets = config.select_files(rev, files) if files else list(config.book_files(rev))
    g = load_glossary(lang.locale)
    for path in targets:
        blocks = english_blocks(rev, path)
        try:
            entries = tfile.load(lang.book, path)
        except tfile.TFileError as e:
            rep.errors.append(f'{path}: {e}')
            continue
        anchors = {b.anchor: b for b in blocks if b.translatable}
        notes_text = _notes_for(lang, path)
        for a, e in entries.items():
            if a not in anchors:
                rep.errors.append(f'{path}/{a}: anchor not in the English file at {rev}')
                continue
            if e.stale:
                (rep.errors if complete else rep.warnings).append(
                    f'{path}/{a}: stale (was={e.was}); needs an update unit')
                continue
            check_translation(lang, anchors[a].kind, anchors[a].content, e.text, f'{path}/{a}',
                              rep, g, acknowledged=a in notes_text)
        if complete:
            miss = [a for a in anchors if a not in entries]
            if miss:
                rep.errors.append(f'{path}: {len(miss)} of {len(anchors)} blocks not translated')
    # locked edits recorded at this revision
    log = lang.root / 'units.jsonl'
    if log.exists():
        for line in log.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r.get('rev') == rev and r.get('locked_edits'):
                rep.locked_edits += [f'{r["unit"]}/{a} ({r.get("agent")})' for a in r['locked_edits']]
    return rep


def _notes_for(lang, path: str) -> str:
    d = lang.root / 'notes'
    if not d.exists():
        return ''
    stem = path[:-3].replace('/', '__') + '__'
    return '\n'.join(p.read_text(encoding='utf-8') for p in sorted(d.glob(stem + 'u*.md')))
