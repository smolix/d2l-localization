"""Work units (DESIGN.md section 5) and work files (TRANSLATING.md section 1)."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config, tfile
from .glossary import _surface_pattern, load as load_glossary
from .protect import strip_tokens
from .segment import FENCE_RE, segment

MIN_WORDS = 1000      # merge `##` sections until a unit has at least this many words
MAX_WORDS = 5000      # ... unless the merge would exceed this
SHORT_PAGE = 2000     # a page with at most this many words is one unit


@functools.lru_cache(maxsize=512)
def english_blocks(rev: str, path: str) -> tuple:
    text = config.show(rev, path)
    if text is None:
        raise config.L10nError(f'{path} does not exist at {rev}')
    return tuple(segment(text))


def words(text: str) -> int:
    return len(text.split())


@dataclass
class Unit:
    path: str
    index: int                       # 1-based
    blocks: list = field(default_factory=list)   # indices into the file's blocks
    title: str = ''

    @property
    def id(self) -> str:
        return f'u{self.index:02d}'

    @property
    def key(self) -> str:
        return f'{self.path}#{self.id}'

    @property
    def filename(self) -> str:
        return self.path[:-3].replace('/', '__') + f'__{self.id}.md'


def units_for(path: str, blocks) -> list:
    """Split one file's translatable blocks into units.

    Main text: one unit per `##` section (the text before the first `##`
    joins the first), small neighbours merged up to MIN_WORDS..MAX_WORDS.
    Slides go to the unit whose section holds the code cells they show
    (`@cell-id` placeholders); a slide without cells follows the previous
    slide. A page without such links spreads its slides by position.
    """
    blocks = list(blocks)
    main = [b for b in blocks if not b.in_slides]
    nsec = max([b.section for b in main] + [0]) + 1
    sec_words = [0] * nsec
    for b in main:
        if b.translatable:
            sec_words[b.section] += words(b.content)
    total = sum(sec_words) + sum(words(b.content) for b in blocks
                                 if b.translatable and b.in_slides)

    # sections -> groups
    groups: list = []
    if total <= SHORT_PAGE:
        groups = [list(range(nsec))]
    else:
        cur, cw = [], 0
        for s in range(nsec):
            w = sec_words[s]
            if cur and (cw >= MIN_WORDS or cw + w > MAX_WORDS) and not (s == 1 and cw < 200):
                groups.append(cur)
                cur, cw = [], 0
            cur.append(s)
            cw += w
        if cur:
            if groups and cw < MIN_WORDS // 3 and sum(sec_words[x] for x in groups[-1]) + cw <= MAX_WORDS:
                groups[-1] += cur
            else:
                groups.append(cur)
    sec_unit = {s: gi for gi, g in enumerate(groups) for s in g}

    # slides -> sections
    cell_sec = {}
    for b in main:
        if b.kind == 'code' and b.cell_id and b.cell_id not in cell_sec:
            cell_sec[b.cell_id] = b.section
    slides: dict = {}
    for k, b in enumerate(blocks):
        if b.in_slides and b.slide >= 0:
            slides.setdefault(b.slide, []).append(k)
    slide_sec = {}
    linked = any(r in cell_sec for b in blocks if b.in_slides for r in b.refs)
    prev = 0
    nslides = len(slides)
    cum = [sum(sec_words[:s + 1]) for s in range(nsec)]
    for j, sid in enumerate(sorted(slides)):
        if linked:
            secs = [cell_sec[r] for k in slides[sid] for r in blocks[k].refs if r in cell_sec]
            sec = max(prev, min(secs)) if secs else prev
        else:
            frac = (j + 0.5) / max(nslides, 1) * (cum[-1] or 1)
            sec = next((s for s in range(nsec) if cum[s] >= frac), nsec - 1)
        slide_sec[sid] = sec
        prev = sec

    units = [Unit(path, gi + 1) for gi in range(len(groups))]
    for k, b in enumerate(blocks):
        if not b.translatable:
            continue
        if not b.in_slides:
            gi = sec_unit[b.section]
        elif b.slide in slide_sec:
            gi = sec_unit[slide_sec[b.slide]]
        else:
            gi = len(groups) - 1
        units[gi].blocks.append(k)
    for u in units:
        u.blocks.sort()
        heads = [blocks[k] for k in u.blocks if blocks[k].kind == 'heading' and not blocks[k].in_slides]
        u.title = heads[0].content.lstrip('#').strip() if heads else path
    return [u for u in units if u.blocks]


# ── unit state ───────────────────────────────────────────────────────

def unit_state(u: Unit, blocks, entries: dict) -> dict:
    """Counts of done / stale / missing blocks in a unit."""
    st = {'done': 0, 'stale': 0, 'missing': 0}
    for k in u.blocks:
        e = entries.get(blocks[k].anchor)
        st['missing' if e is None else 'stale' if e.stale else 'done'] += 1
    return st


def is_open(u: Unit, blocks, entries: dict) -> bool:
    st = unit_state(u, blocks, entries)
    return st['stale'] + st['missing'] > 0


# ── work file ────────────────────────────────────────────────────────

def _read(p: Path, missing: str) -> str:
    return p.read_text(encoding='utf-8').strip() if p.exists() else missing


def chapter_files(rev: str, path: str) -> list:
    d = path.split('/')[0] + '/'
    if path == 'index.md':
        return ['index.md']
    return [f for f in config.book_files(rev) if f.startswith(d)]


def outline(lang, rev: str, path: str) -> str:
    lines = []
    for f in chapter_files(rev, path):
        blocks = english_blocks(rev, f)
        entries = tfile.load(lang.book, f)
        mark = '  <- this page' if f == path else ''
        lines.append(f'- `{f}`{mark}')
        for b in blocks:
            if b.kind == 'heading' and not b.in_slides:
                level = len(b.content) - len(b.content.lstrip('#'))
                if level > 3:
                    continue
                tr = entries.get(b.anchor)
                t = b.content.lstrip('#').strip()
                if tr and not tr.stale:
                    t += ' | ' + tr.text.lstrip('#').strip()
                lines.append('  ' * level + '- ' + t)
    return '\n'.join(lines)


def page_context(rev: str, path: str, code: str = 'pytorch') -> str:
    """The English page, with framework variants of code cells other than
    `code` collapsed to one line (code='full' keeps all, 'none' drops code)."""
    out = []
    for b in english_blocks(rev, path):
        if b.kind != 'code' or code == 'full':
            out.append(b.text)
            continue
        m = re.search(r'^%%tab\s+(.+)$', b.text, re.M)
        fws = [f.strip() for f in m.group(1).split(',')] if m else None
        if code == 'none' or (fws and code not in fws and 'all' not in fws):
            first = b.text.split('\n', 1)[0]
            what = f'{",".join(fws)} variant of ' if fws else ''
            out.append(f'[{what}code cell {first.strip()} omitted]\n')
        else:
            out.append(b.text)
    return ''.join(out)


def _fence_for(text: str) -> str:
    longest = max([len(m.group(2)) for m in
                   (FENCE_RE.match(l) for l in text.split('\n')) if m] + [2])
    return '`' * (longest + 1)


def glossary_brief(lang, text: str) -> str:
    g = load_glossary(lang.locale)
    cs = g.concepts_in(text)
    if not cs:
        return '(no glossary concepts found in this unit)'
    rows = ['| English | Sense | Term | Status | Alternatives |', '|---|---|---|---|---|']
    for c in sorted(cs, key=lambda c: c.en.lower()):
        en = c.en + (f' ({", ".join(c.aliases)})' if c.aliases else '')
        cell = lambda s: (s or '').replace('|', '\\|').replace('\n', ' ')
        rows.append(f'| {cell(en)} | {cell(c.sense)} | {cell(c.preferred or "-")} | '
                    f'{c.status or "no entry"} | {cell(", ".join(c.alternatives))} |')
    return '\n'.join(rows)


LEXICON_BEGIN = '<!-- l10n:lexicon -->'
LEXICON_END = '<!-- l10n:lexicon-end -->'
LEXICON_LINE = re.compile(r'^- (?P<en>.+?) \u2192 ')


def lexicon_surfaces(line: str) -> list:
    """English surface forms of a lexicon line `- en / en2 (sense) → ar — note`."""
    m = LEXICON_LINE.match(line)
    if not m:
        return []
    en = re.sub(r'\s*\([^()]*\)\s*$', '', m.group('en')).replace('*', '')
    return [w.strip() for w in en.split(' / ') if w.strip()]


def decisions_brief(text: str, unit_text: str) -> str:
    """The decisions log for one unit: between the lexicon markers, only the lines
    whose English term occurs in the unit; everything outside them verbatim. A log
    without the markers is returned unchanged."""
    if LEXICON_BEGIN not in text or LEXICON_END not in text:
        return text
    head, rest = text.split(LEXICON_BEGIN, 1)
    body, tail = rest.split(LEXICON_END, 1)
    low = strip_tokens(unit_text)
    lines = [l for l in body.split('\n') if l.startswith('- ')]
    keep = []
    for l in lines:
        forms = lexicon_surfaces(l)
        if any(re.search(r'(?<![\w-])' + _surface_pattern(f) + r'(?![\w-])', low, re.I)
               for f in forms):
            keep.append(l)
    note = (f'({len(keep)} of {len(lines)} lexicon entries occur in this unit; the others '
            f'are omitted. They are binding like the rest of the log.)')
    return '\n'.join([head.rstrip('\n'), '', note, '', *keep, '', tail.lstrip('\n')])


def lexicon_key(line: str) -> str | None:
    m = LEXICON_LINE.match(line)
    return ' '.join(m.group('en').replace('*', '').lower().split()) if m else None


def fold_lexicon(text: str, new_lines: list, keep_sections: bool = False) -> tuple:
    """Merge lexicon lines into the log: a line replaces the line with the same English
    part (term and sense), `- english (sense) → (delete)` removes it, the rest are added;
    the lexicon is re-sorted by English. Unless keep_sections, everything after the end
    marker (the unit sections appended by `apply`) is dropped. Returns (text, counts)."""
    if LEXICON_BEGIN not in text or LEXICON_END not in text:
        raise config.L10nError('decisions.md has no lexicon markers')
    head, rest = text.split(LEXICON_BEGIN, 1)
    body, tail = rest.split(LEXICON_END, 1)
    entries = {}
    for l in body.split('\n'):
        k = lexicon_key(l)
        if k:
            entries[k] = l.rstrip()
    counts = {'added': 0, 'changed': 0, 'deleted': 0, 'unchanged': 0}
    for l in new_lines:
        k = lexicon_key(l)
        if not k:
            continue
        if l.rstrip().endswith('\u2192 (delete)'):
            counts['deleted'] += entries.pop(k, None) is not None
        elif k not in entries:
            entries[k] = l.rstrip(); counts['added'] += 1
        elif entries[k] != l.rstrip():
            entries[k] = l.rstrip(); counts['changed'] += 1
        else:
            counts['unchanged'] += 1
    lines = [entries[k] for k in sorted(entries)]
    out = head + LEXICON_BEGIN + '\n' + '\n'.join(lines) + '\n' + LEXICON_END + '\n'
    if keep_sections and tail.strip():
        out += tail if tail.startswith('\n') else '\n' + tail
    return out, counts


def previous_unit_translation(lang, rev, u: Unit, units, blocks, entries) -> str:
    prev = None
    if u.index > 1:
        prev = (u.path, units[u.index - 2], blocks, entries)
    else:
        from .scope import for_language        # the previous page this language translates
        sc = for_language(lang, rev)
        files = [f for f in config.book_files(rev) if sc.includes(f) or f == u.path]
        k = files.index(u.path) if u.path in files else -1
        if k > 0:
            pp = files[k - 1]
            pb = english_blocks(rev, pp)
            pu = units_for(pp, pb)
            if pu:
                prev = (pp, pu[-1], pb, tfile.load(lang.book, pp))
    if prev is None:
        return '(none: this is the first unit of the book)'
    pp, pu, pb, pe = prev
    parts = []
    for k in pu.blocks:
        e = pe.get(pb[k].anchor)
        if e and not e.stale:
            parts.append(e.text)
    if not parts:
        return f'(the previous unit, {pu.key}, is not translated yet)'
    return f'Previous unit: {pu.key}\n\n' + '\n\n'.join(parts)


def unit_body(u: Unit, blocks, entries: dict, lang_code: str, rev: str) -> str:
    out = [f'<!--unit:{u.key} lang={lang_code} rev={rev}-->', '']
    for k in u.blocks:
        b = blocks[k]
        e = entries.get(b.anchor)
        if e is None:
            out += [f'<!--en:{b.anchor} kind={b.kind}-->', b.content, f'<!--tr:{b.anchor}-->', '']
        elif not e.stale:
            out += [f'<!--en:{b.anchor} kind={b.kind} locked-->', b.content,
                    f'<!--tr:{b.anchor} locked-->', e.text, '']
        else:
            old = old_english(e, u.path)
            out += [f'<!--en:{e.was} kind={b.kind} old-->', old if old is not None else
                    '(old English not found)',
                    f'<!--en:{b.anchor} kind={b.kind}-->', b.content,
                    f'<!--tr:{b.anchor} was={e.was}-->', e.text, '']
    return '\n'.join(out).rstrip('\n') + '\n'


def old_english(e, path: str) -> str | None:
    if not e.src_rev:
        return None
    try:
        for b in english_blocks(e.src_rev, e.src_path or path):
            if b.anchor == e.was:
                return b.content
    except config.L10nError:
        return None
    return None


def work_file(lang, rev: str, u: Unit, units, blocks, entries, code_context='pytorch') -> str:
    style = _read(lang.root / 'style.md', f'(no {lang.code}/style.md yet: follow TRANSLATING.md '
                  'section 8 and record conventions in DECISIONS)')
    unit_text = '\n\n'.join(blocks[k].content for k in u.blocks)
    decisions = decisions_brief(_read(lang.root / 'decisions.md', '(no decisions recorded yet)'),
                                unit_text)
    page = page_context(rev, u.path, code_context)
    fence = _fence_for(page)
    return '\n'.join([
        '=== BRIEF ===', '',
        f'# Translating unit {u.key} into {lang.name} ({lang.code})', '',
        'Follow TRANSLATING.md. Return the UNIT part only (every marker line copied '
        'exactly, each <!--tr:...--> slot filled), then optional DECISIONS and NOTES.', '',
        f'## Style guide ({lang.code}/style.md)', '', style, '',
        f'## Decisions log ({lang.code}/decisions.md)', '', decisions, '',
        f'## Glossary entries for this unit ({lang.locale})', '', glossary_brief(lang, unit_text), '',
        '=== CONTEXT (read, do not translate) ===', '',
        '## Chapter outline (English | translation)', '', outline(lang, rev, u.path), '',
        f'## The whole English page ({u.path} at {rev})', '', fence + 'markdown', page.rstrip('\n'),
        fence, '',
        '## Translation of the previous unit', '',
        previous_unit_translation(lang, rev, u, units, blocks, entries), '',
        '=== UNIT ===', '',
        unit_body(u, blocks, entries, lang.code, rev),
    ])
