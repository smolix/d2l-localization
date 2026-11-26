"""Split an English chapter .md into an ordered list of blocks.

The blocks partition the file: concatenating every block's `text` gives the
file back byte for byte. Translatable blocks carry prose (headings,
paragraphs, list items, table rows, captions, slide text, ...); everything
else (code fences, display math, comments, div fences, tab markers, label
lines, slide placeholders, blank lines) is structure that the build copies
from the English file at the pinned revision.

Kinds (TRANSLATING.md section 6): heading, paragraph, list_item, exercise,
caption, table_row, quote, note, tab_note, slide_title, slide_text, kicker,
html, meta_title (translatable); blank, code, math, comment, div, tab, meta,
table_sep, placeholder, frontmatter, image, verbatim (not translatable).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

TRANSLATABLE = {
    'heading', 'paragraph', 'list_item', 'exercise', 'caption', 'table_row',
    'quote', 'note', 'tab_note', 'slide_title', 'div_title', 'slide_text',
    'kicker', 'html', 'meta_title',
}

FENCE_RE = re.compile(r'^(\s*)(`{3,}|~{3,})(.*)$')
CELL_ID_RE = re.compile(r'(?:^|[\s{])#([a-z][a-z0-9-]*)(?=\s|\}|$)')
DIV_RE = re.compile(r'^:{3,}')
DIV_CLOSE_RE = re.compile(r'^:{3,}\s*$')
TITLE_ATTR_RE = re.compile(r'\btitle="([^"]*)"')
TAB_BEGIN_RE = re.compile(r'^:begin_tab:`([^`]+)`\s*$')
TAB_END_RE = re.compile(r'^:end_tab:\s*$')
META_RE = re.compile(r'^(?::(?:label|eqlabel|width|exlabel):`[^`]*`\s*)+$')
HEADING_RE = re.compile(r'^(#{1,6})\s')
LIST_RE = re.compile(r'^(\s*)([-*+]|\d{1,3}[.)])(\s+|$)')
PLACEHOLDER_RE = re.compile(
    r'^@(?:!)?(?:-)?(?P<id>[a-z][a-z0-9-]*)(?:@\w+)?(?:\[[^\]]*\])?\s*$')
FIG_RE = re.compile(r'^@fig:[a-z][a-z0-9-]*(?:@\w+)?\s*$')
TABLE_RE = re.compile(r'^\s*\|')
TABLE_SEP_RE = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$')
KICKER_RE = re.compile(r'^\[.*\]\{\.kicker\}\s*$')
COVER_KICKER_RE = re.compile(r'^\[Dive into Deep Learning · §[0-9.]*\]\{\.kicker\}\s*$')
HTML_LINE_RE = re.compile(r'^\s*</?[A-Za-z]')
TABLE_CAPTION_RE = re.compile(r'^:(?![a-z_]+:`)(?!:)\S|^: \S')
GENERATED_BEGIN_RE = re.compile(r'<!--\s*@([\w-]+)-begin\b')
FRONT_TRANSLATABLE_RE = re.compile(r'^(pagetitle|title|subtitle|description):\s*\S')
LETTER_RE = re.compile(r'[^\W\d_]')


def normalize(text: str) -> str:
    return ' '.join(text.split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()[:8]


@dataclass
class Block:
    kind: str
    text: str                       # exact source text, including line endings
    line: int                       # 0-based first line
    translatable: bool = False
    anchor: str = ''                # translatable blocks only
    section: int = 0                # index of the `##` section (0 = before the first)
    in_slides: bool = False
    slide: int = -1                 # index of the enclosing slide, if any
    cell_id: str | None = None      # code blocks
    refs: list = field(default_factory=list)   # placeholders: referenced cell ids
    labels: list = field(default_factory=list)  # meta blocks: :label: values

    @property
    def content(self) -> str:
        """The text a translation replaces: `text` without its final newline."""
        return self.text[:-1] if self.text.endswith('\n') else self.text

    @property
    def ending(self) -> str:
        return '\n' if self.text.endswith('\n') else ''

    @property
    def nlines(self) -> int:
        return self.text.count('\n') + (0 if self.text.endswith('\n') else 1)


def _indent(s: str) -> int:
    return len(s) - len(s.lstrip(' '))


def _blank(s: str) -> bool:
    return not s.strip()


def _dollars(s: str) -> int:
    """Number of unescaped `$$` on a line."""
    return len(re.findall(r'(?<!\\)\$\$', s))


def _hard_break(s: str) -> bool:
    """Does this line (at any indentation for fences) end a paragraph?"""
    if FENCE_RE.match(s):
        return True
    if s[:1] in (' ', '\t'):
        return False
    return bool(HEADING_RE.match(s) or DIV_RE.match(s) or TAB_BEGIN_RE.match(s)
                or TAB_END_RE.match(s) or META_RE.match(s) or s.startswith('<!--')
                or PLACEHOLDER_RE.match(s) or FIG_RE.match(s) or TABLE_RE.match(s))


def has_text(content: str) -> bool:
    """Does a candidate block hold any words once protected tokens are removed?"""
    from .protect import strip_tokens
    return bool(LETTER_RE.search(re.sub(r"&#?\w+;", " ", strip_tokens(content))))


class _Ctx:
    def __init__(self):
        self.divs = []          # stack of dicts: {'slide', 'note'}
        self.tab = None
        self.section = 0
        self.exercises = False
        self.in_slides = False
        self.slide = -1
        self.item = False       # inside a list item (continuation after a split)


def segment(text: str) -> list:
    """Segment a whole .md file into blocks (see module docstring)."""
    L = text.splitlines(keepends=True)
    n = len(L)
    blocks: list = []
    ctx = _Ctx()
    i = 0
    prev_text_kind = None

    def raw(k):
        return L[k].rstrip('\n')

    def emit(kind, a, b, **kw):
        nonlocal prev_text_kind
        blk = Block(kind, ''.join(L[a:b]), a, section=ctx.section,
                    in_slides=ctx.in_slides,
                    slide=ctx.slide if any(d['slide'] for d in ctx.divs) else -1, **kw)
        if kind != 'blank':
            ctx.item = kind in ('list_item',) or (ctx.item and blk.text[:1] == ' ')
        if kind in TRANSLATABLE:
            blk.kind = _contextual(kind, ctx)
            if kind in ('slide_title', 'div_title'):
                tm = TITLE_ATTR_RE.search(blk.content)
                blk.translatable = bool(tm and has_text(tm.group(1)))
            else:
                blk.translatable = has_text(blk.content)
            if not blk.translatable:
                blk.kind = 'verbatim'
            prev_text_kind = blk.kind
        elif kind != 'blank':
            prev_text_kind = None
        blocks.append(blk)
        return blk

    # Front matter: one block per line, the title-like keys translatable.
    if n and raw(0) == '---':
        end = next((k for k in range(1, n) if raw(k) in ('---', '...')), None)
        if end is not None:
            for k in range(0, end + 1):
                kind = 'meta_title' if FRONT_TRANSLATABLE_RE.match(raw(k)) else 'frontmatter'
                emit(kind, k, k + 1)
            i = end + 1

    while i < n:
        s = raw(i)
        if _blank(s):
            j = i
            while j < n and _blank(raw(j)):
                j += 1
            emit('blank', i, j)
            i = j
            continue

        m = FENCE_RE.match(s)
        if m:
            ind, fence, info = m.groups()
            ch, ln = fence[0], len(fence)
            close = re.compile(r'^\s*' + re.escape(ch) + '{' + str(ln) + r',}\s*$')
            j = i + 1
            while j < n and not close.match(raw(j)):
                j += 1
            j = min(j + 1, n)
            if info.strip() == '{=html}' and j - i >= 2 and close.match(raw(j - 1)):
                emit('code', i, i + 1)
                _raw_html(L, i + 1, j - 1, emit, raw)
                emit('code', j - 1, j)
                i = j
                continue
            cid = CELL_ID_RE.search(info)
            emit('code', i, j, cell_id=cid.group(1) if cid else None)
            i = j
            continue

        if s.lstrip().startswith('<!--'):
            gen = GENERATED_BEGIN_RE.search(s)
            if gen:
                endpat = re.compile(r'<!--\s*@' + re.escape(gen.group(1)) + r'-end\b.*-->')
                j = i
                while j < n and not endpat.search(raw(j)):
                    j += 1
            else:
                j = i
                start = s.index('<!--') + 4
                while j < n and '-->' not in (raw(j)[start:] if j == i else raw(j)):
                    j += 1
            j = min(j + 1, n)
            if s.strip() == '<!-- slides -->':
                ctx.in_slides = True
            emit('comment', i, j)
            i = j
            continue

        if DIV_RE.match(s):
            if DIV_CLOSE_RE.match(s):
                kind = 'div'
                emit(kind, i, i + 1)
                if ctx.divs:
                    ctx.divs.pop()
            else:
                attrs = s.lstrip(':').strip()
                is_slide = bool(re.match(r'^\{\.(?:sub)?slide\b', attrs))
                if is_slide:
                    ctx.slide += 1
                note = bool(re.search(r'\.(?:d2l-note|callout-[\w-]+)\b', attrs))
                ctx.divs.append({'slide': is_slide, 'note': note})
                if TITLE_ATTR_RE.search(s):
                    emit('slide_title' if is_slide else 'div_title', i, i + 1)
                else:
                    emit('div', i, i + 1)
            i += 1
            continue

        if TAB_BEGIN_RE.match(s):
            ctx.tab = TAB_BEGIN_RE.match(s).group(1)
            emit('tab', i, i + 1)
            i += 1
            continue
        if TAB_END_RE.match(s):
            ctx.tab = None
            emit('tab', i, i + 1)
            i += 1
            continue

        if META_RE.match(s):
            emit('meta', i, i + 1, labels=re.findall(r':(?:label|eqlabel|exlabel):`([^`]*)`', s))
            i += 1
            continue

        hm = HEADING_RE.match(s)
        if hm:
            level = len(hm.group(1))
            if level <= 2 and not ctx.in_slides:
                if level == 2:
                    ctx.section += 1
                ctx.exercises = level == 2 and s[3:].strip().startswith('Exercises')
            emit('heading', i, i + 1)
            i += 1
            continue

        pm = PLACEHOLDER_RE.match(s)
        if pm or FIG_RE.match(s):
            emit('placeholder', i, i + 1, refs=[pm.group('id')] if pm else [])
            i += 1
            continue

        if s.startswith('$$') or (s.lstrip().startswith('$$') and prev_text_kind is None):
            j = i
            if _dollars(s) >= 2 and _dollars(s) % 2 == 0:
                j = i + 1
            else:
                j = i + 1
                while j < n and _dollars(raw(j)) == 0:
                    j += 1
                j = min(j + 1, n)
            emit('math', i, j)
            i = j
            continue

        if TABLE_RE.match(s):
            j = i
            while j < n and TABLE_RE.match(raw(j)):
                emit('table_sep' if TABLE_SEP_RE.match(raw(j)) else 'table_row', j, j + 1)
                j += 1
            i = j
            continue

        if HTML_LINE_RE.match(s):
            emit('html', i, i + 1)
            i += 1
            continue

        lm = LIST_RE.match(s)
        if lm and not (lm.group(2) in '*+-' and not lm.group(3)):
            i0 = len(lm.group(1))
            j = _scan_list_item(L, i, i0)
            emit('list_item', i, j)
            i = j
            continue

        # Paragraph-like: a run of lines up to a blank line or a hard break.
        j = i + 1
        in_math = _dollars(s) % 2 == 1
        while j < n:
            t = raw(j)
            if in_math:
                if _dollars(t) % 2 == 1:
                    in_math = False
                j += 1
                continue
            if _blank(t) or _hard_break(t):
                break
            lt = LIST_RE.match(t)
            if lt and _indent(t) <= 3 and lt.group(3):
                break
            if _dollars(t) % 2 == 1:
                in_math = True
            j += 1
        if s.startswith('!['):
            kind = 'caption' if re.match(r'^!\[[^\]]', s) else 'image'
        elif COVER_KICKER_RE.match(s) and j == i + 1:
            kind = 'verbatim'
        elif KICKER_RE.match(s) and j == i + 1:
            kind = 'kicker'
        elif s.startswith('>'):
            kind = 'quote'
        elif TABLE_CAPTION_RE.match(s):
            kind = 'caption'
        elif s[:1] == ' ' and ctx.item:
            kind = 'list_item'
        else:
            kind = 'paragraph'
        emit(kind, i, j)
        i = j

    _assign_anchors(blocks)
    return blocks


def _raw_html(L, a, b, emit, raw):
    """Segment the inside of a ```{=html} fence: one block per text-bearing line.

    <style>/<script> elements and comments (with generated @x-begin/@x-end
    regions) are kept verbatim.
    """
    i = a
    while i < b:
        s = raw(i)
        if _blank(s):
            j = i
            while j < b and _blank(raw(j)):
                j += 1
            emit('blank', i, j)
            i = j
            continue
        low = s.lstrip().lower()
        for tag in ('style', 'script', 'pre'):
            if low.startswith('<' + tag):
                j = i
                while j < b and '</' + tag not in raw(j).lower():
                    j += 1
                j = min(j + 1, b)
                emit('code', i, j)
                break
        else:
            if low.startswith('<!--'):
                gen = GENERATED_BEGIN_RE.search(s)
                endpat = (re.compile(r'<!--\s*@' + re.escape(gen.group(1)) + r'-end\b.*-->')
                          if gen else re.compile(r'-->'))
                j = i
                while j < b and not endpat.search(raw(j)[(s.index('<!--') + 4) if j == i and not gen else 0:]):
                    j += 1
                j = min(j + 1, b)
                emit('comment', i, j)
            else:
                j = i + 1
                emit('html', i, j)
        i = j


def _scan_list_item(L, i, i0):
    """End line (exclusive) of the list item that starts at line i (indent i0)."""
    n = len(L)
    j = i + 1
    in_math = _dollars(L[i]) % 2 == 1
    while j < n:
        t = L[j].rstrip('\n')
        if in_math:
            if _dollars(t) % 2 == 1:
                in_math = False
            j += 1
            continue
        if _blank(t):
            k = j
            while k < n and _blank(L[k]):
                k += 1
            if k < n and _indent(L[k]) > i0 and not _hard_break(L[k].rstrip('\n')):
                j = k
                continue
            break
        if FENCE_RE.match(t) or _hard_break(t):
            break
        lt = LIST_RE.match(t)
        if lt and _indent(t) <= i0 and lt.group(3):
            break
        if _dollars(t) % 2 == 1:
            in_math = True
        j += 1
    return j


def _contextual(kind: str, ctx: _Ctx) -> str:
    if kind in ('heading', 'kicker', 'slide_title', 'div_title', 'html', 'meta_title',
                'table_row'):
        return kind
    if any(d['slide'] for d in ctx.divs) or ctx.in_slides:
        return 'caption' if kind == 'caption' else 'slide_text'
    if any(d['note'] for d in ctx.divs):
        return 'note'
    if ctx.tab:
        return 'tab_note'
    if ctx.exercises and kind == 'list_item':
        return 'exercise'
    return kind


def _assign_anchors(blocks):
    seen: dict = {}
    for b in blocks:
        if not b.translatable:
            continue
        h = text_hash(b.content)
        seen[h] = seen.get(h, 0) + 1
        b.anchor = h if seen[h] == 1 else f'{h}-{seen[h]}'


def translatable(blocks) -> list:
    return [b for b in blocks if b.translatable]


def reconstruct(blocks, translations: dict) -> str:
    """Rebuild the file, replacing each translatable block found in `translations`.

    `translations` maps anchor -> text (without the final newline). Blocks
    without an entry keep their English text.
    """
    out = []
    for b in blocks:
        if b.translatable and b.anchor in translations:
            out.append(translations[b.anchor] + b.ending)
        else:
            out.append(b.text)
    return ''.join(out)
