"""Protected tokens (TRANSLATING.md section 4) and per-block checks."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# Order matters: earlier alternatives win at the same position.
_TOKEN_RE = re.compile(r'''
  (?P<comment><!--.*?-->)
| (?P<ref>:[a-z_]+:`[^`]*`)
| (?P<code>(?P<ticks>`+)(?:.+?)(?P=ticks))
| (?P<dmath>(?<!\\)\$\$.+?(?<!\\)\$\$)
| (?P<esc>\\[$\\`*_\[\](){}#|<>~^])
| (?P<math>(?<!\\)\$(?=\S)(?:\\.|[^$\\])+?(?<=[^\s\\])\$|\\\(.+?\\\))
| (?P<secref>§\{[^}]*\})
| (?P<target>\]\((?:[^()\s]|\([^()\s]*\))*(?:\s+"[^"]*")?\))
| (?P<attr>\{[.#][^{}\n]*\}|\{=\w+\})
| (?P<autolink><https?://[^>\s]+>)
| (?P<filled><span\ class="d2l-(?:build-date|site-url)">[^<]*</span>)
| (?P<tag></?[A-Za-z][\w:-]*(?:\s+[^<>]*?)?/?>)
| (?P<url>https?://[^\s)<>\]]+)
''', re.X | re.S)

CLASSES = ('comment', 'ref', 'code', 'math', 'secref', 'target', 'attr', 'link', 'tag')
_TRANSLATABLE_ATTR_RE = re.compile(r'\b(alt|title|aria-label)="[^"]*"')
NUMBER_RE = re.compile(r'(?<![\w.])\d+(?:[.,]\d+)*(?![\w])')
BIDI_RE = re.compile('[‎‏‪-‮⁦-⁩؜]')


@dataclass
class CheckOptions:
    """Checker switches; a language may override them in languages.yml (`check:`).

    cjk_adjacent: a reference role (:numref:, :eqref:, :cite:, ...) or inline
        math directly touching a CJK letter (kana, kanji, hangul) is an error,
        because pandoc does not recognise the role there (``:numref:`x`で``).
        'letters' (default) flags kana/kanji/hangul only, full-width
        punctuation such as （） and 、 is fine; 'all' flags any adjacent
        non-ASCII, non-space character; 'off' disables the check.
    order: protected tokens of one class in a different order. Multiset
        equality is always an error; the order is 'warning' (default), 'error'
        or 'off'. A NOTE on the block that mentions its anchor acknowledges
        a reorder (see apply).
    numbers: numerals of the English missing from the translation: 'warning'
        (default) or 'off'. `N×` and en-dash ranges count as numbers, not
        protected tokens: only their digits are compared (40–55 -> 40〜55 passes).
    """
    cjk_adjacent: str = 'letters'
    order: str = 'warning'
    numbers: str = 'warning'

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(**{k: str(v) for k, v in d.items() if k in cls.__dataclass_fields__})


_CJK_LETTER_RE = re.compile(
    '[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff'
    '\uff66-\uff9f\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\u3005\u3006\u30fc]')


def cjk_adjacent(text: str, mode: str = 'letters') -> list:
    """Roles or inline math set directly against a CJK character."""
    if mode == 'off':
        return []
    bad = []
    for t in tokens(text):
        if not (t.cls == 'ref' or (t.cls == 'math' and not t.text.startswith('$$'))):
            continue
        for ch in (text[t.start - 1] if t.start else '', text[t.end] if t.end < len(text) else ''):
            if not ch:
                continue
            hit = (_CJK_LETTER_RE.match(ch) if mode == 'letters'
                   else (ord(ch) > 127 and not ch.isspace()))
            if hit:
                bad.append(t.text)
                break
    return bad


@dataclass
class Token:
    cls: str
    text: str
    start: int
    end: int


def tokens(text: str) -> list:
    out = []
    for m in _TOKEN_RE.finditer(text):
        g = m.lastgroup
        if g == 'ticks':
            g = 'code'
        if g == 'esc':
            continue
        t = m.group(0)
        if g == 'dmath':
            g = 'math'
        elif g == 'filled':
            g = 'tag'
        elif g in ('autolink', 'url'):
            g = 'link'
        elif g == 'tag':
            t = _TRANSLATABLE_ATTR_RE.sub(lambda a: a.group(1) + '="…"', t)
        out.append(Token(g, t, m.start(), m.end()))
    return out


def strip_tokens(text: str, repl: str = ' ') -> str:
    """The text with every protected token replaced by `repl` (escapes kept)."""
    parts, last = [], 0
    for m in _TOKEN_RE.finditer(text):
        if m.lastgroup == 'esc':
            continue
        parts.append(text[last:m.start()])
        parts.append(repl)
        last = m.end()
    parts.append(text[last:])
    return ''.join(parts)


def closing_dollar_digit(text: str) -> list:
    """Inline-math spans whose closing `$` is directly followed by a digit."""
    bad = []
    for t in tokens(text):
        if t.cls == 'math' and t.text.startswith('$') and not t.text.startswith('$$'):
            if t.end < len(text) and text[t.end].isdigit():
                bad.append(t.text)
    return bad


def _emphasis_counts(s: str):
    s = re.sub(r'(?m)^\s*([-*+]|\d+[.)])\s+', ' ', s)    # list markers
    s = s.replace('\\*', '')
    strong = len(re.findall(r'\*\*', s))
    single = len(re.findall(r'(?<!\*)\*(?!\*)', s))
    return strong, single


def balance_problems(text: str) -> list:
    s = strip_tokens(text, repl='X')
    probs = []
    strong, single = _emphasis_counts(s)
    if strong % 2:
        probs.append('unbalanced ** emphasis')
    if single % 2:
        probs.append('unbalanced * emphasis')
    if s.count('[') != s.count(']'):
        probs.append('unbalanced [ ]')
    return probs


_LIST_PREFIX_RE = re.compile(r'^(\s*)([-*+]|\d{1,3}[.)])(\s+)')
_HEADING_PREFIX_RE = re.compile(r'^(#{1,6}\s+)')


def _prefix(kind: str, s: str) -> str:
    """The structural start of a block that a translation must keep."""
    if kind == 'heading':
        m = _HEADING_PREFIX_RE.match(s)
        return m.group(1).rstrip() + ' ' if m else ''
    m = _LIST_PREFIX_RE.match(s)
    if m:
        return m.group(1) + m.group(2)
    ind = s[:len(s) - len(s.lstrip(' '))]
    rest = s[len(ind):]
    for p, kinds in (('![', None), ('>', None), ('|', None), ('[', ('kicker',)),
                     (':', ('caption',))):
        if rest.startswith(p) and (kinds is None or kind in kinds):
            return ind + p
    return ind


def _cells(s: str) -> int:
    return len(re.findall(r'(?<!\\)\|', strip_tokens(s, 'X')))


def check_block(kind: str, english: str, tr: str, opts: CheckOptions | None = None) -> tuple:
    """Errors and warnings for one translated block against its English."""
    opts = opts or CheckOptions()
    errors, warnings = [], []
    if not tr.strip():
        return ['empty translation'], []
    if kind in ('slide_title', 'div_title'):
        blank = lambda s: re.sub(r'\btitle="[^"]*"', 'title=""', s)
        if blank(english) != blank(tr):
            errors.append('div fence changed outside the title attribute')
        if len(re.findall(r'\btitle="', tr)) != 1 or tr.count('"') % 2:
            errors.append('title attribute quotes broken')
        em = re.search(r'\btitle="([^"]*)"', english)
        tm = re.search(r'\btitle="([^"]*)"', tr)
        english, tr = (em.group(1) if em else ''), (tm.group(1) if tm else '')
    elif kind == 'meta_title':
        if english.split(':', 1)[0] != tr.split(':', 1)[0]:
            errors.append('front-matter key changed')

    et, tt = tokens(english), tokens(tr)
    for cls in CLASSES:
        a = [' '.join(t.text.split()) for t in et if t.cls == cls]
        b = [' '.join(t.text.split()) for t in tt if t.cls == cls]
        if Counter(a) != Counter(b):
            missing = list((Counter(a) - Counter(b)).elements())
            extra = list((Counter(b) - Counter(a)).elements())
            msg = f'{cls} tokens differ'
            if missing:
                msg += ' | missing: ' + ' '.join(_short(x) for x in missing[:4])
            if extra:
                msg += ' | added: ' + ' '.join(_short(x) for x in extra[:4])
            errors.append(msg)
        elif a != b and opts.order != 'off':
            (errors if opts.order == 'error' else warnings).append(f'{cls} tokens reordered')

    if kind not in ('slide_title', 'div_title', 'html'):
        pe, pt = _prefix(kind, english), _prefix(kind, tr)
        if pe != pt:
            errors.append(f'line prefix changed: {pe!r} -> {pt!r}')
    if kind == 'table_row' and _cells(english) != _cells(tr):
        errors.append('table cell count changed')
    if kind == 'list_item' or kind == 'exercise':
        ne = len(re.findall(r'(?m)^\s*([-*+]|\d{1,3}[.)])\s', english))
        nt = len(re.findall(r'(?m)^\s*([-*+]|\d{1,3}[.)])\s', tr))
        if ne != nt:
            warnings.append(f'list markers: {ne} in English, {nt} in translation')

    be = set(balance_problems(english))
    for p in balance_problems(tr):
        if p not in be:
            errors.append(p)

    if BIDI_RE.search(tr):
        errors.append('bidi control character')
    if closing_dollar_digit(tr) and not closing_dollar_digit(english):
        errors.append('closing $ followed by a digit: ' + _short(closing_dollar_digit(tr)[0]))

    adj = cjk_adjacent(tr, opts.cjk_adjacent)
    if adj:
        errors.append('role or inline math touching a CJK character (add a space): '
                      + ' '.join(_short(x, 30) for x in adj[:3]))

    if opts.numbers == 'off':
        return errors, warnings
    ne = Counter(NUMBER_RE.findall(strip_tokens(english)))
    nt = Counter(NUMBER_RE.findall(strip_tokens(tr)))
    if ne != nt:
        miss = list((ne - nt).elements())
        if miss:
            warnings.append('numbers not found: ' + ' '.join(miss[:6]))
    return errors, warnings


def _short(s: str, n: int = 40) -> str:
    s = ' '.join(s.split())
    return s if len(s) <= n else s[:n - 1] + '…'
