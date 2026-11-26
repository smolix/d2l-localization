"""Pseudo-locale: wrap the visible text of each block in ⟦…⟧, keep every token.

Used to test the pipeline end to end (language `xx`): every translated string
is visibly marked, and `l10n check` must pass on the result.
"""

from __future__ import annotations

import re

from .protect import _TOKEN_RE

L, R = '⟦', '⟧'


def _covered(s: str) -> list:
    covered = [False] * len(s)
    for m in _TOKEN_RE.finditer(s):
        if m.lastgroup != 'esc':
            for k in range(m.start(), m.end()):
                covered[k] = True
    return covered


def _wrap_text(s: str) -> str:
    """Wrap the stretch from the first to the last non-token character."""
    covered = _covered(s)
    free = [k for k, ch in enumerate(s) if not covered[k] and not ch.isspace()]
    if not free:
        return s
    a, b = free[0], free[-1] + 1
    return s[:a] + L + s[a:b] + R + s[b:]


def _bracket_end(s: str, start: int) -> int:
    """Index of the `]` matching the `[` at s[start]."""
    depth = 0
    k = start
    while k < len(s):
        if s[k] == '\\':
            k += 2
            continue
        if s[k] == '[':
            depth += 1
        elif s[k] == ']':
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return -1


def pseudo(kind: str, content: str) -> str:
    if kind == 'heading':
        m = re.match(r'^(#+\s+)(.*?)(\s*\{[^{}]*\})?$', content)
        return m.group(1) + _wrap_text(m.group(2)) + (m.group(3) or '')
    if kind in ('slide_title', 'div_title'):
        return re.sub(r'\btitle="([^"]*)"', lambda m: f'title="{_wrap_text(m.group(1))}"',
                      content, count=1)
    if kind == 'meta_title':
        m = re.match(r'^(\w+:\s*)(["\']?)(.*?)(\2)\s*$', content)
        return m.group(1) + m.group(2) + L + m.group(3) + R + m.group(4)
    if kind == 'table_row':
        cov = _covered(content)
        cuts = [k for k, ch in enumerate(content) if ch == '|' and not cov[k]
                and (k == 0 or content[k - 1] != '\\')]
        out, last = [], 0
        for k in cuts + [len(content)]:
            out.append(_wrap_text(content[last:k]) if content[last:k].strip() else content[last:k])
            out.append(content[k:k + 1])
            last = k + 1
        return ''.join(out)
    if kind == 'html':
        return re.sub(r'>([^<>]+)<', lambda m: '>' + _wrap_text(m.group(1)) + '<', content) \
            if '>' in content else _wrap_text(content)
    if kind == 'kicker' or (kind in ('caption', 'slide_text') and content.startswith('![')):
        start = 1 if content.startswith('![') else 0
        end = _bracket_end(content, start)
        if end > 0:
            return content[:start + 1] + _wrap_text(content[start + 1:end]) + content[end:]
    m = re.match(r'^(\s*(?:[-*+]|\d{1,3}[.)])\s+|\s*>\s?' + ('|:\\s?' if kind == 'caption' else '') + ')?',
                 content)
    head = m.group(0) if m.group(1) else ''
    if not head:
        ind = len(content) - len(content.lstrip(' '))
        head = content[:ind]
    return head + _wrap_text(content[len(head):])
