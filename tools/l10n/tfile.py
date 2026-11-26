"""The translated file: <lang>/book/<english path>.

It holds only translatable blocks, in English order, each under its anchor:

    <!--en:3f2a91c0-->
    ## 損失関数

    <!--en:a90f13c2 was=8c11d0e4 from=en-2026.11.01-->
    An older translation, of the English block 8c11d0e4 at en-2026.11.01.

The anchor is the hash of the English block at the aligned revision (with an
occurrence suffix `-2`, `-3`, ... for repeats in the file). `was`/`from`
(and `src` when the English file was renamed) mark a stale translation left by
`l10n bump`: it still translates the old English block and is shown only after
an update unit refreshes it. The build reconstructs the full page from the
English file at the aligned revision, so code, display math, comments and
structure are never stored here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ANCHOR_LINE_RE = re.compile(
    r'^<!--en:(?P<anchor>[0-9a-f]{8}(?:-\d+)?)(?P<attrs>(?: [a-z]+=[^\s>]+)*)-->$')


@dataclass
class Entry:
    text: str
    was: str | None = None
    src_rev: str | None = None
    src_path: str | None = None

    @property
    def stale(self) -> bool:
        return self.was is not None


class TFileError(ValueError):
    pass


def parse(text: str) -> dict:
    """anchor -> Entry, in file order."""
    entries: dict = {}
    cur, buf = None, []

    def flush():
        if cur is None:
            body = '\n'.join(buf).strip()
            if body:
                raise TFileError('text before the first anchor: ' + body[:60])
            return
        anchor, attrs = cur
        if anchor in entries:
            raise TFileError(f'duplicate anchor {anchor}')
        entries[anchor] = Entry('\n'.join(buf).strip('\n'), attrs.get('was'),
                                attrs.get('from'), attrs.get('src'))

    for line in text.split('\n'):
        m = ANCHOR_LINE_RE.match(line)
        if m:
            flush()
            attrs = dict(a.split('=', 1) for a in m.group('attrs').split())
            cur, buf = (m.group('anchor'), attrs), []
        else:
            buf.append(line)
    flush()
    return entries


def render(entries: dict) -> str:
    out = []
    for anchor, e in entries.items():
        attrs = ''
        if e.was:
            attrs += f' was={e.was}'
            if e.src_rev:
                attrs += f' from={e.src_rev}'
            if e.src_path:
                attrs += f' src={e.src_path}'
        out.append(f'<!--en:{anchor}{attrs}-->\n{e.text}\n')
    return '\n'.join(out)


def path_for(book_dir: Path, english_path: str) -> Path:
    return book_dir / english_path


def load(book_dir: Path, english_path: str) -> dict:
    p = path_for(book_dir, english_path)
    if not p.exists():
        return {}
    return parse(p.read_text(encoding='utf-8'))


def save(book_dir: Path, english_path: str, entries: dict, order=None):
    """Write entries, in the order of `order` (English anchors) when given."""
    p = path_for(book_dir, english_path)
    if order is not None:
        ordered = {a: entries[a] for a in order if a in entries}
        ordered.update({a: e for a, e in entries.items() if a not in ordered})
        entries = ordered
    if not entries:
        if p.exists():
            p.unlink()
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(entries), encoding='utf-8')


def translations(entries: dict, stale: str = 'english') -> dict:
    """anchor -> text to show; stale entries only when stale='translation'."""
    return {a: e.text for a, e in entries.items() if not e.stale or stale == 'translation'}
