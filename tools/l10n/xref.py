"""Section and chapter references in the word order of the language.

Quarto words a reference as prefix + number: "Section 2.1.3", or in Japanese
"セクション 2.1.3" (its language file, or a `quarto.crossref-*-prefix`
override in <lang>/strings.yml). Quarto has no suffix form, so a language
that puts the word after or around the number sets templates in
<lang>/strings.yml (localization-side keys, not in the tag's i18n/en.yml):

    xref.sec: "{n} 節"        # sections and subsections: 2.1.3 節
    xref.ch: "第 {n} 章"      # chapters: 第 3 章

Unset, a reference keeps Quarto's form. A space of the template is a no-break
space in the output (`&nbsp;` in HTML, `~` in LaTeX), so number and word do
not part at a line break. The build (build.py, DESIGN.md section 7) applies
the templates to

* the HTML book after make (`rewrite_book`): the text of each `quarto-xref`
  link, `<span>セクション 2.1.3</span>`, after the tag's
  fix_crossref_numbers.py has renumbered it (it needs Quarto's words: it
  finds the references by them and turns "Chapter 8.1" into "Section 8.1" for
  section pages), and the same phrases in search.json;
* the LaTeX of the PDFs (`rewrite_tex`), run by the worktree's
  build_one_pdf.sh after fix_latex.py and before xelatex: gen_pdf.py writes
  `<pdf.section>~\\ref{sec-x}` and `<pdf.chapter>~\\ref{sec-chap-x}`
  (fix_latex.py has by then replaced the references to the unnumbered front
  matter by their titles); the part PDFs are cut from this .tex;
* the links to English pages of out-of-scope chapters (scope.ref_text).

This file uses the standard library only: the build copies it into the
worktree as .l10n/xref.py, next to the build's settings .l10n/xref.json, and
build_one_pdf.sh runs

    python3 .l10n/xref.py tex _pdf/<fw>/Dive-into-Deep-Learning.tex
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

KINDS = ('sec', 'ch')
NUM = r'\d+(?:\.\d+)*'
NBSP = '\xa0'


def templates(flat: dict) -> dict:
    """{'sec': ..., 'ch': ...}: the xref.* templates a flat strings dict sets."""
    out = {}
    for kind in KINDS:
        v = flat.get(f'xref.{kind}')
        if v is None:
            continue
        if not isinstance(v, str) or v.count('{n}') != 1 or not v.replace('{n}', '').strip():
            raise ValueError(f'xref.{kind}: expected a string with one {{n}} and a word, '
                             f'got {v!r}')
        out[kind] = v
    return out


def fill(template: str, n: str, space: str = NBSP) -> str:
    """The template with the number (or LaTeX `\\ref{...}`) n; its spaces
    become `space`."""
    head, tail = template.split('{n}')
    return head.replace(' ', space) + n + tail.replace(' ', space)


def _alt(words) -> str:
    return '|'.join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))


def _kind(cfg: dict, words: str, word: str, num: str) -> str | None:
    """The kind of a reference that Quarto (or gen_pdf) wrote with `word`.
    If one word serves both kinds, a dotted number is a section."""
    kinds = [k for k in KINDS if cfg[words].get(k) == word]
    if len(kinds) > 1:
        kinds = ['sec' if '.' in num else 'ch']
    kind = kinds[0] if kinds else None
    return kind if kind in cfg['templates'] else None


# ── HTML ─────────────────────────────────────────────────────────────

def _html_re(cfg: dict):
    words = [cfg['html_words'][k] for k in cfg['templates'] if cfg['html_words'].get(k)]
    if not words:
        return None
    return re.compile(r'(<a\b[^>]*\bquarto-xref\b[^>]*>\s*<span>)(' + _alt(words)
                      + r')(?: |&nbsp;|\xa0)(' + NUM + r')(</span>)')


def rewrite_html(text: str, cfg: dict) -> tuple:
    """(text, n): the quarto-xref link texts in the template form."""
    pat = _html_re(cfg)
    if pat is None:
        return text, 0
    n = 0

    def sub(m):
        nonlocal n
        kind = _kind(cfg, 'html_words', m.group(2), m.group(3))
        if kind is None:
            return m.group(0)
        n += 1
        return m.group(1) + fill(cfg['templates'][kind], m.group(3), '&nbsp;') + m.group(4)
    return pat.sub(sub, text), n


def rewrite_search(data: list, cfg: dict) -> int:
    """Rewrite search.json entries in place (plain text: "セクション 2.1.3")."""
    words = [cfg['html_words'][k] for k in cfg['templates'] if cfg['html_words'].get(k)]
    if not words:
        return 0
    pat = re.compile(r'(' + _alt(words) + r')[ \xa0](' + NUM + r')')
    n = 0

    def sub(m):
        nonlocal n
        kind = _kind(cfg, 'html_words', m.group(1), m.group(2))
        if kind is None:
            return m.group(0)
        n += 1
        return fill(cfg['templates'][kind], m.group(2), NBSP)
    for entry in data:
        if not isinstance(entry, dict):
            continue
        for field in ('text', 'title', 'section'):
            v = entry.get(field)
            if isinstance(v, str) and v:
                entry[field] = pat.sub(sub, v)
    return n


def left_html(text: str, cfg: dict) -> int:
    """quarto-xref link texts still in Quarto's form (for the report)."""
    pat = _html_re(cfg)
    return len(pat.findall(text)) if pat else 0


SKIP_DIRS = {'site_libs', 'notebooks', 'pdf', 'static', 'img'}


def rewrite_book(book: Path, cfg: dict) -> dict:
    """Rewrite the HTML pages and search.json of a rendered _book/ (idempotent)."""
    stats = {'files': 0, 'refs': 0, 'search': 0, 'left': 0}
    words = [cfg['html_words'][k] for k in cfg['templates'] if cfg['html_words'].get(k)]
    for p in sorted(book.rglob('*.html')):
        if p.relative_to(book).parts[0] in SKIP_DIRS:
            continue
        text = p.read_text(encoding='utf-8')
        if not any(w in text for w in words):
            continue
        new, n = rewrite_html(text, cfg)
        stats['left'] += left_html(new, cfg)
        if n:
            p.write_text(new, encoding='utf-8')
            stats['files'] += 1
            stats['refs'] += n
    search = book / 'search.json'
    if search.exists():
        data = json.loads(search.read_text(encoding='utf-8'))
        n = rewrite_search(data, cfg)
        if n:
            search.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            stats['search'] = n
    return stats


# ── LaTeX ────────────────────────────────────────────────────────────

def rewrite_tex(text: str, cfg: dict) -> tuple:
    """(text, n): gen_pdf.py's `<word>~\\ref{sec-...}` in the template form.
    The label decides the kind: sec-chap-* is a chapter (gen_pdf.py)."""
    words = [cfg['tex_words'][k] for k in cfg['templates'] if cfg['tex_words'].get(k)]
    if not words:
        return text, 0
    pat = re.compile(r'(' + _alt(words) + r')~(\\ref\{(sec-[^}]+)\})')
    n = 0

    def sub(m):
        nonlocal n
        kind = 'ch' if m.group(3).startswith('sec-chap-') else 'sec'
        if kind not in cfg['templates'] or cfg['tex_words'].get(kind) != m.group(1):
            return m.group(0)
        n += 1
        return fill(cfg['templates'][kind], m.group(2), '~')
    return pat.sub(sub, text), n


def load(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding='utf-8'))
    for key in ('templates', 'html_words', 'tex_words'):
        cfg.setdefault(key, {})
    return cfg


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] != 'tex':
        print('usage: xref.py tex FILE.tex [xref.json]', file=sys.stderr)
        return 2
    tex = Path(argv[1])
    cfg = load(Path(argv[2]) if len(argv) > 2 else Path(__file__).with_name('xref.json'))
    text = tex.read_text(encoding='utf-8')
    new, n = rewrite_tex(text, cfg)
    if new != text:
        tex.write_text(new, encoding='utf-8')
    print(f'l10n xref: {n} section/chapter reference(s) reworded in {tex}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
