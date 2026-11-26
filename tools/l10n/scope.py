"""Translation scope (DESIGN.md section 13): which parts of the book a language
translates, and how a build shows the rest.

`scope:` in languages.yml (a default for every language, overridable per
language) names the part directories (`chapter_basics`) or chapter
directories (`chapter_lm-agents`) of the pinned tag's `_quarto.yml` that are
translated; `all` (the default when the key is absent) is the whole book.
The landing page `index.md`, top-level pages and files outside the book's
part structure are always in scope. Membership comes from the parts of the
tag's `_quarto.yml`, so a renumbering of the English book changes nothing.

Out of scope, in a build of the language (`build.py` calls `prepare` before
make and `postprocess` after it):

* each out-of-scope chapter renders as ONE placeholder page (its index page,
  same path, English number, translated title if one exists): "not yet
  translated" and the chapter's section list, linking to the English pages;
  an out-of-scope part page is a placeholder that lists its chapters. The
  placeholders carry `<meta name="robots" content="noindex">`;
* the section pages of out-of-scope chapters are not built: they are dropped
  from `_quarto.yml` (their sources stay on disk, so the d2l library, the
  notebooks and the freshness gate are unchanged);
* `:numref:`/`:ref:`/`:eqref:` from translated pages into labels of
  out-of-scope pages become links to the English page on d2l.ai with the
  English number and the "(English)" marker;
* the sidebar marks out-of-scope parts and chapters "(English)";
* PDFs leave out-of-scope chapters out (the tag's `PDF_EXCLUDED_FILES`
  mechanism; `fix_latex.py` keeps each remaining chapter's English number),
  slides and notebook zips skip them, and part PDFs are built for in-scope
  parts only.

The tag's tools have no scope switch, so `prepare` applies a few exact,
anchored edits to the worktree's copies (`PATCHES`); an anchor that is not
found stops the build with a message instead of building something wrong.
"""

from __future__ import annotations

import functools
import html
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import config, tfile
from .units import english_blocks

ALL = 'all'

# UI strings the localization side adds (not in the tag's i18n/en.yml):
# English defaults; `l10n strings <lang>` puts them into <lang>/strings.yml.
STRINGS = {
    'l10n.english_marker': '(English)',
    'l10n.chapter_not_translated': 'This chapter has not been translated yet. '
                                   'It is available in English:',
    'l10n.part_not_translated': 'This part has not been translated yet. '
                                'Its chapters are available in English:',
}

MARKER_CLASS = 'd2l-english-marker'
MARKER_STYLE = 'font-size:0.8em;opacity:0.7;'
PLACEHOLDER_COMMENT = '<!-- l10n: placeholder for a chapter outside the translation scope -->'
NOINDEX = '<meta name="robots" content="noindex">'


# ── book structure from the tag's _quarto.yml ────────────────────────

@dataclass
class Chapter:
    dir: str
    files: list                 # .md files in _quarto.yml order (index.md first)
    part: str | None            # the part directory, None for a top-level entry

    @property
    def page(self) -> str:
        return f'{self.dir}/index.md'


@dataclass
class Part:
    dir: str
    page: str                   # chapter_basics/index.md
    chapters: list              # chapter directories


@dataclass
class Structure:
    parts: dict                 # dir -> Part, book order
    chapters: dict              # dir -> Chapter, book order
    top: list                   # top-level .md entries (index.md, references.md)


@functools.lru_cache(maxsize=16)
def structure(rev: str) -> Structure:
    text = config.show(rev, '_quarto.yml')
    if text is None:
        raise config.L10nError(f'_quarto.yml does not exist at {rev}')
    cfg = yaml.safe_load(text) or {}
    parts, chapters, top = {}, {}, []

    def add_chapter(qmd, part):
        rel = qmd[:-4] + '.md' if qmd.endswith('.qmd') else qmd
        d = rel.split('/')[0] if '/' in rel else None
        if d is None:
            top.append(rel)
            return
        ch = chapters.setdefault(d, Chapter(d, [], part))
        ch.files.append(rel)
        if part and d not in parts[part].chapters:
            parts[part].chapters.append(d)

    for item in (cfg.get('book') or {}).get('chapters') or []:
        if isinstance(item, str):
            add_chapter(item, None)
        elif isinstance(item, dict) and 'part' in item:
            page = item['part'][:-4] + '.md'
            d = page.split('/')[0]
            parts[d] = Part(d, page, [])
            for c in item.get('chapters') or []:
                add_chapter(c, d)
    return Structure(parts, chapters, top)


# ── scope ────────────────────────────────────────────────────────────

@dataclass
class Scope:
    code: str
    rev: str
    spec: object
    st: Structure
    in_parts: set = field(default_factory=set)
    in_chapters: set = field(default_factory=set)

    @property
    def full(self) -> bool:
        return (set(self.st.parts) <= self.in_parts
                and set(self.st.chapters) <= self.in_chapters)

    def includes(self, path: str) -> bool:
        """True if the book file `path` is translated in this scope."""
        d = path.split('/')[0] if '/' in path else None
        if d is None:
            return True
        if d in self.st.chapters:
            return d in self.in_chapters
        if d in self.st.parts:
            return d in self.in_parts
        return True                          # outside the part structure

    @property
    def out_chapters(self) -> list:
        return [c for d, c in self.st.chapters.items() if d not in self.in_chapters]

    @property
    def out_parts(self) -> list:
        return [p for d, p in self.st.parts.items() if d not in self.in_parts]

    @property
    def unbuilt(self) -> list:
        """Listed section pages of out-of-scope chapters: not rendered at all."""
        return [f for c in self.out_chapters for f in c.files if f != c.page]

    @property
    def placeholders(self) -> list:
        """Out-of-scope part pages and chapter pages, rendered as placeholders."""
        return [p.page for p in self.out_parts] + \
            [c.page for c in self.out_chapters if c.page in c.files]

    @property
    def out_listed(self) -> list:
        """Every out-of-scope page listed in _quarto.yml (book order)."""
        out = []
        for p in self.st.parts.values():
            if p.dir not in self.in_parts:
                out.append(p.page)
            for d in p.chapters:
                if d not in self.in_chapters:
                    out += self.st.chapters[d].files
        for d, c in self.st.chapters.items():
            if c.part is None and d not in self.in_chapters:
                out += c.files
        return out

    def summary(self) -> str:
        if self.full:
            return 'scope: the whole book'
        files = config.book_files(self.rev)
        n_in = sum(1 for f in files if self.includes(f))
        out = []
        for p in self.st.parts.values():
            gone = [d for d in p.chapters if d not in self.in_chapters]
            if gone and len(gone) == len(p.chapters):
                out.append(f'{p.dir} ({len(gone)} chapter{"s" if len(gone) > 1 else ""})')
            else:
                out += gone
        return f'scope: {n_in} of {len(files)} files; not translated: ' + ', '.join(out)


def resolve(code: str, spec, rev: str) -> Scope:
    st = structure(rev)
    sc = Scope(code, rev, spec, st)
    if spec is None or spec == ALL or spec == [ALL]:
        sc.in_parts, sc.in_chapters = set(st.parts), set(st.chapters)
        return sc
    names = [spec] if isinstance(spec, str) else list(spec)
    for n in names:
        n = str(n).rstrip('/')
        if n in st.parts:
            sc.in_parts.add(n)
            sc.in_chapters |= set(st.parts[n].chapters)
        elif n in st.chapters:
            sc.in_chapters.add(n)
            if st.chapters[n].part:
                sc.in_parts.add(st.chapters[n].part)
        else:
            raise config.L10nError(
                f'scope of {code}: {n!r} is neither a part nor a chapter directory of '
                f'_quarto.yml at {rev} (parts: {", ".join(st.parts)})')
    # chapters outside every part (top-level entries) are always translated
    sc.in_chapters |= {d for d, c in st.chapters.items() if c.part is None}
    return sc


def for_language(lang, rev: str | None = None) -> Scope:
    rev = rev or lang.aligned
    if not rev:
        raise config.L10nError(f'{lang.code} has no pinned revision')
    return resolve(lang.code, lang.scope, rev)


# ── UI strings of the localization side ──────────────────────────────

def ui_strings(lang) -> dict:
    """STRINGS with the language's values from <lang>/strings.yml."""
    out = dict(STRINGS)
    path = lang.root / 'strings.yml'
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        flat = _flatten(data)
        out.update({k: v for k, v in flat.items() if k in STRINGS and isinstance(v, str)})
    return out


def _flatten(d, prefix=''):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flatten(v, f'{prefix}{k}.'))
        else:
            out[f'{prefix}{k}'] = v
    return out


def build_locale_file(lang, wt: Path) -> Path | None:
    """The D2L_LOCALE_FILE for the tag's tools: <lang>/strings.yml without
    the localization-side keys (the tag's i18n.py warns about unknown keys)."""
    path = lang.root / 'strings.yml'
    if not path.exists():
        return None
    text = path.read_text(encoding='utf-8')
    if not re.search(r'(?m)^l10n\.', text):
        return path.resolve()
    flat = {k: v for k, v in _flatten(yaml.safe_load(text) or {}).items()
            if not k.startswith('l10n.')}
    out = wt / '.l10n' / 'strings.yml'
    out.parent.mkdir(exist_ok=True)
    out.write_text('# Generated by l10n build from ' + str(path) + ' (l10n.* keys removed)\n'
                   + yaml.safe_dump(flat, allow_unicode=True, sort_keys=False, width=1000),
                   encoding='utf-8')
    return out


# ── English labels, titles, references ───────────────────────────────

LABEL_RE = re.compile(r':(label|eqlabel):`([^`]+)`')
# The tag's d2l_preprocess.translate_directives turns these into @-refs.
REF_RE = re.compile(r'(?<!`):(numref|ref|eqref):`{1,2}([^`\s]+)`{1,2}')
FENCE_RE = re.compile(r'^\s*(`{3,}|~{3,})')


def _outside_fences(text: str):
    """Yield (is_code, chunk) over the lines of text."""
    fence = None
    buf, code = [], False
    for line in text.splitlines(keepends=True):
        m = FENCE_RE.match(line)
        if fence is None and m:
            if buf:
                yield code, ''.join(buf)
            buf, code, fence = [line], True, m.group(1)
            continue
        buf.append(line)
        if fence is not None and m and m.group(1)[0] == fence[0] \
                and len(m.group(1)) >= len(fence):
            yield True, ''.join(buf)
            buf, code, fence = [], False, None
    if buf:
        yield code, ''.join(buf)


def h1(text: str):
    """(title, label) of a page's `# ` heading (label on the next non-blank line)."""
    lines = text.split('\n')
    fence = None
    for i, line in enumerate(lines):
        m = FENCE_RE.match(line)
        if m:
            fence = None if fence else m.group(1)
            continue
        if fence or not line.startswith('# '):
            continue
        title = re.sub(r'\s*(\{#[^}]*\}|:label:`[^`]*`)', '', line[2:]).strip()
        label = None
        for nxt in lines[i + 1:]:
            if nxt.strip():
                lm = re.match(r'^:label:`([^`]+)`\s*$', nxt)
                label = lm.group(1) if lm else None
                break
        return title, label
    return '', None


def out_labels(sc: Scope) -> dict:
    """{label: (file, 'label'|'eqlabel')} defined in out-of-scope listed pages
    and not also in an in-scope page."""
    labels = {}
    for rel in sc.out_listed:
        text = config.show(sc.rev, rel) or ''
        for kind, label in LABEL_RE.findall(text):
            labels.setdefault(label, (rel, kind))
    return labels


def references(text: str) -> set:
    """Labels referenced by :numref:/:ref:/:eqref: outside code fences."""
    out = set()
    for code, chunk in _outside_fences(text):
        if not code:
            out |= {m.group(2) for m in REF_RE.finditer(chunk)}
    return out


def english_url(base: str, rel: str, anchor: str | None = None) -> str:
    page = rel[:-3] + '.html'
    return f'{base.rstrip("/")}/{page}' + (f'#{anchor}' if anchor else '')


def _md_text(s: str) -> str:
    return s.replace('\\', '\\\\').replace('[', '\\[').replace(']', '\\]')


def ref_text(info: dict, terms: dict) -> str:
    """'Section 18.3' as Quarto words it in the build language. A narrow
    no-break space joins word and number: the tag's fix_crossref_numbers.py
    renumbers 'Section 18' (space or &nbsp;) as a local reference."""
    word = {'chapter': 'Chapter', 'section': 'Section', 'fig': 'Figure', 'tbl': 'Table',
            'eq': 'Equation'}.get(info['kind'], 'Section')
    if info.get('number'):
        return f'{terms.get(word, word)} {info["number"]}'
    return info.get('title') or info['id']


def ref_link(info: dict, terms: dict, marker: str, base: str) -> str:
    url = english_url(base, info['file'], None if info.get('h1') else info['id'])
    return (f'[{_md_text(ref_text(info, terms))} {_md_text(marker)}]({url})'
            '{.d2l-english-ref hreflang="en"}')


def rewrite_refs(text: str, table: dict, terms: dict, marker: str, base: str) -> tuple:
    """Replace references to labels in `table` by English links; (text, n)."""
    n = 0

    def sub(m):
        nonlocal n
        info = table.get(m.group(2))
        if info is None:
            return m.group(0)
        n += 1
        return ref_link(info, terms, marker, base)

    out = []
    for code, chunk in _outside_fences(text):
        out.append(chunk if code else REF_RE.sub(sub, chunk))
    return ''.join(out), n


def run_helper(wt: Path, env: dict, labels: dict) -> dict:
    """Ask the worktree's own tools for words, numbering and label numbers."""
    req = {'labels': {}}
    for label, (rel, kind) in sorted(labels.items()):
        req['labels'].setdefault(rel, []).append([label, kind])
    with tempfile.TemporaryDirectory(prefix='l10n-scope-') as tmp:
        q, a = Path(tmp) / 'req.json', Path(tmp) / 'ans.json'
        q.write_text(json.dumps(req), encoding='utf-8')
        r = subprocess.run(['python3', str(Path(__file__).with_name('_wt_helper.py')),
                            str(q), str(a)], cwd=wt, env=env, capture_output=True, text=True)
        if r.returncode or not a.exists():
            raise config.L10nError('scope helper failed in the worktree:\n'
                                   + (r.stderr or r.stdout)[-2000:])
        return json.loads(a.read_text(encoding='utf-8'))


def label_table(sc: Scope, wanted: set, helper: dict) -> dict:
    """{label: info} for the referenced out-of-scope labels."""
    labels = out_labels(sc)
    numbering = helper['numbering']
    table = {}
    for label in sorted(wanted):
        if label not in labels:
            continue
        rel, _ = labels[label]
        title, h1label = h1(config.show(sc.rev, rel) or '')
        if label == h1label:
            nums = numbering.get(rel)
            table[label] = {'file': rel, 'id': label, 'h1': True, 'title': title,
                            'kind': 'chapter' if nums and len(nums) == 1 else 'section',
                            'number': '.'.join(str(n) for n in nums) if nums else None}
        elif label in helper['labels']:
            table[label] = dict(helper['labels'][label], h1=False)
    return table


# ── placeholders ─────────────────────────────────────────────────────

def translated_h1(lang, rev: str, rel: str) -> str | None:
    """The translated `# ` heading of a page, if <lang>/book has one."""
    entries = tfile.load(lang.book, rel)
    if not entries:
        return None
    for b in english_blocks(rev, rel):
        if b.kind == 'heading' and not b.in_slides and b.content.startswith('# '):
            e = entries.get(b.anchor)
            if e and not e.stale:
                return e.text.lstrip('#').strip()
            return None
    return None


def _num(nums) -> str:
    return '.'.join(str(n) for n in nums) if nums else ''


def _item(nums, title, url) -> str:
    label = f'{_num(nums)} {title}'.strip()
    return f'- [{_md_text(label)}]({url}){{hreflang="en"}}'


def placeholder(lang, sc: Scope, rel: str, numbering: dict, strings: dict, base: str) -> str:
    """The placeholder page for an out-of-scope chapter page or part page."""
    en_title, label = h1(config.show(sc.rev, rel) or '')
    title = translated_h1(lang, sc.rev, rel) or en_title
    d = rel.split('/')[0]
    lines = [f'# {title}']
    if label:
        lines.append(f':label:`{label}`')
    lines += ['', PLACEHOLDER_COMMENT, '']
    if d in sc.st.parts and sc.st.parts[d].page == rel:
        text = strings['l10n.part_not_translated']
        pages = [sc.st.chapters[c].page for c in sc.st.parts[d].chapters
                 if c not in sc.in_chapters]
    else:
        text = strings['l10n.chapter_not_translated']
        pages = list(sc.st.chapters[d].files)
    lines += ['::: {.callout-note appearance="simple"}', text, ':::', '']
    for p in pages:
        t, _ = h1(config.show(sc.rev, p) or '')
        lines.append(_item(numbering.get(p), t, english_url(base, p)))
    return '\n'.join(lines) + '\n'


# ── _quarto.yml ──────────────────────────────────────────────────────

def filter_quarto(text: str, sc: Scope) -> str:
    """Drop the unbuilt section pages and the out-of-scope part PDFs."""
    for rel in sc.unbuilt:
        qmd = rel[:-3] + '.qmd'
        text, n = re.subn(r'(?m)^[ \t]*- ' + re.escape(qmd) + r'[ \t]*\n', '', text)
        if n != 1:
            raise config.L10nError(f'_quarto.yml: cannot drop {qmd} ({n} matches)')
    for p in sc.out_parts:
        slug = p.dir[len('chapter_'):] if p.dir.startswith('chapter_') else p.dir
        text = re.sub(r'(?m)^([ \t]*)- text: [^\n]*\n\1  href: /pdf/parts/Dive-into-Deep-'
                      r'Learning-' + re.escape(slug) + r'-[a-z]+\.pdf[ \t]*\n', '', text)
    cfg = yaml.safe_load(text)
    listed = []
    for item in cfg['book']['chapters']:
        if isinstance(item, dict):
            listed += item.get('chapters') or []
        else:
            listed.append(item)
    left = {c[:-4] + '.md' for c in listed} & set(sc.unbuilt)
    if left:
        raise config.L10nError(f'_quarto.yml still lists {sorted(left)[:3]}')
    return text


# ── edits to the worktree's tools ────────────────────────────────────

def _pylist(items) -> str:
    return '{\n' + ''.join(f'    {json.dumps(x)},\n' for x in sorted(items)) + '}'


def patches(sc: Scope, pdf_parts: list) -> list:
    """[(path, old, new)]: exact edits to the tag's tools (see module doc)."""
    out_files = sorted(set(sc.out_listed))
    block = ('# >>> l10n scope: written by d2l-localization `l10n build`, not in the tag.\n'
             '# Out-of-scope pages: left out of the PDFs (their section pages are not\n'
             '# rendered at all; their chapter pages are HTML placeholders).\n'
             f'L10N_OUT_OF_SCOPE = frozenset({_pylist(out_files)})\n'
             f'L10N_UNBUILT = frozenset({_pylist(sc.unbuilt)})\n'
             'PDF_EXCLUDED_FILES = PDF_EXCLUDED_FILES | L10N_OUT_OF_SCOPE\n'
             '# <<< l10n scope\n')
    anchor_pdf = 'PDF_CHAPTER_FILES = [\n    rel for rel in CHAPTER_FILES if rel not in PDF_EXCLUDED_FILES\n]\n'
    guard = ("        dict_order = [rel[:-3] for rel in CHAPTER_NUMBERING\n"
             "                      if rel != 'chapter_references/zreferences.md']\n")
    guard_new = ("        dict_order = [rel[:-3] for rel in CHAPTER_NUMBERING\n"
                 "                      if rel != 'chapter_references/zreferences.md'\n"
                 "                      and rel not in __import__('d2l_preprocess').L10N_UNBUILT]"
                 "  # l10n scope\n")
    decks = '    for rel in CHAPTER_NUMBERING:\n        relp = Path(rel)\n'
    decks_new = ('    for rel in CHAPTER_NUMBERING:\n'
                 "        if rel in __import__('d2l_preprocess').L10N_OUT_OF_SCOPE:  # l10n scope\n"
                 '            continue\n'
                 '        relp = Path(rel)\n')
    parts = 'MIN_PART_PDF_PAGES = 20\n'
    parts_new = (f'PDF_PARTS = [p for p in PDF_PARTS if p in {pdf_parts!r}]  # l10n scope\n'
                 + parts)
    return [('tools/d2l_preprocess.py', anchor_pdf, block + anchor_pdf),
            ('tools/gen_pdf.py', guard, guard_new),
            ('tools/check_book_artifacts.py', decks, decks_new),
            ('tools/check_book_artifacts.py', parts, parts_new)]


def apply_patches(wt: Path, edits: list):
    for rel, old, new in edits:
        p = wt / rel
        text = p.read_text(encoding='utf-8')
        if text.count(old) != 1:
            raise config.L10nError(
                f'scope: cannot edit {rel} of this English tag (anchor found {text.count(old)} '
                'times); the tag\'s tools changed, update PATCHES in tools/l10n/scope.py')
        p.write_text(text.replace(old, new), encoding='utf-8')


def tag_pdf_parts(wt: Path) -> list:
    """The part-PDF slugs of the tag (make/render.mk PDF_PARTS)."""
    for name in ('make/render.mk', 'Makefile'):
        p = wt / name
        if p.exists():
            m = re.search(r'(?m)^PDF_PARTS\s*:?=\s*(.+)$', p.read_text(encoding='utf-8'))
            if m:
                return m.group(1).split()
    raise config.L10nError('scope: PDF_PARTS not found in the tag\'s make/render.mk')


# ── before make ──────────────────────────────────────────────────────

@dataclass
class Prepared:
    make_vars: list
    html_placeholders: list
    html_unbuilt: list
    out_dirs: list
    refs: int
    files_with_refs: list
    marker: str


def prepare(lang, sc: Scope, wt: Path, env: dict) -> Prepared:
    """Placeholders, English links, _quarto.yml, tool edits, stale outputs."""
    strings = ui_strings(lang)
    base = config.get_language('en').url
    marker = strings['l10n.english_marker']
    in_files = [f for f in config.book_files(sc.rev) if sc.includes(f) and (wt / f).exists()]
    texts = {f: (wt / f).read_text(encoding='utf-8') for f in in_files}
    wanted = set().union(*(references(t) for t in texts.values())) if texts else set()
    labels = out_labels(sc)
    in_defined = set()
    for t in texts.values():
        in_defined |= {lb for _, lb in LABEL_RE.findall(t)}
    wanted = {lb for lb in wanted if lb in labels and lb not in in_defined}
    need = {lb: labels[lb] for lb in wanted if h1(config.show(sc.rev, labels[lb][0]) or '')[1] != lb}
    helper = run_helper(wt, env, need)
    table = label_table(sc, wanted, helper)
    nrefs, touched = 0, []
    for f, t in texts.items():
        new, n = rewrite_refs(t, table, helper['terms'], marker, base)
        if n:
            (wt / f).write_text(new, encoding='utf-8')
            nrefs += n
            touched.append(f)
    for rel in sc.placeholders:
        (wt / rel).write_text(placeholder(lang, sc, rel, helper['numbering'], strings, base),
                              encoding='utf-8')
    q = wt / '_quarto.yml'
    q.write_text(filter_quarto(q.read_text(encoding='utf-8'), sc), encoding='utf-8')
    slugs = tag_pdf_parts(wt)
    pdf_parts = [s for s in slugs if f'chapter_{s}' not in {p.dir for p in sc.out_parts}]
    apply_patches(wt, patches(sc, pdf_parts))
    out_dirs = [c.dir for c in sc.out_chapters]
    _remove_outputs(wt, sc, out_dirs)
    slides = [f for f in helper['numbering'] if sc.includes(f)]
    make_vars = [f'PDF_PARTS={" ".join(pdf_parts)}', f'SLIDES_FILTER={" ".join(slides)}']
    return Prepared(make_vars, [r[:-3] + '.html' for r in sc.placeholders],
                    [r[:-3] + '.html' for r in sc.unbuilt], out_dirs, nrefs, touched, marker)


def _remove_outputs(wt: Path, sc: Scope, out_dirs: list):
    """Decks and pages an earlier (unscoped) build left in the worktree."""
    import shutil
    for root in [wt / '_slides', wt / '_book' / 'slides']:
        if root.is_dir():
            for fw in root.iterdir():
                for d in out_dirs:
                    if (fw / d).is_dir():
                        shutil.rmtree(fw / d)
    for rel in sc.unbuilt:
        (wt / '_book' / (rel[:-3] + '.html')).unlink(missing_ok=True)


def make_args(prep: Prepared | None) -> str:
    return ''.join(' ' + shlex.quote(v) for v in (prep.make_vars if prep else []))


# ── after make ───────────────────────────────────────────────────────

SIDEBAR_LINK = re.compile(r'(<a href="([^"]+)" class="sidebar-item-text sidebar-link[^"]*">'
                          r'\s*<span class="menu-text">)(.*?)(</span></a>)')
HREF = re.compile(r'href="([^"#?:]+\.html)(#[^"]*)?"')
SKIP_DIRS = {'slides', 'site_libs', 'notebooks', 'pdf', 'static', 'img'}
IMGREF = re.compile(r'\.\./img/([\w./-]+\.(?:svg|png|jpg|jpeg|gif))')


def _resolve(page: str, href: str) -> str:
    if href.startswith('/'):
        return posixpath.normpath(href.lstrip('/'))
    return posixpath.normpath(posixpath.join(posixpath.dirname(page), href))


def postprocess(lang, wt: Path, prep: Prepared) -> dict:
    """Edit the rendered _book/ (idempotent): markers, noindex, English links,
    no unbuilt pages, notebook zips without out-of-scope chapters."""
    book = wt / '_book'
    base = config.get_language('en').url.rstrip('/')
    stats = {'deleted': 0, 'markers': 0, 'noindex': 0, 'links': 0, 'search': 0, 'zips': 0,
             'unresolved': []}
    for rel in prep.html_unbuilt:
        p = book / rel
        if p.exists():
            p.unlink()
            stats['deleted'] += 1
    placeholders, unbuilt = set(prep.html_placeholders), set(prep.html_unbuilt)
    span = (f' <span class="{MARKER_CLASS}" style="{MARKER_STYLE}">'
            f'{html.escape(prep.marker)}</span>')
    for p in sorted(book.rglob('*.html')):
        rel = p.relative_to(book).as_posix()
        if rel.split('/')[0] in SKIP_DIRS:
            continue
        text = p.read_text(encoding='utf-8')
        new = text

        def mark(m, rel=rel):
            target = _resolve(rel, m.group(2).split('#')[0])
            if target in placeholders and MARKER_CLASS not in m.group(3):
                stats['markers'] += 1
                return m.group(1) + m.group(3) + span + m.group(4)
            return m.group(0)
        new = SIDEBAR_LINK.sub(mark, new)

        def link(m, rel=rel):
            target = _resolve(rel, m.group(1))
            if target in unbuilt:
                stats['links'] += 1
                return f'href="{base}/{target}{m.group(2) or ""}" hreflang="en"'
            return m.group(0)
        new = HREF.sub(link, new)
        if rel in placeholders and NOINDEX not in new:
            new, k = re.subn(r'(<meta charset="utf-8">)', r'\1\n' + NOINDEX, new, count=1)
            if not k:
                new = new.replace('<head>', '<head>\n' + NOINDEX, 1)
            stats['noindex'] += 1
        if rel not in placeholders:
            stats['unresolved'] += [f'{rel}: {x}' for x in
                                    sorted(set(re.findall(r'\?@[a-z][\w-]*', new)))]
        if new != text:
            p.write_text(new, encoding='utf-8')
    search = book / 'search.json'
    if search.exists():
        data = json.loads(search.read_text(encoding='utf-8'))
        keep = [e for e in data if e.get('href', '').split('#')[0] not in unbuilt]
        if len(keep) != len(data):
            stats['search'] = len(data) - len(keep)
            search.write_text(json.dumps(keep, ensure_ascii=False), encoding='utf-8')
    for z in sorted((book / 'notebooks').glob('d2l-*.zip')):
        if filter_zip(z, prep.out_dirs):
            stats['zips'] += 1
    return stats


def filter_zip(path: Path, out_dirs: list) -> bool:
    """Drop out-of-scope chapters (and figures only they use) from a notebook
    zip, keeping entry order, timestamps and attributes (deterministic)."""
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        data = {i.filename: z.read(i) for i in infos}
    if not infos:
        return False
    top = infos[0].filename.split('/')[0]
    drop = {f'{top}/{d}/' for d in out_dirs}
    kept = [i for i in infos if not any(i.filename.startswith(d) for d in drop)]
    if len(kept) == len(infos):
        return False
    refs = set()
    for i in kept:
        if i.filename.endswith('.ipynb'):
            refs |= set(IMGREF.findall(data[i.filename].decode('utf-8', 'replace')))
    img = f'{top}/img/'
    kept = [i for i in kept if not i.filename.startswith(img) or i.filename[len(img):] in refs]
    tmp = path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for i in kept:
            z.writestr(i, data[i.filename])
    os.replace(tmp, path)
    return True


def report(stats: dict, prep: Prepared, out=sys.stdout):
    print(f'scope: {len(prep.html_placeholders)} placeholder page(s), '
          f'{len(prep.html_unbuilt)} section page(s) not built, {prep.refs} reference(s) '
          f'to English in {len(prep.files_with_refs)} file(s); after make: '
          f'{stats["markers"]} sidebar marker(s), {stats["noindex"]} noindex, '
          f'{stats["links"]} link(s) to English, {stats["deleted"]} stale page(s) deleted, '
          f'{stats["zips"]} notebook zip(s) filtered', file=out)
    for u in stats['unresolved'][:20]:
        print(f'warning: unresolved reference {u}', file=out)
