"""Paths, languages.yml, and access to the English book (d2l-neu) by revision."""

from __future__ import annotations

import functools
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
D2L_NEU = Path(os.environ.get('D2L_NEU', '/home/smola/d2l-neu'))
LANGUAGES_YML = REPO / 'languages.yml'
GLOSSARY_DIR = REPO / 'glossary'


class L10nError(Exception):
    """A user-facing error: printed without a traceback."""


@dataclass
class Language:
    code: str
    locale: str
    name: str
    bucket: str | None = None
    hosts: list = field(default_factory=list)
    dir: str = 'ltr'
    pinned: str | None = None
    pinned_next: str | None = None
    forum: str = 'english'
    forum_category: int | None = None
    fonts: dict | None = None
    test: bool = False
    menu: bool | None = None
    scope: object = 'all'           # `all` or a list of part/chapter directories (scope.py)
    raw: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return REPO / self.code

    @property
    def book(self) -> Path:
        return self.root / 'book'

    @property
    def aligned(self) -> str | None:
        """The English revision the files in <lang>/book/ are aligned with."""
        return self.pinned_next or self.pinned

    @property
    def url(self) -> str:
        if self.code == 'en':
            return 'https://d2l.ai'
        return f'https://{self.hosts[0]}' if self.hosts else ''


def load_languages(path: Path | None = None) -> dict:
    path = path or LANGUAGES_YML
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    langs = {}
    default_scope = data.get('scope', 'all')
    for entry, test in ([(e, False) for e in data.get('languages') or []]
                        + [(e, True) for e in data.get('test_languages') or []]):
        known = {k: entry[k] for k in entry if k in Language.__dataclass_fields__
                 and k not in ('raw', 'test')}
        lang = Language(**known, test=test, raw=dict(entry))
        if 'scope' not in entry:
            lang.scope = default_scope
        langs[lang.code] = lang
    return {'languages': langs, 'old_editions': data.get('old_editions') or [],
            'scope': default_scope}


def get_language(code: str) -> Language:
    langs = load_languages()['languages']
    if code not in langs:
        raise L10nError(f'unknown language {code!r}; add it to languages.yml')
    return langs[code]


def set_language_field(code: str, key: str, value, path: Path | None = None):
    """Set (or with value=None remove) one key in a language's flow-style entry.

    languages.yml keeps one `- {code: xx, ...}` line per language and carries
    comments, so it is edited textually rather than re-dumped.
    """
    path = path or LANGUAGES_YML
    lines = path.read_text(encoding='utf-8').split('\n')
    pat = re.compile(r'^(\s*-\s*\{\s*code:\s*' + re.escape(code) + r'\s*,)(.*)\}(.*)$')
    for i, line in enumerate(lines):
        m = pat.match(line)
        if not m:
            continue
        head, body, tail = m.groups()
        parts = _split_flow(body)
        out, done = [], False
        for p in parts:
            k = p.split(':', 1)[0].strip()
            if k == key:
                done = True
                if value is not None:
                    out.append(f' {key}: {_yaml_scalar(value)}')
            else:
                out.append(p)
        if not done and value is not None:
            out.append(f' {key}: {_yaml_scalar(value)}')
        lines[i] = head + ','.join(out) + '}' + tail
        path.write_text('\n'.join(lines), encoding='utf-8')
        return
    raise L10nError(f'language {code!r} not found in {path}')


def _split_flow(body: str) -> list:
    parts, depth, cur, quote = [], 0, '', None
    for ch in body:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in '"\'':
            quote = ch
        elif ch in '[{':
            depth += 1
        elif ch in ']}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def _yaml_scalar(v) -> str:
    if v is None:
        return 'null'
    s = str(v)
    if re.fullmatch(r'[A-Za-z0-9_.\-/]+', s) and s not in ('null', 'true', 'false'):
        return s
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


# ── English book access ──────────────────────────────────────────────

def git(*args, cwd: Path = D2L_NEU, check=True) -> str:
    r = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise L10nError(f'git {" ".join(args)} failed: {r.stderr.strip()}')
    return r.stdout


@functools.lru_cache(maxsize=None)
def resolve_rev(rev: str) -> str:
    """Full commit SHA of a revision of d2l-neu."""
    return git('rev-parse', '--verify', f'{rev}^{{commit}}').strip()


def stable_rev(rev: str) -> str:
    """The name to record for a revision: a tag stays a tag, anything else a SHA."""
    if git('tag', '--list', rev).strip() == rev:
        return rev
    return resolve_rev(rev)


@functools.lru_cache(maxsize=4096)
def show(rev: str, path: str) -> str | None:
    """Contents of `path` at `rev` (None if absent)."""
    r = subprocess.run(['git', 'show', f'{resolve_rev(rev)}:{path}'], cwd=D2L_NEU,
                       capture_output=True)
    if r.returncode != 0:
        return None
    return r.stdout.decode('utf-8')


@functools.lru_cache(maxsize=64)
def list_files(rev: str) -> tuple:
    out = git('ls-tree', '-r', '--name-only', resolve_rev(rev))
    return tuple(out.split('\n')[:-1] if out else [])


def is_book_file(path: str) -> bool:
    return path == 'index.md' or bool(re.fullmatch(r'chapter_[^/]+/[^/]+\.md', path))


@functools.lru_cache(maxsize=64)
def book_files(rev: str) -> tuple:
    """Translatable book sources at `rev`, in book order (_quarto.yml), then the rest."""
    files = [p for p in list_files(rev) if is_book_file(p)]
    have = set(files)
    order = []
    quarto = show(rev, '_quarto.yml') or ''
    for m in re.finditer(r'([\w./-]+)\.qmd\b', quarto):
        p = m.group(1) + '.md'
        if p in have and p not in order:
            order.append(p)
    rest = sorted(have - set(order))
    return tuple(order + rest)


def select_files(rev: str, patterns, scope=None) -> list:
    """Book files at `rev` matching any of `patterns` (paths, globs, chapter dirs).

    With a `scope` (scope.Scope), only the files it translates: a pattern
    that matches nothing else is an error.
    """
    import fnmatch
    files = list(book_files(rev))
    if scope is not None:
        files_all, files = files, [f for f in files if scope.includes(f)]
    if not patterns:
        return files
    out = []
    for pat in patterns:
        pat = pat.rstrip('/')
        match = lambda f: (f == pat or fnmatch.fnmatch(f, pat) or f.startswith(pat + '/')
                           or f == pat + '.md')
        hits = [f for f in files if match(f)]
        if not hits:
            if scope is not None and any(match(f) for f in files_all):
                raise L10nError(f'{pat!r} is outside the translation scope of {scope.code} '
                                '(languages.yml `scope:`)')
            raise L10nError(f'no book file at {rev} matches {pat!r}')
        out += [h for h in hits if h not in out]
    return [f for f in files if f in out]
