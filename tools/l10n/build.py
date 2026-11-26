"""Build and deploy one language; the language menu; the UI strings file.

Contract with d2l-neu (DESIGN.md section 7):
  * a build is a `git worktree add` of d2l-neu at the aligned revision, with
    .venv-*, .env and data symlinked to the main checkout's;
  * every chapter .md with a translation is overwritten by the reconstructed
    translated page (same path);
  * D2L_LANG=<code> and D2L_LOCALE_FILE=<lang>/strings.yml select the UI
    strings; then `make all-quick` (or `make html`).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import sys
from pathlib import Path

import yaml

from . import config, tfile
from .pseudo import L as PL, R as PR
from .segment import reconstruct
from .units import english_blocks

STATE = config.REPO / '_state'
BUILD_ROOT = Path(os.environ.get('L10N_BUILD_ROOT', str(config.REPO / '_build')))


# ── overlay ──────────────────────────────────────────────────────────

def translated_pages(lang, rev: str, stale: str = 'english') -> dict:
    """english path -> reconstructed translated text, for files with any translation."""
    out = {}
    if not lang.book.exists():
        return out
    book = set(config.book_files(rev))
    for p in sorted(lang.book.rglob('*.md')):
        rel = str(p.relative_to(lang.book))
        if rel not in book:
            raise config.L10nError(f'{lang.code}/book/{rel}: no such English file at {rev}')
        entries = tfile.load(lang.book, rel)
        tr = tfile.translations(entries, stale)
        blocks = english_blocks(rev, rel)
        unknown = [a for a in entries if a not in {b.anchor for b in blocks}]
        if unknown:
            raise config.L10nError(f'{rel}: anchors not in the English file at {rev}: '
                                   + ' '.join(unknown[:5]) + ' (run l10n check)')
        out[rel] = reconstruct(blocks, tr)
    return out


def prepare_worktree(lang, rev: str, workdir: Path | None = None) -> Path:
    sha = config.resolve_rev(rev)
    wt = (workdir or BUILD_ROOT / lang.code).resolve()
    if (wt / '.git').exists():
        config.git('checkout', '--detach', '-f', sha, cwd=wt)
        config.git('checkout', '-f', '--', '.', cwd=wt)
    else:
        wt.parent.mkdir(parents=True, exist_ok=True)
        if wt.exists() and any(wt.iterdir()):
            raise config.L10nError(f'{wt} exists and is not a worktree')
        config.git('worktree', 'add', '--detach', str(wt), sha)
    for name in [p.name for p in config.D2L_NEU.glob('.venv-*')] + ['.env', 'data']:
        src, dst = config.D2L_NEU / name, wt / name
        if src.exists() and not dst.exists() and not dst.is_symlink():
            dst.symlink_to(src)
    # The venvs are the main checkout's. A fresh checkout's pyproject.toml/uv.lock
    # are newer than their .synced stamps, so make would `uv sync` the shared
    # venvs to this revision's lock; date the files back so it does not.
    for name in ('pyproject.toml', 'uv.lock', 'tools/check_runtime_deps.py'):
        if (wt / name).exists():
            os.utime(wt / name, (0, 0))
    main_lock = config.D2L_NEU / 'uv.lock'
    if main_lock.exists() and (wt / 'uv.lock').exists() and \
            main_lock.read_bytes() != (wt / 'uv.lock').read_bytes():
        print(f'warning: uv.lock at {rev} differs from {main_lock}; the build uses the main '
              'checkout\'s venvs as they are', file=sys.stderr)
    lfs = subprocess.run(['git', 'lfs', 'ls-files'], cwd=wt, capture_output=True, text=True)
    if lfs.returncode == 0 and any(l.split()[1:2] == ['-'] for l in lfs.stdout.splitlines()):
        print('materializing Git-LFS files (git lfs pull) ...', file=sys.stderr)
        config.git('lfs', 'pull', cwd=wt)
    return wt


def overlay(lang, rev: str, wt: Path, stale='english') -> list:
    pages = translated_pages(lang, rev, stale)
    for rel, text in pages.items():
        (wt / rel).write_text(text, encoding='utf-8')
    return sorted(pages)


def build_env(lang) -> dict:
    env = dict(os.environ)
    env['D2L_LANG'] = lang.code
    strings = lang.root / 'strings.yml'
    if strings.exists():
        env['D2L_LOCALE_FILE'] = str(strings.resolve())
    else:
        print(f'note: no {lang.code}/strings.yml; UI strings stay English '
              f'(run l10n strings {lang.code})', file=sys.stderr)
        env.pop('D2L_LOCALE_FILE', None)
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env.setdefault(v, '8')
    env.setdefault('UPLOAD_JOBS', '8')
    return env


def build(lang, rev: str | None = None, html_only=False, workdir=None, target=None,
          stale='english', dry_run=False) -> Path:
    rev = rev or lang.aligned
    if not rev:
        raise config.L10nError(f'{lang.code} has no pinned revision')
    if lang.pinned and lang.pinned_next and rev == lang.pinned_next:
        print(f'note: building the in-progress bump target {rev} '
              f'(pinned: {lang.pinned}); stale blocks show English', file=sys.stderr)
    wt = prepare_worktree(lang, rev, workdir)
    pages = overlay(lang, rev, wt, stale)
    print(f'{len(pages)} translated page(s) overlaid in {wt}')
    target = target or ('html' if html_only else 'all-quick')
    # build-guide 4.14: raise the process limit; nice keeps the shared box responsive
    cmd = ['bash', '-c', f'ulimit -u 8000 2>/dev/null || ulimit -u $(ulimit -Hu); '
           f'exec nice -n 10 make {target}']
    log = wt / f'l10n-build-{lang.code}.log'
    print(f'running: D2L_LANG={lang.code} make {target}  (in {wt}, log: {log})')
    if dry_run:
        return wt
    with log.open('w') as f:
        p = subprocess.Popen(cmd, cwd=wt, env=build_env(lang), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line)
            sys.stdout.write(line)
        rc = p.wait()
    if rc:
        raise config.L10nError(f'make {target} failed (rc={rc}); see {log}')
    return wt


def deploy(lang, workdir=None, delete=True, dry_run=False):
    if lang.code == 'en':
        raise config.L10nError('English is deployed from d2l-neu, not from here')
    if not lang.bucket:
        raise config.L10nError(f'{lang.code} has no bucket in languages.yml')
    wt = (workdir or BUILD_ROOT / lang.code).resolve()
    book = wt / '_book'
    if not (book / 'index.html').exists():
        raise config.L10nError(f'{book} has no build; run l10n build {lang.code}')
    STATE.mkdir(exist_ok=True)
    manifest = f'.upload-manifest-{lang.bucket}.txt'
    saved = STATE / manifest
    if saved.exists():
        shutil.copy2(saved, wt / manifest)
    env = build_env(lang)
    env['R2_BUCKET'] = lang.bucket
    env['BOOK_DIR'] = '_book'
    cmd = [str(wt / 'tools' / 'upload_r2.sh')]
    if delete:
        cmd.append('--delete')
    if dry_run:
        cmd.append('--dry-run')
    print(f'running: R2_BUCKET={lang.bucket} {" ".join(cmd)}  (in {wt})')
    rc = subprocess.run(cmd, cwd=wt, env=env).returncode
    if not dry_run and (wt / manifest).exists():
        shutil.copy2(wt / manifest, saved)
    if rc:
        raise config.L10nError(f'upload failed (rc={rc})')


# ── language menu ────────────────────────────────────────────────────

LANGUAGES_URL = 'https://d2l-data.smola.org/site/languages.json'


def live_pages() -> dict:
    """{code: pages} from the published languages.json. Each language's
    `pages` list is written by d2l-neu's tools/update_language_pages.py when
    that language's site is uploaded; the menu links to the same page in
    another language only if its list has the page (no 404 rule exists)."""
    try:
        req = urllib.request.Request(LANGUAGES_URL, headers={'User-Agent': 'd2l-l10n',
                                                             'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=30) as r:
            live = json.load(r)
    except (urllib.error.URLError, ValueError, OSError):
        return {}
    return {l['code']: l['pages'] for l in live.get('languages', [])
            if isinstance(l, dict) and l.get('code') and isinstance(l.get('pages'), list)}


def menu_data(include_unpublished=False) -> dict:
    """languages.json: the language list from languages.yml, keeping each
    language's `pages` from the published file (a language without a list
    links to its home page)."""
    data = config.load_languages()
    pages = live_pages()
    langs = []
    for lang in data['languages'].values():
        if lang.test:
            continue
        show = lang.menu if lang.menu is not None else (lang.code == 'en' or lang.pinned)
        if show or include_unpublished:
            entry = {'code': lang.code, 'name': lang.name, 'url': lang.url}
            if lang.code in pages:
                entry['pages'] = pages[lang.code]
            langs.append(entry)
    old = [{'name': o['name'], 'url': o['url']} for o in data['old_editions']]
    return {'languages': langs, 'old_editions': old}


def write_menu(out: Path, include_unpublished=False) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(menu_data(include_unpublished), ensure_ascii=False, indent=2) + '\n',
                   encoding='utf-8')
    return out


def upload_menu(path: Path, dry_run=False, env_file=Path.home() / '.config/cloudflare/d2l-data.env'):
    if not env_file.exists():
        raise config.L10nError(f'{env_file} not found')
    env = dict(os.environ)
    for line in env_file.read_text().splitlines():
        m = re.match(r'^([A-Z_][A-Z0-9_]*)=(.*)$', line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"\'')
    endpoint = env.get('AWS_ENDPOINT_URL_S3')
    cmd = ['aws', 's3', 'cp', str(path), 's3://d2l-data/site/languages.json',
           '--content-type', 'application/json; charset=utf-8',
           '--cache-control', 'public, max-age=300', '--region', 'auto']
    if endpoint:
        cmd += ['--endpoint-url', endpoint]
    if dry_run:
        cmd.append('--dryrun')
    print('running: ' + ' '.join(c for c in cmd if not c.startswith('https://')))
    if subprocess.run(cmd, env=env).returncode:
        raise config.L10nError('menu upload failed')


# ── UI strings ───────────────────────────────────────────────────────

SETTING_PREFIXES = ('code', 'lang', 'dir', 'site.', 'forum.', 'fonts.', 'pdf.lang', 'quarto.')
TODO = '# TODO translate'
TODO_UPDATE = '# TODO update: English changed'

FONT_DEFAULTS = {
    'ja': {'fonts.html.sans': "'Source Sans 3', 'Noto Sans JP', -apple-system, BlinkMacSystemFont, "
                              "'Hiragino Sans', 'Yu Gothic', sans-serif",
           'fonts.html.google': ['Noto Sans JP:wght@400;500;700'],
           'fonts.pdf.cjk': 'Noto Serif CJK JP', 'fonts.pdf.cjk_sans': 'Noto Sans CJK JP',
           'fonts.pdf.cjk_mono': 'Noto Sans Mono CJK JP'},
    'zh': {'fonts.html.sans': "'Source Sans 3', 'Noto Sans SC', -apple-system, BlinkMacSystemFont, "
                              "'PingFang SC', 'Microsoft YaHei', sans-serif",
           'fonts.html.google': ['Noto Sans SC:wght@400;500;700'],
           'fonts.pdf.cjk': 'Noto Serif CJK SC', 'fonts.pdf.cjk_sans': 'Noto Sans CJK SC',
           'fonts.pdf.cjk_mono': 'Noto Sans Mono CJK SC'},
    'ko': {'fonts.html.sans': "'Source Sans 3', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, "
                              "'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif",
           'fonts.html.google': ['Noto Sans KR:wght@400;500;700'],
           'fonts.pdf.cjk': 'Noto Serif CJK KR', 'fonts.pdf.cjk_sans': 'Noto Sans CJK KR',
           'fonts.pdf.cjk_mono': 'Noto Sans Mono CJK KR'},
}


def is_setting(key: str) -> bool:
    return any(key == s or (s.endswith('.') and key.startswith(s)) for s in SETTING_PREFIXES)


def _flatten(d, prefix=''):
    out = {}
    for k, v in (d or {}).items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def _entries(text: str) -> list:
    """en.yml as [('comment', lines) | ('key', key, lines)] preserving layout."""
    out = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^([A-Za-z0-9_.\-]+):(\s|$)', line)
        if not m:
            out.append(('comment', line))
            i += 1
            continue
        j = i + 1
        while j < len(lines) and (lines[j].startswith((' ', '\t')) or
                                  (not lines[j].strip() and j + 1 < len(lines)
                                   and lines[j + 1].startswith((' ', '\t')))):
            j += 1
        out.append(('key', m.group(1), lines[i:j]))
        i = j
    return out


def _render_value(key: str, v) -> list:
    if isinstance(v, str) and '\n' in v:
        body = v
        if body.endswith('\n') and not body.endswith('\n\n'):
            chomp, body = '', body[:-1]
        elif not body.endswith('\n'):
            chomp = '-'
        else:
            chomp = '+'
            body = body[:-1]
        return [f'{key}: |2{chomp}'] + [('  ' + l) if l else '' for l in body.split('\n')]
    return [f'{key}: {json.dumps(v, ensure_ascii=False)}']


def _todo_keys(text: str) -> set:
    return {m.group(1) for m in re.finditer(r'(?m)^([A-Za-z0-9_.\-]+):.*# TODO translate\s*$', text)}


def _update_keys(text: str) -> set:
    return {m.group(1) for m in re.finditer(r'(?m)^([A-Za-z0-9_.\-]+):.*# TODO update\b.*$', text)}


def language_settings(lang) -> dict:
    s = {'code': lang.code, 'lang': lang.locale, 'dir': lang.dir,
         'site.hosts': list(lang.hosts or []),
         'forum.mode': lang.forum or 'english'}
    if lang.forum == 'own' and lang.forum_category:
        s['forum.category'] = lang.forum_category
    return s


def strings(lang, rev: str | None = None, en_file: Path | None = None, pseudo=False) -> tuple:
    """Create or update <lang>/strings.yml. Returns (path, added, removed, todo)."""
    if en_file:
        en_text = en_file.read_text(encoding='utf-8')
    else:
        rev = rev or lang.aligned
        if not rev:
            raise config.L10nError(f'{lang.code} has no pinned revision; pass --rev or --en-file')
        en_text = config.show(rev, 'i18n/en.yml')
        if en_text is None:
            raise config.L10nError(f'i18n/en.yml does not exist at {rev}')
    en = _flatten(yaml.safe_load(en_text))
    path = lang.root / 'strings.yml'
    old_text = path.read_text(encoding='utf-8') if path.exists() else ''
    old = _flatten(yaml.safe_load(old_text)) if old_text else {}
    todo_old = _todo_keys(old_text)
    snap_path = lang.root / 'strings.en.json'      # the English each translation was made from
    snap = json.loads(snap_path.read_text(encoding='utf-8')) if snap_path.exists() else {}
    changed = set(k for k in en if k in old and k in snap and snap[k] != en[k]
                  and not is_setting(k)) | (_update_keys(old_text) & set(en))
    settings = language_settings(lang)
    fonts = FONT_DEFAULTS.get(lang.code, {})
    added = [k for k in en if k not in old]
    removed = [k for k in old if k not in en and not k.startswith('quarto.')]
    out = [f'# UI strings and settings for {lang.name} ({lang.code}): the D2L_LOCALE_FILE of '
           f'`l10n build {lang.code}`.',
           f'# Generated from d2l-neu i18n/en.yml by `l10n strings {lang.code}`; edit the values. '
           f'Keys marked "{TODO}" still hold the English text.', '']
    todo = []
    for ent in _entries(en_text):
        if ent[0] == 'comment':
            out.append(ent[1])
            continue
        _, key, _lines = ent
        ev = en[key] if key in en else None
        if is_setting(key):
            v = settings[key] if key in settings else old.get(key, fonts.get(key, ev))
            out += _render_value(key, v)
            continue
        if key in old and key not in todo_old:
            v = old[key]
            mark = False
        elif pseudo and isinstance(ev, str):
            v = f'{PL}{ev}{PR}'
            mark = False
        else:
            v = ev
            mark = isinstance(ev, str)
        lines = _render_value(key, v)
        if mark:
            lines[0] += '  ' + TODO
            todo.append(key)
        elif key in changed:
            lines[0] += '  ' + TODO_UPDATE
            todo.append(key)
        out += lines
    extra = {k: v for k, v in old.items() if k.startswith('quarto.') and k not in en}
    if extra:
        out += ['', '# Quarto language overrides (keys of Quarto share/language/_language.yml)']
        for k, v in extra.items():
            out += _render_value(k, v)
    text = '\n'.join(out).rstrip('\n') + '\n'
    data = _flatten(yaml.safe_load(text))
    for k in en:
        if k not in data:
            raise config.L10nError(f'strings: generated file lost key {k}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    snap_path.write_text(json.dumps({k: v for k, v in en.items() if not is_setting(k)},
                                    ensure_ascii=False, indent=1, sort_keys=True) + '\n',
                         encoding='utf-8')
    return path, added if old_text else [], removed, todo
