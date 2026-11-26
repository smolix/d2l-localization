"""Command line: python -m l10n <command> ... (or bin/l10n)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, tfile
from .config import L10nError


def _lang(args):
    lang = config.get_language(args.lang)
    if getattr(args, 'rev', None):
        lang.pinned_next = config.stable_rev(args.rev)
    return lang


def _need_rev(lang):
    if not lang.aligned:
        raise L10nError(f'{lang.code} has no pinned revision: run `l10n pin {lang.code} <rev>` '
                        'or pass --rev')
    return lang.aligned


def _print_report(rep, quiet_warnings=False, limit=200):
    for e in rep.errors[:limit]:
        print('ERROR   ' + e)
    if len(rep.errors) > limit:
        print(f'ERROR   ... {len(rep.errors) - limit} more')
    if not quiet_warnings:
        for w in rep.warnings[:limit]:
            print('warning ' + w)
        if len(rep.warnings) > limit:
            print(f'warning ... {len(rep.warnings) - limit} more')
    for le in rep.locked_edits:
        print('locked  edited locked block: ' + le)
    print(f'{len(rep.errors)} error(s), {len(rep.warnings)} warning(s), '
          f'{len(rep.locked_edits)} locked-block edit(s)')


def cmd_status(args):
    from .answers import file_status
    lang = _lang(args)
    rev = _need_rev(lang)
    files = config.select_files(rev, args.files)
    tot = {'blocks': 0, 'done': 0, 'stale': 0, 'missing': 0, 'orphan': 0, 'words': 0, 'words_done': 0}
    rows = []
    for f in files:
        st = file_status(lang, rev, f)
        for k in tot:
            tot[k] += st[k]
        if args.all or st['done'] or st['stale'] or st['orphan']:
            rows.append((f, st))
    print(f'{lang.name} ({lang.code}) against {rev}'
          + (f' (bump from {lang.pinned} in progress)' if lang.pinned_next and lang.pinned else ''))
    print(f'{"file":64} {"done":>6} {"stale":>6} {"miss":>6} {"orph":>5} {"words%":>7}')
    for f, st in rows:
        pct = 100 * st['words_done'] / st['words'] if st['words'] else 100.0
        print(f'{f:64} {st["done"]:6} {st["stale"]:6} {st["missing"]:6} {st["orphan"]:5} {pct:6.1f}%')
    pct = 100 * tot['words_done'] / tot['words'] if tot['words'] else 0.0
    print(f'{"TOTAL (" + str(len(files)) + " files)":64} {tot["done"]:6} {tot["stale"]:6} '
          f'{tot["missing"]:6} {tot["orphan"]:5} {pct:6.1f}%')


def cmd_pin(args):
    lang = config.get_language(args.lang)
    rev = config.stable_rev(args.rev)
    if lang.aligned and lang.book.exists() and any(lang.book.rglob('*.md')) and not args.force:
        raise L10nError(f'{lang.code} already has translations aligned with {lang.aligned}; '
                        'use `l10n bump` to move them (or --force)')
    config.set_language_field(lang.code, 'pinned', rev)
    config.set_language_field(lang.code, 'pinned_next', None)
    print(f'{lang.code}: pinned {rev}')


def emit_units(lang, rev, files, out: Path, limit=None, code='pytorch', include_closed=False):
    from .units import english_blocks, is_open, units_for, work_file
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for f in files:
        blocks = english_blocks(rev, f)
        entries = tfile.load(lang.book, f)
        units = units_for(f, blocks)
        for u in units:
            if not include_closed and not is_open(u, blocks, entries):
                continue
            p = out / u.filename
            p.write_text(work_file(lang, rev, u, units, blocks, entries, code), encoding='utf-8')
            written.append(p)
            if limit and len(written) >= limit:
                return written
    return written


def cmd_emit(args):
    lang = _lang(args)
    rev = _need_rev(lang)
    files = config.select_files(rev, args.files)
    out = Path(args.out) if args.out else config.REPO / 'work' / lang.code
    written = emit_units(lang, rev, files, out, args.units, args.code, args.all)
    for p in written:
        print(p)
    print(f'{len(written)} work file(s) in {out}')


def cmd_apply(args):
    from .answers import AnswerError, apply_answer, parse_answer
    lang = _lang(args)
    _need_rev(lang)
    bad = 0
    files = []
    for a in args.answers:                       # a directory means its *.md files
        files += sorted(Path(a).glob('*.md')) if Path(a).is_dir() else [Path(a)]
    for a in files:
        text = Path(a).read_text(encoding='utf-8')
        try:
            answers = parse_answer(text, source=str(a))
        except AnswerError as e:
            print(f'ERROR   {e}')
            bad += 1
            continue
        for ans in answers:
            rep = apply_answer(lang, ans, args.agent, args.model, args.force, args.dry_run)
            state = 'checked' if args.dry_run else ('applied' if rep.ok() or args.force else 'REJECTED')
            print(f'{ans.path}#{ans.uid}: {state}')
            _print_report(rep, args.quiet)
            bad += 0 if rep.ok() else 1
    if bad and not args.force:
        sys.exit(1)


def cmd_check(args):
    from .answers import check_language
    lang = _lang(args)
    rev = _need_rev(lang)
    rep = check_language(lang, args.files, complete=args.complete or args.promote)
    _print_report(rep, args.quiet)
    if not rep.ok():
        sys.exit(1)
    if args.promote:
        if not lang.pinned_next or lang.pinned_next == lang.pinned:
            print(f'{lang.code}: no bump in progress; pinned stays {lang.pinned}')
        else:
            config.set_language_field(lang.code, 'pinned', rev)
            config.set_language_field(lang.code, 'pinned_next', None)
            print(f'{lang.code}: pinned {rev}')


def cmd_bump(args):
    from .bump import bump_language
    lang = config.get_language(args.lang)
    res = bump_language(lang, args.to, dry_run=args.dry_run)
    t = {'unchanged': 0, 'changed': 0, 'new': 0, 'deleted': 0}
    for r in res:
        for k in t:
            t[k] += getattr(r, k)
        if r.changed or r.new or r.deleted or r.new_path != r.old_path:
            arrow = '' if r.new_path == r.old_path else f' -> {r.new_path or "(deleted)"}'
            print(f'{r.old_path}{arrow}: {r.unchanged} unchanged, {r.changed} changed, '
                  f'{r.new} new, {r.deleted} deleted')
    print(f'TOTAL {len(res)} files: {t["unchanged"]} unchanged, {t["changed"]} changed '
          f'(update), {t["new"]} new, {t["deleted"]} deleted')
    if args.dry_run:
        return
    lang = config.get_language(args.lang)
    print(f'{lang.code}: aligned with {lang.aligned} (pinned stays {lang.pinned} until '
          f'`l10n check {lang.code} --promote`)')
    if not args.no_emit:
        files = sorted({r.new_path for r in res if r.new_path})
        out = Path(args.out) if args.out else config.REPO / 'work' / lang.code
        written = emit_units(lang, lang.aligned, files, out)
        print(f'{len(written)} update work file(s) in {out}')


def cmd_pseudo(args):
    from .pseudo import pseudo
    from .units import english_blocks
    lang = _lang(args)
    if not lang.test and not args.force:
        raise L10nError(f'{lang.code} is not a test language (test_languages in languages.yml)')
    rev = _need_rev(lang)
    n = 0
    for f in config.select_files(rev, args.files):
        blocks = english_blocks(rev, f)
        entries = {b.anchor: tfile.Entry(pseudo(b.kind, b.content)) for b in blocks if b.translatable}
        tfile.save(lang.book, f, entries)
        n += 1
    print(f'{n} file(s) pseudo-translated into {lang.book}')


def cmd_build(args):
    from .build import build
    lang = _lang(args)
    wt = build(lang, None, args.html_only, Path(args.workdir) if args.workdir else None,
               args.target, args.stale, args.dry_run)
    print(f'built in {wt}')


def cmd_deploy(args):
    from .build import deploy
    lang = config.get_language(args.lang)
    deploy(lang, Path(args.workdir) if args.workdir else None, not args.no_delete, args.dry_run)


def cmd_menu(args):
    from .build import upload_menu, write_menu
    out = Path(args.out) if args.out else config.REPO / 'site' / 'languages.json'
    write_menu(out, args.all)
    print(out.read_text(encoding='utf-8'), end='')
    if args.upload or args.dry_run:
        upload_menu(out, dry_run=args.dry_run)


def cmd_strings(args):
    from .build import strings
    lang = _lang(args)
    path, added, removed, todo = strings(lang, None if args.en_file else lang.aligned,
                                         Path(args.en_file) if args.en_file else None, args.pseudo)
    print(f'{path}: {len(added)} new key(s), {len(removed)} removed, {len(todo)} to translate')
    for k in added:
        print(f'  + {k}')
    for k in removed:
        print(f'  - {k}')


def cmd_roundtrip(args):
    from .segment import reconstruct, segment, translatable
    rev = args.rev or 'HEAD'
    files = config.select_files(rev, args.files)
    bad = 0
    nblocks = 0
    for f in files:
        text = config.show(rev, f)
        blocks = segment(text)
        tr = {b.anchor: b.content for b in translatable(blocks)}
        nblocks += len(tr)
        stored = tfile.parse(tfile.render({a: tfile.Entry(t) for a, t in tr.items()}))
        out = reconstruct(blocks, {a: e.text for a, e in stored.items()})
        if out != text or ''.join(b.text for b in blocks) != text:
            print(f'MISMATCH {f}')
            bad += 1
    print(f'{len(files)} files, {nblocks} translatable blocks at {rev}: '
          f'{len(files) - bad} byte-identical, {bad} mismatched')
    if bad:
        sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='l10n', description='Translation tools for Dive into Deep '
                                 'Learning (see DESIGN.md, TRANSLATING.md).')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def lang_cmd(name, func, help, rev=True):
        p = sub.add_parser(name, help=help, description=help)
        p.add_argument('lang')
        if rev:
            p.add_argument('--rev', help='use this English revision instead of the pinned one')
        p.set_defaults(func=func)
        return p

    p = lang_cmd('status', cmd_status, 'coverage per file')
    p.add_argument('--files', nargs='*')
    p.add_argument('--all', action='store_true', help='list untranslated files too')

    p = lang_cmd('pin', cmd_pin, 'set the English revision of a language with no translations yet',
                 rev=False)
    p.add_argument('rev')
    p.add_argument('--force', action='store_true')

    p = lang_cmd('emit', cmd_emit, 'write work files for units that need translation')
    p.add_argument('--files', nargs='*')
    p.add_argument('--units', type=int, help='at most N units')
    p.add_argument('--out', help='output directory (default work/<lang>/)')
    p.add_argument('--code', default='pytorch', choices=['pytorch', 'jax', 'tensorflow', 'mxnet',
                                                         'full', 'none'],
                   help='code shown in the CONTEXT page (default: pytorch variants only)')
    p.add_argument('--all', action='store_true', help='also units with nothing open')

    p = lang_cmd('apply', cmd_apply, 'check translator answers and merge them')
    p.add_argument('answers', nargs='+')
    p.add_argument('--agent', help='who translated (e.g. "claude fable-5.2", "human: N.N.")')
    p.add_argument('--model', help='model id, if not part of --agent')
    p.add_argument('--force', action='store_true', help='merge despite errors')
    p.add_argument('--dry-run', action='store_true', help='check only')
    p.add_argument('--quiet', action='store_true', help='hide warnings')

    p = lang_cmd('check', cmd_check, 'check every translated file')
    p.add_argument('--files', nargs='*')
    p.add_argument('--complete', action='store_true', help='also require every block translated')
    p.add_argument('--promote', action='store_true',
                   help='after a clean --complete check, move pinned to the bump target')
    p.add_argument('--quiet', action='store_true', help='hide warnings')

    p = lang_cmd('bump', cmd_bump, 'align translations with a newer English revision', rev=False)
    p.add_argument('--to', required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-emit', action='store_true')
    p.add_argument('--out')

    p = lang_cmd('pseudo', cmd_pseudo, 'pseudo-translate files (test languages): ⟦text⟧')
    p.add_argument('--files', nargs='*')
    p.add_argument('--force', action='store_true')

    p = lang_cmd('build', cmd_build, 'build the language from a d2l-neu worktree')
    p.add_argument('--html-only', action='store_true', help='make html instead of make all-quick')
    p.add_argument('--target', help='another make target')
    p.add_argument('--workdir', help='worktree path (default _build/<lang>)')
    p.add_argument('--stale', default='english', choices=['english', 'translation'],
                   help='what stale blocks show (default: English)')
    p.add_argument('--dry-run', action='store_true', help='prepare the worktree, do not run make')

    p = lang_cmd('deploy', cmd_deploy, 'upload the build to the language bucket', rev=False)
    p.add_argument('--workdir')
    p.add_argument('--no-delete', action='store_true')
    p.add_argument('--dry-run', action='store_true')

    p = sub.add_parser('compare', help='side-by-side page of several agents\' translations',
                       description='Check every answers-<agent>/ directory in DIR against the work '
                                   'files in DIR and write one HTML page that shows them side by side.')
    p.add_argument('lang')
    p.add_argument('dir', help='directory with the work files and answers-<agent>/ directories')
    p.add_argument('--out', help='HTML file (default DIR/compare.html)')
    p.set_defaults(func=lambda a: __import__('l10n.compare', fromlist=['compare']).compare(a.lang, a.dir, a.out))

    p = sub.add_parser('menu', help='write site/languages.json (and upload it)')
    p.add_argument('--out')
    p.add_argument('--all', action='store_true', help='include languages without a release')
    p.add_argument('--upload', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='upload dry run')
    p.set_defaults(func=cmd_menu)

    p = lang_cmd('strings', cmd_strings, 'create/update <lang>/strings.yml from i18n/en.yml')
    p.add_argument('--en-file', help='read this en.yml instead of the pinned revision\'s')
    p.add_argument('--pseudo', action='store_true', help='new keys get ⟦English⟧')

    p = sub.add_parser('roundtrip', help='segment and rebuild every book file with English as '
                       'the translation; must be byte-identical')
    p.add_argument('--rev')
    p.add_argument('--files', nargs='*')
    p.set_defaults(func=cmd_roundtrip)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except L10nError as e:
        print(f'l10n: {e}', file=sys.stderr)
        sys.exit(2)
