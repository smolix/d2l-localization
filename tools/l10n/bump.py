"""Move a language from its aligned English revision to a newer one (DESIGN.md section 4)."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from . import config, tfile
from .segment import normalize
from .units import english_blocks

SIMILAR = 0.3      # minimum word-level similarity for "changed" (else deleted + new)


@dataclass
class FileResult:
    old_path: str
    new_path: str | None
    unchanged: int = 0
    changed: int = 0
    new: int = 0
    deleted: int = 0
    dropped: list = field(default_factory=list)   # old anchors whose translation was dropped


def _key(b):
    """Alignment key: text hash for prose, cell ID for code, labels for meta."""
    if b.translatable:
        return 't:' + b.anchor.split('-')[0]
    if b.kind == 'code' and b.cell_id:
        return 'c:' + b.cell_id
    if b.kind == 'meta' and b.labels:
        return 'l:' + ','.join(b.labels)
    if b.kind == 'blank':
        return 'blank'
    return b.kind + ':' + normalize(b.text)[:80]


def _labels(blocks) -> set:
    return {lab for b in blocks if b.kind == 'meta' for lab in b.labels}


def rename_map(old_rev: str, new_rev: str, old_files) -> dict:
    """old path -> new path (None if gone), by git rename detection, then labels."""
    new_files = set(config.book_files(new_rev))
    out = {}
    git_ren = {}
    diff = config.git('diff', '--name-status', '-M50%', config.resolve_rev(old_rev),
                      config.resolve_rev(new_rev), '--', 'index.md', 'chapter_*')
    for line in diff.splitlines():
        parts = line.split('\t')
        if parts[0].startswith('R') and len(parts) == 3:
            git_ren[parts[1]] = parts[2]
    label_index = None
    for p in old_files:
        if p in new_files:
            out[p] = p
        elif p in git_ren and git_ren[p] in new_files:
            out[p] = git_ren[p]
        else:
            if label_index is None:
                label_index = {}
                for q in new_files:
                    for lab in _labels(english_blocks(new_rev, q)):
                        label_index.setdefault(lab, q)
            labs = [b.labels[0] for b in english_blocks(old_rev, p) if b.kind == 'meta' and b.labels]
            hit = next((label_index[l] for l in labs if l in label_index), None)
            out[p] = hit
    return out


def align(old_blocks, new_blocks) -> tuple:
    """Pairs (old index, new index, 'unchanged'|'changed'), plus unmatched new/old indices."""
    a = [_key(b) for b in old_blocks]
    b = [_key(x) for x in new_blocks]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pairs, new_only, old_only = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if old_blocks[i].translatable:
                    pairs.append((i, j, 'unchanged'))
            continue
        olds = [i for i in range(i1, i2) if old_blocks[i].translatable]
        news = [j for j in range(j1, j2) if new_blocks[j].translatable]
        matched = _match_region(old_blocks, new_blocks, olds, news)
        mo = {i for i, _ in matched}
        mn = {j for _, j in matched}
        pairs += [(i, j, 'changed') for i, j in matched]
        new_only += [j for j in news if j not in mn]
        old_only += [i for i in olds if i not in mo]
    # identical text moved elsewhere (outside its aligned region) is unchanged
    by_hash = {}
    for i in old_only:
        by_hash.setdefault(old_blocks[i].anchor.split('-')[0], []).append(i)
    still_new = []
    for j in new_only:
        h = new_blocks[j].anchor.split('-')[0]
        if by_hash.get(h):
            i = by_hash[h].pop(0)
            old_only.remove(i)
            pairs.append((i, j, 'unchanged'))
        else:
            still_new.append(j)
    return pairs, still_new, old_only


def _match_region(old_blocks, new_blocks, olds, news) -> list:
    """Monotone pairing of old and new prose blocks maximising similarity (DP)."""
    if not olds or not news:
        return []
    n, m = len(olds), len(news)
    sim = [[0.0] * m for _ in range(n)]
    for x, i in enumerate(olds):
        for y, j in enumerate(news):
            ob, nb = old_blocks[i], new_blocks[j]
            if ob.kind != nb.kind and not ({ob.kind, nb.kind} <= {'paragraph', 'list_item',
                                                                  'exercise', 'tab_note', 'note'}):
                continue
            s = difflib.SequenceMatcher(None, ob.content.split(), nb.content.split(),
                                        autojunk=False).ratio()
            sim[x][y] = s if s >= SIMILAR else 0.0
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for x in range(n - 1, -1, -1):
        for y in range(m - 1, -1, -1):
            best[x][y] = max(best[x + 1][y], best[x][y + 1],
                             best[x + 1][y + 1] + sim[x][y] if sim[x][y] else 0.0)
    out, x, y = [], 0, 0
    while x < n and y < m:
        if sim[x][y] and best[x][y] == best[x + 1][y + 1] + sim[x][y]:
            out.append((olds[x], news[y]))
            x, y = x + 1, y + 1
        elif best[x][y] == best[x + 1][y]:
            x += 1
        else:
            y += 1
    return out


def bump_language(lang, to_rev: str, dry_run=False) -> list:
    old_rev = lang.aligned
    if not old_rev:
        raise config.L10nError(f'{lang.code} has no pinned revision to bump from')
    new_rev = config.stable_rev(to_rev)
    have = sorted(str(p.relative_to(lang.book)) for p in lang.book.rglob('*.md')) \
        if lang.book.exists() else []
    renames = rename_map(old_rev, new_rev, have)
    results = []
    written: dict = {}
    for old_path in have:
        entries = tfile.load(lang.book, old_path)
        new_path = renames.get(old_path)
        res = FileResult(old_path, new_path)
        results.append(res)
        if new_path is None:
            res.deleted = len(entries)
            res.dropped = list(entries)
            continue
        ob = english_blocks(old_rev, old_path)
        nb = english_blocks(new_rev, new_path)
        pairs, new_only, old_only = align(list(ob), list(nb))
        out = written.setdefault(new_path, {})
        for i, j, how in pairs:
            e = entries.get(ob[i].anchor)
            if e is None:
                continue
            na = nb[j].anchor
            if how == 'unchanged' and not e.stale:
                out[na] = tfile.Entry(e.text)
                res.unchanged += 1
            elif how == 'unchanged':      # still stale from an earlier bump
                out[na] = e
                res.changed += 1
            else:
                out[na] = e if e.stale else tfile.Entry(
                    e.text, was=ob[i].anchor, src_rev=old_rev,
                    src_path=old_path if old_path != new_path else None)
                res.changed += 1
        res.new = sum(1 for j in new_only)
        res.deleted = sum(1 for i in old_only if ob[i].anchor in entries)
        res.dropped = [ob[i].anchor for i in old_only if ob[i].anchor in entries]
    if not dry_run:
        for old_path in have:
            if old_path not in written:
                tfile.save(lang.book, old_path, {})
        for new_path, entries in written.items():
            order = [b.anchor for b in english_blocks(new_rev, new_path) if b.translatable]
            tfile.save(lang.book, new_path, entries, order)
        config.set_language_field(lang.code, 'pinned_next', new_rev)
    return results
