#!/usr/bin/env python3
"""Apply a glossary review's corrections to terms/<locale>.jsonl.

    python glossary/scripts/apply_review.py [--date 2026-11-26] [locale ...]

Reads review/<locale>-corrections.jsonl (one line per entry to change:
concept_id, current, proposed, new_status, severity, reason, sources) and
rewrites terms/<locale>.jsonl:

- `preferred` becomes `proposed`; the replaced term is dropped from
  `alternatives` and recorded in `notes` with the severity and reason.
- A line whose `proposed` equals `current` changes only the status, or (reason
  starting "ALT:") deletes the quoted alternatives it names.
- The old evidence attests the old term, so it is dropped. `verified_external`
  is kept only when the line cites a web page: each URL is registered in
  sources/review-<date>.jsonl and becomes the evidence; otherwise the entry is
  `proposed`.

A line whose `current` no longer matches the entry is skipped and reported, so
running the script twice changes nothing.
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlparse

BASE = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(l) for l in path.open(encoding='utf-8') if l.strip()]


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def nfc(s):
    return unicodedata.normalize('NFC', s.strip())


def source_record(url, locale, date):
    sid = f'review-{locale}-' + hashlib.sha1(url.encode()).hexdigest()[:10]
    p = urlparse(url)
    title = unquote(p.path.rstrip('/').rsplit('/', 1)[-1] or p.netloc).replace('_', ' ')
    return sid, {'id': sid, 'title': f'{p.netloc}: {title}'[:200], 'locale': locale,
                 'kind': 'web page cited by the glossary review', 'url': url,
                 'accessed': date, 'publisher': p.netloc,
                 'notes': f'Cited in review/{locale}-corrections.jsonl ({date}).'}


def apply(locale, date, new_sources, existing_sources):
    terms_path = BASE / 'terms' / f'{locale}.jsonl'
    rows = read_jsonl(terms_path)
    by_id = {r['concept_id']: r for r in rows}
    stats = {'changed': 0, 'status_only': 0, 'alt_removed': 0, 'skipped': 0,
             'verified_external': 0, 'downgraded': 0}
    for c in read_jsonl(BASE / 'review' / f'{locale}-corrections.jsonl'):
        r = by_id[c['concept_id']]
        current, proposed = nfc(c['current']), nfc(c['proposed'])
        note = f"Review {date} ({c['severity']})"
        if proposed == current:
            quoted = re.findall(r'["“«]([^"”»]+)["”»]', c['reason'])
            if c['reason'].startswith('ALT:') and quoted:
                before = list(r['alternatives'])
                r['alternatives'] = [a for a in r['alternatives'] if a not in quoted]
                if r['alternatives'] == before:
                    stats['skipped'] += 1
                    continue
                stats['alt_removed'] += 1
                r['notes'] = (r['notes'] + ' ' if r['notes'] else '') + f"{note}: {c['reason']}"
                continue
            if r['status'] == c['new_status']:
                stats['skipped'] += 1
                continue
            stats['status_only'] += 1
        else:
            if r['preferred'] != current:
                if r['preferred'] == proposed:
                    stats['skipped'] += 1
                    continue
                print(f'{locale}/{c["concept_id"]}: current {current!r} != {r["preferred"]!r}; skipped',
                      file=sys.stderr)
                stats['skipped'] += 1
                continue
            r['preferred'] = proposed
            r['alternatives'] = [a for a in r['alternatives'] if a not in (current, proposed)]
            note += f': replaced "{current}". {c["reason"]}'
            stats['changed'] += 1
        if proposed == current:
            note += f': status {r["status"]} -> {c["new_status"]}. {c["reason"]}'
        old_sources = ', '.join(r['source_ids'])
        if old_sources:
            note += f' (Former evidence: {old_sources}.)'
        urls = [s for s in c.get('sources', []) if isinstance(s, str) and s.startswith('http')]
        known = [s for s in c.get('sources', []) if s in existing_sources]
        status = c['new_status']
        evidence, source_ids = [], []
        if status == 'verified_external':
            for url in urls:
                sid, rec = source_record(url, locale, date)
                new_sources[sid] = rec
                source_ids.append(sid)
            source_ids += [s for s in known if existing_sources[s].get('locale') in (locale, 'multilingual')]
            source_ids = list(dict.fromkeys(source_ids))
            evidence = [{'source_id': sid, 'locator': 'page',
                         'source_term': proposed,
                         'context': f'Glossary review {date}: {c["reason"][:400]}'}
                        for sid in source_ids]
            if source_ids:
                stats['verified_external'] += 1
            else:
                status = 'proposed'
                stats['downgraded'] += 1
        r['status'], r['evidence'], r['source_ids'] = status, evidence, source_ids
        r['notes'] = (r['notes'] + ' ' if r['notes'] else '') + note
    write_jsonl(terms_path, rows)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('locales', nargs='*')
    ap.add_argument('--date', default='2026-11-26')
    a = ap.parse_args()
    locales = a.locales or sorted(p.name.removesuffix('-corrections.jsonl')
                                  for p in (BASE / 'review').glob('*-corrections.jsonl'))
    existing = {}
    for p in (BASE / 'sources').glob('*.jsonl'):
        existing.update({s['id']: s for s in read_jsonl(p)})
    out = BASE / 'sources' / f'review-{a.date}.jsonl'
    new_sources = {s['id']: s for s in read_jsonl(out)} if out.exists() else {}
    for loc in locales:
        print(loc, apply(loc, a.date, new_sources, existing))
    write_jsonl(out, [new_sources[k] for k in sorted(new_sources)])


if __name__ == '__main__':
    main()
