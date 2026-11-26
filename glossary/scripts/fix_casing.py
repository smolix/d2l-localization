#!/usr/bin/env python3
"""Lowercase heading-style capitals in tr and de glossary entries (review 2026-11-26).

tr: every word is lowercased unless it is a proper name (matches a capitalised
    word of the English concept), an acronym, or has internal capitals.
de: a leading inflected adjective is lowercased, also from names (bayessche,
    gaußsches, per Duden); English words of the concept (Monte Carlo) are kept ("Stochastischer Gradientenabstieg" ->
    "stochastischer Gradientenabstieg"); nouns keep their capitals.
Prints each change; --write rewrites terms/<locale>.jsonl.
"""
import json, re, sys
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
C = {json.loads(l)['id']: json.loads(l) for l in open(BASE / 'concepts.jsonl', encoding='utf-8')}

def tr_lower(w):
    return w.replace('I', 'ı').replace('İ', 'i').lower()

def names(cid):
    c = C[cid]
    return [t for s in [c['en']] + c.get('aliases_en', []) for t in re.findall(r"[A-Z][^\W\d_A-Z]+", s)]

TR_NAMES = ('Öklid', 'Jakobi', 'Hesse', 'Gauss', 'Laplace', 'Lagrange')  # Turkish spellings of names

def is_name(w, cid):
    core = re.split(r"['’]", w)[0]
    return (core.startswith(TR_NAMES)
            or any(core == n or (len(n) >= 3 and core.lower().startswith(n.lower()[:4])) for n in names(cid)))

def fix_tr(term, cid):
    out = []
    for part in re.split(r'(\s+|-)', term):
        if (len(part) > 1 and part[0].isupper() and part[1:].isalpha() and part[1:] == part[1:].lower()
                and not is_name(part, cid)):
            part = part[0].replace('I', 'ı').replace('İ', 'i').lower() + part[1:]
        out.append(part)
    return ''.join(out)

def fix_de(term, cid):
    words = term.split(' ')
    if len(words) < 2 or not words[0][:1].isupper() or not words[1][:1].isupper():
        return term
    head = words[0].split('-')[-1]
    english = {w.lower() for s in [C[cid]['en']] + C[cid].get('aliases_en', []) for w in re.findall(r"[A-Za-z]+", s)}
    if re.search(r'(e|er|es|en|em)$', head) and head.lower() not in english:
        words[0] = words[0][0].lower() + words[0][1:]
    return ' '.join(words)

def main():
    write = '--write' in sys.argv
    for loc, fix in (('tr', fix_tr), ('de', fix_de)):
        p = BASE / 'terms' / f'{loc}.jsonl'
        rows = [json.loads(l) for l in open(p, encoding='utf-8')]
        n = 0
        for r in rows:
            if not r['preferred']:
                continue
            new = fix(r['preferred'], r['concept_id'])
            alts = [fix(a, r['concept_id']) for a in r['alternatives']]
            if new != r['preferred'] or alts != r['alternatives']:
                n += 1
                print(f'{loc}\t{r["concept_id"]}\t{r["preferred"]}\t{new}')
                r['preferred'], r['alternatives'] = new, list(dict.fromkeys(a for a in alts if a != new))
        print(f'{loc}: {n} entries', file=sys.stderr)
        if write:
            with open(p, 'w', encoding='utf-8') as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

main()
