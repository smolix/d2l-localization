"""Glossary lookup: English concepts in a text, and their target terms."""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field

from .config import GLOSSARY_DIR
from .protect import strip_tokens

BINDING = {'verified_d2l', 'verified_external', 'invariant'}


@dataclass
class Concept:
    id: str
    en: str
    aliases: list
    sense: str
    preferred: str | None = None
    alternatives: list = field(default_factory=list)
    status: str | None = None
    notes: str = ''
    check: bool = True       # False: common English word; listed in briefs, not enforced

    @property
    def binding(self) -> bool:
        return self.status in BINDING

    def forms(self) -> list:
        return [f for f in [self.preferred, *self.alternatives] if f]


class Glossary:
    def __init__(self, locale: str | None, base=GLOSSARY_DIR):
        self.locale = locale
        self.concepts: dict = {}
        cpath = base / 'concepts.jsonl'
        if not cpath.exists():
            self.regex = None
            return
        for line in cpath.read_text(encoding='utf-8').splitlines():
            if line.strip():
                c = json.loads(line)
                self.concepts[c['id']] = Concept(c['id'], c['en'], c.get('aliases_en') or [],
                                                 c.get('sense', ''),
                                                 check=c.get('check', True) is not False)
        redirects = {}
        rpath = base / 'concept-redirects.json'
        if rpath.exists():
            redirects = json.loads(rpath.read_text(encoding='utf-8'))
        tpath = base / 'terms' / f'{locale}.jsonl'
        if locale and tpath.exists():
            for line in tpath.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                t = json.loads(line)
                cid = redirects.get(t['concept_id'], t['concept_id'])
                c = self.concepts.get(cid)
                if c:
                    c.preferred = t.get('preferred')
                    c.alternatives = [a if isinstance(a, str) else a.get('term', '')
                                      for a in t.get('alternatives') or []]
                    c.status = t.get('status')
                    c.notes = t.get('notes') or ''
        surfaces: dict = {}
        for c in self.concepts.values():
            for s in [c.en, *c.aliases]:
                if s:
                    surfaces.setdefault(s.lower(), []).append(c.id)
        self.surfaces = surfaces
        self.norm = {' '.join(re.split(r'[\s-]+', k)): k for k in surfaces}
        alts = sorted(surfaces, key=len, reverse=True)
        body = '|'.join(_surface_pattern(s) for s in alts)
        self.regex = re.compile(r'(?<![\w-])(?:' + body + r')(?![\w-])', re.I)

    def find(self, text: str) -> list:
        """[(surface, [concept ids])] for each concept mention, leftmost-longest."""
        if not self.regex:
            return []
        out = []
        for m in self.regex.finditer(strip_tokens(text)):
            key = _canonical(m.group(0), self.norm)
            if key:
                out.append((m.group(0), self.surfaces[key]))
        return out

    def concepts_in(self, text: str) -> list:
        seen, out = set(), []
        for _, ids in self.find(text):
            for cid in ids:
                if cid not in seen:
                    seen.add(cid)
                    out.append(self.concepts[cid])
        return out

    def violations(self, english: str, translation: str) -> list:
        """Binding concepts in the English whose target term is absent."""
        out = []
        low = ' '.join(translation.lower().split())
        for surface, ids in self.find(english):
            cands = [self.concepts[c] for c in ids]
            if not any(c.binding and c.check for c in cands):
                continue
            forms = [f for c in cands for f in c.forms()]
            if forms and not any(_present(f, low) for f in forms):
                out.append(f'{surface!r} -> {" / ".join(c.preferred for c in cands if c.preferred)}')
        return sorted(set(out))


def _present(form: str, low: str) -> bool:
    """Is the target term in the (lower-cased, whitespace-normalised) translation?

    A line break inside the term does not hide it. In scripts with spaces, an
    inflected form also counts: each word of the term may change its last
    letter (last two for words over six letters) and take an ending ("champ récepteur" in "champs récepteurs",
    "matriz" in "matrices", "convolução" in "convoluções", "katman" in "katmanları").
    """
    f = ' '.join(form.lower().split())
    if f in low:
        return True
    words = f.split()
    if not all(re.fullmatch(r'[^\W\d_]+', w) for w in words):
        return False
    stems = [re.escape(w[:-2] if len(w) > 6 else w[:-1] if len(w) > 4 else w) + r'[^\W\d_]*'
             for w in words]
    return re.search(r'(?<![^\W\d_])' + r'\s+'.join(stems), low) is not None


def _surface_pattern(s: str) -> str:
    """Regex for an English surface form: flexible whitespace/hyphen, plural."""
    words = re.split(r'[\s-]+', s)
    pat = r'[\s-]+'.join(re.escape(w) for w in words)
    if re.search(r'[a-z]$', s, re.I) and not s.isupper():
        pat += r'(?:s|es)?'
    return pat


def _canonical(found: str, norm: dict) -> str | None:
    f = ' '.join(re.split(r'[\s-]+', found.lower()))
    for cand in (f, f[:-1], f[:-2]):
        if cand in norm:
            return norm[cand]
    return None


@functools.lru_cache(maxsize=16)
def load(locale: str | None) -> Glossary:
    return Glossary(locale)
