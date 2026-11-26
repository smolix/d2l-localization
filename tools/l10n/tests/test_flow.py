"""emit -> answer -> apply -> check -> bump, on real d2l-neu revisions."""

import json
import re
import unittest

import yaml

from l10n import cli, config, tfile
from l10n.answers import apply_answer, check_language, parse_answer
from l10n.build import menu_data, strings, translated_pages
from l10n.bump import align
from l10n.pseudo import pseudo
from l10n.segment import normalize, segment
from l10n.units import english_blocks, units_for, work_file

from . import HAVE_BOOK, SLOW, TempRepo

OLD = 'a89a23879'           # main on 2026-11-05
NEW = 'e92716458'           # main on 2026-11-25
PAGE = 'chapter_linear-regression/weight-decay.md'
MARK = re.compile(r'^<!--(en|tr):([0-9a-f]{8}(?:-\d+)?)((?: [a-z]+(?:=[^\s>]+)?)*)-->$')


def fill(work: str) -> str:
    """Act as a translator: pseudo-translate every open slot of a work file."""
    unit = work[work.index('=== UNIT ===') + len('=== UNIT ==='):]
    lines = unit.split('\n')
    eng, kind, cur = {}, {}, None
    for l in lines:
        m = MARK.match(l)
        if m:
            cur = m.groups()
            if cur[0] == 'en' and ' old' not in cur[2]:
                eng[cur[1]] = []
                kind[cur[1]] = re.search(r'kind=(\S+)', cur[2]).group(1)
        elif cur and cur[0] == 'en' and ' old' not in cur[2]:
            eng[cur[1]].append(l)
    out, k = [], 0
    while k < len(lines):
        m = MARK.match(lines[k])
        out.append(lines[k])
        k += 1
        if m and m.group(1) == 'tr' and 'locked' not in m.group(3):
            while k < len(lines) and not MARK.match(lines[k]):
                k += 1
            out += [pseudo(kind[m.group(2)], '\n'.join(eng[m.group(2)]).strip('\n')), '']
    return '\n'.join(out) + '\n=== DECISIONS ===\n- "loss" -> ⟦loss⟧\n\n=== NOTES ===\n- none\n'


@unittest.skipUnless(HAVE_BOOK, 'd2l-neu checkout not found')
class TestFlow(TempRepo):
    def lang(self, code='xx'):
        return config.get_language(code)

    def test_emit_apply_check(self):
        config.set_language_field('xx', 'pinned', NEW)
        lang = self.lang()
        blocks = english_blocks(NEW, PAGE)
        units = units_for(PAGE, blocks)
        covered = sorted(k for u in units for k in u.blocks)
        self.assertEqual(covered, [k for k, b in enumerate(blocks) if b.translatable])
        for u in units:
            w = work_file(lang, NEW, u, units, blocks, tfile.load(lang.book, PAGE))
            self.assertIn('=== BRIEF ===', w)
            self.assertIn(f'<!--unit:{PAGE}#{u.id} lang=xx rev={NEW}-->', w)
            for ans in parse_answer(fill(w)):
                rep = apply_answer(lang, ans, 'test-agent', 'pseudo')
                self.assertEqual(rep.errors, [])
        rep = check_language(lang, [PAGE], complete=True)
        self.assertEqual(rep.errors, [])
        page = translated_pages(lang, NEW)[PAGE]
        self.assertEqual(len(segment(page)), len(blocks))
        self.assertIn('⟦', page)
        self.assertNotIn('<!--en:', page)
        # structure untouched: every non-translatable block is byte-identical
        eng = config.show(NEW, PAGE)
        self.assertEqual([b.text for b in segment(eng) if not b.translatable],
                         [b.text for b in segment(page) if not b.translatable])
        log = [json.loads(l) for l in (lang.root / 'units.jsonl').read_text().splitlines()]
        self.assertEqual({r['agent'] for r in log}, {'test-agent'})
        self.assertIn('⟦loss⟧', (lang.root / 'decisions.md').read_text())

    def test_reject_and_locked_edit(self):
        config.set_language_field('xx', 'pinned', NEW)
        lang = self.lang()
        blocks = english_blocks(NEW, PAGE)
        units = units_for(PAGE, blocks)
        u = units[0]
        w = work_file(lang, NEW, u, units, blocks, {})
        bad = fill(w).replace('⟧', '⟧ $z$', 1)       # add a math token
        rep = apply_answer(lang, parse_answer(bad)[0], 'a', None)
        self.assertTrue(rep.errors)
        self.assertFalse((lang.book / PAGE).exists())
        apply_answer(lang, parse_answer(fill(w))[0], 'a', None)
        entries = tfile.load(lang.book, PAGE)
        w2 = work_file(lang, NEW, u, units, blocks, entries)
        self.assertIn('locked-->', w2)
        first = blocks[u.blocks[0]]
        edited = w2.replace(entries[first.anchor].text, entries[first.anchor].text + ' X', 1)
        rep = apply_answer(lang, parse_answer(edited)[0], 'b', None)
        self.assertEqual(rep.locked_edits, [first.anchor])
        self.assertEqual(check_language(lang).locked_edits[0].split('/')[-1].split(' ')[0],
                         first.anchor)

    @unittest.skipUnless(SLOW, 'L10N_SLOW_TESTS=0')
    def test_bump_real_revisions(self):
        config.set_language_field('xx', 'pinned', OLD)
        files = ['chapter_preliminaries/ndarray.md', 'chapter_linear-regression/linear-regression.md',
                 PAGE]
        cli.main(['pseudo', 'xx', '--files', *files])
        cli.main(['bump', 'xx', '--to', NEW, '--out', str(self.tmp / 'work')])
        lang = self.lang()
        self.assertEqual(lang.pinned_next, config.resolve_rev(NEW))
        n = {'unchanged': 0, 'stale': 0}
        for f in files:
            nb = {b.anchor: b for b in english_blocks(lang.aligned, f) if b.translatable}
            ob = {b.anchor: b for b in english_blocks(OLD, f) if b.translatable}
            for a, e in tfile.load(lang.book, f).items():
                self.assertIn(a, nb)
                plain = normalize(e.text.replace('⟦', '').replace('⟧', ''))
                if e.stale:
                    n['stale'] += 1
                    self.assertEqual(plain, normalize(ob[e.was].content))
                else:
                    n['unchanged'] += 1
                    self.assertEqual(plain, normalize(nb[a].content))
        self.assertGreater(n['unchanged'], 20)
        self.assertGreater(n['stale'], 20)
        works = sorted((self.tmp / 'work').glob('*.md'))
        self.assertTrue(works)
        self.assertTrue(any(' old-->' in w.read_text() for w in works))
        for w in works:
            for ans in parse_answer(fill(w.read_text())):
                self.assertEqual(apply_answer(lang, ans, 'updater', None).errors, [])
        cli.main(['check', 'xx', '--files', *files, '--promote'])
        lang = self.lang()
        self.assertEqual(lang.pinned, config.resolve_rev(NEW))
        self.assertIsNone(lang.pinned_next)


class TestAlign(unittest.TestCase):
    def test_classes(self):
        old = segment('# Title\n\nKeep this one.\n\nChange this sentence a little.\n\n'
                      '```{.python .input #c1}\nx = 1\n```\n\nDrop me entirely please.\n')
        new = segment('# Title\n\nKeep this one.\n\nChange this sentence a little bit.\n\n'
                      '```{.python .input #c1}\nx = 2\n```\n\nA brand new paragraph here.\n\n'
                      'Another new one.\n')
        pairs, new_only, old_only = align(old, new)
        how = {old[i].content: (new[j].content, h) for i, j, h in pairs}
        self.assertEqual(how['Keep this one.'][1], 'unchanged')
        self.assertEqual(how['Change this sentence a little.'],
                         ('Change this sentence a little bit.', 'changed'))
        self.assertEqual([new[j].content for j in new_only],
                         ['A brand new paragraph here.', 'Another new one.'])
        self.assertEqual([old[i].content for i in old_only], ['Drop me entirely please.'])


class TestStringsMenu(TempRepo):
    EN = ('# catalog\ncode: en\nlang: en\ndir: ltr\nsite.hosts: [d2l.ai]\nforum.mode: english\n'
          'forum.category: 5\nfonts.pdf.cjk: null\n\n# Navbar\nnav.slides: Slides\n'
          'nav.pdf: PDF   # short\nslides.footer: "{sec} {title}  ·  {host}"\n'
          'slides.readme: |\n  {title}, line one\n\n      indented code\n  last\n'
          'slides.lead: >-\n  A folded\n  value.\n')

    def test_strings_cycle(self):
        en = self.tmp / 'en.yml'
        en.write_text(self.EN, encoding='utf-8')
        lang = config.get_language('ja')
        path, _, _, todo = strings(lang, en_file=en)
        data = yaml.safe_load(path.read_text())
        ref = yaml.safe_load(self.EN)
        self.assertEqual(set(data), set(ref))
        self.assertEqual(data['slides.readme'], ref['slides.readme'])
        self.assertEqual(data['lang'], 'ja')
        self.assertEqual(data['site.hosts'], ['ja.d2l.ai'])
        self.assertEqual(data['fonts.pdf.cjk'], 'Noto Serif CJK JP')
        self.assertIn('nav.slides', todo)
        text = path.read_text().replace('nav.slides: "Slides"  # TODO translate',
                                        'nav.slides: "スライド"')
        path.write_text(text)
        en.write_text(self.EN.replace('nav.pdf: PDF   # short\n', '') + 'nav.new: New\n')
        path, added, removed, todo = strings(lang, en_file=en)
        data = yaml.safe_load(path.read_text())
        self.assertEqual(data['nav.slides'], 'スライド')
        self.assertEqual(added, ['nav.new'])
        self.assertEqual(removed, ['nav.pdf'])
        self.assertNotIn('nav.slides', todo)
        self.assertIn('nav.new: "New"  # TODO translate', path.read_text())
        zh = config.get_language('zh')
        data = yaml.safe_load(strings(zh, en_file=en)[0].read_text())
        self.assertEqual((data['forum.mode'], data['forum.category']), ('own', 16))

    def test_menu(self):
        m = menu_data()
        self.assertEqual([x['code'] for x in m['languages']], ['en', 'zh'])
        self.assertEqual(m['languages'][0]['url'], 'https://d2l.ai')
        self.assertEqual(m['languages'][1], {'code': 'zh', 'name': '中文', 'url': 'https://zh.d2l.ai'})
        self.assertEqual(m['old_editions'], [{'name': 'English (old)', 'url': 'https://en-v2.d2l.ai'}])
        self.assertEqual(len(menu_data(include_unpublished=True)['languages']), 3)


if __name__ == '__main__':
    unittest.main()
