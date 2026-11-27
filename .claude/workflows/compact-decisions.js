export const meta = {
  name: 'compact-decisions',
  description: 'Compact a language\'s running decisions log into a lexicon plus a style.md conventions section (args: {lang, date})',
  whenToUse: 'A <lang>/decisions.md without lexicon markers has grown past ~60 KB (l10n apply warns), or a language starts a new translation run on an old log. DESIGN.md section 5.',
  phases: [
    { title: 'Split', detail: 'l10n compact <lang> split', model: 'sonnet' },
    { title: 'Extract', detail: 'one Sonnet agent per ~70 KB chunk → term entries + convention rules', model: 'sonnet' },
    { title: 'Merge', detail: 'l10n compact <lang> merge → conflicting renderings', model: 'sonnet' },
    { title: 'Resolve', detail: 'Opus settles conflicts; Opus writes the style.md conventions section', model: 'opus' },
    { title: 'Assemble', detail: 'l10n compact <lang> assemble; verify', model: 'sonnet' },
  ],
}

// Models: extraction is mechanical (Sonnet); resolving competing renderings and writing
// style rules needs judgement in the target language (Opus). Do not use the language's
// translator model here: the compaction must not re-decide the translation's style.
const { lang, date } = args
if (!lang || !date) throw new Error('args: {lang: "<code>", date: "YYYY-MM-DD"}')
const REPO = '/Users/smola/Repositories/github/d2l-localization'
const D = `${REPO}/work/${lang}/compact`
const ENV = `Work in ${REPO}. Set \`export D2L_NEU=/Users/smola/Repositories/github/d2l-neu\` before bin/l10n commands. Do not commit or change git state.`
const BATCH = 250   // conflict groups per resolver agent

phase('Split')
const sp = await agent(`${ENV} Run \`rm -rf ${D} && bin/l10n compact ${lang} split ${D}\` and return the number of chunks it wrote (chunk00.md … in ${D}). If it fails, return 0 and the error.`,
  { label: `split:${lang}`, phase: 'Split', model: 'sonnet', effort: 'low',
    schema: { type: 'object', properties: { chunks: { type: 'integer' }, error: { type: 'string' } }, required: ['chunks'] } })
if (!sp || !sp.chunks) throw new Error(`split failed: ${sp && sp.error}`)

const extract = i => {
  const n = String(i).padStart(2, '0')
  return `${ENV}

You are compacting the translation decisions log of *Dive into Deep Learning* for language \`${lang}\` (${lang}/decisions.md). The log is a running list, one "## <unit> (agent, date)" section per translated unit, sometimes with editorial or "chapter check" sections that override earlier choices. It is being replaced by (a) a compact lexicon, one line per English term, and (b) a short list of convention rules for ${lang}/style.md. Skim the headings of ${lang}/style.md so you know what is already a style rule.

Input: ${D}/chunk${n}.md (one chronological slice; read all of it).

Write two files:
1. ${D}/ext${n}.jsonl — one JSON object per line per term decision worth keeping:
   {"en": "...", "sense": "...", "tr": "...", "note": "...", "src": "<section heading without ##>"}
   - en: the English surface form as it appears in running text, base form (singular noun, bare verb), lowercase except names and acronyms, no quotes. Variants of one concept in one entry separated by " / ".
   - sense: empty unless the English word has several senses in the book (bias, kernel, return, regime, scale, …); then a few words naming this sense.
   - tr: the ${lang} rendering exactly as the log writes it, without the log's quotation marks — COPY the characters, do not retype from memory. The main form; plural, slide or variant forms go in note.
   - note: at most ~15 words (plural, slide form, "not X (= other sense)", gloss policy). Empty if nothing.
   - Within the chunk, when a later entry changes an earlier one for the same term (editorial/chapter-check sections are authoritative), output only the final choice with its src.
2. ${D}/conv${n}.md — one bullet per convention rule that is not a single term mapping (how to render a kind of construct: headings, recurring section titles, slides, captions, credit lines, names, numbers, glosses, typography, recurring phrase patterns), one or two sentences with one example, then "(src: <section>)". Skip rules ${lang}/style.md already states.

DROP: bookkeeping lines ("English glosses given in …"); everyday one-off phrases that are not fixed terms of the book; choices superseded later in the chunk. Keep anything technical, any recurring phrase, any disambiguation, any title a later page must repeat. When in doubt, keep.

Every line of the JSONL must be valid JSON with literal UTF-8 (no \\u escapes). Verify: python3 -c "import json;[json.loads(l) for l in open('${D}/ext${n}.jsonl') if l.strip()]". Return the counts.`
}

phase('Extract')
const ex = await parallel(Array.from({ length: sp.chunks }, (_, i) => () =>
  agent(extract(i), { label: `extract:${lang}:${i}`, phase: 'Extract', model: 'sonnet', effort: 'medium',
    schema: { type: 'object', properties: { entries: { type: 'integer' }, conventions: { type: 'integer' }, remarks: { type: 'string' } }, required: ['entries'] } })))
const failed = ex.map((r, i) => r ? null : i).filter(i => i !== null)
if (failed.length) throw new Error(`extraction failed for chunks ${failed.join(', ')}; resume the workflow`)

phase('Merge')
const mg = await agent(`${ENV} Run \`bin/l10n compact ${lang} merge ${D}\` and return its counts (entries, keys, conflicts).`,
  { label: `merge:${lang}`, phase: 'Merge', model: 'sonnet', effort: 'low',
    schema: { type: 'object', properties: { entries: { type: 'integer' }, keys: { type: 'integer' }, conflicts: { type: 'integer' } }, required: ['conflicts'] } })
log(`${lang}: ${mg.entries} entries, ${mg.keys} English keys, ${mg.conflicts} conflicts`)

const resolve = (lo, hi, k) => `${ENV}

The ${lang} decisions log of *Dive into Deep Learning* is being compacted into a lexicon with one line per English term and sense. ${D}/conflicts.md lists English keys with competing renderings; each group "### Cnnn <key>" lists candidates in chronological order as [chunk.line] with en, sense, ${lang}, note and src (editorial or "chapter check" sections are authoritative). Handle groups C${String(lo).padStart(3, '0')} to C${String(hi - 1).padStart(3, '0')} only.

For each group decide the final entry or entries:
- Same term in different grammatical forms → ONE entry with the base form a glossary would list; variants in note.
- Genuinely different senses → one entry per sense with a short sense label.
- Competing choices for one sense → the later decision wins, an editorial/chapter-check decision beats a unit entry, and a choice agreeing with ${lang}/style.md or the glossary (${REPO}/glossary/terms/<locale>.jsonl for this language; see languages.yml) beats one that does not. Put the loser in note as "not X" only when a translator might reach for it.
- Copy the ${lang} characters from the candidates; do not retype.

Write ${D}/resolved-${k}.jsonl: one JSON object per final entry, {"key": "<group key after ### Cnnn>", "en", "sense", "tr", "note"}; every group key in your range at least once. Validate with python3 (json.loads per line; all keys of your range present). Return the number of lines and anything a human should look at.`

const conventions = `${ENV}

The ${lang} decisions log of *Dive into Deep Learning* is being compacted. Its convention rules were extracted into ${D}/conv00.md … conv${String(sp.chunks - 1).padStart(2, '0')}.md (chronological; later and editorial rules win). Read ${lang}/style.md in full, then all conv files. Write ${D}/style-section.md, a new section for ${lang}/style.md headed exactly "## X. Conventions from the translated chapters" with subsections "### X.1 …", "### X.2 …" (the tool replaces X with the next section number). One rule per bullet or table row, grouped under a few subheadings (headings and recurring titles; slides; exercises; figures, captions, tables and credit lines; names and Latin terms; numbers and units; glosses and quotations; notation, math and code in prose; consistency): the decision, a short reason if not obvious, one example (English → ${lang}), in the terse style of ${lang}/style.md. Merge duplicates, leave out what style.md already says, resolve contradictions in favour of the later or editorial rule; single term mappings do not belong here. At most ~250 lines: every translator reads this with every unit. If a rule contradicts style.md itself, keep style.md and list the contradiction in ${D}/style-contradictions.md (create it, empty if none). Do not edit repository files. Return the line count and remarks.`

phase('Resolve')
const jobs = []
for (let lo = 0, k = 0; lo < mg.conflicts; lo += BATCH, k++) {
  const hi = Math.min(lo + BATCH, mg.conflicts)
  jobs.push(() => agent(resolve(lo, hi, k), { label: `resolve:${lang}:${k}`, phase: 'Resolve', model: 'opus', effort: 'medium',
    schema: { type: 'object', properties: { written: { type: 'integer' }, remarks: { type: 'string' } }, required: ['written'] } }))
}
jobs.push(() => agent(conventions, { label: `style:${lang}`, phase: 'Resolve', model: 'opus', effort: 'medium',
  schema: { type: 'object', properties: { written: { type: 'integer' }, remarks: { type: 'string' } }, required: ['written'] } }))
const rs = await parallel(jobs)
if (rs.some(r => !r)) throw new Error('a resolve/style agent failed; resume the workflow')

phase('Assemble')
const as = await agent(`${ENV} Run \`bin/l10n compact ${lang} assemble ${D} --date ${date}\`. If it reports missing conflict keys or malformed lines, fix the offending resolved-*.jsonl lines (copying from ${D}/conflicts.md) and rerun. Then verify: \`(cd tools && python3 -m unittest discover -s l10n/tests -t .)\` passes, and \`bin/l10n check ${lang} --quiet | tail -1\` reports 0 errors. Return the assemble output, the check line, and the sizes of ${lang}/decisions.md and ${lang}/style.md (wc -c).`,
  { label: `assemble:${lang}`, phase: 'Assemble', model: 'sonnet', effort: 'low' })

return { lang, split: sp, merge: mg, resolvers: rs, assemble: as }
