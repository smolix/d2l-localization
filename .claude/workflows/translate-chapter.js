export const meta = {
  name: 'translate-chapter',
  description: 'Translate one d2l chapter into a language: one translator per page (units in order), per-page review, editorial pass that folds the decisions into the lexicon',
  whenToUse: 'Translating a chapter end to end (DESIGN.md section 5). args: {lang, chapter, pages, date, translator: {agent, modelId?, model?, effort?}, reviewer?: {model?, effort?}, editor?: {model?, effort?}, reviewChecks?, repo?, d2lNeu?}. Run one chapter at a time; commit after each.',
  phases: [
    { title: 'Translate', detail: 'one agent per page: emit --units 1 → answer → apply, until the page is done' },
    { title: 'Review', detail: 'independent block-by-block accuracy check per page' },
    { title: 'Editorial pass', detail: 'cross-page consistency; fold the chapter\'s decisions into the lexicon (l10n fold) and style.md' },
  ],
}

// Example (Arabic, 2026-11-27): translator {agent: "claude fable-5.2 (medium)", modelId: "claude-fable-5-2",
// model: "fable", effort: "medium"}, reviewer {model: "opus", effort: "medium"}, editor {model: "fable",
// effort: "medium"}. The part's index page goes with the part's first chapter.
const { lang, chapter, pages, date } = args
const T = args.translator || {}
const R = args.reviewer || {}
const E = args.editor || {}
if (!lang || !chapter || !pages || !pages.length || !date || !T.agent)
  throw new Error('args: {lang, chapter, pages: [...], date, translator: {agent: "<who> (<effort>)", ...}}')
const REPO = args.repo || '/Users/smola/Repositories/github/d2l-localization'
const NEU = args.d2lNeu || '/Users/smola/Repositories/github/d2l-neu'
const ENV = `Work in ${REPO}. Before every bin/l10n command run \`export D2L_NEU=${NEU}\`. Never commit, never touch git state, never edit languages.yml or glossary/.`
const AGENT = `--agent "${T.agent}"` + (T.modelId ? ` --model ${T.modelId}` : '')
const stem = p => p.replace(/\.md$/, '').replace('/', '__')
const opts = (o, extra) => Object.assign({}, extra, o.model ? { model: o.model } : {}, o.effort ? { effort: o.effort } : {})
const CONV = '"Conventions from the translated chapters" section'

const TR_SCHEMA = { type: 'object', properties: {
  page: { type: 'string' },
  units_applied: { type: 'array', items: { type: 'string' } },
  complete: { type: 'boolean', description: 'emit reports 0 work files for this page' },
  warnings: { type: 'integer' },
  problems: { type: 'string', description: 'anything unresolved; empty if none' },
}, required: ['page', 'units_applied', 'complete', 'problems'] }

const RV_SCHEMA = { type: 'object', properties: {
  page: { type: 'string' },
  blocks_reviewed: { type: 'integer' },
  fixes: { type: 'array', items: { type: 'object', properties: {
    hash: { type: 'string' }, problem: { type: 'string' }, fix: { type: 'string' } }, required: ['hash', 'problem'] } },
  terms_seen: { type: 'string', description: 'key term choices on this page that the editor should compare across pages' },
  check_clean: { type: 'boolean' },
}, required: ['page', 'fixes', 'check_clean'] }

function translatePrompt(page) {
  const s = stem(page)
  return `You are the ${lang} translator for one page of *Dive into Deep Learning*: \`${page}\` (chapter ${chapter}). ${ENV}

First read TRANSLATING.md in full. Then translate every unit of this page, in order, one at a time:

1. \`bin/l10n emit ${lang} --files ${page} --units 1\` writes the page's next untranslated unit to work/${lang}/${s}__uNN.md (it prints the path). If it reports 0 work files, the page is done: stop.
2. Read the work file in parts (grep -n for "## Decisions log", "## Glossary entries for this unit", "=== CONTEXT", "=== UNIT ===" to find them). On the first unit read the style guide in full, including its ${CONV}. On every unit read the decisions-log part in full (only the lexicon lines whose English occurs in this unit, plus this chapter's entries so far), the glossary entries, the whole CONTEXT (it holds your translation of the previous section) and the UNIT.
3. Write your answer (the UNIT part with every marker copied exactly and every slot filled, then optional === DECISIONS === and === NOTES === exactly as TRANSLATING.md §2 says; DECISIONS bullets in the lexicon form \`- english (sense) → term — note\`) to work/${lang}/answers/${s}/${s}__uNN.md.
4. \`bin/l10n apply ${lang} <answer> --dry-run\`. Fix every error. Read every warning: fix real problems; for a false match keep the correct translation and name the block and the false match in NOTES. Then apply: \`bin/l10n apply ${lang} <answer> ${AGENT}\`.
5. Go back to step 1.

Follow ${lang}/style.md and the decisions log exactly; protected tokens byte-identical; glossary and decisions log binding. Never re-apply a unit you already applied, even if another page later logs a different term: report the difference under problems; the editor reconciles it. Edit only files under work/${lang}/answers/${s}/ (apply writes ${lang}/book, notes, decisions.md and units.jsonl). Other agents translate other pages at the same time; do not touch their files.

Return: the units you applied, whether emit now reports 0 work files for this page, the total warnings, and any unresolved problem.`
}

function reviewPrompt(page) {
  const s = stem(page)
  return `You are an independent reviewer of the ${lang} translation of one page of *Dive into Deep Learning*: \`${page}\`. Another agent translated it; find and fix real errors, do not restyle. ${ENV}

Read ${lang}/style.md and TRANSLATING.md §4. Then \`bin/l10n emit ${lang} --files ${page} --all --out work/${lang}/review/${s}\`: each work file shows every English block with its current translation (locked). Read in each file the "## Glossary entries for this unit" section and the UNIT part; consult the CONTEXT's English page only where a block depends on surrounding code or equations. Check each block against its English for:
- meaning: anything added, dropped or changed; negations, connectives, quantities, signs, comparisons, referents;
- terminology against the unit's glossary entries (right sense; one term per concept within the page);
- the checkable rules of ${lang}/style.md${args.reviewChecks ? `, in particular: ${args.reviewChecks}` : ''};
- broken grammar or sentences that do not read naturally.

Fix each real error directly in ${lang}/book/${page}, only the translation text under its \`<!--en:HASH-->\` anchor; never an anchor line, never another page, never ${lang}/decisions.md (report term choices instead). Then \`bin/l10n check ${lang} --files ${page}\` and fix every error your edits caused. Return the fixes (hash, problem, fix), blocks reviewed, the term choices the editor should compare across pages, and whether the check is clean.`
}

function editorPrompt(reviews, translations) {
  return `You are the editor for the ${lang} translation of chapter ${chapter} of *Dive into Deep Learning* (DESIGN.md §5 "Editorial pass", TRANSLATING.md §11). Every page has been translated and reviewed. ${ENV}

Pages: ${pages.join(', ')}.
Translators' reports: ${JSON.stringify(translations)}
Reviewers' reports: ${JSON.stringify(reviews)}

Read ${lang}/style.md (including its ${CONV}), the unit sections after the lexicon end marker in ${lang}/decisions.md (this chapter's entries; grep the lexicon for any other term), and every page of the chapter in ${lang}/book/ (use \`bin/l10n emit ${lang} --files <page> --all --out work/${lang}/editor/<stem>\` where you need the English). Make the chapter one text: one term per concept across its pages and with earlier chapters (choose what the lexicon or glossary supports, else the better term, and edit the other pages); the same rendering of recurring titles, slide headings, credit lines and captions; one register and smooth transitions. Edit only translation text under existing anchors.

Then fold the chapter's decisions (do not edit ${lang}/decisions.md by hand):
1. Write work/${lang}/fold-${chapter}.md: one lexicon line \`- english / variant (sense) → term — note\` per term mapping of this chapter's sections, in its final form after your edits (English in base form as in running text; a sense only for words with several senses; repeat an existing English part exactly to change it; \`- <english part> → (delete)\` to remove). Leave out bookkeeping, one-off everyday phrases and superseded choices. Copy the target text exactly.
2. Add each convention rule of those sections that is not a single term to the ${CONV} of ${lang}/style.md (create that section at the end, numbered after the last one, if it does not exist), unless it is already there.
3. \`bin/l10n fold ${lang} work/${lang}/fold-${chapter}.md\`; check the counts and that the file now ends with the lexicon end marker.
Run \`bin/l10n check ${lang} --files ${pages.join(' ')}\` and fix all errors. Return: terms unified, pages edited, lexicon lines added/changed, style rules added, the final check line.`
}

phase('Translate')
const results = await pipeline(
  pages,
  page => agent(translatePrompt(page), opts(T, { label: `tr:${page}`, phase: 'Translate', schema: TR_SCHEMA })),
  (tr, page) => {
    if (!tr) { log(`translator for ${page} returned nothing`); return null }
    if (!tr.complete) log(`${page}: translator reports page incomplete — ${tr.problems}`)
    return agent(reviewPrompt(page), opts(R, { label: `rv:${page}`, phase: 'Review', schema: RV_SCHEMA }))
      .then(rv => ({ page, tr, rv }))
  },
)
const done = results.filter(Boolean)
const missing = pages.filter(p => !done.find(d => d.page === p))
if (missing.length) log(`pages without a finished translate+review: ${missing.join(', ')}`)

phase('Editorial pass')
const summary = await agent(editorPrompt(done.map(d => d.rv), done.map(d => d.tr)), opts(E, { label: `edit:${chapter}`, phase: 'Editorial pass' }))

return { lang, chapter, missing, pages: done.map(d => ({ page: d.page, units: d.tr.units_applied, complete: d.tr.complete, problems: d.tr.problems, fixes: d.rv ? d.rv.fixes.length : null, check_clean: d.rv ? d.rv.check_clean : null })), summary }
