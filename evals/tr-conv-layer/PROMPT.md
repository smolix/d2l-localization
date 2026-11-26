Translate one page of *Dive into Deep Learning* into Turkish, following the
project's translation instructions exactly. Replace `{NAME}` below with your
agent name (lower case, e.g. `gemini-3-pro`).

1. Read `TRANSLATING.md` at the root of this repository in full.
2. The page is split into three units, translated in order:
   - `evals/tr-conv-layer/chapter_convolutional-neural-networks__conv-layer__u01.md`
   - `evals/tr-conv-layer/chapter_convolutional-neural-networks__conv-layer__u02.md`
   - `evals/tr-conv-layer/chapter_convolutional-neural-networks__conv-layer__u03.md`

   Each work file is self-contained. Its BRIEF holds the style guide (or, where the language has none yet, a note to follow TRANSLATING.md §8 and record your conventions in DECISIONS),
   the decisions log, the glossary entries for the unit, the chapter outline
   and the whole English page; then come CONTEXT and UNIT.
3. For each unit, write your answer as TRANSLATING.md §2 specifies: the UNIT
   part with every marker line copied exactly and every slot filled, then
   optional DECISIONS and NOTES. Save it to `evals/tr-conv-layer/answers-{NAME}/`
   under the same file name as the work file.
4. The decisions log in the briefs is empty. Carry your own DECISIONS from
   earlier units forward, and keep the page consistent.
5. If you can run commands and a checkout of the English book (smolix/d2l-neu)
   is available, check your answers from the repository root (set `D2L_NEU`
   to that checkout if it is not at `/home/smola/d2l-neu`):

   ```bash
   bin/l10n apply tr evals/tr-conv-layer/answers-{NAME} --rev 370f7b3a15d0557b44a27c52e06ab7f5b30584e1 --dry-run
   ```

   Fix every error. A warning can be a false match; never change a correct
   translation only to silence a warning, but name the false match in NOTES.
   Never run `l10n apply` without `--dry-run`. Do not edit any other file.
   Keep any helper scripts or drafts in a directory of your own (e.g.
   `work/{NAME}/`, which git ignores): other agents may run the same test at
   the same time.
   (Python 3.12+ with PyYAML is needed.) Without the English checkout, skip
   this step and go through TRANSLATING.md §9 by hand; the comparison runs
   the check later.

When you finish, list the three answer files and the remaining check output.
