# Greek (`el`) glossary review, 2026-11-26

Reviewer pass over the drafted `terms/el.jsonl` (1,191 records, Modern Greek, monotonic). No source was inspected: every record stays `proposed` with empty `source_ids` and `evidence`. The judgments below are those of a terminology editor working from knowledge of Greek university teaching and textbook usage; they are not attestations.

Result: **120 preferred terms changed**, **159 further records changed in alternatives or notes only** (75 of them a mechanical repair, see Grammar and orthography), 912 records unchanged. The draft is kept at `scratchpad/gloss-el/el.opus-draft.jsonl` of the drafting session; the edit script is `review/apply.py` next to it.

## What changed and why

### Family consistency

| Family | Decision | Records touched |
| --- | --- | --- |
| variance | `διακύμανση` everywhere, so that it shares a stem with `συνδιακύμανση` and gives the usual `μεροληψία–διακύμανση`; `διασπορά` is the alternative | variance, variance reduction, bias-variance decomposition, VP/VE, posterior/sample/estimator/noise/conditional/total/predictive/gradient variance, variance schedule |
| Gaussian | adjective `γκαουσιανός` in every compound; the draft mixed it with `κατανομή Gauss` | Gaussian distribution, multivariate Gaussian |
| embedding | `ενσωμάτωση` everywhere (inflects, plural `ενσωματώσεις`); the draft had Latin `embedding` in ten compounds and `διανυσματική αναπαράσταση λέξεων` for word embedding | embedding, word/position/rotary/patch/user/item embedding, embedding matrix/layer/dimension/table |
| descent | `κατάβαση` everywhere; the draft switched to `κάθοδος` in two compounds | descent direction, descent lemma |
| conditional | `δεσμευμένος` for probability objects (probability, distribution, entropy, expectation, variance, independence, likelihood, mutual information); `υπό συνθήκη` for models and procedures | conditional generation/model/denoising/velocity/flow matching, class conditioning |
| predictive | `προβλεπτικός` (from `πρόβλεψη`), not `προγνωστικός` | posterior predictive (4 records), predictive distribution/variance |
| matching | `ταίριασμα` for both score matching and flow matching; the draft translated one and kept the other in Latin | flow matching, conditional flow matching, denoising score matching |
| cross- | `διασταυρούμενη` (entropy, validation, attention); cross-correlation keeps the established `ετεροσυσχέτιση` | cross-validation |
| action / actor | `δράση` / `δράστης`; `ενέργεια` is left to energy (energy function, energy-based model, free energy) | action, action-value function, state-action pair |
| discount | `προεξόφληση` (`συντελεστής προεξόφλησης`, `προεξοφλημένος`) | discount factor, NDCG |
| context | `συμφραζόμενα` in all language-model compounds | in-context learning, long-context |
| invariance / equivariance | `αμεταβλητότητα` / `ισομεταβλητότητα ως προς τη μετατόπιση` | translation equivariance/invariance, spatial invariance |
| forward / backward / reverse | `εμπρόσθιος`; backward pass `οπίσθιο πέρασμα`; diffusion reverse `αντίστροφος`; reverse-mode differentiation `ανάστροφη λειτουργία` (a backward sweep, not an inverse) | backward pass, reverse-time SDE |
| mode (of a distribution) | `κορυφή`, as the draft already had in posterior mode | mode collapse, mode dropping |
| bound | `φράγμα` | upper confidence bound, ELBO |
| decomposition / factorization | `ανάλυση` / `παραγοντοποίηση` | Cholesky decomposition |
| importance | `σπουδαιότητα` (significance keeps `σημαντικότητα`) | importance weight |
| stride | `διασκελισμός`, leaving `βήμα` to step | stride, strided convolution |
| clip / crop / truncate | `ψαλιδισμός` / `περικοπή` / `περικομμένος`, with `σφάλμα αποκοπής` kept as the fixed numerical-analysis term | gradient clipping |
| masking | object `μάσκα`, "masked X" `X με μάσκα`, the act `απόκρυψη` | attention masking, causal masking |
| normalizing flow | `ροή κανονικοποίησης` | normalizing flow, continuous normalizing flow |
| statistic | `στατιστική συνάρτηση` / `ελεγχοσυνάρτηση` | sufficient statistic, test statistic |
| preconditioning | `προσταθεροποίηση` / `προσταθεροποιητής` | preconditioner, preconditioning |
| tokenization | `κατάτμηση σε tokens`, `κατάτμηση σε υπολεξικές μονάδες` | tokenization, subword tokenization, subword |
| bottleneck, pipeline | Latin `bottleneck`, `pipeline` in every compound | bottleneck, bottleneck block, information bottleneck, input pipeline, pipeline parallelism |

Left as drafted after checking: the normalization (`κανονικοποίηση`) versus regularization (`ομαλοποίηση`) split, `παρτίδα` / `μίνι-παρτίδα`, the three bias senses (`πόλωση` additive; `μεροληψία` statistical, inductive, implicit, selection, exposure, bias correction), encoder / decoder / autoencoder, the gate names, policy compounds, the schedule family (`πρόγραμμα`), and the Markov compounds (`αλυσίδα Markov` next to `μαρκοβιανή διαδικασία αποφάσεων`; both patterns are normal Greek).

### Wrong sense

- multi-armed bandit: `πολύοπλος ληστής` ("many-weaponed") → `ληστής πολλών βραχιόνων`.
- ill-conditioned: `κακώς ορισμένος` (ill-defined) → `κακής κατάστασης`; the same wrong alternative removed from ill-conditioning.
- quasi-Newton: `ημι-Newton` (semi-) → `οιονεί Newton`.
- transpose: the concept is the operation → `αναστροφή` (result: `ανάστροφος πίνακας`).
- random horizontal flip: `αναστροφή` (transpose) → `κατοπτρισμός`.
- experience replay: `επανάληψη` (iteration) → `αναπαραγωγή εμπειρίας`.
- exponential averaging: `εκθετική εξομάλυνση` (smoothing) → `εκθετικός μέσος όρος`.
- conjugate gradient: the concept is the method → `μέθοδος συζυγών κλίσεων`.
- training compute: `υπολογισμός εκπαίδευσης` → `υπολογιστικό κόστος εκπαίδευσης`.
- weight tying: `δέσμευση βαρών` (conditioning) → `δέσιμο βαρών`.
- noise level: `επίπεδο θορύβου` → `στάθμη θορύβου` (keeps `επίπεδο` for layer).
- Wrong alternatives removed: `διευκρινιστής` (discriminator), `μπεϋζιανή επαγωγή` (Bayesian inference), `ελλειπτικότητα` (missingness), `στοιχειώδες γινόμενο` (Hadamard), `διαπλοκή` (perplexity), `κρατημένα δεδομένα` (held-out data).

### Better standard term

accuracy `ορθότητα` and precision `ακρίβεια`; residual `κατάλοιπο`; span `γραμμική θήκη`; alignment `στοίχιση`; quantization `κβάντιση`; F1 score `μέτρο F1`; soft-thresholding `μαλακή κατωφλίωση`; backtracking `οπισθοδρόμηση`; random seed `φύτρο τυχαίων αριθμών`; fine-tuning `λεπτή ρύθμιση`; generator `γεννήτορας`; terminal reward `τερματική ανταμοιβή`; policy evaluation `αξιολόγηση πολιτικής`; grid search `αναζήτηση σε πλέγμα`; perceptual loss `αντιληπτική απώλεια`; concept drift `ολίσθηση έννοιας`; BPTT `οπισθοδιάδοση μέσω του χρόνου`; sequence-to-sequence `ακολουθίας σε ακολουθία`; MCMC `Monte Carlo με αλυσίδες Markov`.

### Grammar and orthography

- 75 records had the drafter's note split into `alternatives` (items beginning with `| `); the text was moved back into `notes`.
- `διασταλμένη` → `διεσταλμένη συνέλιξη`; `επιλύτης` → `επιλυτής` (the draft had both accents); `δεξί` → `δεξιό ιδιάζον διάνυσμα`; `υποδιπλασίαση` → `υποδιπλασιασμός`; ELBO `τεκμηρίων` → `τεκμηρίου`; `μείωση του ρυθμού μάθησης` → `μείωση ρυθμού μάθησης` (article-free lemma like its neighbours).
- Notes now state gender and inflection for Latin-script nouns: neuter and without Greek endings (`το token`, `τα tokens`, `των tokens`; `το prompt`, `το dropout`, `το broadcasting`, `το checkpointing`, `το bottleneck`, `το pipeline`); `ο transformer`, `ο buffer`; feminine by gloss `η perplexity`, `η KV cache`.
- Genitive and adverbial modifiers are marked as placed after the noun (`μοντέλο ακολουθίας σε ακολουθία`, `πίνακας κακής κατάστασης`, `διάχυση διατήρησης διακύμανσης`, `εργασίες κατάντη`); `εγγύς` is marked indeclinable.

### Latin retention

- Moved to Greek: embedding family, tokenization, precision, flow-matching family (English in parentheses on first use).
- Moved to Latin, because the Greek draft term would not be recognised: bottleneck (was `στενωπός`), ground truth (was `πραγματική τιμή αναφοράς`), hinge (`απώλεια hinge`), ridge (`παλινδρόμηση ridge`), pipeline (was `αγωγός`).
- Kept in Latin as drafted: token, prompt, dropout, softmax, logit, transformer, perplexity, teacher forcing, broadcasting, checkpointing, bootstrap, bootstrapping, prefill, KV cache, stop-gradient, roofline, the collective operations, and all model, optimizer and metric names. Mixed compounds with a Greek head and a Latin name or loan are kept where Greek prose writes them that way (`πολυεπίπεδο perceptron`, `κωδικοποίηση one-hot`, `συνάρτηση score`, `μπλοκ transformer`, `tokens εκπαίδευσης`, `buffer ορμής`).

## Decisions on the drafter's list

1. gradient descent: `κατάβαση κλίσης` confirmed and extended to descent direction and descent lemma; `κάθοδος κλίσης` is the alternative.
2. regularization: `ομαλοποίηση` confirmed; `κανονικοποίηση` stays reserved for normalization.
3. precision: changed. accuracy `ορθότητα`, precision `ακρίβεια`, recall `ανάκληση`; numerical precision is also `ακρίβεια`, as in English.
4. layer: `επίπεδο` confirmed, `στρώμα` alternative; noise level moved to `στάθμη` to reduce the level/layer clash.
5. pooling: `συγκέντρωση` confirmed, with `pooling` in parentheses on first use.
6. batch / minibatch: `παρτίδα` / `μίνι-παρτίδα` confirmed.
7. embedding: changed to `ενσωμάτωση` throughout; the inconsistency with word embedding is gone.
8. bias: `πόλωση` (additive) and `μεροληψία` (all statistical and learning-preference senses) confirmed.
9. weight decay: `εξασθένηση βαρών` confirmed; schedule decay is `μείωση`.
10. fine-tuning: changed to `λεπτή ρύθμιση`.
11. GAN discriminator: `διακριτής` confirmed; generator changed to `γεννήτορας`.
12. variational: `μεταβολικός` confirmed for inference, autoencoder and objective; ELBO `κάτω φράγμα τεκμηρίου`, acronym after first use.
13. RL return: `απόδοση` confirmed (write `επίδοση` for performance in RL passages); discount factor changed to `συντελεστής προεξόφλησης`.
14. equivariance / invariance: `ισομεταβλητότητα` / `αμεταβλητότητα ως προς τη μετατόπιση`.
15. proximal operator: `εγγύς τελεστής` confirmed for the whole proximal family; `τελεστής εγγύτητας` alternative.

Runners-up: mode collapse → `κατάρρευση κορυφών`; downstream task `εργασία κατάντη` kept; throughput `ρυθμαπόδοση` kept; generative `παραγωγικό` kept; MLP `πολυεπίπεδο perceptron` kept; receptive field `υποδεκτικό πεδίο` kept.

Consistency questions: Gaussian unified on the adjective; inference split confirmed (`συμπερασματολογία` statistical, `συμπερασμός` for running a model and for NLI); ODE and SDE stay as Latin acronyms.

Merged or ambiguous pairs: dot / inner product (`εσωτερικό γινόμενο`), downsampling / subsampling (`υποδειγματοληψία`), RL critic / GAN critic (`κριτής`): merges accepted, with notes. transpose: changed to the operation. attention value `τιμή`: accepted. residual: now `κατάλοιπο`, separate from `υπολειμματικός` for connections. alignment: now `στοίχιση`. bootstrap / bootstrapping: both Latin, senses separated in the notes. score: `βαθμολογία` versus Latin `score` for the density score confirmed; F1 moved to `μέτρο F1`. negative feedback (recommender sense), slice (`τμήμα`) and decode (generation stage): accepted.

## Entries that most need a native-speaker check

1. accuracy `ορθότητα` / precision `ακρίβεια`: the largest change by frequency in the book.
2. embedding `ενσωμάτωση` (versus Latin `embedding`).
3. layer `επίπεδο` versus `στρώμα`.
4. action `δράση` versus `ενέργεια`; return `απόδοση`.
5. discount factor `συντελεστής προεξόφλησης` versus the common calque `συντελεστής έκπτωσης`.
6. stride `διασκελισμός` versus `βήμα`.
7. fine-tuning `λεπτή ρύθμιση`.
8. generator `γεννήτορας` versus `γεννήτρια`.
9. multi-armed bandit `ληστής πολλών βραχιόνων`; contextual bandit `ληστής με συμφραζόμενα`.
10. mode collapse `κατάρρευση κορυφών`, mode dropping `παράλειψη κορυφών`.
11. random seed `φύτρο τυχαίων αριθμών`.
12. preconditioner `προσταθεροποιητής` versus `προρρυθμιστής`.
13. subgradient `υποκλίση` (kept; rare in print).
14. outer product `εξωτερικό γινόμενο` (also the cross product in Greek school usage).
15. downstream task `εργασία κατάντη`.

The 830 records that still carry the drafter's stock note were read and left as they are: single standard terms with no choice to record.
