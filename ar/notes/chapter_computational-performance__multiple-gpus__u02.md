# Notes on chapter_computational-performance/multiple-gpus.md#u02 (claude fable-5.2 (medium), 2026-11-28 01:00 UTC)

- 4955fec1, e844cf9e: "a 1-float allreduce" is عدد واحد (style §6), so the digit 1 is not in the block.
- 694faba1, 3dda73ed: "1B-parameter", "the 1B GPT" are written مليار معلَمة (style §15.7), without the digit 1.
- c882b4d1: "1 GPU" / "2 GPUs" are وحدة GPU واحدة / وحدتي GPU (style §6); the enumerators (a), (b) take في الحالة (§15.2).
- 3dda73ed: "Suppose" is rendered as a statement of the setup, since slides take no imperative.
- c8157c1f: the glossary's قانون التحجيم is the scaling law of models; here "Scaling Laws" are Amdahl's and Gustafson's laws of strong and weak scaling, so قوانين التناسب (lexicon: scaling with number of GPUs → تناسب).
- 4a712844: "$N = 11.2$M" is written $N = 11.2$ مليون (style §15.7); the math span that the English breaks across two lines is copied with its line break.
- c4cdbb08: the arrow is inside the protected math span "$\to 1$" and cannot be turned for right-to-left reading.
- 6778fe69, 37794ebb, af35d421: "link" is وصلة, as on this page's first unit and on the hardware page; the chapter index logged الرابط بين وحدات GPU.
- 541113b8, c4cdbb08: "start-up latency / overhead" stays زمن استجابة البدء / عبء البدء as in this page's first unit; the hardware page logged زمن الاستجابة الابتدائي.
- 3dda73ed: warning "glossary term not used: tokens" is a false match; the term is there in the accusative (رمزًا مميزًا) after $32{,}768$.
