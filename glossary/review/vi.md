# Vietnamese (vi) glossary review

Reviewed 2026-11-26 against `glossary/terms/vi.jsonl` (1,184 entries: 212
`verified_d2l`, 64 `verified_external`, 3 `invariant`, 905 `proposed`).
Corrections are in `vi-corrections.jsonl` (195 lines). All 279 non-proposed
entries and all 155 proposed entries with frequency ≥ 50 were checked; the
other 750 proposed entries were all read once (not sampled), with less
source-checking per entry.

## Verdict

The Vietnamese glossary is in much better shape than the Japanese one, because
most of its `verified_d2l` entries come from a human-edited term list and not
from machine-translated prose. It is still not usable as binding constraints
without the corrections: `verified_d2l` mixes two sources of very different
quality, `verified_external` is mostly copied headings, and the proposals often
contradict the human glossary they sit next to.

| Status | Entries | Flagged | Substantive (not casing/typo) | Of which wrong sense or MT |
|---|---|---|---|---|
| `verified_d2l` from `glossary.md` | 181 | 26 | 21 | 1 |
| `verified_d2l` from chapter prose | 31 | 11 | 11 | 6 |
| `verified_external` | 64 | 48 | 17 | 9 |
| `proposed`, frequency ≥ 50 | 155 | 47 | 44 | 3 |
| `proposed`, frequency < 50 | 750 | 63 | 58 | 4 |

By severity: 8 wrong sense, 15 calque/MT, 56 non-standard, 72 inconsistent,
44 minor. Six lines keep the preferred term and only delete a bad alternative.

## Provenance: how the 212 `verified_d2l` terms were chosen

There are two different Vietnamese D2L texts in `d2l-ai/d2l-vi`, and the
glossary draws on both under one status.

1. **The community edition (2019–2021), human.** Translated by the
   aivivn / Machine Learning Cơ Bản group ("MLBVN Team" in `config.ini`;
   maintainers Vũ Hữu Tiệp and Đoàn Võ Duy Thanh; about 50 GitHub
   contributors), served at `d2l.aivivn.com`. Its `glossary.md` has 459 rows.
   Terms were added by pull request and argued in GitHub threads: 225 rows link
   to their discussion (the `git.io` links resolve to `aivivn/d2l-vn` pull
   requests), and `CONTRIBUTING.md` requires every translation PR to follow the
   list and pass review. 181 of the 212 entries (plus the 3 `invariant`) come
   from this file. They are deliberate, reviewed choices.
2. **The text now on `main` and `vi.d2l.ai` (April 2022), machine-translated.**
   `config.ini` says `translator = aws`; the commit "Initialise vi.d2l.ai"
   (2022-04-26) replaced the human chapters with a translation of v0.17.5. The
   prose shows it: "Điều đó đang được nói" (that being said), "đại lý học tập
   tăng cường" (RL agent as a sales agent), "học tập củng cố" next to "học tăng
   cường", chapter title "Sơ bộ" for Preliminaries. The 31 `verified_d2l`
   entries with a chapter locator come from this text. Six are wrong: "câu trả
   lời câu hỏi" (question answering), "bộ mã hóa tự động" (autoencoder),
   "thông tin lẫn nhau" (mutual information), "Tăng cường hình ảnh" with the
   alternative "Hình ảnh Augmentation", "tự giám sát" (the noun is missing),
   and the truncated "dịch chuyển phân phố". Five more contradict the human
   glossary ("chuẩn hóa lớp", "kết nối dư", "nhúng", "nhúng từ", "Euclide").

The last human commit is `5a2e43ca27` (2021-04-07); I checked nine of its
chapters to settle disputed terms (perceptron đa tầng, chuẩn hóa theo tầng,
thông tin tương hỗ, bộ tự mã hóa, embedding từ, kết nối phần dư, hàm hợp lý).
`DISCOVERY.md` should record this split, and future extraction should read the
human chapters at that commit (files `*_vn.md`), never `main`.

**Do the human terms match current usage?** Mostly yes: học tăng cường, lan
truyền ngược, tích chập, quá khớp, tinh chỉnh, siêu tham số, tiên nghiệm / hậu
nghiệm, hợp lý cực đại are what Vietnamese ML writing uses today. The group
also made a set of house choices that differ from other common usage. They are
consistent and defensible, and I left them alone unless noted: tầng (layer,
keeping lớp for class), hạ gradient, điều chuẩn (others: chính quy hóa), hệ số
điều chỉnh (bias parameter), mạng nơ-ron hồi tiếp, gọt gradient, sải bước,
khung chứa, hệ thống đề xuất (others: gợi ý), and English kept for batch,
minibatch, epoch, dropout, token, embedding, logit, precision, recall,
perplexity. Five choices have dated or lost: cơ chế tập trung (attention; the
glossary here already uses chú ý, which is right), học truyền tải (now học
chuyển giao), biểu diễn vị trí (now mã hóa vị trí), định thời tốc độ học, and
the word order of trung bình bình phương sai số.

`verified_external` has two sources. The Stanford CS229/CS230 cheatsheet
translation is human but supplies headings, so about 30 entries are capitalised
("Hồi quy", "Tích chập", "Cổng quên"), and it has real errors: "truy vết" for
the trace of a matrix (it is "vết"), "Họ số mũ", "Điểm Bleu", "Độ rộng Beam".
The Google ML glossary pages read as machine translation and use a vocabulary
that conflicts with D2L-vi: 11 of its 20 entries are flagged ("tổn thất xác
thực", "hạng mục dương", "thời gian chờ" for latency, the bare acronym "MOE",
the alternatives "nhân viên hỗ trợ" for agent and "tổn thất khớp nối" for hinge
loss). Treat Google's Vietnamese pages as unverified.

## Systematic problems

1. **Glosses instead of terms (proposed, about 45 entries, mostly high
   frequency).** gradient → "gradient của hàm mất mát", token → "đơn vị từ
   vựng", entropy → "entropy thông tin", Adam → "thuật toán tối ưu Adam",
   Transformer → "kiến trúc Transformer", DDIM, Mamba, RMSNorm, ALiBi, GELU,
   InfoNCE, n-gram, GPU all expanded into descriptions. As constraints these
   would be inserted at every occurrence, and some are wrong (a gradient is not
   always of a loss; DDIM's "implicit" became "ẩn"). Names and acronyms stay
   in Latin script.
2. **Proposals ignore the human glossary.** minibatch → "lô nhỏ", dropout →
   "phương pháp bỏ nút ngẫu nhiên", logit → "điểm logit", pretraining → "huấn
   luyện trước" (glossary: tiền huấn luyện), non-maximum suppression,
   up/downsampling, sequence-to-sequence, categorical. About 275 glossary rows were
   never matched to a concept.
3. **Glossary markup copied into terms.** "epoch (khi huấn luyện)", "hàm lồi
   (chặt)" (which says strictly convex), "(câu) truy vấn", "mô hình (hoá) ngôn
   ngữ", "tràn (số) dưới", and typos carried over ("biễu diễn one-hot").
4. **One English head word, several Vietnamese ones.** layer: tầng / lớp
   (perceptron đa lớp, chuẩn hóa lớp, lớp nhúng); loss: mất mát / tổn thất;
   likelihood: khả năng / hợp lý / độ hợp lý; error: lỗi / sai số;
   generalization: tổng quát hoá / khái quát; contrastive: đối sánh / tương
   phản; flow: dòng / dòng chảy / luồng; marginal: biên / cận biên; score
   function: hàm điểm / hàm điểm số; schedule: định thời / lịch / lịch trình;
   class: lớp / hạng mục; proximal, penalty, residual, Jacobi/Jacobian,
   Euclid/Euclide.
5. **Sense collisions.** Handled correctly: bias (hệ số điều chỉnh / độ chệch
   / thiên kiến), kernel (hạt nhân for both senses, "kernel" for a GPU kernel),
   RL return (tổng phần thưởng), value, state, policy, reward, label, mean,
   expectation, regularization. Not handled: `shape` and size are both "kích
   thước" (kích thước batch, kích thước bước); "hiệu chỉnh" serves calibration
   and correction; actor and critic are translated alone ("bộ tác động", "bộ
   đánh giá") but left in English in "actor-critic mềm".
6. **Orthography.** vectơ / vector, tenxơ / tensor, nơron / nơ-ron, hoá / hóa,
   mômen / mô-men, cosin / Cosine / cô-sin, sentence-case headings. The
   corrections apply the human edition's spelling (vector, tensor, nơ-ron, hóa).

## Decisions for the author or a native reviewer

1. **vector / tensor or vectơ / tenxơ.** The human edition and the ML community
   write vector and tensor; school textbooks and vi.wikipedia write vectơ and
   tenxơ. The corrections assume the first (11 lines); reversing it is a single
   change to "vector riêng".
2. **embedding as a noun**: "embedding" (human glossary), "biểu diễn nhúng" or
   "vector nhúng". Bare "nhúng" is a verb. About ten compounds follow.
3. **generalization**: khái quát hóa (two verified glossary compounds) or tổng
   quát hóa (more common today). Whichever is chosen, change the other side.
4. **local / global minimum**: the glossary's "cực tiểu" / "giá trị nhỏ nhất"
   or the explicit "cực tiểu cục bộ" / "cực tiểu toàn cục" (recommended).
5. **actor, critic, episode, return, prompt, off-policy / on-policy**: Vietnamese
   RL and LLM writing mostly keeps these in English. The corrections keep
   actor, critic and prompt, and leave "lượt tương tác" (episode) and "tổng phần
   thưởng tích lũy" (return) for a reviewer who knows the RL literature.
6. **shape**: keep "kích thước" (human glossary) or move to "hình dạng" so that
   size and shape differ. Not in the corrections file.
7. **error**: "lỗi" or "sai số" for training / test / generalization error.
8. **Statistics and optimisation terms needing a specialist**: ước lượng vững
   (consistent estimator), họ mũ, gần kề (proximal), tốc độ–méo
   (rate–distortion), null space ("không gian nghiệm không" is odd; "không gian
   hạch" or "hạt nhân" are the textbook candidates), năng lực thống kê (power),
   cảnh quan mất mát (loss landscape, a calque the human group chose),
   "lấy mẫu từ chối", "độ bất định do thiếu hiểu biết".
9. **Keep the house terms listed above** (tầng, điều chuẩn, hạ gradient, hệ số
   điều chỉnh, hồi tiếp)? I recommend yes: they are consistent, were argued in
   public, and readers of the earlier edition know them.

## Sources

- Human D2L-vi glossary and chapters: <https://github.com/d2l-ai/d2l-vi/blob/c775d6b4998e6243ec5d11f950e67679555a2c74/glossary.md>, <https://github.com/d2l-ai/d2l-vi/tree/5a2e43ca27> (`*_vn.md`), `CONTRIBUTING.md` and `STYLE_GUIDE.md` on the `release` branch; discussion links such as <https://git.io/Jvojp> → `aivivn/d2l-vn` PR 136.
- MT status of `main`: `config.ini` (`translator = aws`) and commit `834cc8be07` at <https://github.com/d2l-ai/d2l-vi>; <https://vi.d2l.ai/>.
- vi.wikipedia article titles checked through the API: Vết (đại số tuyến tính), Phương pháp nhân tử Lagrange, Sai số toàn phương trung bình, Học chuyển giao, Bộ tự mã hóa, Học tự giám sát, Hợp lý cực đại, Ma trận Jacobi, Quá trình quyết định Markov, Cơ chế chú ý, Vectơ, Tenxơ, Hệ thống gợi ý, Kỹ thuật tạo lệnh.
- Not verified: I could not load the Vietnamese rendering of Google's ML glossary (the fetch returned English), so the judgement that it is machine translation rests on the entries and alternatives recorded in `vi.jsonl`. `d2l.aivivn.com` itself is gone (see `DISCOVERY.md`), so the human edition was read from git history only.
