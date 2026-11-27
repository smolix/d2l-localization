# Vietnamese: corrections from the style guide (2026-11-26)

Source: `vi/style.md` §13 ("Proposed glossary corrections"). Applied to
`glossary/terms/vi.jsonl` as proposals; each changed record's `notes` ends with
"Style guide 2026-11-26: …". No record changed status, and no attested
(`verified_*`) preferred term changed.

| Concept | Before | After | Reason |
|---|---|---|---|
| kernel-size | kích thước nhân tích chập | kích thước hạt nhân | *kernel* is the verified *hạt nhân*; §10.1 rejects *nhân tích chập*; all three test translators overrode the entry. |
| scalar (verified_d2l) | số vô hướng; alternatives none | unchanged; alternative *vô hướng* | Conflict: attested entry. The adjective ("scalar bias" → hệ số điều chỉnh vô hướng, §10.1) would otherwise warn every time. |
| off-policy | ngoài chính sách (off-policy) | ngoài chính sách | A parenthesis in a term is inserted at every occurrence; the first-use gloss is §4.3's job. |
| on-policy | theo chính sách (on-policy) | theo chính sách | Same. |
| soft-actor-critic | actor-critic mềm (SAC) | actor-critic mềm | Same; SAC is written where the English writes it (§4.2). |
| softmax | hàm softmax | softmax; alternative hàm softmax | Matches hồi quy softmax, chuẩn hóa softmax; the head noun is added where needed (§4.2). |
| sigmoid | hàm sigmoid | sigmoid; alternative hàm sigmoid | Same. |
| return (RL) | tổng phần thưởng tích lũy | lợi tức; alternative tổng phần thưởng tích lũy | Length; **to confirm** with a reader of Vietnamese RL literature (§11.9). |
| null-space | không gian nghiệm không | không gian hạt nhân | Review calls the old form odd; **to confirm** with a specialist (review also lists *không gian hạch*). |
| hessian | ma trận Hessian | ma trận Hesse; alternative ma trận Hessian | One pattern with ma trận Jacobi: noun + the person's name (§8); **to confirm**. |
| jacobian | ma trận Jacobi | unchanged | Already follows the pattern. |
| image-patch | mảnh ảnh | unchanged; alternative mảnh | §10.1 uses *mảnh* for a patch (ma trận mảnh). |
| dense layer | no entry | none (new alias) | Same concept as `fully-connected-layer`, whose preferred is already tầng kết nối đầy đủ; needs "dense layer" as an English alias in `concepts.jsonl` (§11.5, to confirm). |
| standard normal distribution | no entry | new concept | phân phối chuẩn tắc. |
| image edge, edge detection | no entry | new concepts | biên, phát hiện biên (§10.1, §11.4). |
| patch, patch matrix, flatten, banded matrix, effective receptive field | no entries | new concepts (patch: alias of image-patch) | mảnh, ma trận mảnh, trải phẳng, ma trận dải, vùng tiếp nhận hiệu dụng. |
| implement | no entry | new concept | lập trình (§10.1, §11.15). |

Not changed, for a later pass: `hessian-vector-product` (tích Hessian–vector)
and `hessian-eigenvalue` (trị riêng của Hessian) still use *Hessian*;
`monte-carlo-return` (tổng phần thưởng Monte Carlo) still uses the old
*return*. They should follow if the author confirms Hesse and lợi tức.
