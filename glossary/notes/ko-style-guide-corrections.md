# Korean: glossary corrections from the style guide (2026-11-26)

Applied from `ko/style.md` §12 to `glossary/terms/ko.jsonl`. Every entry was
`proposed`; all stay `proposed`. Each record's `notes` gives the reason and the
former term. No alternatives were added: the glossary check accepts every
alternative, so keeping the former term there would let both forms through.

| Concept | Before | After | Reason |
|---|---|---|---|
| model-parameter | 모델 매개변수 | 모델 파라미터 | §5.1, one family with 하이퍼파라미터 (to confirm, §11.7) |
| parameter-count | 매개변수 수 | 파라미터 수 | §5.1 |
| parameter-update | 매개변수 갱신 | 파라미터 갱신 | §5.1 |
| active-parameters | 활성 매개변수 수 | 활성 파라미터 수 | §5.1 |
| natural-parameter | 자연 매개변수 | 자연 파라미터 | §5.1 |
| forward-pass | 순방향 패스 | 순전파 | §9.1, established pair (to confirm, §11.9) |
| backward-pass | 역방향 패스 | 역전파 | §9.1; now shares its term with backpropagation |
| regularization-strength | 규제 강도 | 정칙화 강도 | §9.1, regularization is 정칙화 |
| regularization-path | 정규화 경로 | 정칙화 경로 | §9.1, 정규화 is normalization |
| regularized-objective | 정규화된 목적함수 | 정칙화된 목적 함수 | §9.1 and §2.1 |
| normal-distribution | 정규분포 | 정규 분포 | §2.1 spacing |
| exponential-distribution | 지수분포 | 지수 분포 | §2.1 spacing |
| probability-density | 확률밀도 | 확률 밀도 | §2.1 spacing |
| latent-variable | 잠재변수 | 잠재 변수 | §2.1 spacing |
| convex-function | 볼록함수 | 볼록 함수 | §2.1 spacing |
| conjugate-prior | 켤레 사전분포 | 켤레 사전 분포 | §2.1 spacing |
| forward-posterior | 순방향 사후분포 | 순방향 사후 분포 | §2.1 spacing |
| variational-objective | 변분 목적함수 | 변분 목적 함수 | §2.1 spacing |
| backtracking-line-search | 역추적 선탐색 | 역추적 선 탐색 | §2.1 spacing |
| pooling-window | 풀링 창 | 풀링 윈도 | §9.1, sliding window is 윈도 |
| newton-step | 뉴턴 단계 | 뉴턴 스텝 | §9.1, optimization step is 스텝 |
| residual-network | 잔차 네트워크 | 잔차 신경망 | §9.1, neural network is 신경망 |
| fully-convolutional-network | 완전 합성곱 네트워크 | 완전 합성곱 신경망 | §9.1 |
| soft-thresholding | 소프트 임계값 처리 | 소프트 임곗값 처리 | §1, 사이시옷 |
| gaussian-distribution (check only) | unchanged (`verified_d2l`, 가우시안 분포) | none | Resolved by the normal-distribution split and the respacing above: "standard normal distribution" now matches 정규 분포, which 표준 정규 분포 contains |

## Follow-up: same rules, entries not listed in §12

Applied at the coordinator's request. I swept every entry's preferred term and
alternatives for a solid 목적함수, 사전분포, 사후분포, 정규분포, 분포함수 or
밀도함수, and for 규제 or 정규화 used for regularization. The sweep found
exactly these 11, all `proposed`. Every other 정규화 entry means normalization
(batch, layer, spectral, flows, constants) and stays as it is.

| Concept | Before | After | Reason |
|---|---|---|---|
| implicit-regularization | 암묵적 정규화 | 암묵적 정칙화 | §9.1, regularization is 정칙화 |
| denoising-objective | 잡음 제거 목적함수 | 잡음 제거 목적 함수 | §2.1 spacing |
| adversarial-objective | 적대적 목적함수 | 적대적 목적 함수 | §2.1 spacing |
| convex-objective | 볼록 목적함수 | 볼록 목적 함수 | §2.1 spacing |
| noisy-objective | 잡음이 있는 목적함수 | 잡음이 있는 목적 함수 | §2.1 spacing |
| objective-value | 목적함수 값 | 목적 함수 값 | §2.1 spacing |
| gaussian-prior | 가우시안 사전분포 | 가우시안 사전 분포 | §2.1 spacing |
| gaussian-posterior | 가우시안 사후분포 | 가우시안 사후 분포 | §2.1 spacing |
| hyperprior | 초사전분포 | 초사전 분포 | §2.1 spacing; the prefix 초- stays attached |
| gaussian-process-prior | 가우시안 프로세스 사전분포 | 가우시안 프로세스 사전 분포 | §2.1 spacing |
| gaussian-process-posterior | 가우시안 프로세스 사후분포 | 가우시안 프로세스 사후 분포 | §2.1 spacing |
