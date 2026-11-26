# Korean (ko) glossary review

Reviewed 2026-11-26 against `glossary/terms/ko.jsonl` (1,184 entries: 252 `verified_d2l`, 88 `verified_external`, 844 `proposed`). All 340 verified entries and all 844 proposed entries were read against the sense in `concepts.jsonl`. Corrections: `glossary/review/ko-corrections.jsonl` (210 lines).

## Verdict

The glossary is not usable as a constraint file in its current state. The `verified_d2l` tier is the least reliable of the three, and the three tiers contradict each other on the most frequent head words (gradient, attention, likelihood, convolution, layer, training). The `proposed` tier is the best of the three: its terms are mostly the ones a Korean ML author would write.

| Tier | Entries | Changed | wrong sense | calque / MT | non-standard | inconsistent | minor |
|---|---:|---:|---:|---:|---:|---:|---:|
| verified_d2l | 252 | 92 (37%) | 7 | 27 | 9 | 37 | 12 |
| verified_external | 88 | 19 (22%) | 1 | 1 | 4 | 7 | 6 |
| proposed | 844 | 99 (12%) | 1 | 0 | 24 | 45 | 29 |
| **Total** | 1,184 | 210 (18%) | 9 | 28 | 37 | 89 | 47 |

Of the 131 proposed entries with frequency ≥ 50, 34 change. Wrong in the plain sense (wrong meaning, MT calque, or a term nobody writes): 43 of 252 `verified_d2l` and 6 of 88 `verified_external`. The rest of the changes make one choice where the glossary currently holds two or three.

## Provenance

`d2l-ai/d2l-ko` `config.ini` has `translator = aws` (release 0.17.1): the 167 chapter files are machine translation, committed in bulk in December 2021. The repository also carries `TERMINOLOGY.md`, which is a different thing: a human-edited term list from the earlier community translation (commits by Muhyun Kim, 2019, and Donghwi Park, 2020). It has 160 rows, about 100 of them empty.

The 252 `verified_d2l` entries split accordingly:

- **209 from MT chapter prose**: 80 need changing (38%). All 7 wrong senses and all 27 calques are here. This evidence shows only that an MT system once produced the string.
- **43 from `TERMINOLOGY.md`**: 12 need changing, none of them wrong in meaning. They are legitimate but dated or minority choices (그래디언트, 학습 속도, 오버피팅/언더피팅, 오차 행렬, 뉴럴 네트워크, 컨볼루셔널 층, 크로스-엔트로피, 특성, 공변량 변화).

Of the 88 `verified_external`, 60 come from the Korean CS229/CS230 cheatsheet translations (volunteer work, uneven) and 28 from Google ML Glossary pages that declare machine translation. The worst external entry, 규범 for norm (frequency 1,068), is from the cheatsheet.

Recommendation: demote every prose-sourced `verified_d2l` entry to `proposed` unless separately confirmed; keep `TERMINOLOGY.md` entries as attested human choices, not as binding.

## Systematic problems

1. **Wrong senses from MT** (verified): 관절 확률 (joint probability, "anatomical joint"), 접합 분포 (joint distribution), 번역 불변성 (translation invariance, "language translation"), 음수 샘플링 (negative sampling, "negative number"), 규범 (norm), 욕심 많은 검색 (greedy search), 서프라이즈 (surprisal), 스타일 전송 (style transfer), 잔여 블록/연결 (residual), 지수 붕괴 (exponential decay), 합계 상한 (union bound).
2. **Broken extraction** (verified): the preferred term is a fragment or a sentence. 학습 for preconditioning, 시간 for backpropagation through time, 네트워크 for Network in Network, a full definition sentence for trace, `가양성 (false positive)`, `null 가설`, `레이블 shift`.
3. **Head words rendered two or three ways.** Each row needs one ruling; the corrections file applies the recommended one.

   | English | In the glossary now | Recommended |
   |---|---|---|
   | gradient | 그래디언트 (3), 기울기 (15), 경사 (8) | 기울기; 경사 only in 경사 하강법, 정책 경사 |
   | attention | 어텐션 (16), 주의/주의력 (6) | 어텐션 |
   | likelihood | 가능도 (3), 우도 (9) | 가능도 |
   | convolution | 합성곱, 컨볼루션, 컨벌루션, 컨볼루셔널 | 합성곱 |
   | layer | 층, 계층, 레이어 | 층 |
   | training | 훈련 (10), 학습 (5) | 훈련; 학습 = learning |
   | noise | 잡음 (21), 노이즈 (2) | 잡음 |
   | cross- | 교차, 크로스- | 교차 |
   | feature | 특성, 특징 | 특징 |
   | forward / reverse | 정방향, 역방향, 역, 역시간 | 순방향 / 역방향 |
   | search | 검색, 탐색 | 탐색; 검색 = retrieval |
   | decay | 감쇠, 감소, 붕괴 | 감쇠 |
   | shift | 이동, 변화, `shift` | 이동 |
   | projection | 투영, 사영 | 사영 (see decisions) |

4. **Sense collisions.**
   - regularization and normalization are both 정규화. Proposed: 정칙화 for regularization (it is the old term list's own choice).
   - exploration and search are both 탐색. Proposed: 탐험 for RL exploration.
   - rank (행렬의 계수) and coefficient (계수). Proposed: 랭크, as in the glossary's own 저랭크.
   - maximum likelihood estimator and estimation share one string. Proposed: 추정량 for the estimator.
   - RL return is 누적 보상 in one entry and 수익 in another (몬테카를로 수익); the common 반환값 would collide with a function's return value.
   - Checked and consistent: bias (편향 / 통계적 편향), score (스코어 for the density score, 점수 for metrics), value (값 in attention, 가치 in RL), kernel (커널 once the transliterated prefix is dropped), state, reward, policy, token, embedding, label, mean.
5. **Glosses in place of terms** (proposed, about 25 entries): 그래픽 처리 장치 for GPU, "AdamW 최적화 알고리즘", "BERT 모델", "Reflow를 통한 궤적 재학습", "제곱평균제곱근 정규화" for RMSNorm, "BLEU 번역 평가 지표", 로짓 점수, 행동 선택, 부트스트랩 재표본추출. A translator substituting these produces nonsense in running text. Names and acronyms stay in Latin script.
6. **Transliterated names** (verified): 아담, 아다그라드, "RMS 프롭", 렐루, "Leaky 렐루", 글로벌 벡터. The proposed tier already keeps AdamW, He, Xavier, GELU in Latin script.
7. **Orthography.** 사이시옷 is applied in 특잇값, 최솟값, 기댓값 (compounds) but not in 고유값, 특이값 분해, 기대값; 엔코더 next to 인코더; 포아송 for 푸아송; spacing varies (경사하강법 / 경사 하강법, 단위행렬 / 대칭 행렬, 거짓음성).

## Decisions for the author or a native reviewer

1. **gradient: 기울기 or 그래디언트.** Both are in wide use. The corrections take 기울기 (Korean Wikipedia, most translated textbooks); 그래디언트 is the old D2L choice. It is the most frequent technical word in the book (3,176 occurrences); 15 proposed compounds already assume 기울기.
2. **likelihood: 가능도 or 우도.** Corrections take 가능도; statisticians of the older school write 우도.
3. **regularization: 정칙화 or 규제.** Either removes the collision with 정규화. 규제 is common in recent translated books; 정칙화 is the old term list's choice.
4. **training: 훈련 or 학습**, and with it 사전 훈련 or 사전 학습 for pretraining (Wikipedia has 사전 학습). The corrections keep 훈련 because it separates training from learning.
5. **feature: 특징 or 특성.**
6. **RL return: 수익, 리턴, 반환값 or 이득**, and exploration: 탐험 or 탐색.
7. **surprisal: 놀람도 or 자기 정보량.** No settled Korean term was found; `information-content` already holds 정보량.
8. **projection: 사영 or 투영** for the mathematical sense, and whether a layer's output projection follows it.
9. **preconditioning / preconditioner: 선조건화 / 선조건자**, or 예조건화, or 사전 조건화.
10. **parameter: 매개변수, hyperparameter: 하이퍼파라미터.** The mix is common in Korean texts and was left alone; confirm it is intended.
11. **Loanword or native, left unchanged pending a ruling:** 가우시안 프로세스 (other processes are 과정), 플로 매칭 against 정규화 흐름 / 확률 흐름, 멀티 헤드 against 다중 쿼리 / 다중 스케일, 능형 회귀 against 릿지 회귀, 헤시안 against 헤세 행렬, 네트워크 in names (심층 Q 네트워크) next to 신경망.
12. **Possible collisions left unchanged:** 연결 serves both concatenation and connection; 외적 (outer product) is also the everyday word for the cross product; 병목 serves bottleneck and compute-/memory-bound; 임계값 would be 임곗값 under the same spelling rule as 고윳값; 모양 for tensor shape (some Korean documentation keeps `shape`).

## Sources checked

Korean Wikipedia articles 노름 공간, 기울기 (벡터), 가능도, 결합분포, 연쇄 법칙, 대각합, 장단기 메모리, 합성곱 신경망, 트랜스포머 (기계 학습), 경사 하강법, 강화 학습, 특잇값 분해, 상호정보, 분배 함수, 치환 적분; the PyTorch Korea user group's translation glossary (`PyTorchKorea/tutorials-kr`, `TRANSLATION_GUIDE.md`, human-maintained); `d2l-ai/d2l-ko` `config.ini`, `TERMINOLOGY.md` and their commit history; the Korean Hugging Face Transformers documentation (decoding terms); wikidocs pages returned by search (반환값, 보정). URLs are in the corrections file.

Limits: an entry is marked `verified_external` only where a fetched page shows the term in this sense (29 entries). The other 181 corrections are `proposed`. The Korean Mathematical Society and Korean Statistical Society term lists were not consulted and would settle items 2, 8 and 9. This review was not done by a native speaker; the choices between two established terms (the decisions above) should be confirmed by one.
