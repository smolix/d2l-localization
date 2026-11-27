<!--en:b8107830-->
# 強化学習

<!--en:52cecb39-->
これまで本書で扱ってきたモデルはすべて、静止したデータセット上で訓練されていた。モデルが何を予測しても、次にどの例が現れるかは一切変わらなかったのである。ここで、数学の問題に取り組む言語モデルを考えてみよう。モデルは 1,000 トークンにわたる推論を書き連ねるが、答えが正しいかどうかを判定器が告げるのは最後だけである。どのトークンにも個別のラベルはなく、1 トークン出力するごとにモデルが次に書ける内容が変化する。動く台車の上で棒のバランスを取る制御器も同じ立場にある。台車を押すたびに次に補正すべき角度が変わり、得られる報酬（reward）は棒が倒れずに立っている時間だけである。:numref:`fig_rl_agent_env` は、意思決定主体である*エージェント*と、それが働きかける世界である*環境*（environment）の間のこのループを描いている。

<!--en:4af4acea-->
![エージェントと環境の間のループ。ステップ $t$ において、エージェントは方策 $\pi$ に従って行動 $a_t$（青）を選択する。遷移確率 $P$ と報酬 $r$ に規定される環境は、報酬 $r_t$ と次の状態 $s_{t+1}$（オレンジ）を返す。このやり取りを繰り返すことで、軌跡 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ が生成される。](../img/mdl-rl-agent-env.svg)

<!--en:f2184032-->
ここから 2 つの問いが生じる。遅れて届く報酬に対してどの意思決定が貢献したのか、そして何がよい意思決定かを誰にも示されないまま、学習者はどのようにしてそれを見つけ出せるのか。1 つ目の問いに対し、ルールが既知の場合の答えを 1950 年代に示したのが Richard Bellman である。各状態（state）にこれから得られる報酬によってスコアを付けることで、長いホライズンにわたる問題を 1 ステップの先読みの連鎖へと変換したのである :cite:`BellmanDPBook`。1980 年代には Andrew Barto、Richard Sutton、Chris Watkins が試行錯誤によってそれらのスコアを学習し :cite:`Barto.Sutton.Anderson.1983,Sutton.1988,Watkins.1989`、続いて Ronald Williams が意思決定の規則である*方策*（policy）を試行から得られた報酬によって直接改善する方法を示した :cite:`Williams.1992`。学習者は自らデータを収集するため、犯した失敗によって次に学ぶ対象そのものが変化する。モデルが自らの誤りから立て直す練習を一切しない場合、模倣だけでは不十分になる理由もここにある。

<!--en:18e46521-->
本章は 3 つの段階で進む。第 1 に、マルコフ決定過程（Markov decision process）を定式化し（:numref:`sec_mdp`）、ルールが既知の場合の厳密な最適方策（optimal policy）を計算し（:numref:`sec_valueiter`）、サンプリングされた遷移から行動価値（action value）を学習し（:numref:`sec_qlearning`）、探索（exploration）のリグレットを定量化する（:numref:`sec_exploration`）。第 2 に、長いホライズンにおいてエキスパートのデモンストレーションが見落とす問題を調べ（:numref:`sec_imitation`）、既知の物理法則のもとで連続系の動的計画法（dynamic programming）がどのようになるかを、線形二次レギュレータ（linear-quadratic regulator、LQR）、反復 LQR（iterative LQR、iLQR）、およびリシーディングホライズン型のモデル予測制御（model predictive control、MPC）を通じて学ぶ（:numref:`sec_optimal_control`）。最後に、スコア関数（score function）の勾配によって方策を直接最適化し（:numref:`sec_policygradient`）、ベースラインとグループ相対重みによって分散（variance）を抑え（:numref:`sec_baselines`）、ルックアップテーブルをニューラルネットワークに置き換えたうえで、本章と次章の諸手法を体系的に整理する（:numref:`sec_deeprl`）。

<!--en:a2aa64c4-->
すべての例は PyTorch と JAX で動作し、その大半はあらゆる推定値を厳密解と照らし合わせて検証できるほど小規模である。続く :numref:`chap_deep_rl` ではブートストラップ型クリティック、経験再生（experience replay）、学習されたダイナミクスモデル、固定されたオフラインデータセットを扱い、「言語モデル」パートでは選好に基づく事後学習（post-training）をより深く掘り下げる。

<!--en:5dd576ec-->
## 参考資料と関連文献 {.unnumbered}

<!--en:0124b728-->
以下に挙げる無料公開の資料では、理論的な解説、講義コース、実装、および実証的な指針が得られる。

<!--en:311cd888-->
**教科書**

<!--en:cabb7b05-->
- Sutton and Barto『Reinforcement Learning: An Introduction』:cite:`Sutton.Barto.2018` は、価値手法、方策手法、プランニング、関数近似を包括的に解説した入門書である。Sutton と Barto は、強化学習の基礎を築いた貢献により 2024 年のチューリング賞を受賞した。

<!--en:1725f7f1-->
- Szepesvári『Algorithms for Reinforcement Learning』:cite:`Szepesvari.2010` は、本章のアルゴリズムとその理論保証を 100 ページ未満でまとめている。

<!--en:2a027965-->
- Bertsekas『A Course in Reinforcement Learning』:cite:`Bertsekas.2025` は、最適制御と動的計画法の観点から強化学習を展開している。

<!--en:fe304d58-->
- Agarwal、Jiang、Kakade、Sun『Reinforcement Learning: Theory and Algorithms』:cite:`Agarwal.Jiang.Kakade.ea.2019` は、本章で定性的に述べるサンプル計算量のレートを与えている。

<!--en:e1e65e47-->
- Lattimore、Szepesvári『Bandit Algorithms』:cite:`Lattimore.Szepesvari.2020` は、:numref:`sec_exploration` の背後にある理論の全体像を扱っている。

<!--en:e53a40b7-->
- :numref:`sec_optimal_control` については、:citet:`Kalman.1960` および :citet:`Anderson.Moore.2007` が線形二次レギュレータ（LQR）を展開し、微分動的計画法 :cite:`Mayne.1966` と反復 LQR :cite:`Li.Todorov.2004` が後ろ向きのリカッチパスを非線形軌跡最適化へと拡張している。また、:citet:`Rawlings.Mayne.Diehl.2017` は制約付きのリシーディングホライズン型モデル予測制御を網羅し、:citet:`Tedrake.2023` は劣駆動ロボティクスと軌跡最適化を扱っている。

<!--en:3e1e13d6-->
**講義コース**

<!--en:13842fe7-->
- [Berkeley CS285: Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) は、方策勾配、価値手法、モデルベース学習、オフライン強化学習を大学院レベルで扱っている。

<!--en:6bc85c3c-->
- [Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/) は、基礎、探索、および理論保証を伴う方策勾配を扱い、対象範囲が本章に最も近い講義コースである。

<!--en:f4d069ff-->
- [Stanford CS224R: Deep Reinforcement Learning](https://cs224r.stanford.edu/) は、模倣学習やオフライン手法を含め、実践面を扱っている。

<!--en:982afea8-->
- [CMU 10-703: Deep Reinforcement Learning and Control](https://cmudeeprl.github.io/703website/) は、アルゴリズムを制御および物理的な身体性と結びつけて扱っている。

<!--en:4947b987-->
- [MIT 6.7920: Reinforcement Learning: Foundations and Methods](https://web.mit.edu/6.7920/www/) は、:numref:`sec_valueiter` で導入する動的計画法の理論を展開している。

<!--en:ce8a4e45-->
**注釈付きの実装**

<!--en:217fb7cd-->
- [CleanRL](https://github.com/vwxyzjn/cleanrl) :cite:`Huang.Dossa.Ye.ea.2022` は、:numref:`sec_deeprl` 以降で扱う主要な深層強化学習アルゴリズムについて、ベンチマーク検証済みの単一ファイル実装を提供している。

<!--en:ed5acff1-->
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) :cite:`Raffin.Hill.Gleave.ea.2021` は同じアルゴリズム群のライブラリ実装を提供しており、付属の Zoo には調整済みのハイパーパラメータが用意されている。

<!--en:76ce63c4-->
- [Gymnasium](https://gymnasium.farama.org/) :cite:`Towers.Kwiatkowski.Terry.ea.2024` は、これらの章全体で使用する環境インターフェースのドキュメントを提供している。

<!--en:c6b3100d-->
**実証的な実践**

<!--en:beeca0d6-->
- [Spinning Up in Deep RL](https://spinningup.openai.com/) :cite:`Achiam.2018` は、スコア関数から SAC までの簡潔なカリキュラムであり、動作するコードと並べて読むことを想定して書かれている。

<!--en:e4f655c4-->
- [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) :cite:`Huang.Dossa.Raffin.ea.2022` は、アルゴリズムの説明では省略されがちな実装上の選択が PPO の性能と再現性にどう影響するかを記録している。

<!--en:b7b234df-->
強化学習の基礎<br>**マルコフ決定過程・動的計画法・時間差分・探索・模倣・最適制御・方策勾配・分散低減**

<!--en:649a629f-->
::: {.slide title="行動によって変化する学習者の次のデータ"}

<!--en:8619d0ac-->
[逐次意思決定のループ]{.kicker}

<!--en:d613a313-->
教師あり学習ではデータセットは静止しているが、逐次的に行動するエージェントの意思決定は将来の状態を変化させ、次に届くデータを決定する。

<!--en:690f2004-->
- **方策 $\pi(a_t \mid s_t)$：**現在の状態 $s_t$ を行動 $a_t$（モータトルク、グリッド上の移動、生成トークンなど）へ写像する

<!--en:d78787d6-->
- **環境のダイナミクス $P(s_{t+1} \mid s_t, a_t)$：**状態を確率的に時間発展させ、スカラーの報酬 $r_t$ を出力する

<!--en:8a74da1b-->
- **閉ループのフィードバック：**相互作用を繰り返すことで軌跡 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ が生成され、その分布は $\pi$ に依存する

<!--en:b64e2aef-->
::: {.slide title="遅延した信頼割り当てと試行錯誤による探索"}

<!--en:cf471910-->
[2 つの中核課題]{.kicker}

<!--en:d511f929-->
- **時間的信頼割り当て：**長いロールアウトの最後にしか報酬が得られないとき、過去のどの意思決定が功績や失策に値するのか。

<!--en:26aa3d27-->
- **ベルマンの原理（1957）：**すべての状態に将来の割引期待収益でスコアを付け、ホライズン $1/(1-\gamma)$ の問題を 1 ステップの先読みへと変換する

<!--en:1019ab95-->
- **教師なしの探索：**価値の高い行動の活用と、不確実な行動の探索のバランスを取る

<!--en:84417a2a-->
::: {.slide title="ベンチマーク 1：滑る氷上で逆転する最短経路の論理"}

<!--en:4a73abd6-->
[表形式 MDP のベンチマーク]{.kicker}

<!--en:a18fa7ed-->
最初の共通ベンチマークは $4 \times 4$ の `FrozenLake-v1` である。機械精度まで厳密に解けるほど小さい一方、素朴な貪欲ヒューリスティクスが通用しないだけの確率性を備えている。

<!--en:3396d493-->
- **滑る遷移カーネル：**各コマンドは確率 $1/3$ で意図した方向へ進み、左右の直角方向へそれぞれ確率 $1/3$ で滑る

<!--en:e58ddad8-->
- **最短経路対滑りを考慮した最適解：**滑らない氷上での 6 手の最短経路をたどると、滑って隣接する穴に落ちるため、成功率は全エピソードのわずか **$4.5\%$** にとどまる

<!--en:9e23e6ba-->
- **リスク最小化制御：**最適方策 $\pi^*$ は全エピソードの **$73.0\%$** でゴールに到達し、穴に落ちるリスクを最小化するため 4 つのセルでゴールから*遠ざかる*方向を指示する

<!--en:6e4f071c-->
::: {.slide title="ベンチマーク 2：検証器 MDP としての言語生成"}

<!--en:015b2589-->
[言語モデル事後学習のベンチマーク]{.kicker}

<!--en:80215e22-->
- **1 ステップへの縮約表現：**決定論的なトークン追加と終端報酬 $r(x, y) \in \{0, 1\}$ のもとでは、応答全体 $y$ がプロンプト $x$ に対する 1 つの行動として働く

<!--en:ee8e9eb3-->
- **参照方策と上限の比較：**一様なソフトマックス参照方策の正解率は $6.2\%$（`2+2` と `3*4` で $\approx 1/8$、解けない `8*8` と `9*9` で $0$）であり、上限は $50\%$ となる

<!--en:c0cf26e0-->
- **報酬ハッキングの危険：**9 番目の候補は保険をかけた回答（*「4, 12, 64 or 81」*）であり、厳密な判定器はスコアを $0$ とするが、部分文字列の一致を見る甘い判定器はすべてのプロンプトに対して $1.0$ を与えてしまう

<!--en:b9813a17-->
::: {.slide title="第 §{num:chap_reinforcement_learning} 章のロードマップ：厳密な表からニューラルクリティックまで"}

<!--en:deaa4fa5-->
[アーキテクチャとアルゴリズムの展開]{.kicker}

<!--en:c7209fe6-->
- **既知から未知の物理法則へ（§{sec_mdp}〜§{num:sec_exploration}）：**MDP を定式化し、ベルマン縮小スイープを解き、単一のサンプルから $Q^*$ を学習し、UCB とトンプソンサンプリングで探索のリグレットを抑える

<!--en:20140ac7-->
- **デモンストレーションと連続系の物理（§{sec_imitation}〜§{num:sec_optimal_control}）：**行動クローニングにおける $O(\varepsilon T^2)$ のシフトを診断し（DAgger）、リカッチ LQR、iLQR、リシーディングホライズン MPC で連続時間の線形・非線形制御を解く

<!--en:92eabaa7-->
- **方策勾配とニューラルクリティック（§{sec_policygradient}〜§{num:sec_deeprl}）：**REINFORCE、ベースライン、RLOO、GRPO、および `tanh` MLP クリティックを用いて確率的方策を微分する

<!--en:9d54645d-->
::: {.slide title="概念チェック：逐次意思決定問題の定式化"}

<!--en:929b7671-->
[理解度チェック]{.kicker}

<!--en:ceb8c25b-->
§{sec_mdp} に入る前に、以下の 3 つの基本的な問いに取り組んでみよう。

<!--en:49386d38-->
1. **信頼割り当てと割引：**どの割引率 $\gamma$ を選べば実効的な計画ホライズン $1/(1-\gamma)$ が $50$ ステップになり、そのとき $50$ ステップ先の報酬にはどれだけの重みがかかるか。

<!--en:e757b251-->
2. **確率的制御：**滑る FrozenLake のセル 9（真上に穴 5 がある）において、*下*を指示しても 3 回中 2 回は横へ滑るにもかかわらず、最適方策が*下*を指示するのはなぜか。

<!--en:096f282a-->
3. **報酬の仕様：**正解が部分文字列として含まれるかどうかだけを調べる甘い検証器が、4 プロンプトの算術タスクで破綻するのはなぜか。

<!--en:557b98b6-->
**解答：**

<!--en:045bda33-->
1. $1/(1-\gamma) = 50$ とおくと $\gamma = 0.98$ が得られ、1 ホライズン先の報酬にかかる重みは $0.98^{50} \approx e^{-1} \approx 0.368$ となる。

<!--en:8a8ac5e9-->
2. セル 9 から*左*、*右*、または*上*を指示すると確率 $1/3$ で穴 5 に滑り落ちるが、*下*は 1 ステップで穴に落ちるリスクが $0$ であり、安全なセル 8、13、10 のいずれかに着地する。

<!--en:e0662bdd-->
3. オプティマイザは仕様の抜け穴を突き、すべてのプロンプトに対して保険をかけた回答*「one of 4, 12, 64 or 81」*を出力することで、算術を解かずに $1.0$ を獲得してしまう。
