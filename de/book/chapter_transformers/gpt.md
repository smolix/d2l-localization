<!--en:e6e75e3c-->
# Sprachmodelle mit reinem Decoder (GPT)

<!--en:944e72b3-->
Ein einzelner `TransformerBlock` (:numref:`sec_transformer-block`) bildet eine Sequenz $d$-dimensionaler Tokenrepräsentationen auf eine Sequenz derselben Form $(B, T, d)$ ab. Für einen autoregressiven **Generative Pre-trained Transformer (GPT)** :cite:`Radford.Narasimhan.Salimans.ea.2018,Radford.Wu.Child.ea.2019,brown2020language` ordnen wir $L$ kausale Transformer-Blöcke zwischen zwei Schichten an (:numref:`fig_gpt-stack`):

<!--en:262e4135-->
1. **Eingabeeinbettung:** Eine Einbettungstabelle (engl. embedding table) $\mathbf{E} \in \mathbb{R}^{V \times d}$ bildet diskrete Token-IDs $x_1, \dots, x_T \in \{0, \dots, V-1\}$ auf Vektoren in $\mathbb{R}^d$ ab. Verwendet die Attention keine Rotary Position Embeddings, können wir gelernte absolute Positionseinbettungen $\mathbf{P} \in \mathbb{R}^{T_{\max} \times d}$ addieren.

<!--en:d5a9b5f5-->
2. **Abschließende Normalisierung und Ausgabekopf des Sprachmodells:** Nach $L$ kausalen `TransformerBlock`-Schichten stabilisiert eine letzte Normalisierungsschicht den Residualstrom (engl. residual stream) $\mathbf{H}^{(L)}$. Ein linearer Kopf berechnet für jede Position $t$ die Logits des nächsten Tokens durch die Projektion $\mathbf{Z} = \textrm{Norm}(\mathbf{H}^{(L)})\mathbf{E}^\top \in \mathbb{R}^{B \times T \times V}$. Dabei **binden** wir die Gewichte der Ausgabeprojektion an die Einbettungstabelle der Eingabe $\mathbf{E}$ :cite:`Press.Wolf.2017,Inan.Khosravi.Socher.2017`.

<!--en:81ff1f71-->
## Von den Blöcken zum Sprachmodell

<!--en:fd61d568-->
![Ein GPT von unten nach oben. Die Tabelle $\mathbf{E}$ bettet die Token-IDs in den Residualstrom ein. Die $L$ kausalen `TransformerBlock`-Schichten lesen aus diesem Strom und schreiben in ihn. Die abschließende Normalisierung und die transponierte Tabelle $\mathbf{E}^\top$ wandeln den Vektor jeder Position in Logits über dem Vokabular um. In der gezeigten modernen Konfiguration rotiert RoPE die Abfragen und Schlüssel in jeder Attention-Schicht. GPT-2 dagegen addiert gelernte Positionsvektoren einmalig am Eingang.](../img/mdl-transformers-gpt-stack.svg)

<!--en:b237ddab-->
In jedem Block fasst `CausalAttention` die Projektionen für Abfragen, Schlüssel und Werte zusammen und unterstützt **Grouped-Query-Attention** (`num_kv_heads` $\le$ `num_heads`, :numref:`sec_multihead-attention`). Optional rotiert **RoPE** die Abfragen und Schlüssel (`rope=True`, :numref:`sec_positional-information`). Die kausale Maske beschränkt die Attention an Position $t$ auf die Positionen $s \le t$:

<!--en:e7d6e73a-->
Unsere Klasse `GPT` stapelt `num_blks` konfigurierbare `TransformerBlock`-Module. Die Voreinstellungen sind `norm='rms'`, `act='swiglu'`, `pos='rope'` und `pre_norm=True`. Die Klasse unterstützt auch die Einstellungen von GPT-2 (`norm='layer'`, `act='gelu'`, `pos='learned'`, `bias=True`) und bindet die Ausgabeprojektion an die Einbettungstabelle `token_emb`:

<!--en:84d78175-->
Wie bei GPT-2 :cite:`Radford.Wu.Child.ea.2019` und in :numref:`subsec_signal-propagation` zieht `init_weights` die Gewichte der linearen Schichten und der Einbettungen aus $\mathcal{N}(0, 0.02^2)$. Die beiden Ausgabeprojektionen in den Residualstrom jedes Blocks (`attention.W_o` und `ffn.W_2`) skaliert die Funktion mit $1 / \sqrt{2L}$:

<!--en:fb50dd93-->
## Training eines bytebasierten GPT auf WikiText-103

<!--en:9cc86c11-->
Damit unsere Experimente mit neu initialisierten Modellen in sich geschlossen und frei von Tokenizer-Artefakten bleiben, trainieren wir direkt auf den UTF-8-Rohbytes ($V = 256$) von **WikiText-103** :cite:`Merity.Xiong.Bradbury.ea.2016`. Jeder Minibatch enthält $B = 64$ Fenster der Länge $T = 128$ Byte ($8{,}192$ Token pro Schritt):

<!--en:814b7e8f-->
Außerdem definieren wir den Hilfskonstruktor `gpt(...)`. Er erzeugt ein frisch initialisiertes `GPT`, mit dem das Training beginnen kann:

<!--en:866ed5c7-->
`gpt` verschiebt das Modell auf die GPU, kompiliert es mit `torch.compile` und nutzt für fp32-Matrixprodukte die tf32-Tensorkerne.

<!--en:5817e1c3-->
`gpt` zieht für jedes Modell mithilfe von `d2l.get_key()` neue zufällige Gewichte.

<!--en:733fa8be-->
### Die autoregressive Zielfunktion

<!--en:95a106fa-->
Für ein Eingabefenster $\mathbf{x} = (x_1, \dots, x_T)$ und das verschobene Zielfenster $\mathbf{y} = (x_2, \dots, x_{T+1})$ stellt die kausale Maske sicher, dass Zeile $t$ der Ausgabelogits $\mathbf{z}_t \in \mathbb{R}^V$ nur von $(x_1, \dots, x_t)$ abhängt. Beim Training minimieren wir den mittleren Kreuzentropieverlust (engl. cross-entropy loss) parallel über alle $T$ Positionen:

<!--en:c6c365dc-->
Unmittelbar nach der Initialisierung, noch vor dem Training, weist das Modell jedem Byte ungefähr dieselbe Wahrscheinlichkeit $1/V$ zu. Der anfängliche Kreuzentropieverlust liegt daher nahe bei $\ln V = \ln 256 \approx 5.55$ Nat pro Byte. Betrachten wir zunächst ein gegeneinander verschobenes Paar aus Eingabe- und Zielfenster `(X[0], Y[0])` zusammen mit einer kausalen $5 \times 5$-Maske in unterer Dreiecksform:

<!--en:9498ccb7-->
Wir werten unser frisch initialisiertes `GPT` mit 6 Blöcken ($\approx 4.79\textrm{M}$ Parameter) auf einem Validierungsminibatch aus. Wie erwartet liegt die Kreuzentropie vor dem Training nahe bei $\ln 256 \approx 5.55$ Nat pro Byte:

<!--en:ce741cf0-->
### Das Trainingsrezept (`LMOptimizer`)

<!--en:e2373fd9-->
Beim modernen Pre-Training großer Sprachmodelle verwenden wir **AdamW** (:numref:`sec_adam`). Die Standardwerte der Momentum-Parameter sind $\beta_1 = 0.9, \beta_2 = 0.999$, bei Läufen mit großen Batches auch $\beta_2 = 0.95$. Dazu kommt **selektiver Gewichtszerfall** (engl. selective weight decay): Der Gewichtszerfall $\lambda = 0.1$ wirkt nur auf zweidimensionale lineare Projektionsmatrizen. Einbettungen, Bias-Terme und eindimensionale Skalierungsfaktoren der Normalisierung sind ausgenommen. Hinzu kommen eine **lineare Aufwärmphase mit anschließendem Kosinus-Abfall der Lernrate** (:numref:`sec_scheduler`) und **Gradienten-Clipping anhand der globalen Norm** ($\|\nabla \mathcal{L}\|_2 \le 1.0$):

<!--en:eaaa29d2-->
Wir trainieren unser bytebasiertes `GPT` mit 6 Blöcken ($d = 256, H = 8, L = 6$, $\approx 4.8\textrm{M}$ Parameter) über 1000 Schritte ($\approx 8.2\textrm{M}$ Byte). Dabei fällt die Kreuzentropie auf den Validierungsdaten deutlich unter die empirische Unigrammentropie ($3.17$ Nat pro Byte) und sogar unter die empirische Bigrammentropie ($2.45$ Nat pro Byte) englischer Wikipedia-Texte:

<!--en:43e6d2c9-->
Zum Vergleich berechnen wir die empirische Unigramm- und Bigrammentropie der Bytes direkt aus den ersten $10\textrm{M}$ Byte des Trainingskorpus. So prüfen wir, wie weit der Transformer mit 6 Blöcken über lokale Zeichenstatistiken hinauskommt:

<!--en:37066329-->
## Kontrollierte Vergleiche

<!--en:4f14bb99-->
### Streuung zwischen Trainingsläufen und FFN-Aktivierung (`SwiGLU` oder `GELU`)

<!--en:f68c6a84-->
Um Verbesserungen durch die Architektur von zufälligen Schwankungen zu unterscheiden, trainieren wir unser Standardmodell mit `SwiGLU` und ein `GELU`-Modell mit gleicher Parameteranzahl (`act='gelu'`) jeweils mit drei unabhängigen Zufallsstartwerten:

<!--en:34a102dc-->
Die Ergebnisse verschiedener Zufallsstartwerte schwanken nur um $\sim 0.01$ Nat pro Byte. Dagegen schlägt `SwiGLU` bei gleicher Parameteranzahl und gleichem FLOP-Budget `GELU` durchweg um $\sim 0.06$ Nat pro Byte.

<!--en:1f5e6636-->
Mit `train_lm` können wir auch die andere Entwurfsentscheidung aus :numref:`sec_transformer-block` untersuchen: Wo soll die Normalisierung stehen? Bei der Initialisierung zeigte sich in :numref:`subsec_signal-propagation` kein Problem mit der nachgeschalteten Normalisierung (Post-Norm) bei der Gewichtsskala $0.02$, die `init_weights` verwendet. Die Token bleiben unterscheidbar und die Gradienten der Abfragen verschwinden nicht. Ob Post-Norm auch beim *Training* ebenso gut abschneidet, ist eine andere Frage. Wir trainieren dasselbe Modell mit Post-Norm-Blöcken (`pre_norm=False`) dreimal: mit 500 Aufwärmschritten, mit den 100 Schritten des Hauptlaufs und ganz ohne Aufwärmphase. Die maximale Lernrate entspricht in allen drei Fällen der des Hauptlaufs:

<!--en:538ce59d-->
Die Antwort hängt von der Aufwärmphase ab. Mit 100 Aufwärmschritten erreicht das Modell mit vorgeschalteter Normalisierung (Pre-Norm) etwa $1.34$ Nat pro Byte. Das Post-Norm-Modell endet bei $3.17$, ebenso wie im Lauf ohne Aufwärmphase. Diesen Wert kennen wir bereits: Er entspricht der oben berechneten Unigrammentropie. Beide Modelle haben also gelernt, wie häufig jedes Byte vorkommt, aber nichts über die vorausgehenden Bytes. Die ersten Updates sind zu groß. Steigt die Lernrate in wenigen Schritten auf ihr Maximum, werden die Ausgaben der Attention- und FFN-Teilschichten deutlich größer als der normalisierte Residualstrom, zu dem sie addiert werden. Nach den Normalisierungen haben alle Token nahezu denselben Endvektor. Aus demselben Vektor an jeder Position lassen sich nur die Bytehäufigkeiten vorhersagen. Mit der Streuung der Tokenrepräsentationen schrumpfen auch die Gradienten, die sie wieder auseinanderziehen könnten (:numref:`subsec_signal-propagation`). Daher erholt sich das Training nicht.

<!--en:eae6523d-->
Mit 500 Aufwärmschritten bleiben die Ausgaben der Teilschichten kleiner als der Residualstrom. Das Post-Norm-Modell erreicht etwa $1.38$ Nat pro Byte und liegt damit knapp hinter dem Pre-Norm-Modell. Die Aufwärmphase, die für Pre-Norm genügt, reicht für Post-Norm also nicht aus. Wo die Grenze liegt, können wir nicht sagen. Sie ist unscharf: Bei mittellangen Aufwärmphasen lernen manche Modelle, bei anderen kollabieren die Tokenrepräsentationen. Genau dieses Verhalten beschrieben :citet:`xiong2020layer` für den ursprünglichen Transformer: Post-Norm benötigt eine lange Aufwärmphase, während Pre-Norm auch ohne sie auskommt und schneller konvergiert.

<!--en:64302a93-->
### Vergleich mit einem rekurrenten Modell gleicher Größe

<!--en:b3e05a88-->
Wie schneidet unser `GPT` mit 6 Blöcken gegenüber dem rekurrenten Sprachmodell (`d2l.RNNLM`, :numref:`sec_rnn-scratch`) ab, wenn beide **dieselbe Parameteranzahl** ($d_{\textrm{rnn}} = 1920$, $\approx 4.7\textrm{M}$ Parameter) und gleich viele Trainingsschritte haben? Wir variieren die Lernrate des RNN über $\{10^{-3}, 2\times 10^{-3}, 4\times 10^{-3}\}$ und vergleichen den Validierungsverlust als Funktion der Fensterposition $t \in \{0, \dots, 127\}$ (`loss_by_position`):

<!--en:ce65c29b-->
Wir trainieren `RNNLM` für jede der maximalen Lernraten $\{10^{-3}, 2\times 10^{-3}, 4\times 10^{-3}\}$ mit zwei Zufallsstartwerten. Dann vergleichen wir den besten rekurrenten Lauf mit `GPT` an den Fensterpositionen $t \in \{0, 1, 4, 16, 64, 127\}$:

<!--en:ff339a19-->
An Position $t = 0$, an der nur ein Byte Kontext vorliegt, beginnen `GPT` und `RNNLM` mit nahezu demselben Verlust. Mit längerem Präfix ($t \to 127$) sinkt der Validierungsverlust des Transformers steiler als der des RNN. Die Self-Attention greift in $O(1)$ Schritten auf jedes frühere Byte zu. Das RNN muss dagegen die Vorgeschichte über $t$ aufeinanderfolgende rekurrente Schritte verdichten.

<!--en:dd17b6a9-->
## Sampling aus dem Sprachmodell

<!--en:0adf0f45-->
Bei der autoregressiven Generierung (`GPT.generate`) hängen wir in jedem Schritt ein Token an. Dazu verarbeiten wir das aktuelle Präfix mit dem Modell und dividieren die Logits $\mathbf{z}_t$ der letzten Position durch eine **Softmax-Temperatur** $\tau > 0$. Optional setzen wir alle Logits bis auf die **Top-$k$** auf $-\infty$ :cite:`Fan.Lewis.Dauphin.2018`. Das nächste Token ziehen wir aus $\textrm{softmax}(\mathbf{z}_t / \tau)$. Gieriges Dekodieren (engl. greedy decoding), der Grenzfall $\tau \to 0$, entspricht der Einstellung `top_k=1`:

<!--en:1411ac05-->
Wir ziehen Fortsetzungen von je 120 Byte für `'The history of the '` mit drei Einstellungen für `(temperature, top_k)`. So sehen wir, wie eine niedrigere Temperatur und ein Top-$k$-Filter die lokale Kohärenz auf Kosten der Vielfalt erhöhen:

<!--en:55654c67-->
## Laden des vortrainierten GPT-2 (`124M`)

<!--en:f0f02b4c-->
Unsere Klasse `GPT` unterstützt `norm='layer'`, `act='gelu'`, `pos='learned'` und `bias=True`. Damit können wir die Architektur **GPT-2 Small** von OpenAI mit 124 Millionen Parametern ($V = 50{,}257, d = 768, H = 12, L = 12, T_{\max} = 1024$) instanziieren. Die offiziellen Gewichte aus `model.safetensors` und den bytebasierten BPE-Tokenizer (`gpt2_tokenizer`) können wir direkt laden:

<!--en:7b9fd19b-->
:Zuordnung der Schlüssel im GPT-2-Checkpoint von OpenAI im Format `safetensors` zu den Modulattributen unserer Klasse `GPT` (die Gewichte von `Conv1D` haben die Form `(in, out)`).

<!--en:c5fb60ea-->
| Tensor im Checkpoint | Attribut von `GPT` | Anmerkung |

<!--en:1ca501b3-->
| `wte.weight` | `token_emb` | Gemeinsame Gewichte mit dem Ausgabekopf (`V x d`) |

<!--en:eb6ceffb-->
| `wpe.weight` | `pos_emb` | 1024 gelernte Positionsvektoren |

<!--en:313df5ab-->
| `h.i.ln_1`, `h.i.ln_2` | `blks[i].norm1`, `blks[i].norm2` | LayerNorm vor der Attention und vor dem FFN |

<!--en:e3970ec6-->
| `h.i.attn.c_attn` | `blks[i].attention.W_qkv` | Fusionierte QKV-Projektion (`Conv1D`) |

<!--en:69291e63-->
| `h.i.attn.c_proj` | `blks[i].attention.W_o` | Ausgabeprojektion (`Conv1D`) |

<!--en:003065c0-->
| `h.i.mlp.c_fc`, `h.i.mlp.c_proj` | `blks[i].ffn.W_1`, `blks[i].ffn.W_2` | Zweischichtiges MLP mit GELU (`Conv1D`) |

<!--en:98bf1e30-->
| `ln_f` | `norm` | Abschließende LayerNorm vor dem Ausgabekopf |

<!--en:4824c297-->
`nn.Linear` speichert die Gewichte in der Form `(out, in)`. Daher transponiert `load_gpt2` jede Gewichtsmatrix von `Conv1D`.

<!--en:8859f6c7-->
Die Gewichtsmatrix von `nnx.Linear` hat die Form `(in, out)`, genau wie bei `Conv1D`. Daher kopiert `load_gpt2` die Gewichte unverändert.

<!--en:b438f4fa-->
Wir prüfen den geladenen Checkpoint, indem wir seine Kreuzentropie auf dem ersten Absatz von *The Time Machine* berechnen ($\approx 3.9$ Nat pro Token, entsprechend einer Perplexität von $\approx 49$) und Fortsetzungen erzeugen:

<!--en:1c822607-->
Nach der Prüfung der vortrainierten Gewichte übergeben wir einen Prompt an `gpt2.generate`. Gieriges Dekodieren (`top_k=1`) liefert eine deterministische Fortsetzung:

<!--en:b394b9f9-->
Wechseln wir für 40 Subworttoken zum stochastischen Top-$k$-Sampling (`temperature=0.8, top_k=50`), erhalten wir eine flüssige, abwechslungsreiche Fortsetzung:

<!--en:02e337a2-->
## Die moderne Architektur

<!--en:2b297386-->
Der Vergleich der Konfiguration von `load_gpt2()` aus dem Jahr 2019 mit unserem Standardmodell `GPT` zeigt vier wesentliche Neuerungen, die moderne Sprachmodelle mit reinem Decoder und offenen Gewichten auszeichnen (:numref:`tab_modern-recipe`):

<!--en:1a5e47de-->
1. **Normalisierung:** Eine vorgeschaltete `RMSNorm` ohne Bias-Terme (:numref:`subsec_rmsnorm`) ersetzt `LayerNorm`. Oft kommt eine **QK-Norm** pro Kopf (:numref:`subsec_qk-norm`) oder eine Normalisierung der Ausgaben der Teilschichten hinzu (OLMo 2, Gemma 3).

<!--en:db5a8885-->
2. **Positionen:** **Rotary Position Embeddings (RoPE)** (:numref:`sec_positional-information`) in jeder Attention-Schicht ersetzen die feste, gelernte Positionstabelle `wpe`.

<!--en:14c0d153-->
3. **Feedforward-Netz:** **`SwiGLU`** (oder `GeGLU`) mit Gating und ohne Bias-Terme ersetzt ein zweischichtiges MLP mit `GELU`.

<!--en:13a78e81-->
4. **Cache bei der Inferenz und Skalierung der Modellkapazität:** **Grouped-Query-Attention (GQA)**, **Attention mit gleitendem Fenster** oder **Kompression durch eine niederrangige latente Darstellung (MLA)** verkleinern den KV-Cache (:numref:`sec_kv-cache`). Schichten mit einer **Mixture of Experts (MoE)** (:numref:`sec_moe`) erhöhen die Gesamtzahl der Parameter bei gleichbleibenden FLOPs pro Token.

<!--en:6e007488-->
:Architekturen von sieben Modellfamilien mit offenen Gewichten nach ihren technischen Berichten aus den Jahren 2023 bis 2025. Attention und Cache: :numref:`sec_kv-cache-reduction`. Normalisierung: :numref:`sec_transformer-block`. Positionen: :numref:`sec_positional-information`. Mixture of Experts: :numref:`sec_moe`.

<!--en:ccbb7ecb-->
| Modell | Attention und Cache | Normalisierung | Positionen | FFN | Dicht oder MoE |

<!--en:bd862735-->
| Mistral 7B (2023) :cite:`Jiang.Sablayrolles.Mensch.ea.2023` | GQA $32{:}8$, gleitendes Fenster der Länge 4096 | RMSNorm, vorgeschaltet | RoPE, Basis $10^4$ | SwiGLU | Dicht |

<!--en:da1f1d7c-->
| Llama 3 (2024) | GQA $32{:}8$ (8B), $64{:}8$ (70B), $128{:}8$ (405B) | RMSNorm, vorgeschaltet | RoPE, Basis $5 \times 10^5$ | SwiGLU | Dicht |

<!--en:dcc4d5fa-->
| Qwen3 (2025) :cite:`Yang.Li.Yang.ea.2025` | GQA: $64{:}8$ dicht (32B), $64{:}4$ MoE (235B-A22B) | RMSNorm, vorgeschaltet, QK-Norm | RoPE, Basis $10^6$, YaRN | SwiGLU | Dicht und MoE (128 Experten, 8 aktiv) |

<!--en:b5734b4e-->
| OLMo 2/3 (2024/25) :cite:`OLMo.2025,OLMo3.2025` | MHA (7B), GQA (32B). OLMo 3: Fenster der Länge 4096 in 3 von 4 Schichten | RMSNorm auf den Ausgaben der Teilschichten, QK-Norm | RoPE | SwiGLU | Dicht |

<!--en:b51c73e1-->
| DeepSeek-V3 (2024) :cite:`Liu.Feng.Xue.ea.2024` | MLA: Kompression von KV auf einen 512-dimensionalen latenten Vektor | RMSNorm, vorgeschaltet | Entkoppeltes RoPE (gemeinsamer 64-dimensionaler Schlüssel) | SwiGLU | MoE: 256 Experten + 1 gemeinsamer Experte, 8 aktiv |

<!--en:d498fb36-->
| Gemma 3 (2025) :cite:`Gemma.Team.2025` | GQA, lokal zu global $5{:}1$, Fensterlänge 1024 | RMSNorm, vor- und nachgeschaltet, QK-Norm | RoPE, Basis $10^6$ global, $10^4$ lokal | GeGLU | Dicht |

<!--en:d9ddcc60-->
| gpt-oss (2025) :cite:`OpenAI.2025` | GQA $64{:}8$, Fensterlänge 128 in jeder zweiten Schicht, gelernte Attention-Senken | RMSNorm, vorgeschaltet | RoPE, YaRN | SwiGLU mit begrenztem Wertebereich | MoE: 128 oder 32 Experten, 4 aktiv |

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:db030636-->
Ein **GPT mit reinem Decoder** ordnet $L$ kausale `TransformerBlock`-Schichten zwischen der Einbettungstabelle für Token $\mathbf{E} \in \mathbb{R}^{V \times d}$ und einer abschließenden Normalisierung mit gewichtsgebundener Ausgabeprojektion $\mathbf{E}^\top$ an. In einem einzigen parallelen Vorwärtsdurchlauf berechnet es an allen $T$ Sequenzpositionen die Logits für das nächste Token (:eqref:`eq_gpt-loss`). Beim Training verwenden wir **AdamW** mit selektivem Gewichtszerfall nur für zweidimensionale Matrizen, eine lineare Aufwärmphase, Kosinus-Abfall und Gradienten-Clipping (`LMOptimizer`). Damit unterschreitet unser bytebasiertes `GPT` mit 4.8 Millionen Parametern innerhalb von 1000 Schritten die Bigrammentropie englischer Texte deutlich. In kontrollierten Vergleichen mit gleicher Parameteranzahl und gleichem Rechenaufwand schlägt `SwiGLU` die Variante `GELU` um rund $0.06$ Nat pro Byte. Die Streuung zwischen Zufallsstartwerten beträgt nur $0.01$ Nat. Der Vorsprung von `GPT` gegenüber einem gleich großen `RNNLM` wächst mit der Kontextposition $t$ stetig. Moderne große Sprachmodelle mit offenen Gewichten (:numref:`tab_modern-recipe`) kombinieren ebenfalls vorgeschaltete `RMSNorm` mit `RoPE` und `SwiGLU` oder `GeGLU`. Sie unterscheiden sich vor allem darin, wie sie den KV-Cache komprimieren (:numref:`sec_kv-cache`) und ob sie das dichte FFN durch eine Mixture of Experts ersetzen (:numref:`sec_moe`).

<!--en:2294243a-->
## Übungen

<!--en:57d0fb9f-->
1. **Ablationsstudie zum Trainingsrezept (`LMOptimizer`).** `LMOptimizer` kombiniert vier Bestandteile: selektiven Gewichtszerfall ($\lambda = 0.1$ nur für zweidimensionale Gewichtsmatrizen), eine lineare Aufwärmphase (100 Schritte), Kosinus-Abfall und Gradienten-Clipping anhand der globalen Norm ($\|\nabla \mathcal{L}\|_2 \le 1.0$). Trainieren Sie unser bytebasiertes `GPT` mit 6 Blöcken für jeweils 1000 Schritte. Ändern Sie pro Lauf nur einen Bestandteil: (a) Gewichtszerfall für *alle* Parameter einschließlich Einbettungen und RMSNorm-Verstärkungsfaktoren, (b) keine Aufwärmphase (`warmup=0`), (c) konstante Lernrate (durchgehend `lr=2e-3`) und (d) kein Gradienten-Clipping. Welche einzelne Änderung erhöht den Validierungsverlust am stärksten?
   :exlabel:`ex_gpt-recipe`

<!--en:15ac0867-->
2. **Zuordnung der Tensoren im GPT-2-Checkpoint prüfen.** Die Datei `model.safetensors` von GPT-2 enthält 148 Gewichts- und Bias-Tensoren sowie 12 Puffer für kausale Masken aus der ursprünglichen Implementierung (`h.i.attn.bias`). Schreiben Sie eine kurze Prüfung, die bestätigt, dass `d2l.num_params(gpt2)` mit der Gesamtzahl der Elemente in den 148 Parametertensoren der Datei übereinstimmt ($124{,}439{,}808$). Erklären Sie anhand der Anordnung der Gewichte in linearen Schichten Ihres Frameworks, ob Sie `c_attn.weight` beim Laden transponieren müssen.
   :exlabel:`ex_gpt-checkpoint-mismatches`

<!--en:99c816e7-->
3. **Baseline ohne Blöcke mit gebundenen Einbettungen ($L = 0$).** Betrachten Sie `GPT` mit `num_blks=0`. Das Modell bettet das Eingabetoken $x_t$ als $\mathbf{e}_{x_t}$ ein, normalisiert den Vektor mit `RMSNorm` und multipliziert ihn mit $\mathbf{E}^\top$. Daraus ergeben sich die Logits $\mathbf{z}_t \propto \mathbf{e}_{x_t} \mathbf{E}^\top$.
   1. Welche Klasse statistischer Sprachmodelle (welche $n$-Gramm-Ordnung) kann ein Modell mit 0 Blöcken darstellen?
   2. Warum erzwingt die Bindung der Eingabe- und Ausgabeeinbettungen ($\mathbf{W}_{\textrm{LM}} = \mathbf{E}^\top$), dass in jeder Zeile der symmetrischen Matrix $\mathbf{E}\mathbf{E}^\top$ der Diagonaleintrag für $x_{t+1} = x_t$ maximal ist, wenn alle Zeilen von $\mathbf{E}$ dieselbe Norm haben? Wie bricht ein Modell ohne Gewichtsbindung oder mit 1 Block diese Symmetrie?
   :exlabel:`ex_gpt-tied-head`

<!--en:eafdc1b3-->
4. **Bit pro Byte bei unterschiedlichen Tokenizern (bytebasiertes GPT und GPT-2 mit BPE).** Ein bytebasiertes Modell gibt die Kreuzentropie in Nat pro Byte an, GPT-2 dagegen in Nat pro Subworttoken.
   1. Betrachten Sie eine beliebige Passage mit $N_B$ UTF-8-Bytes, die `enc.encode` in $N_T$ Subworttoken zerlegt. Zeigen Sie, dass sich die **Bit pro Byte (bpb)** eines Subwortmodells mit mittlerer negativer Log-Likelihood $\mathcal{L}_{\textrm{tok}}$ pro Token (in Nat pro Token) exakt als $\textrm{bpb} = \frac{N_T \, \mathcal{L}_{\textrm{tok}}}{N_B \ln 2}$ berechnen lassen.
   2. Bestimmen Sie für unser bytebasiertes `model` mit 4.8 Millionen Parametern und für das vortrainierte `gpt2` mit 124 Millionen Parametern die Bit pro Byte auf den ersten 10 000 Byte des Validierungsdatensatzes von WikiText-103.
   :exlabel:`ex_gpt-bits-per-byte`

<!--en:25995c8e-->
5. **Top-$p$-Sampling (Nucleus-Sampling).** Erweitern Sie `GPT.generate` um ein optionales Argument `top_p` ($0 < p < 1$) :cite:`Holtzman.Buys.Du.ea.2020`. Sortieren Sie dazu die mit der Temperatur skalierten Wahrscheinlichkeiten absteigend. Maskieren Sie alle Token jenseits des kürzesten Präfixes, dessen kumulierte Wahrscheinlichkeit $p$ übersteigt. Vergleichen Sie die Fortsetzungen von `gpt2` für den Prompt `"Alan Turing theorized that computers would"` bei `top_p=0.9, temperature=0.8` mit denen bei `temperature=1.2` ohne Filter.

<!--en:637cae41-->
6. **Stabilität der Lernrate mit QK-Norm.** Trainieren Sie unser bytebasiertes `GPT` mit 6 Blöcken für die maximalen Lernraten $\eta \in \{2\times 10^{-3}, 6\times 10^{-3}, 1.5\times 10^{-2}\}$. Vergleichen Sie die Standardversion von `CausalAttention` mit einer Variante, die `RMSNorm(d_h)` auf `q` und `k` in `CausalAttention.qkv` anwendet (:numref:`subsec_qk-norm`). Wie verändert die QK-Norm die höchste stabile Lernrate?
   :exlabel:`ex_gpt-qk-norm-sensitivity`

<!--en:02c69a8b-->
[Diskussion](https://d2l.discourse.group/t/36775)

<!--en:a300e618-->
Sprachmodelle mit reinem Decoder (GPT)<br>**kausale Blöcke und gebundene Einbettungen · bytebasiertes Pre-Training · SwiGLU und RNN im Vergleich · Sampling · Laden von GPT-2 · moderne Architektur**

<!--en:d95e6fda-->
::: {.slide title="Ein Fenster der Länge T liefert T Vorhersageaufgaben"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:c814502d-->
- Ein autoregressives Sprachmodell zerlegt die Wahrscheinlichkeit einer Sequenz mit der Kettenregel $P(x_{1:T+1}) = \prod_{t=1}^T P(x_{t+1} \mid x_{1:t})$.

<!--en:8899f996-->
- **Paralleles Teacher-Forcing:** Statt $T$ separater Vorwärtsdurchläufe für Präfixe der Längen $1, \dots, T$ berechnet ein **GPT mit reinem Decoder** alle $T$ Präfixvorhersagen in **einem parallelen Vorwärtsdurchlauf** mit kausaler Maske.

<!--en:74e8b455-->
::: {.slide title="GPT-Architektur: Einbettungen, kausale Blöcke und gebundener Ausgabekopf"}

<!--en:c12b0ef7-->
[Architektur im Überblick]{.kicker}

<!--en:aa875cfc-->
- **Eingabeeinbettung ($\mathbf{E} \in \mathbb{R}^{V \times d}$):** Bildet diskrete Token-IDs $x_1, \dots, x_T$ auf die Anfangsvektoren des Residualstroms in $\mathbb{R}^{B \times T \times d}$ ab.

<!--en:2505e568-->
- **Kausaler Transformer-Stapel ($L$ Blöcke):** Jeder `TransformerBlock` wendet kausale Self-Attention und ein positionsweises FFN auf den Residualstrom an.

<!--en:29167e93-->
- **Abschließende Normalisierung und Ausgabekopf mit Gewichtsbindung ($\mathbf{E}^\top$):** Projiziert die normalisierten Vektoren des Residualstroms auf die Logits des Vokabulars $\mathbf{Z} = \mathrm{Norm}(\mathbf{H}^{(L)})\mathbf{E}^\top \in \mathbb{R}^{B \times T \times V}$. Die Wiederverwendung von $\mathbf{E}$ spart $V d$ Parameter ($38.6\text{M}$ bei GPT-2 Small).

<!--en:f6e0f154-->
::: {.slide title="Fusionierte kausale Self-Attention mit RoPE und GQA"}

<!--en:d731f9ec-->
[Kausale Attention]{.kicker}

<!--en:50e1ad04-->
- **Fusionierte QKV-Projektion:** Multipliziert $\mathbf{X}$ mit einer einzigen Matrix $\mathbf{W}_{qkv} \in \mathbb{R}^{d \times (d + 2 H_{\text{kv}} d_h)}$ und teilt das Ergebnis auf $H$ Query-Köpfe und $H_{\text{kv}} \le H$ Schlüssel-Wert-Köpfe auf.

<!--en:87cd8898-->
- **RoPE und kausale Maske (`is_causal=True`):** RoPE rotiert Abfragen und Schlüssel um positionsabhängige Winkel. Die Maske setzt die Scores für $s > t$ auf $-\infty$, sodass $\partial \mathbf{z}_t / \partial \mathbf{x}_s = \mathbf{0}$ gilt.

<!--en:05975dd2-->
::: {.slide title="Positionsinformation: gelernte Tabellen und RoPE im Vergleich"}

<!--en:caa96def-->
[Positionskodierungen]{.kicker}

<!--en:55e78ab5-->
- **`pos='learned'` (GPT-2, 2019):** Addiert am Eingang des Stapels einmalig die Zeile $\mathbf{p}_t \in \mathbb{R}^d$ aus `pos_emb` ($T_{\max} \times d$). Die feste Obergrenze liegt bei $T_{\max} = 1{,}024$.

<!--en:c7ff61d5-->
- **`pos='rope'` (Standard, Llama, Qwen):** Rotiert $(\mathbf{q}_t, \mathbf{k}_s)$ in *jeder* Attention-Schicht mithilfe der vorberechneten Tabellen aus `d2l.rope_tables(T, head_width)`. Die relativen Scores eignen sich für Sequenzen von $32\text{K}$ bis $1\text{M}$ Token.

<!--en:832273c2-->
::: {.slide title="Der gesamte GPT-Vorwärtsdurchlauf in zehn Zeilen" only="pytorch"}

<!--en:cb6399b9-->
[Implementierung des Vorwärtsdurchlaufs]{.kicker}

<!--en:3194b30b-->
Für Token-IDs `X` der Form `(batch, T)` liefert `GPT.forward` gleichzeitig an jeder Position die Logits für das nächste Token. Das Ergebnis hat die Form `(batch, T, V)`:

<!--en:d104e599-->
Die Gewichtsbindung koppelt die Ausgabeprojektion an `self.token_emb.weight`. Dadurch erhalten seltene Token Gradientenupdates, wenn sie im Eingabepräfix erscheinen, aber auch dann, wenn sie im Nenner der Softmax-Funktion als Alternativen eingehen.

<!--en:98568517-->
::: {.slide title="Der gesamte GPT-Vorwärtsdurchlauf in zehn Zeilen" only="jax"}

<!--en:cb6399b9-2-->
[Implementierung des Vorwärtsdurchlaufs]{.kicker}

<!--en:4f084454-->
Für Token-IDs `X` der Form `(batch, T)` liefert `GPT.__call__` gleichzeitig an jeder Position die Logits für das nächste Token. Das Ergebnis hat die Form `(batch, T, V)`:

<!--en:4b3ce629-->
Die Gewichtsbindung koppelt die Ausgabeprojektion über `self.token_emb.attend` an die Einbettungstabelle. Dadurch erhalten seltene Token Gradientenupdates, wenn sie im Eingabepräfix erscheinen, aber auch dann, wenn sie im Nenner der Softmax-Funktion als Alternativen eingehen.

<!--en:1dd498d7-->
::: {.slide title="Tiefenabhängige Skalierung der Anfangsgewichte (1/√(2L))"}

<!--en:1580e77b-->
[Initialisierung]{.kicker}

<!--en:ddce39a5-->
- `init_weights` zieht die Gewichte für Einbettungen und Projektionen aus $\mathcal{N}(0, 0.02^2)$. Die beiden Matrizen, die in den Residualstrom schreiben (`W_o`, `W_2`), skaliert die Funktion mit $1/\sqrt{2L}$.

<!--en:54bfb32a-->
- Obwohl $2L$ Teilschichten in den Pre-Norm-Residualstrom schreiben, bleibt dessen Varianz bei der Initialisierung so schon ab Block 1 nahe an der Varianz der Einbettungen.

<!--en:63091fa0-->
::: {.slide title="Kreuzentropie als autoregressive Zielfunktion"}

<!--en:f6f8a32f-->
[Trainingsziel]{.kicker}

<!--en:b5d924cb-->
Für ein Eingabefenster $\mathbf{x} = (x_1, \dots, x_T)$ und die verschobenen Zielwerte $\mathbf{y} = (x_2, \dots, x_{T+1})$ minimiert das Training die mittlere negative Log-Likelihood über alle $B \times T$ Positionen:

<!--en:d40cd8f4-->
::: {.slide title="Bytebasierte Sprachmodellierung auf WikiText-103"}

<!--en:327a54bd-->
[Korpus und Prüfung der Initialisierung]{.kicker}

<!--en:5174a181-->
Ein bytebasiertes `GPT` mit 6 Blöcken ($V = 256, d = 256, H = 8, L = 6, T = 128$, $4.79\text{M}$ Parameter) wird auf $270\text{M}$ UTF-8-Rohbytes aus **WikiText-103** trainiert:

<!--en:279360d5-->
**Plausibilitätsprüfung vor dem Training:** Bei Schritt $0$ ergibt sich aus den zufälligen Logits für jedes Byte ungefähr dieselbe Wahrscheinlichkeit $1/256$. Der anfängliche Verlust ($5.5\text{--}5.7\text{ nats/byte}$) liegt also nahe bei $\ln 256 \approx 5.55\text{ nats/byte}$.

<!--en:6116d67d-->
::: {.slide title="Ein moderner Optimierer für das Pre-Training (LMOptimizer)"}

<!--en:821003f6-->
[Optimierungsrezept]{.kicker}

<!--en:63fe4672-->
- `LMOptimizer` kombiniert AdamW ($\eta_{\max} = 2 \times 10^{-3}$), eine lineare Aufwärmphase mit anschließendem Kosinus-Abfall und Gradienten-Clipping anhand der globalen Norm ($\le 1.0$):
  $$\eta_t = \eta_{\max} \cdot \frac{1}{2}\Bigl(1 + \cos\bigl(\pi \tfrac{t - t_w}{S - t_w}\bigr)\Bigr)$$

<!--en:f55885ee-->
- **Selektiver Gewichtszerfall ($\lambda = 0.1$ nur für zweidimensionale Matrizen):** Ausgenommen sind Einbettungen (`*emb*`), eindimensionale Verstärkungsfaktoren von `RMSNorm` und Bias-Terme.

<!--en:f9117ee3-->
::: {.slide title="Trainingskurve und empirische Entropien als Vergleichswerte"}

<!--en:cba18bb6-->
[Trainingsergebnisse]{.kicker}

<!--en:995f03af-->
- **Schnelle Konvergenz:** In $1{,}000$ Schritten ($8.2\text{M}$ Byte, $\sim 3\%$ des Korpus) fällt die Kreuzentropie auf den Validierungsdaten von $\approx 5.6$ auf **$1.34\text{ nats/byte}$** ($\approx 1.93\text{ bits/byte}$).

<!--en:40c26715-->
- **Vergleichswerte aus Häufigkeiten derselben Bytes:**
  - Gleichverteiltes Raten ($\ln 256$): **$5.55\text{ nats/byte}$**
  - Empirische Unigrammhäufigkeiten: **$3.17\text{ nats/byte}$**
  - Empirische Bigrammübergänge: **$2.45\text{ nats/byte}$**

<!--en:fc6f082c-->
::: {.slide title="Trainingskurve und empirische Entropien als Vergleichswerte (Fortsetzung)"}

<!--en:cba18bb6-2-->
[Trainingsergebnisse]{.kicker}

<!--en:c9c6f0e7-->
Die Unigramm- und Bigrammvergleichswerte ergeben sich aus den Häufigkeiten einzelner Bytes und Bytepaare in den ersten $10\text{M}$ Byte der Trainingsdaten:

<!--en:02569252-->
::: {.slide title="Kontrollierte Ablation: Streuung durch Zufallsstartwerte und Gewinn durch SwiGLU"}

<!--en:29a2cbee-->
[Versuchsbedingungen]{.kicker}

<!--en:7f34a42f-->
Um Architekturverbesserungen von zufälligen Schwankungen bei der Initialisierung zu unterscheiden, läuft das Training für `act='swiglu'` und die Variante `act='gelu'` mit gleicher Parameteranzahl jeweils mit drei unabhängigen Zufallsstartwerten:

<!--en:3accbf68-->
- **Geringe Streuung zwischen Läufen:** Bei drei Zufallsstartwerten schwankt der Validierungsverlust um **weniger als $0.01\text{ nats/byte}$**.

<!--en:9c1c50a4-->
- **Statistisch klarer Vorteil für `SwiGLU`:** Bei gleicher Parameteranzahl ($4.79\text{M}$) und gleicher Zeit pro Schritt schlägt `SwiGLU` die Variante `GELU` um **$\sim 0.06\text{ nats/byte}$** ($> 5\times$ so viel wie die Streuung zwischen Zufallsstartwerten).

<!--en:314fd56a-->
- **Versuch mit Post-Norm:** Blöcke mit `pre_norm=False` erreichen mit denselben 100 Aufwärmschritten und auch ohne Aufwärmphase nur die Unigrammentropie von **$3.17\text{ nats/byte}$**. Die Ausgaben der Teilschichten übersteigen den normalisierten Residualstrom, alle Token erhalten denselben Endvektor und das Training erholt sich nicht. Mit 500 Aufwärmschritten erreicht Post-Norm etwa $1.38$ und liegt knapp hinter Pre-Norm ($1.34$).

<!--en:cab31162-->
::: {.slide title="Vergleich mit einem rekurrenten Modell gleicher Parameteranzahl"}

<!--en:05a118e8-->
[Transformer und RNNLM im Vergleich]{.kicker}

<!--en:d5a89296-->
Ein `d2l.RNNLM` vergleichbarer Größe ($d_{\text{rnn}} = 1{,}920$, $4.74\text{M}$ Parameter) durchläuft das Training mit drei Lernraten. Verglichen wird der Validierungsverlust an den Fensterpositionen $t \in \{0, 1, 4, 16, 64, 127\}$:

<!--en:2416d618-->
**Warum der Abstand mit der Kontextposition $t$ wächst:** Bei $t = 0$ (1 Byte Kontext) erreichen beide Modelle etwa $2.5\text{ nats}$ (Bigrammgrenze: $2.45$). Bei $t = 16 \dots 127$ pendelt sich das RNN bei etwa $1.4\text{ nats}$ ein, weil es die Vorgeschichte in $t$ aufeinanderfolgenden $\tanh$-Schritten verdichten muss. Dagegen greift `GPT` in $O(1)$ Schritten direkt auf jedes frühere Byte zu (etwa $1.3\text{ nats}$).

<!--en:690e79ce-->
::: {.slide title="Autoregressive Dekodierung mit Temperatur und Top-k"}

<!--en:80006646-->
[Funktionsweise des Samplings]{.kicker}

<!--en:4a09287d-->
- Bei jedem Dekodierschritt dividiert `GPT.generate` die Logits $\mathbf{z}_t \in \mathbb{R}^V$ durch die **Softmax-Temperatur $\tau > 0$** und beschränkt die Auswahl auf die **Top-$k$**-Kandidaten:
  $$p_v = \frac{\exp(z_{t,v} / \tau)}{\sum_{u \in \mathcal{V}_k} \exp(z_{t,u} / \tau)}$$

<!--en:c56ebfd4-->
::: {.slide title="Autoregressive Dekodierung mit Temperatur und Top-k (Fortsetzung)"}

<!--en:80006646-2-->
[Funktionsweise des Samplings]{.kicker}

<!--en:6e268d0e-->
Bei $\tau = 0.7, k = 8$ konzentriert sich die Wahrscheinlichkeitsmasse auf erkennbare englische Wörter. Bei $\tau = 2.0$ wird die Verteilung flacher und der Text zunehmend unverständlich:

<!--en:64c4534e-->
::: {.slide title="Laden des vortrainierten GPT-2 Small von OpenAI (124M)"}

<!--en:532c94b7-->
[Checkpoint laden]{.kicker}

<!--en:96082955-->
- Mit den Einstellungen von GPT-2 aus dem Jahr 2019 (`norm='layer'`, `act='gelu'`, `pos='learned'`, `bias=True`) erzeugt `d2l.GPT` exakt die Architektur von **GPT-2 Small**:

<!--en:a2d4aab0-->
::: {.slide title="Fehlerquelle in PyTorch: Transponieren der Conv1D-Gewichte" only="pytorch"}

<!--en:704e16fe-->
[Anordnung der Gewichte]{.kicker}

<!--en:3c51e097-->
OpenAIs ursprünglicher TensorFlow-Code für GPT-2 verwendete `Conv1D`-Schichten, deren Gewichtsmatrizen die Form `(in_features, out_features)` haben. In PyTorch verwendet `nn.Linear` die Form `(out_features, in_features)`:

<!--en:99cc9585-->
**Leicht zu übersehender Fehler:** Da `c_proj` die quadratische Form `(768, 768)` hat, besteht die Matrix auch ohne `.T` jede Prüfung der Form. Erst eine Prüfung der Perplexität anhand echter Texte deckt die vertauschten Achsen auf.

<!--en:4b5b80fb-->
::: {.slide title="Übereinstimmende Gewichtsanordnung in Conv1D und nnx.Linear bei JAX" only="jax"}

<!--en:704e16fe-2-->
[Anordnung der Gewichte]{.kicker}

<!--en:0f5cbe6e-->
OpenAIs ursprünglicher TensorFlow-Code für GPT-2 verwendete `Conv1D`-Schichten mit Gewichtsmatrizen der Form `(in_features, out_features)`. Genau diese Anordnung verwendet auch `nnx.Linear` in Flax:

<!--en:bc0819c5-->
**Leicht zu übersehender Fehler:** Wer bei den Gewichten linearer Schichten die Konventionen `(in, out)` und `(out, in)` verwechselt, vertauscht bei quadratischen Projektionen wie `c_proj` mit der Form `(768, 768)` die Achsen, ohne einen Formfehler auszulösen. Geladene Checkpoints daher immer anhand der Perplexität auf zurückgehaltenen Daten prüfen.

<!--en:6ee29397-->
::: {.slide title="Prüfung des geladenen GPT-2 Small anhand von Perplexität und Text"}

<!--en:23e12e5c-->
[Checkpoint prüfen]{.kicker}

<!--en:ed3b2023-->
Mit seinen `124.4M` Parametern erreicht das geladene Modell `gpt2` auf *The Time Machine* eine Kreuzentropie von $3.89\text{ nats}$ pro Token (Perplexität $e^{3.89} \approx 49$ gegenüber $50{,}257$ bei gleichverteiltem Raten). Sowohl gieriges Dekodieren als auch Sampling liefern zusammenhängende Fortsetzungen:

<!--en:520559a1-->
::: {.slide title="Fortsetzungen von GPT-2 Small mit gierigem Dekodieren und Sampling"}

<!--en:23e12e5c-2-->
[Checkpoint prüfen]{.kicker}

<!--en:0a8c9a1a-->
::: {.slide title="Architektur moderner großer Sprachmodelle mit offenen Gewichten"}

<!--en:3ec0f888-->
[Vergleich aktueller Modelle]{.kicker}

<!--en:7901b569-->
- Alle wichtigen Modellfamilien mit reinem Decoder und offenen Gewichten (2023–2025) teilen die Grundarchitektur unseres Standardmodells `GPT` (`RMSNorm`, `RoPE`, `SwiGLU`).

<!--en:2320b78c-->
- **Zwei wesentliche Unterschiede zwischen aktuellen Modellen:**
  1. **Kompression des KV-Caches (§{sec_kv-cache}):** GQA ($32{:}8$ oder $64{:}8$), lokale und globale Fenster im Verhältnis $5{:}1$ (Gemma 3) oder MLA mit niedrigem Rang (DeepSeek-V3).
  2. **Sparsität im FFN (§{sec_moe}):** Dichtes `SwiGLU` oder feingranulare MoE ($128$ bis $256$ Experten).

<!--en:fa3aece6-->
::: {.slide title="Verständnisfragen zu Parametern, Perplexität und Symmetrie"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:4ef5ca81-->
Drei Rechnungen zur Selbstkontrolle:

<!--en:63d671b9-->
1. **Größe von GPT-2 Medium:** GPT-2 Medium hat $L = 24$ Blöcke, die Breite $d = 1024$, ein Vokabular der Größe $V = 50{,}257$, $T_{\max} = 1024$ gelernte Positionen und einen Ausgabekopf mit Gewichtsbindung. Die Gesamtzahl der Parameter schätzen.

<!--en:eaf8bb5f-->
2. **Perplexität als effektiver Verzweigungsfaktor:** Unser bytebasiertes GPT erreicht $1.35\text{ nats/byte}$, GPT-2 Small auf *The Time Machine* dagegen $3.89\text{ nats/token}$. Wie vielen gleich wahrscheinlichen Möglichkeiten entspricht der jeweilige Verlust?

<!--en:845adf55-->
3. **Gewichtsbindung ohne Blöcke:** Es gelte `num_blks=0` und alle Zeilen von $\mathbf{E}$ haben die Norm eins. Welches nächste Token $x_{t+1}$ erhält in $\mathbf{z}_t = \mathbf{e}_{x_t}\mathbf{E}^\top$ stets das höchste Logit?

<!--en:557b98b6-->
**Lösungen:**

<!--en:41a969fa-->
1. **$\approx 355\text{M}$ Parameter:** $24 \times 12(1024)^2 \approx 302.0\text{M}$ in den Blöcken $+ 50{,}257 \times 1024 \approx 51.5\text{M}$ in `token_emb` $+ 1.0\text{M}$ in `pos_emb`.

<!--en:6f4822f9-->
2. **$e^{1.35} \approx 3.9$ Byte** von $256$ für unser Bytemodell, **$e^{3.89} \approx 49$ Subworttoken** von $50{,}257$ für GPT-2.

<!--en:c7bc7e4a-->
3. **$x_{t+1} = x_t$ (Wiederholung des aktuellen Tokens):** In der symmetrischen Gram-Matrix $\mathbf{E}\mathbf{E}^\top$ ist das Skalarprodukt $\mathbf{e}_i^\top \mathbf{e}_j$ auf der Diagonalen $j = i$ maximal. Attention und FFNs brechen diese Symmetrie.

<!--en:8312f4f3-->
::: {.slide title="Zusammenfassung: Aufbau und Training eines GPT mit reinem Decoder"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:a933f8f2-->
- **Kausale Blöcke und gebundener Ausgabekopf:** Mit fusionierter Projektion $\mathbf{W}_{qkv}$, RoPE, kausaler Maske in unterer Dreiecksform und gemeinsam genutzter Matrix $\mathbf{E} \in \mathbb{R}^{V \times d}$ lassen sich alle $T$ Präfixvorhersagen in einem Durchlauf trainieren.

<!--en:5bf8b1b5-->
- **Initialisierung mit `1/√(2L)` und `LMOptimizer`:** AdamW mit Gewichtszerfall nur für zweidimensionale Matrizen, Aufwärmphase und Kosinus-Abfall erreicht $1.34\text{ nats/byte}$ in $1{,}000$ Schritten.

<!--en:9c33501a-->
- **Kontrollierte Vergleiche und Laden von GPT-2:** `SwiGLU` schlägt `GELU` um $0.06\text{ nats}$. Unser Modell lädt die offiziellen Gewichte von GPT-2 mit `124M` Parametern direkt.
