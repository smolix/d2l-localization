<!--en:ec4f9b5c-->
# Muon

<!--en:2de61e91-->
Alle bisher betrachteten Optimierer (SGD, Momentum und AdamW) behandeln die Parameter eines neuronalen Netzes als einen einzigen unstrukturierten Vektor. Selbst AdamW führt zwei skalare Mittelwerte $(m_{t,ij}, v_{t,ij})$ unabhängig für jeden Eintrag $W_{ij}$ einer Gewichtsmatrix $\mathbf{W}$. Würden wir die Einträge von $\mathbf{W}$ beliebig vertauschen, so würde AdamW für jeden Eintrag genau dieselben Schritte machen.

<!--en:5e701ddc-->
Mehr als $80\%$ der Parameter eines Transformers liegen jedoch in zweidimensionalen Projektionsmatrizen $\mathbf{W} \in \mathbb{R}^{m \times n}$ der versteckten Schichten. Als lineare Operatoren bilden sie die Aktivierungen $\mathbf{x} \in \mathbb{R}^n$ einer Schicht auf $\mathbf{y} = \mathbf{W}\mathbf{x} \in \mathbb{R}^m$ ab. In welcher Richtung verläuft der steilste Abstieg (engl. steepest descent), wenn wir die Gewichtsmatrix *als linearen Operator betrachten*?

<!--en:fa3eee24-->
Wie :citet:`Bernstein.Newhouse.2024` betonen, hängt der steilste Abstieg davon ab, mit welcher Norm wir den Schritt begrenzen. Messen wir ihn in der euklidischen Norm ($\ell_2$), so folgt SGD unmittelbar der Richtung des negativen Gradienten. Begrenzen wir dagegen mit der $\ell_\infty$-Norm jede Koordinate einzeln, erhalten wir den Vorzeichenabstieg (engl. sign descent), den Grenzfall von Adam ohne Gedächtnis. Wir können auch die größte relative Änderung der Schichtausgabe begrenzen, die ein Schritt $\Delta\mathbf{W}$ bewirken kann. Das Maximum von $\|\Delta\mathbf{W}\mathbf{x}\|_2 / \|\mathbf{x}\|_2$ ist die **Spektralnorm** $\|\Delta\mathbf{W}\|_2$. Für die Richtung des steilsten Abstiegs ersetzen wir dann jeden von null verschiedenen Singulärwert der Gradientenmatrix durch $1$. So erhalten wir den **orthogonalisierten Gradienten** $\mathbf{U}\mathbf{V}^\top$.

<!--en:7de100ae-->
Eine exakte Singulärwertzerlegung (engl. singular value decomposition, SVD) bei jedem Schritt dauert selbst auf GPUs zu lange. Mit fünf Anwendungen eines ungeraden Matrixpolynoms (**Newton-Schulz-Iteration**) lässt sich $\mathbf{U}\mathbf{V}^\top$ allein durch Matrizenmultiplikationen annähern. Wenden wir dieses Verfahren auf die Nesterov-Momentum-Puffer der versteckten 2D-Gewichtsmatrizen an und überlassen Einbettungen, eindimensionale Parameter und den Ausgabekopf AdamW, so erhalten wir **Muon** (*Momentum Orthogonalized by Newton–Schulz*, :citet:`Jordan.Jin.Boza.ea.2024`). In diesem Abschnitt leiten wir den steilsten Abstieg für die Kugeln der euklidischen Norm, der $\ell_\infty$-Norm und der Spektralnorm her. Wir untersuchen die abgestimmte Newton-Schulz-Iteration fünften Grades, implementieren den hybriden Optimierer Muon+AdamW von Grund auf und vergleichen ihn auf `TinyLM` mit AdamW.

<!--en:d6c2575c-->
## Normen und steilster Abstieg

<!--en:0db74f83-->
### Steilster Abstieg in einer Normkugel

<!--en:abd395cc-->
Wir gehen von der Taylor-Entwicklung erster Ordnung $f(\mathbf{x} + \mathbf{d}) \approx f(\mathbf{x}) + \langle \mathbf{g}, \mathbf{d} \rangle$ mit dem Gradienten $\mathbf{g} = \nabla f(\mathbf{x})$ aus. Für einen Radius $\eta > 0$ und eine Norm $\|\cdot\|$ löst der Schritt des steilsten Abstiegs das folgende Problem:

<!--en:a2e52008-->
Je nach Geometrie der Norm ergeben sich grundverschiedene Abstiegsrichtungen :cite:`Bernstein.Newhouse.2024` (:numref:`fig_opt_norm_balls` und :numref:`tab_muon_norms`):

<!--en:ec32b180-->
1. **Euklidische Kugel ($\|\mathbf{d}\|_2 \le \eta$):** Nach der Cauchy-Schwarz-Ungleichung zeigt der Minimierer in Richtung des negativen Gradienten. Der Schritt $\mathbf{d}^\star = -\eta\,\mathbf{g}/\|\mathbf{g}\|_2$ senkt die linearisierte Zielfunktion um $\eta\|\mathbf{g}\|_2$. Das ist die Richtung von SGD.

<!--en:f5ff7920-->
2. **Achsenparalleler Würfel ($\|\mathbf{d}\|_\infty = \max_i |d_i| \le \eta$):** Minimieren wir $\sum_i g_i d_i$ unter der Nebenbedingung $|d_i| \le \eta$, so können wir jede Koordinate an ihre Grenze setzen. Der Schritt $\mathbf{d}^\star = -\eta\,\operatorname{sign}(\mathbf{g})$ senkt die linearisierte Zielfunktion um den durch die duale $\ell_1$-Norm bestimmten Betrag $\eta\|\mathbf{g}\|_1$. Das ist der Vorzeichenabstieg, der Grenzfall von Adam ohne Gedächtnis ($\beta_1 = \beta_2 = 0$, siehe :numref:`sec_adam`).

<!--en:857bbe50-->
### Matrizen und Spektralnorm

<!--en:979843e1-->
Betrachten wir eine versteckte Gewichtsmatrix $\mathbf{W} \in \mathbb{R}^{m \times n}$ mit der Abbildungsvorschrift $\mathbf{x} \mapsto \mathbf{W}\mathbf{x}$. Für eine Störung $\Delta\mathbf{W}$ ist die Änderung der Ausgabe durch $\|\Delta\mathbf{W}\mathbf{x}\|_2 \le \|\Delta\mathbf{W}\|_2 \|\mathbf{x}\|_2$ beschränkt. Dabei ist $\|\Delta\mathbf{W}\|_2 = \sigma_1(\Delta\mathbf{W})$ die **Spektralnorm** (Operatornorm, :numref:`subsec_lin-alg-eigenvalues`). Die Gradientenmatrix $\mathbf{G} \in \mathbb{R}^{m \times n}$ habe die reduzierte Singulärwertzerlegung (:numref:`sec_mdl-svd-low-rank`)

<!--en:f0bd4b85-->
mit den Matrizen $\mathbf{U} \in \mathbb{R}^{m \times r}$ und $\mathbf{V} \in \mathbb{R}^{n \times r}$, deren Spalten jeweils orthonormal sind, und mit $\sigma_1 \ge \cdots \ge \sigma_r > 0$. Die Frobeniusnorm $\|\mathbf{G}\|_F = \sqrt{\sum_i \sigma_i^2}$ ist die euklidische Norm der als Vektor aufgefassten Matrix. Bei vergleichbar großen Singulärwerten kann sie die Spektralnorm $\sigma_1$ um einen Faktor von bis zu $\sqrt{\min(m, n)}$ übersteigen.

<!--en:976b8dc3-->
Lösen wir :eqref:`eq_muon-ball` unter der Nebenbedingung $\|\Delta\mathbf{W}\|_2 \le \eta$ für die Spektralnorm, so erhalten wir den **orthogonalisierten Gradienten**:

<!--en:67bdc3a1-->
Dabei wird jeder von null verschiedene Singulärwert von $\mathbf{G}$ auf $1$ gesetzt. Das lässt sich direkt beweisen. Für jede Matrix $\mathbf{A}$ mit $\|\mathbf{A}\|_2 \le 1$ folgt aus der zyklischen Invarianz der Spur

<!--en:a8bc9ce4-->
denn aus $\|(\mathbf{U}^\top \mathbf{A} \mathbf{V})\|_2 \le \|\mathbf{A}\|_2 \le 1$ folgt $|(\mathbf{U}^\top \mathbf{A} \mathbf{V})_{ii}| \le 1$ (Übung :exref:`ex_muon-diagonal-entry`). Setzen wir $\mathbf{A} = -\mathbf{U}\mathbf{V}^\top$, so gilt $\mathbf{U}^\top\mathbf{A}\mathbf{V} = -\mathbf{I}_r$. Damit erreichen wir das Minimum $-\eta\sum_i \sigma_i = -\eta\|\mathbf{G}\|_*$, wobei die **Nuklearnorm** die duale Norm ist.

<!--en:85910aca-->
![Richtungen des steilsten Abstiegs in drei Normkugeln: euklidische $\ell_2$-Kugel (Gradientenrichtung), achsenparalleler $\ell_\infty$-Würfel (Vorzeichenabstieg) und Kugel der Spektralnorm $\|\cdot\|_2$ (orthogonalisierter Gradient $\mathbf{U}\mathbf{V}^{\top}$ mit dem Wert $1$ für jeden aktiven Singulärwert).](../img/mdl-opt-norm-balls.svg)

<!--en:25b7a7e1-->
Adam gleicht die Schrittbeträge für die *einzelnen Matrixeinträge* an. Dabei kann sich der Schritt auf eine Richtung vom Rang $1$ beschränken, denn $\operatorname{sign}\!\begin{bmatrix}3 & 1\\ 1 & 1\end{bmatrix} = \begin{bmatrix}1 & 1\\ 1 & 1\end{bmatrix}$ hat die Singulärwerte $2$ und $0$. Der spektrale steilste Abstieg $-\eta\mathbf{U}\mathbf{V}^\top$ dagegen macht einen Schritt derselben Weite $\eta$ entlang jeder aktiven Singulärrichtung $(\mathbf{u}_i, \mathbf{v}_i)$ des Gradienten. :numref:`fig_opt_muon_spectrum` zeigt die Auswirkung auf das Spektrum. Im Gradienten überwiegen einige wenige Richtungen, im resultierenden Schritt haben alle dasselbe Gewicht.

<!--en:289952b7-->
![Singulärwerte der Gradientenmatrix $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ relativ zum größten Singulärwert, $\sigma_i/\sigma_1$ (links), und Singulärwerte der orthogonalisierten Matrix $\mathbf{U}\mathbf{V}^\top$ (rechts). Die Orthogonalisierung erhält die Singulärvektoren und setzt jeden von null verschiedenen Singulärwert auf $1$. Damit hat der Schritt in jeder dieser Richtungen die Weite $\eta$.](../img/mdl-opt-muon-spectrum.svg)

<!--en:16789a4c-->
:Steilster Abstieg für drei Normgeometrien.

<!--en:6558932e-->
| Normkugel für den Schritt | Richtung des steilsten Abstiegs | Abnahme der linearisierten Zielfunktion (duale Norm) | Geeignete Parametergruppe |

<!--en:db71bc3b-->
| Euklidische Kugel ($\|\mathbf{d}\|_2 \leq \eta$) | $-\eta\, \mathbf{g} / \|\mathbf{g}\|_2$ | $\eta\, \|\mathbf{g}\|_2$ | Unstrukturierter Vektor (SGD) |

<!--en:82087f1e-->
| Achsenparalleler Würfel ($\|\mathbf{d}\|_\infty \leq \eta$) | $-\eta\, \mathrm{sign}(\mathbf{g})$ | $\eta\, \|\mathbf{g}\|_1$ | Heterogene Koordinaten / Einbettungen (AdamW) |

<!--en:312654d2-->
| Kugel der Spektralnorm ($\|\Delta \mathbf{W}\|_2 \leq \eta$) | $-\eta\, \mathbf{U}\mathbf{V}^\top$ | $\eta \|\mathbf{G}\|_* = \eta \sum_i \sigma_i$ | Dichte 2D-Gewichtsmatrizen versteckter Schichten (Muon) |

<!--en:8733ed8a-->
### Zusammenhang mit Matrixvorkonditionierern (Shampoo) und Aufteilung der Parameter

<!--en:1a572e43-->
Der orthogonalisierte Gradient $\mathbf{U}\mathbf{V}^\top$ ergibt sich auch als Grenzfall der Matrixvorkonditionierung mit Kronecker-Faktoren ohne Gedächtnis. **Shampoo** :cite:`Gupta.Koren.Singer.2018` summiert linke und rechte Gram-Matrizen zu $\mathbf{L}_t = \sum_{s \le t} \mathbf{G}_s \mathbf{G}_s^\top$ und $\mathbf{R}_t = \sum_{s \le t} \mathbf{G}_s^\top \mathbf{G}_s$ auf und konditioniert den Gradienten mit $\mathbf{L}_t^{-1/4}\mathbf{G}_t\mathbf{R}_t^{-1/4}$ vor. Betrachten wir nur einen einzigen Gradienten $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ ohne Beiträge aus früheren Schritten, so kürzen sich die Singulärwerte exakt heraus:

<!--en:e600834e-->
Muon orthogonalisiert einen gewöhnlichen Momentum-Puffer für das erste Moment. Damit entfällt das Speichern und Invertieren von $\mathbf{L}_t$ und $\mathbf{R}_t$. Für die Vorkonditionierung der gesamten Matrix benötigt Muon also nur einen Momentum-Puffer pro Matrix (:numref:`subsec_mdl-preconditioning-ladder`).

<!--en:7e131a93-->
Die Geometrie der Spektralnorm gilt speziell für dichte lineare Abbildungen zwischen Aktivierungsräumen. Deshalb wenden wir Muon **nur auf die 2D-Gewichtsmatrizen der versteckten Schichten** an (die Attention-Matrizen `qkv`/`proj` und die MLP-Gewichte). Mit AdamW optimieren wir die Einbettungstabellen, in denen eine One-Hot-Kodierung jeweils eine Zeile auswählt, die eindimensionalen Bias- und LayerNorm-Skalierungsparameter sowie den abschließenden Ausgabekopf für das Vokabular :cite:`Jordan.Jin.Boza.ea.2024`.

<!--en:adae62e2-->
## Orthogonalisierung mit dem Newton-Schulz-Verfahren

<!--en:4fd247fa-->
Die Berechnung von $\mathbf{U}\mathbf{V}^\top$ durch eine explizite SVD dauert im Training zu lange. Wir nutzen daher die Tatsache, dass jedes **ungerade Matrixpolynom** auf die Singulärwerte von $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ einzeln wirkt und dabei die Singulärvektoren erhält:

<!--en:2c498b6f-->
Das folgt aus $\mathbf{X}\mathbf{X}^\top = \mathbf{U}\boldsymbol{\Sigma}^2\mathbf{U}^\top$. Normieren wir zunächst gemäß $\mathbf{X} \leftarrow \mathbf{X} / \|\mathbf{X}\|_F$, so liegen alle Singulärwerte im Intervall $[0, 1]$. Nun wenden wir wiederholt ein ungerades skalares Polynom $p(x)$ an, das die Werte aus $(0, 1]$ in Richtung $1$ verschiebt (**Newton-Schulz-Iteration**, :numref:`subsec_mdl-orthogonalized-updates`). So orthogonalisieren wir $\mathbf{X}$ allein durch schnelle Matrizenmultiplikationen.

<!--en:2a0774f7-->
Beim klassischen kubischen Polynom $p(x) = \frac{3}{2}x - \frac{1}{2}x^3$ ist die Anfangssteigung $p'(0) = 1.5$ so klein, dass die kleinen Singulärwerte in $5$ Schritten zu langsam wachsen. Deshalb verwenden :citet:`Jordan.Jin.Boza.ea.2024` ein abgestimmtes Polynom fünften Grades mit der großen Anfangssteigung $p'(0) = 3.4445$:

<!--en:328dcff1-->
Wie :numref:`fig_opt_ns_maps` zeigt, hebt die fünfmalige Anwendung von :eqref:`eq_muon-quintic` jeden Singulärwert aus $[0.01, 1]$ in das schmale Intervall $[0.68, 1.13]$ (und jeden Singulärwert über $0.0015$ auf mindestens $0.68$). Wir verzichten auf die exakte asymptotische Konvergenz gegen $1$ und gewinnen dafür in nur $5$ Schritten eine starke Verstärkung der schwachen Singulärrichtungen.

<!--en:1afde417-->
![Wirkung der Newton-Schulz-Polynome auf die Singulärwerte. Links: einmalige Anwendung des klassischen kubischen Polynoms und des abgestimmten Polynoms fünften Grades :eqref:`eq_muon-quintic`. Rechts: fünf aufeinanderfolgende Iterationen auf $[10^{-3}, 1]$ im Vergleich mit der Variante Polar Express, die mehrere Polynome verwendet :cite:`Amsel.Persson.Musco.ea.2026`.](../img/mdl-opt-ns-maps.svg)

<!--en:252a0f4f-->
Für eine $m \times n$-Matrix ($m \le n$) und eine Batchgröße von $B$ Token kosten fünf Newton-Schulz-Iterationen $5(4m^2n + 2m^3)$ FLOPs. Zum Vergleich benötigen der Vorwärtsdurchlauf und der Rückwärtsdurchlauf zusammen $6mnB$ FLOPs. Der relative Mehraufwand beträgt $\sim 5m/B$ ($<1\%$ bei praxisüblichen großen Batchgrößen von $B \sim 5 \times 10^5$ und $\sim 12\%$ bei `TinyLM` mit $B = 4{,}096$). Die Anwendung von `newton_schulz` auf eine $96 \times 64$-Matrix mit gaußverteilten Einträgen zeigt, wie die Iterationen das Singulärwertspektrum zusammendrängen. Es liegt im Intervall $[0.68, 1.20]$ nach $k = 3$ Iterationen und im Intervall $[0.68, 1.13]$ nach $k = 5$ Iterationen:

<!--en:10a6c2b9-->
Bei Gradienten aus realen Netzen kommen Muon zwei Eigenschaften zugute. Erstens orthogonalisiert Muon den Nesterov-Momentum-Puffer $\mathbf{G}_t + \beta\mathbf{M}_t$, der über $\sim 1/(1-\beta) = 20$ Schritte bei $\beta = 0.95$ mittelt. Damit hängt die Richtung nicht allein vom verrauschten Gradienten eines einzelnen Batches ab. Zweitens erfahren Singulärwerte unter $0.15\%$ der Frobeniusnorm nur eine lineare Verstärkung. Hier reicht selbst der Faktor $3.4445^5 \approx 485$ nicht aus, um $0.68$ zu erreichen. Solche Werte bleiben also deutlich unter $1$. Das dämpft am unteren Ende des Spektrums gerade die Richtungen, die nur Rauschen enthalten.

<!--en:c8b784e4-->
## Muon von Grund auf

<!--en:64a6cdef-->
### Aktualisierung und Abgleich der quadratischen Mittelwerte

<!--en:59724c9e-->
Für eine versteckte $m \times n$-Gewichtsmatrix $\mathbf{W}$ mit dem Gradienten $\mathbf{G}_t$ aktualisiert Muon den Momentum-Puffer $\mathbf{M}_t$ ($\beta = 0.95$). Anschließend wendet Muon fünf Newton-Schulz-Iterationen $\mathrm{NS}_5$ auf den vorausberechneten Nesterov-Gradienten $\mathbf{G}_t + \beta\mathbf{M}_t$ an. Dazu kommt der entkoppelte Gewichtszerfall (engl. decoupled weight decay) :cite:`Liu.Su.Yao.ea.2025`:

<!--en:f833cb5b-->
Warum multiplizieren wir mit $0.2\sqrt{\max(m, n)}$? Eine orthogonale $m \times n$-Matrix $\mathbf{U}\mathbf{V}^\top$ hat $\min(m, n)$ Singulärwerte vom Betrag eins und die Frobeniusnorm $\sqrt{\min(m, n)}$. Der quadratische Mittelwert (engl. root mean square, RMS) ihrer Einträge beträgt daher $\sqrt{\min(m, n)/(mn)} = 1/\sqrt{\max(m, n)}$. Multiplizieren wir mit $0.2\sqrt{\max(m, n)}$, so beträgt der RMS-Wert der Einträge des Schritts unabhängig von den Matrixdimensionen $0.2\,\eta$. Das entspricht dem empirischen Wert $\sim 0.2\,\eta$, den wir für AdamW in :numref:`sec_adam` gemessen haben. Beide Optimierer können daher dieselbe Lernrate $\eta$ und denselben Gewichtszerfall $\lambda$ verwenden :cite:`Liu.Su.Yao.ea.2025`.

<!--en:ce156b91-->
Wir haben Muon selbst implementiert, um jeden Schritt nachzuvollziehen. Auch das Framework bietet eine Implementierung an. Wie hängen deren Einstellungen mit dem Faktor $0.2\sqrt{\max(m, n)}$ aus :eqref:`eq_muon-update` zusammen? Dazu setzen wir alle Gewichte auf null und machen mit einer $96 \times 64$-Gradientenmatrix mit gaußverteilten Einträgen einen einzigen Schritt. Wir vergleichen den Schritt unserer Implementierung mit dem der Bibliothek:

<!--en:4e9314eb-->
Die beiden Schritte stimmen bis auf etwa ein Prozent überein. Der RMS-Wert ihrer Einträge beträgt jeweils rund $0.18\,\eta$. Das liegt etwas unter dem Nennwert $0.2\,\eta$, denn nach fünf Newton-Schulz-Iterationen verteilen sich die Singulärwerte noch zwischen $0.68$ und $1.13$, statt genau $1$ zu betragen. Für die folgenden Experimente verwenden wir weiterhin unsere eigene Implementierung.

<!--en:2aed377b-->
`torch.optim.Muon` akzeptiert nur zweidimensionale Parameter. Für Bias-Terme, Normalisierungsparameter und Einbettungen brauchen wir also weiterhin einen zweiten Optimierer. Die Einstellung `adjust_lr_fn='match_rms_adamw'` entspricht dem Faktor $0.2\sqrt{\max(m, n)}$. Mit der Standardeinstellung `'original'` multipliziert der Optimierer stattdessen mit $\sqrt{\max(1, m/n)}$. Der Standardwert für den Gewichtszerfall ist $0.1$. Die Bibliothek führt die Newton-Schulz-Iteration in bfloat16 aus. Das erklärt den Unterschied von etwa einem Prozent zwischen den beiden Schritten.

<!--en:b096c4d0-->
Bei `optax.contrib.muon` gibt `consistent_rms` die Konstante im Skalierungsfaktor an. Der Wert $0.2$ ergibt $0.2\sqrt{\max(m, n)}$. Ohne diese Angabe wird der Schritt mit $\sqrt{\max(1, \text{fan-out}/\text{fan-in})}$ multipliziert. Die Funktion wendet Muon auf die zweidimensionalen Parameter und Adam auf alle übrigen an. Wir können ihr daher das ganze Modell übergeben.

<!--en:4daae54f-->
### Parametergruppen und hybrider Optimierer

<!--en:97795973-->
Bei `TinyLM` übernimmt Muon die $8$ internen Blockmatrizen (`qkv`, `proj`, `mlp1` und `mlp2` in $2$ Blöcken). AdamW übernimmt die übrigen $22$ Tensoren (Einbettungen, LayerNorm-Vektoren, Bias-Terme und Ausgabekopf). Damit entfallen $393{,}216$ der insgesamt $470{,}784$ Parameter ($84\%$) auf Muon. Muon speichert für jeden Parameter nur einen Momentum-Puffer, während AdamW zwei Momente speichert. Dadurch sinkt der gesamte Speicherbedarf für den Optimiererzustand beim hybriden Optimierer um $42\%$.

<!--en:058ac43a-->
Wir kombinieren nun `Muon` (mit entkoppeltem Gewichtszerfall für die versteckten 2D-Matrizen) und `AdamW` (ohne Gewichtszerfall für die übrigen Parameter) mit `muon_adamw` zu einem einzigen hybriden Optimierer:

<!--en:c87b66e2-->
### Vergleich mit AdamW

<!--en:192729aa-->
Wir vergleichen die Kombination Muon+AdamW mit dem üblichen AdamW für die Lernraten $\eta \in \{10^{-3}, 3 \times 10^{-3}, 10^{-2}, 3 \times 10^{-2}, 10^{-1}\}$. Für die versteckten Matrizen gilt $\lambda = 0.1$. Beide Verfahren starten mit denselben Gewichten und laufen $2{,}000$ Schritte lang. Alle $250$ Schritte messen wir den Validierungsverlust. Zunächst führen wir für das Referenzverfahren `adamw` die Lernratensuche mit `run_lm` aus:

<!--en:fda4616d-->
Anschließend testen wir die Kombination `muon_adamw` mit genau denselben Lernraten aus `lrs` und demselben Startwert `seed` für den Zufallsgenerator:

<!--en:bd9a3371-->
Für jeden Optimierer wählen wir die Lernrate mit dem geringsten Validierungsverlust (`best_adamw` und `best_muon`). Wir plotten die geglätteten Trainings- und Validierungskurven. In einer Tabelle halten wir fest, bei welchem Schritt der Validierungsverlust erstmals unter $1.6$ Nat fällt und welcher Lernratenbereich nutzbar ist:

<!--en:08ca519c-->
Bei `TinyLM` liegt der Validierungsverlust von Muon+AdamW um $>0.1$ Nat niedriger als bei AdamW, und zwar bereits bei Schritt $500$. Die Schwelle von $1.6$ Nat unterschreitet Muon+AdamW mindestens einen Checkpoint ($250$ Schritte) früher als AdamW. Der nutzbare Lernratenbereich reicht bis $\eta = 3 \times 10^{-2}$. Bei dieser Lernrate endet AdamW deutlich über $2$ Nat. Der Grund ist, dass frühe Gradienten vom Rang $1$ nach eintragsweiser Normalisierung eine um den Faktor $\sqrt{\min(m, n)}$ größere Spektralnorm haben können als ein orthogonalisierter Schritt.

<!--en:425d6803-->
Bei größeren Modellen fällt der Vorteil geringer aus und ist weniger gut belegt. Bei vollständigem Lernratenabfall haben :citet:`Wen.Hall.Ma.ea.2025` gemessen, um welchen Faktor die Schrittzahl gegenüber einem sorgfältig abgestimmten AdamW sinkt. Mit wachsendem Modell fällt dieser Faktor von etwa $1.4\times$ bei $0.1\text{B}$ Parametern auf etwa $1.1\times$ bei $1.2\text{B}$ Parametern. :citet:`Liu.Su.Yao.ea.2025` berichten aus ihren eigenen Skalierungsexperimenten von einer etwa doppelt so hohen Recheneffizienz wie bei AdamW. Kimi K2 mit einer Billion Parametern wurde mit Muon und einer Begrenzung der Attention-Logits trainiert :cite:`Kimi.Team.2025`. Ein AdamW-Lauf dieser Größe fehlt für den Vergleich. Die Speicherersparnis beim Optimiererzustand beträgt unabhängig von der Modellgröße fast die Hälfte.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:79bec545-->
Der steilste Abstieg hängt von der Normkugel ab, die den Schritt begrenzt. Die euklidische $\ell_2$-Kugel liefert SGD, der achsenparallele $\ell_\infty$-Würfel den Vorzeichenabstieg, den Adam glättet. Die Kugel der Spektralnorm $\|\Delta\mathbf{W}\|_2 \le \eta$ liefert den Schritt entlang des orthogonalisierten Gradienten $-\eta\mathbf{U}\mathbf{V}^\top$. Er hat in jeder aktiven Singulärrichtung des Matrixoperators dieselbe Weite. **Muon** nähert $\mathbf{U}\mathbf{V}^\top$ für den Nesterov-Momentum-Puffer jeder versteckten 2D-Gewichtsmatrix an. Dazu nutzt Muon fünf Iterationen des abgestimmten Newton-Schulz-Polynoms fünften Grades :eqref:`eq_muon-quintic`. Die Skalierung mit $0.2\sqrt{\max(m, n)}$ gleicht den RMS-Wert der Schritteinträge an den von AdamW an. Muon wendet entkoppelten Gewichtszerfall an und überlässt Einbettungen, 1D-Vektoren und den Ausgabekopf AdamW. Bei `TinyLM` beschleunigt dieser hybride Optimierer Muon+AdamW die anfängliche Konvergenz. Er bleibt bei $3\times$ so hohen Lernraten wie AdamW stabil und senkt den Speicherbedarf für den Optimiererzustand um $42\%$.

<!--en:2294243a-->
## Übungen

<!--en:9a685631-->
1. **Schranke für Diagonaleinträge und Eindeutigkeit des spektralen Schritts.** Die Matrizen $\mathbf{U} \in \mathbb{R}^{m \times r}$ und $\mathbf{V} \in \mathbb{R}^{n \times r}$ haben jeweils orthonormale Spalten. Zeigen Sie, dass $\|\mathbf{U}^\top\mathbf{A}\mathbf{V}\|_2 \le \|\mathbf{A}\|_2$ und $|(\mathbf{U}^\top\mathbf{A}\mathbf{V})_{ii}| \le \|\mathbf{A}\|_2$ gelten. Bestimmen Sie, wann der Minimierer $\Delta\mathbf{W}^\star = -\eta\mathbf{U}\mathbf{V}^\top$ in :eqref:`eq_muon-spectral-step` eindeutig ist.
   :exlabel:`ex_muon-diagonal-entry`

<!--en:c75f5092-->
1. [code] **Empirischer RMS-Wert der Schritteinträge und Singulärwertspektrum.** Erweitern Sie `Muon` für `TinyLM` so, dass Sie den RMS-Wert der Einträge von $\Delta\mathbf{W} / \eta$ über $2{,}000$ Schritte verfolgen können. Vergleichen Sie das Singulärwertspektrum eines einzelnen Minibatch-Gradienten $\mathbf{G}_t$ mit dem des Momentum-Puffers $\mathbf{M}_t$ bei Schritt $0$ und bei Schritt $500$.
   :exlabel:`ex_muon-step-size`

<!--en:6560717a-->
1. [code] **Iterationszahl und Polynomgrad beim Newton-Schulz-Verfahren.** Vergleichen Sie die Iterationszahlen `num_iters` $\in \{1, 3, 5, 10\}$ für das abgestimmte Polynom fünften Grades :eqref:`eq_muon-quintic` mit $5$ Iterationen des klassischen kubischen Polynoms $p(x) = 1.5x - 0.5x^3$ auf `TinyLM`. Geben Sie den Validierungsverlust und die tatsächlich verstrichene Zeit pro Schritt an.
   :exlabel:`ex_muon-iterations`

<!--en:b648b5d9-->
1. [code] **Warum Einbettungen bei AdamW bleiben.**
   :exlabel:`ex_muon-embeddings`
    1. Eine $m \times n$-Gradientenmatrix $\mathbf{M}$ habe nur $r \le n$ von null verschiedene, linear unabhängige Zeilen. Diese entsprechen $r$ verschiedenen Token in einem Minibatch. Zeigen Sie, dass jede aktive Zeile von $\mathbf{U}\mathbf{V}^\top$ die $\ell_2$-Norm eins hat, unabhängig davon, ob ihr Token einmal oder $500$ Mal vorkam.
    1. Verschieben Sie `token_emb` bei `TinyLM` in die Parametergruppe von Muon. Prüfen Sie empirisch, ob wie erwartet die Perplexität auf den Validierungsdaten höher ausfällt, als wenn `token_emb` in der AdamW-Gruppe bleibt.

<!--en:cd9d3531-->
[Diskussion](https://d2l.discourse.group/t/36764)

<!--en:c6c95b7f-->
Matrixvorkonditionierung und Muon<br>**Steilster Abstieg in Normkugeln · Spektralnorm und UVᵀ · Newton-Schulz-Polynom fünften Grades · RMS-Abgleich · hybrider Optimierer Muon+AdamW**

<!--en:d7fc58d1-->
::: {.slide title="Gewichtsmatrizen als lineare Operatoren statt als flache Vektoren"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:b53781d0-->
- **Eintragsweise Verfahren ignorieren Permutationen:** SGD, Momentum und AdamW behandeln $\mathbf{W} \in \mathbb{R}^{m \times n}$ als $mn$ unabhängige Koordinaten. Eine Permutation der Einträge von $\mathbf{W}$ ändert die Schritte von AdamW für die einzelnen Einträge nicht.

<!--en:f52b4369-->
- **Operatorgeometrie ($83.5\%$ der Parameter von `TinyLM`):** Interne 2D-Gewichtsmatrizen bilden Schichtaktivierungen gemäß $\mathbf{x} \in \mathbb{R}^n \mapsto \mathbf{y} = \mathbf{W}\mathbf{x} \in \mathbb{R}^m$ ab.

<!--en:29776c00-->
- **Zentrale Frage:** Welche Richtung hat der steilste Abstieg, wenn die Änderung der Aktivierung $\Delta\mathbf{y} = \Delta\mathbf{W}\,\mathbf{x}$ beschränkt ist?

<!--en:861bf260-->
::: {.slide title="Abhängigkeit des steilsten Abstiegs von der Normkugel"}

<!--en:847a5857-->
[Geometrie des steilsten Abstiegs]{.kicker}

<!--en:06b9879f-->
Für die Taylor-Entwicklung erster Ordnung $f(\mathbf{x} + \mathbf{d}) \approx f(\mathbf{x}) + \langle\mathbf{g}, \mathbf{d}\rangle$ löst der Schritt des steilsten Abstiegs bei Radius $\eta > 0$ und Norm $\|\cdot\|$ folgendes Problem (Bernstein & Newhouse, 2024):

<!--en:d499e476-->
- **1. Euklidische Kugel ($\|\mathbf{d}\|_2 \le \eta$):** Das Minimum liegt in Richtung des negativen Gradienten $\mathbf{d}^\star = -\eta\,\frac{\mathbf{g}}{\|\mathbf{g}\|_2}$ (Richtung von **SGD**).

<!--en:0f68f2a3-->
- **2. Achsenparalleler Würfel ($\|\mathbf{d}\|_\infty = \max_i |d_i| \le \eta$):** Jede Koordinate erreicht ihre Grenze. Eine Ecke des Würfels liefert den Schritt $\mathbf{d}^\star = -\eta\,\operatorname{sign}(\mathbf{g})$ (**Vorzeichenabstieg**, durch **Adam** geglättet).

<!--en:8fd1ec29-->
::: {.slide title="Die Spektralnorm als Maß für Matrixschritte"}

<!--en:1197854b-->
[Geometrie von Matrixoperatoren]{.kicker}

<!--en:c145343f-->
- **Die Spektralnorm begrenzt die Änderung der Aktivierung:**
  $$\|\Delta\mathbf{W}\,\mathbf{x}\|_2 \le \|\Delta\mathbf{W}\|_2\,\|\mathbf{x}\|_2, \quad \|\Delta\mathbf{W}\|_2 = \sigma_1(\Delta\mathbf{W})$$

<!--en:e77f880f-->
- **SVD und Frobeniusnorm:** Für $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \sum_{i=1}^r \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ misst $\|\mathbf{G}\|_2 = \sigma_1$ die maximale Streckung in einer Richtung. Dagegen überschätzt $\|\mathbf{G}\|_F = \sqrt{\sum_i \sigma_i^2}$ die Streckung eines orthogonalen Schritts vom Rang $r=128$ um den Faktor $\sqrt{128} \approx 11.3\times$.

<!--en:51faf65e-->
::: {.slide title="Steilster Abstieg in der Spektralnorm mit UVᵀ"}

<!--en:f1b01b59-->
[Der orthogonalisierte Gradient]{.kicker}

<!--en:44525c27-->
- **Der spektrale steilste Abstieg ersetzt jeden aktiven Singulärwert durch $1$:**
  $$\Delta\mathbf{W}^\star = \operatorname*{argmin}_{\|\Delta\mathbf{W}\|_2 \le \eta}\; \langle\mathbf{G}, \Delta\mathbf{W}\rangle = -\eta\,\mathbf{U}\mathbf{V}^\top$$

<!--en:669a550c-->
- **Beweis über die Dualität der Spurpaarung:** Für $\|\mathbf{A}\|_2 \le 1$ gilt $\langle\mathbf{G}, \mathbf{A}\rangle = \sum_{i=1}^r \sigma_i (\mathbf{U}^\top\mathbf{A}\mathbf{V})_{ii} \ge -\sum_{i=1}^r \sigma_i = -\|\mathbf{G}\|_*$. Gleichheit gilt für $\mathbf{A} = -\mathbf{U}\mathbf{V}^\top$.

<!--en:88cb1f6d-->
::: {.slide title="Warum der eintragsweise Vorzeichenabstieg auf Rang 1 fallen kann"}

<!--en:3135b2f3-->
[Eintragsweiser und spektraler Ausgleich]{.kicker}

<!--en:3c1b954f-->
- Beispielmatrix $\mathbf{G} = \begin{bmatrix}3 & 1\\ 1 & 1\end{bmatrix}$ ($\sigma_1 \approx 3.41, \sigma_2 \approx 0.59$):
  - **Euklidische Kugel ($-\mathbf{G}/\|\mathbf{G}\|_F$):** Bei den Singulärwerten $(0.99, 0.17)$ kommt die zweite Richtung kaum zum Tragen.
  - **Achsenparalleler Würfel ($-\operatorname{sign}(\mathbf{G}) = -\mathbf{1}\mathbf{1}^\top$):** Die Singulärwerte $(2.00, 0.00)$ ergeben **nur noch Rang 1**.
  - **Spektralnorm ($-\mathbf{U}\mathbf{V}^\top = -\mathbf{I}_2$):** Die Singulärwerte **$(1.00, 1.00)$** ergeben einen vollen Schritt in jeder orthogonalen Merkmalsrichtung.

<!--en:a79a30f8-->
::: {.slide title="Drei Normgeometrien und ihr Zusammenhang mit Shampoo"}

<!--en:18db82cd-->
[Vorkonditionierer im Vergleich]{.kicker}

<!--en:22e5bdca-->
| Normbeschränkung für den Schritt | Richtung des steilsten Abstiegs | Abnahme der linearisierten Zielfunktion | Geeignete Parametergruppe |

<!--en:4e1b7052-->
| **Euklidische Kugel ($\|\mathbf{d}\|_2 \le \eta$)** | $-\eta\,\mathbf{g}/\|\mathbf{g}\|_2$ | $\eta\,\|\mathbf{g}\|_2$ | Unstrukturierter Vektor (SGD) |

<!--en:ce778d11-->
| **Achsenparalleler Würfel ($\|\mathbf{d}\|_\infty \le \eta$)** | $-\eta\,\operatorname{sign}(\mathbf{g})$ | $\eta\,\|\mathbf{g}\|_1$ | Einbettungen, 1D-Normierungsparameter und Bias-Terme, Ausgabekopf (AdamW) |

<!--en:5f8f20a8-->
| **Spektralnorm ($\|\Delta\mathbf{W}\|_2 \le \eta$)** | $-\eta\,\mathbf{U}\mathbf{V}^\top$ | $\eta\,\|\mathbf{G}\|_* = \eta\sum_i \sigma_i$ | Dichte 2D-Projektionsmatrizen versteckter Schichten (Muon) |

<!--en:dab4bb0b-->
- **Muon entspricht Shampoo ohne Gedächtnis:** **Shampoo** (Gupta et al., 2018) summiert die linken und rechten Gram-Matrizen $\mathbf{L}_t = \sum_{s \le t}\mathbf{G}_s\mathbf{G}_s^\top$ und $\mathbf{R}_t = \sum_{s \le t}\mathbf{G}_s^\top\mathbf{G}_s$ auf und verwendet $\mathbf{L}_t^{-1/4}\mathbf{G}_t\mathbf{R}_t^{-1/4}$ zur Vorkonditionierung. Für eine einzelne Matrix $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ gilt:
  $$(\mathbf{G}\mathbf{G}^\top)^{-1/4}\,\mathbf{G}\,(\mathbf{G}^\top\mathbf{G})^{-1/4} = \mathbf{U}\boldsymbol{\Sigma}^{-1/2}\mathbf{U}^\top \cdot \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top \cdot \mathbf{V}\boldsymbol{\Sigma}^{-1/2}\mathbf{V}^\top = \mathbf{U}\mathbf{V}^\top$$

<!--en:0dd6c57f-->
::: {.slide title="Orthogonalisierung ohne SVD durch ungerade Matrixpolynome"}

<!--en:90ef9d24-->
[Newton-Schulz-Iteration]{.kicker}

<!--en:d65255f4-->
- **Ungerade Matrixpolynome erhalten die Singulärvektoren:** Für $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ und $\mathbf{A} = \mathbf{X}\mathbf{X}^\top = \mathbf{U}\boldsymbol{\Sigma}^2\mathbf{U}^\top$ gilt:
  $$p(\mathbf{X}) = (a\mathbf{I} + b\mathbf{A} + c\mathbf{A}^2)\mathbf{X} = \mathbf{U}\,p(\boldsymbol{\Sigma})\,\mathbf{V}^\top$$

<!--en:3cc3ded6-->
- **Zwei Schritte auf der GPU:**
  1. Mit $\mathbf{X}_0 = \mathbf{G} / (\|\mathbf{G}\|_F + \epsilon)$ normieren, sodass $\sigma_i \in [0, 1]$ gilt.
  2. Die Vorschrift $\mathbf{X}_{k+1} = p(\mathbf{X}_k)$ in $5$ Schritten mit schnellen Matrizenmultiplikationen anwenden, sodass $p^{(5)}(\sigma_i) \approx 1$ gilt.

<!--en:4e979f50-->
::: {.slide title="Das abgestimmte Newton-Schulz-Polynom fünften Grades von Jordan et al."}

<!--en:e4128d7a-->
[Schnelle Annäherung in 5 Schritten statt exakter asymptotischer Konvergenz]{.kicker}

<!--en:0a704b16-->
- Das klassische kubische Polynom $p(x) = 1.5x - 0.5x^3$ konvergiert asymptotisch gegen $1$. Wegen seiner geringen Steigung im Ursprung ($p'(0) = 1.5$, also $1.5^5 \approx 7.6$) hebt es kleine Singulärwerte in $5$ Iterationen viel zu langsam an.

<!--en:1aa93dc7-->
- **Das abgestimmte Polynom fünften Grades von Muon** (Jordan et al., 2024) maximiert die Anfangssteigung $p'(0) = 3.4445$ (Verstärkung um $3.4445^5 \approx 485\times$):
  $$p(x) = 3.4445\,x - 4.7750\,x^3 + 2.0315\,x^5$$

<!--en:8b9c617b-->
- Nach $5$ Iterationen liegt jeder Singulärwert aus $[0.0015, 1]$ im Intervall **$[0.68, 1.13]$**.

<!--en:5438a9aa-->
::: {.slide title="Implementierung und Rechenaufwand der Newton-Schulz-Iteration"}

<!--en:ea34d214-->
[Schnelle Orthogonalisierung auf der GPU]{.kicker}

<!--en:bf01a058-->
- **Transponieren hoher Matrizen ($m > n$):** Für $m > n$ liefert die Transposition $\mathbf{X} \leftarrow \mathbf{M}^\top$ eine Gram-Matrix $\mathbf{A} = \mathbf{X}\mathbf{X}^\top$ mit der kleineren Dimension $\min(m,n) \times \min(m,n)$.

<!--en:1b2e0861-->
- **Geringer zusätzlicher Rechenaufwand bei großen Batches:**
  - Für eine $m \times n$-Matrix ($m \le n$) und eine Batchgröße von $B$ Token kosten $5$ Newton-Schulz-Schritte $5(4m^2 n + 2m^3)$ FLOPs. Vorwärts- und Rückwärtsdurchlauf benötigen zusammen $6mnB$ FLOPs.
  - Der Mehraufwand beträgt $\sim \frac{5m}{B}$: **$<1\%$** bei praxisüblichen großen Batches ($B \sim 5 \times 10^5$ Token) und $\sim 12\%$ bei `TinyLM` ($m = 128, B = 4{,}096$).

<!--en:40019407-->
::: {.slide title="Verdichtung des Singulärwertspektrums durch die Newton-Schulz-Iteration"}

<!--en:3aff906a-->
[Abflachung des Spektrums]{.kicker}

<!--en:8abffa1c-->
- Die Anwendung von `newton_schulz` auf eine $96 \times 64$-Matrix $\mathbf{G}$ mit gaußverteilten Einträgen und normierten Singulärwerten im Bereich $[0.02, 0.22]$ verdichtet das Spektrum rasch:
  - **$k = 1$:** Anhebung um bis zu $3.44\times$.
  - **$k = 3$:** Gesamtes Spektrum im Intervall $[0.68, 1.20]$.
  - **$k = 5$:** Abflachung auf das Intervall $[0.68, 1.13]$.

<!--en:549e8cc9-->
- **Dämpfung des Rauschens:** Jeder Singulärwert unter $0.15\%$ von $\|\mathbf{G}\|_F$ ($0.68 / 3.4445^5 \approx 0.0014$) bleibt klein, statt auf $1$ anzuwachsen.

<!--en:5b4296f8-->
::: {.slide title="Der vollständige Muon-Schritt und der RMS-Abgleich für die Einträge"}

<!--en:e14d8b60-->
[Abgleich mit der Lernrate von AdamW]{.kicker}

<!--en:e129c287-->
- **Muon-Schritt ($\beta = 0.95$, Liu et al., 2025):**
  $$\mathbf{W}_{t+1} = (1-\eta\lambda)\mathbf{W}_t - 0.2\eta\sqrt{\max(m,n)}\;\mathrm{NS}_5(\mathbf{G}_t + \beta\mathbf{M}_t)$$

<!--en:408df713-->
- **Warum vor $\mathrm{NS}_5$ mitteln?** Die Mittelung über $20$ Schritte erfasst beständige Singulärrichtungen vor der nichtlinearen Orthogonalisierung.

<!--en:8b31d973-->
- **Warum der Faktor $0.2\sqrt{\max(m, n)}$?** Wegen $\text{RMS}(\mathbf{U}\mathbf{V}^\top) = 1/\sqrt{\max(m,n)}$ setzt er den RMS-Wert der Schritteinträge auf **$0.2\,\eta$**. Das entspricht AdamW, sodass beide Optimierer dieselben Werte für $\eta$ und $\lambda$ verwenden können.

<!--en:757b8914-->
::: {.slide title="Implementierung von Muon von Grund auf"}

<!--en:a4fbb6b9-->
[Implementierung im Code]{.kicker}

<!--en:57287f91-->
`Muon` speichert **einen Momentum-Puffer (`buf`) pro Parameter**. Der Optimiererzustand ist damit halb so groß wie bei AdamW:

<!--en:184a948b-->
::: {.slide title="Aufteilung der Parameter zwischen Muon und AdamW"}

<!--en:493c4f4c-->
[Aufbau des hybriden Optimierers]{.kicker}

<!--en:8fcdee72-->
- Die Geometrie der Spektralnorm gilt speziell für dichte lineare Transformationen zwischen kontinuierlichen Aktivierungsvektoren:
  - **Muon ($8$ Tensoren, $393{,}216$ Parameter, $83.5\%$):** Alle internen 2D-Blockmatrizen (`qkv`, `proj`, `mlp1` und `mlp2` in beiden Transformer-Blöcken) mit entkoppeltem Gewichtszerfall $\lambda = 0.1$.
  - **AdamW ($22$ Tensoren, $77{,}568$ Parameter, $16.5\%$):** Einbettungstabellen für Token und Positionen, eindimensionale Bias- und LayerNorm-Skalierungsparameter sowie der Ausgabekopf `head` für das Vokabular ($\lambda = 0$).

<!--en:2183e6e8-->
- **42% weniger Speicher für den Optimiererzustand:** 1 Puffer für $83.5\%$ der Parameter und 2 Puffer für $16.5\%$ ergeben $0.835(1) + 0.165(2) = 1.165$ Puffer pro Parameter statt $2.0$. Das sind $58\%$ des Zustandsspeichers von AdamW.

<!--en:b3f612d1-->
::: {.slide title="Vergleich von Muon+AdamW und AdamW auf TinyLM"}

<!--en:6a8b4076-->
[Trainings- und Validierungskurven]{.kicker}

<!--en:6b52c6b0-->
- Lernratensuche für `AdamW` und die Kombination `Muon + AdamW` über $\eta \in \{10^{-3}, 3{\times}10^{-3}, 10^{-2}, 3{\times}10^{-2}, 10^{-1}\}$ ($\lambda = 0.1$ für versteckte Matrizen, $2{,}000$ Schritte mit identischen Anfangsgewichten):
  - **Schnelle Fortschritte zu Beginn:** Bei Schritt $500$ liegt der Validierungsverlust von Muon+AdamW um **$>0.10$ Nat** niedriger (etwa $1.7$–$1.75$ gegenüber $1.88$ Nat).
  - **Früheres Unterschreiten der Schwelle:** Die Schwelle von $1.60$ Nat unterschreitet der Validierungsverlust mindestens **$250$ Schritte früher** als bei AdamW.

<!--en:dc723930-->
::: {.slide title="Stabiles Training mit Muon+AdamW bei 3× so hoher Lernrate"}

<!--en:ad21732d-->
[Nutzbarer Lernratenbereich]{.kicker}

<!--en:9d986920-->
- **Lernratensuche für AdamW:**
  - Bei $\eta = 0.03$ ($3\times$ so hoch wie die obere Grenze des nutzbaren Bereichs, $\eta = 0.01$) stagniert AdamW oberhalb von $2.4$ Nat. Frühe Vorzeichenschritte vom Rang 1 haben eine $\sqrt{\min(m,n)} \approx 11\times$ so große Spektralnorm wie ein orthogonalisierter Schritt mit gleichem RMS-Wert der Einträge.

<!--en:921a2dde-->
- **Lernratensuche für Muon+AdamW:**
  - Die Beschränkung der Spektralnorm jedes Schritts für die versteckten Matrizen ermöglicht das Training bei $\eta = 0.03$ (Validierungsverlust etwa $1.67$). Die obere Grenze des nutzbaren Lernratenbereichs steigt um den Faktor $3\times$.

<!--en:8b35c6c5-->
::: {.slide title="Muon bei langem Pre-Training und QK-Clip"}

<!--en:e52359b6-->
[Besonderheiten bei langem Pre-Training]{.kicker}

<!--en:5cd76a59-->
- **Der Faktor, um den die Schrittzahl sinkt, nimmt mit der Modellgröße ab: von ≈1.4× bei 0.1 Mrd. auf ≈1.1× bei 1.2 Mrd. Parametern (Wen et al., 2025).** Kimi K2 (1 Billion Parameter, 2025) wurde mit Muon trainiert. Ein Vergleichslauf mit AdamW fehlt. Lange Läufe erfordern zwei Schutzmaßnahmen:

<!--en:43e4deb9-->
- **1. Entkoppelter Gewichtszerfall ($\lambda > 0$) ist erforderlich:** Da $\mathrm{NS}_5$ die Singulärwerte bei jedem Schritt auf $\sim 1$ normiert, wachsen die Gewichtsnormen $\|\mathbf{W}\|_F$ ohne die Schrumpfung um den Faktor $(1-\eta\lambda)$ unbegrenzt.

<!--en:8e631e30-->
- **2. Bilinearer QK-Clip:** In die Attention-Logits $\mathbf{q}^\top\mathbf{k} = \mathbf{x}^\top\mathbf{W}_Q^\top\mathbf{W}_K\mathbf{x}$ geht ein Produkt zweier Matrizen ein. Eine Neuskalierung von $\mathbf{W}_Q, \mathbf{W}_K$, sobald der maximale Logit den Wert $100$ übersteigt, verhindert die Sättigung der Softmax-Funktion.

<!--en:3ef29bd6-->
::: {.slide title="Verständnisfragen zu Spektralnorm, Polynom fünften Grades und RMS-Skalierung"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:8e74b271-->
Für jede Frage zum Aufbau von Muon den genauen Zahlenwert bestimmen:

<!--en:3b5d9f15-->
1. **RMS-Abgleich für die Einträge:** Bei `TinyLM` hat die erste MLP-Matrix die Form $\mathbf{W} \in \mathbb{R}^{512 \times 128}$. Welche Frobeniusnorm und welchen RMS-Wert der Einträge hat eine exakt orthogonale Matrix $\mathbf{U}\mathbf{V}^\top \in \mathbb{R}^{512 \times 128}$? Welcher RMS-Wert ergibt sich durch den Faktor $0.2\sqrt{\max(512, 128)}$?

<!--en:aab193ca-->
2. **Warum Einbettungen bei AdamW bleiben:** Ein Minibatch enthalte das Token $A$ insgesamt $500$ Mal und das Token $B$ einmal. Dann hat die Gradientenmatrix $\mathbf{G}$ der Einbettung zwei von null verschiedene Zeilen $\mathbf{g}_A \perp \mathbf{g}_B$ mit $\|\mathbf{g}_A\|_2 = 500\,\|\mathbf{g}_B\|_2$. Wie verändert die exakte Orthogonalisierung $\mathbf{U}\mathbf{V}^\top$ die beiden Zeilen, und wie behandelt AdamW sie?

<!--en:80584480-->
3. **Reichweite nach 3 und 5 Newton-Schulz-Schritten:** Das Polynom fünften Grades hat die Anfangssteigung $p'(0) = 3.4445$. Wie groß muss ein normierter Singulärwert $\sigma / \|\mathbf{G}\|_F$ sein, damit er $0.68$ in $3$ beziehungsweise $5$ Iterationen erreicht?

<!--en:de3e4419-->
::: {.slide title="Verständnisfragen zu Spektralnorm, Polynom fünften Grades und RMS-Skalierung (Lösungen)"}

<!--en:929b7671-2-->
[Verständnis prüfen]{.kicker}

<!--en:557b98b6-->
**Lösungen:**

<!--en:d475a1b7-->
1. $\mathbf{U}\mathbf{V}^\top$ hat $\min(512, 128) = 128$ Singulärwerte vom Betrag eins. Daher gilt $\|\mathbf{U}\mathbf{V}^\top\|_F = \sqrt{128}$ bei $512 \times 128 = 65{,}536$ Einträgen. Ihr RMS-Wert ist $\sqrt{\frac{128}{512 \times 128}} = \frac{1}{\sqrt{512}} \approx 0.0442$. Die Multiplikation mit $0.2\sqrt{512} \approx 4.525$ ergibt für die Einträge einen RMS-Wert von genau **$0.2$**.

<!--en:c5cbd933-->
2. Beide aktiven Zeilen von $\mathbf{U}\mathbf{V}^\top$ erhalten die $\ell_2$-Norm eins. Dadurch sind die Richtungen seltener und häufiger Token über das Vokabular hinweg gekoppelt. AdamW dagegen normalisiert jeden Zeileneintrag unabhängig anhand seines eigenen historischen exponentiellen gleitenden Mittels $\sqrt{\hat{v}_{t,ij}}$.

<!--en:b66c8e38-->
3. Drei Schritte verstärken kleine Werte höchstens um den Faktor $3.4445^3 \approx 40.9$. Erforderlich ist daher $\sigma/\|\mathbf{G}\|_F \ge 0.68/40.9 \approx$ **$1.66\%$**. Fünf Schritte verstärken sie um den Faktor $3.4445^5 \approx 485$ und erfassen damit noch Werte ab **$0.14\%$**.

<!--en:cdd0014a-->
::: {.slide title="Zusammenfassung zu Normkugeln, Newton-Schulz-Iteration und Muon"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:6e132752-->
| Konzept / Komponente | Mathematische Formel | Rechenaufwand / Speicherbedarf | Rolle im Deep Learning |

<!--en:cfebe371-->
| **Spektraler steilster Abstieg** | $\operatorname*{argmin}_{\|\Delta\mathbf{W}\|_2 \le \eta}\langle\mathbf{G}, \Delta\mathbf{W}\rangle = -\eta\,\mathbf{U}\mathbf{V}^\top$ | Setzt alle aktiven Singulärwerte von $\mathbf{G}$ auf $1$, Abnahme der Zielfunktion um $\eta\|\mathbf{G}\|_*$ | Begrenzt die maximale Änderung der Aktivierung gemäß $\|\Delta\mathbf{W}\mathbf{x}\|_2 / \|\mathbf{x}\|_2 \le \eta$ |

<!--en:d6207780-->
| **Newton-Schulz-Polynom fünften Grades** | $p(\mathbf{X}) = 3.4445\mathbf{X} - 4.7750\mathbf{A}\mathbf{X} + 2.0315\mathbf{A}^2\mathbf{X}$ | $5$ Iterationen aus Matrizenmultiplikationen heben Singulärwerte mit $\sigma/\|\mathbf{G}\|_F \ge 0.15\%$ ins Intervall $[0.68, 1.13]$ | Nähert $\mathbf{U}\mathbf{V}^\top$ auf GPUs mit $\sim 5m/B < 1\%$ zusätzlichen FLOPs relativ zum Trainingsaufwand an |

<!--en:6394cbe8-->
| **Muon-Schritt und Skalierung** | $\mathbf{W}_{t+1} = (1-\eta\lambda)\mathbf{W}_t - 0.2\eta\sqrt{\max(m,n)}\,\mathrm{NS}_5(\mathbf{G}_t+\beta\mathbf{M}_t)$ | $1$ Momentum-Puffer pro Matrix ($\beta = 0.95$), RMS-Wert der Einträge $\approx 0.2\eta$ | Gleiche Werte für $\eta$ und $\lambda$ wie bei AdamW, um $42\%$ geringerer Speicherbedarf für den Zustand |

<!--en:9496eb5e-->
| **Hybrider Optimierer Muon+AdamW** | Muon für versteckte 2D-Matrizen ($84\%$), AdamW für Einbettungen, Ausgabekopf und 1D-Parameter ($16\%$) | $3\times$ so hohe maximale stabile Lernrate bei `TinyLM`, Schrittzahl bei 0.1 Mrd. Parametern um den Faktor $\approx 1.4\times$ geringer, bei 1.2 Mrd. um den Faktor $\approx 1.1\times$ | Einsatz mit entkoppeltem Gewichtszerfall $\lambda > 0$ (bei extrem großen Modellen auch mit QK-Clip) |
