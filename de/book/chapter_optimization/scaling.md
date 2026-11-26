<!--en:5a3beed9-->
# Skalierung auf größere Modelle

<!--en:408dded7-->
Neuronale Netze werden im Kleinen entwickelt und im Großen eingesetzt. Beim Pre-Training eines Modells mit mehreren Milliarden Parametern ist eine Gittersuche über Lernraten zu teuer. In der Praxis stimmt man die Hyperparameter deshalb an einem kleinen **Stellvertretermodell** (engl. proxy model) derselben Architekturfamilie ab und überträgt sie auf das Zielmodell.

<!--en:4a3d2cb3-->
Deep-Learning-Frameworks verwenden standardmäßig die **Standardparametrisierung** (engl. standard parametrization, SP). Dabei werden die Gewichte jeder Schicht mit der Varianz $1/\text{fan-in}$ initialisiert, und alle Schichten verwenden dieselbe globale Lernrate $\eta$. Die optimale Lernrate *lässt sich nicht auf andere Modellbreiten übertragen*. Mit zunehmender Schichtbreite $n$ verändert ein einziger Adam-Schritt die Präaktivierungen versteckter Schichten um $O(\eta n)$. Denn die Vorzeichen der Updates in den $n$ Termen des Matrix-Vektor-Produkts $\Delta\mathbf{W}\,\mathbf{h}$ richten sich nach den Vorzeichen der eingehenden Aktivierungen $\mathbf{h}$. Deshalb sinkt die optimale Lernrate mit wachsender Modellbreite.

<!--en:c36434dd-->
Die **Maximal Update Parametrization** ($\mu\text{P}$, :citet:`Yang.Hu.2021,Yang.Hu.Babuschkin.ea.2022`) beseitigt diese Drift. Sie skaliert die Lernrate und den Ausgabefaktor jeder Schicht danach, wie deren Fan-in und Fan-out wachsen. So bleiben die Merkmalsänderungen in jeder versteckten Schicht von der Größenordnung $\Theta(1)$, wenn die Breite gegen unendlich geht ($n \to \infty$). Wir messen die Lernratendrift unter SP bei MLPs, deren Breite um einen Faktor $64\times$ variiert ($n = 128$ bis $8{,}192$). Wir leiten die Skalierungsregel für den ersten Schritt her, implementieren $\mu\text{P}$ und überprüfen die Implementierung mit einem **Koordinatentest** (engl. coordinate check) über einen einzigen Schritt :cite:`Yang.Hu.Babuschkin.ea.2022,Dey.Anthony.Hestness.2024`. Anschließend zeigen wir, wie sich die Lernrate ohne erneute Abstimmung auf andere Breiten übertragen lässt, und erweitern die Regeln auf Transformer, Optimierer auf Basis der Spektralnorm und die Netztiefe.

<!--en:bcccc463-->
## Lernrate und Breite bei Standardparametrisierung

<!--en:af4af4fc-->
### Lernratendrift in einer Familie unterschiedlich breiter Netze

<!--en:7589d0a5-->
Betrachten wir ein mehrschichtiges Perzeptron (engl. multilayer perceptron, MLP) mit ReLU und drei versteckten Schichten für Fashion-MNIST. Für die Breite $n$ dieser Schichten wählen wir Werte aus $\{128, 512, 2048, 8192\}$. Damit reicht die Parameteranzahl von $1.35 \times 10^5$ bis $1.41 \times 10^8$. Die Gewichtsmatrizen skalieren unterschiedlich mit $n$:

<!--en:1835b8eb-->
* **Eingabeschicht (`fc_in`, $784 \to n$):** Der Fan-in ist fest ($784$), der Fan-out wächst mit $n$.

<!--en:f8c39e60-->
* **Versteckte Schichten (`fc_h1`, `fc_h2`, $n \to n$):** Fan-in und Fan-out wachsen mit $n$.

<!--en:3875366a-->
* **Ausgabeschicht (`fc_out`, $n \to 10$):** Der Fan-in wächst mit $n$, der Fan-out ist fest ($10$).

<!--en:c63cee1e-->
Bei der Standardparametrisierung (SP) ziehen wir die Einträge jeder Gewichtsmatrix aus $\mathcal{N}(0, 1/\text{fan-in})$ und initialisieren die Bias-Terme mit null. Alle Parameter verwenden dieselbe Adam-Lernrate $\eta$. Zunächst bringen wir die $60{,}000$ Trainingsbilder von Fashion-MNIST in Vektorform, normalisieren sie und übertragen sie auf den Beschleuniger:

<!--en:7e2c9b92-->
Als Nächstes definieren wir `MLP(width)` mit den Schichtbreiten $784 \to n \to n \to n \to 10$. Wir initialisieren die Gewichte mit der Standardabweichung $\text{fan-in}^{-1/2}$ und verwenden für alle Parameter dieselbe Adam-Lernrate:

<!--en:f6247999-->
Wie hängt die optimale Lernrate von der Breite $n$ ab? Um das zu messen, trainiert `train_mlp` das Modell `arch(width)` über $400$ Adam-Schritte ($b = 512$) und gibt den mittleren Verlust der letzten $50$ Schritte zurück. Bei divergenten Läufen begrenzen wir diesen Wert auf $2.5$:

<!--en:e97b1233-->
Mit den Hilfsfunktionen `sweep`, `print_optimum` und `plot_sweep` werten wir `train_mlp` auf einem Gitter aus Breiten, Lernraten und Zufallsstartwerten aus. Durch Anpassung einer Geraden in doppelt logarithmischer Darstellung bestimmen wir die Steigung der optimalen Lernrate $\eta^\star(n)$ in Abhängigkeit von $n$. Außerdem zeichnen wir die über die Zufallsstartwerte gemittelten Verlustkurven:

<!--en:cf023828-->
Nun wenden wir `sweep` auf `MLP` an. Für die Breiten $n \in \{128, 512, 2048, 8192\}$ testen wir $10$ logarithmisch gleichmäßig verteilte Lernraten $\eta \in \{2^{-14}, \ldots, 2^{-5}\}$, jeweils mit denselben beiden Zufallsstartwerten:

<!--en:5b165635-->
Bei der Standardparametrisierung verschieben sich die U-förmigen Verlustkurven mit wachsender Breite stetig nach links. Die optimale Lernrate sinkt um den Faktor $16$–$32\times$, von $\sim 3.9 \times 10^{-3}$ bei Breite $128$ auf $\sim 1$–$2 \times 10^{-4}$ bei Breite $8{,}192$. Die Steigung in doppelt logarithmischer Darstellung beträgt $-0.7$ bis $-0.8$. Übertragen wir die für Breite $128$ optimale Lernrate direkt auf das Modell der Breite $8{,}192$, landen wir weit oben auf dem instabilen rechten Ast seiner Verlustkurve. Damit geht der Vorteil der um den Faktor $1{,}000\times$ höheren Parameteranzahl verloren.

<!--en:73940d49-->
### Drift der optimalen Lernrate beim ersten Adam-Schritt

<!--en:97b482bf-->
Warum muss $\eta^\star$ mit wachsender Breite $n$ sinken? Betrachten wir eine quadratische Gewichtsmatrix $\mathbf{W} \in \mathbb{R}^{n \times n}$ einer versteckten Schicht, die den Aktivierungsvektor $\mathbf{h} \in \mathbb{R}^n$ auf $\mathbf{z} = \mathbf{W}\mathbf{h}$ abbildet. Bei der Initialisierung ist $W_{ij} \sim \mathcal{N}(0, 1/n)$ unabhängig von $\mathbf{h}$. Daher hat jede Koordinate $z_i$ eine Varianz der Größenordnung $\Theta(1)$, unabhängig von $n$.

<!--en:f9c50bae-->
Für ein einzelnes Beispiel mit rückpropagiertem Ausgabegradienten $\boldsymbol{\delta} = \partial \ell / \partial \mathbf{z}$ ist der Gewichtsgradient das äußere Produkt $\mathbf{G} = \boldsymbol{\delta}\mathbf{h}^\top$. Im Schritt $1$ ergibt sich nach Korrektur der Verzerrung das Adam-Update $\Delta\mathbf{W} = -\eta\,\operatorname{sign}(\mathbf{G}) = -\eta\,\operatorname{sign}(\boldsymbol{\delta})\operatorname{sign}(\mathbf{h})^\top$. Für denselben Aktivierungsvektor $\mathbf{h}$ ändert sich die Präaktivierung $z_i$ somit um

<!--en:9c11ca1e-->
wobei $\overline{|h|} = \frac{1}{n}\sum_{j=1}^n |h_j| = \Theta(1)$ gilt. Da $\Delta W_{ij}$ mit $h_j$ korreliert, haben alle $n$ Summanden dasselbe Vorzeichen. Sie addieren sich kohärent ($O(n)$), während sich die Terme mit zufälligen Vorzeichen bei der Initialisierung teilweise aufheben ($O(\sqrt{n})$). Damit die Merkmalsänderung $(\Delta\mathbf{W}\,\mathbf{h})_i = \Theta(1)$ von der Breite $n$ unabhängig bleibt, muss die Lernrate einer versteckten Gewichtsmatrix mit Fan-in $n$ gemäß **$\eta \propto 1/n$** skalieren. In doppelt logarithmischer Darstellung entspricht das einer Steigung von $-1$. Die am gesamten MLP gemessene Steigung von $-0.7$ bis $-0.8$ spiegelt einen Kompromiss zwischen den versteckten $n \times n$-Matrizen einerseits und der Eingabeschicht mit festem Fan-in sowie den Bias-Termen andererseits wider.

<!--en:4f96f93f-->
## Maximal Update Parametrization ($\mu\text{P}$)

<!--en:566a00a4-->
### Skalierungsregeln für $\mu\text{P}$

<!--en:3045c227-->
Eine einzige globale Lernrate trägt dem unterschiedlichen Wachstum des Fan-in der Schichten nicht Rechnung. Die **Maximal Update Parametrization** ($\mu\text{P}$, :citet:`Yang.Hu.2021,Yang.Hu.Babuschkin.ea.2022`) skaliert jede Schicht relativ zu einer **Basisbreite** (engl. base width) $n_{\text{base}}$. Dazu dient der **Breitenfaktor** (engl. width multiplier) $m = n / n_{\text{base}}$ (:numref:`tab_mup-rules`):

<!--en:eeb982ac-->
1. **Eingabegewichte und eindimensionale Bias- und Normierungsparameter (fester Fan-in):** Die Anfangsvarianz $1/\text{fan-in}$ und die Lernrate $\eta$ bleiben unverändert.

<!--en:e7f61de5-->
2. **Versteckte Gewichtsmatrizen (Fan-in und Fan-out $\propto n$):** Die Anfangsvarianz bleibt $1/\text{fan-in} = 1/n$. Wir setzen die Adam-Lernrate auf **$\eta / m$**. Damit gilt bei jeder Breite $(\eta / m)\,n\,\overline{|h|} = \eta\,n_{\text{base}}\,\overline{|h|} = \Theta(1)$.

<!--en:7ed835f9-->
3. **Ausgabematrix (Fan-in $\propto n$, fester Fan-out):** Sowohl $\Delta\mathbf{W}_{\text{out}}\,\mathbf{h}$ als auch $\mathbf{W}_{\text{out}}\,\Delta\mathbf{h}$ tragen jeweils die Summe von $n$ korrelierten Termen zu den Ausgabelogits bei. Die Einträge von $\mathbf{W}_{\text{out}}$ müssen deshalb Beträge der Größenordnung $O(1/n)$ statt $O(1/\sqrt{n})$ haben. Wir setzen die unskalierte Varianz der Gewichtsinitialisierung auf $1/n_{\text{base}}$, multiplizieren die Ausgabelogits im Vorwärtsdurchlauf mit **$1/m$** und belassen die Adam-Lernrate bei $\eta$. Dadurch skalieren die effektiven Ausgabegewichte und ihre Updates wie $O(1/n)$, und die Logits ändern sich bei jeder Breite um $\Theta(1)$.

<!--en:d90f09c6-->
:Regeln der Maximal Update Parametrization ($\mu\text{P}$) für Adam bei Basisbreite $n_{\textrm{base}}$ und Breitenfaktor $m = n/n_{\textrm{base}}$.

<!--en:6387d4da-->
| Parametergruppe | Anfangsvarianz | Adam-Lernrate | Faktor im Vorwärtsdurchlauf |

<!--en:7b26e829-->
| Eingabegewichte und alle Bias-Terme | $1/\textrm{fan-in}$ (Bias-Terme $0$) | $\eta$ | $1$ |

<!--en:61e6392f-->
| Versteckte Gewichtsmatrizen ($n \times n$) | $1/\textrm{fan-in} = 1/n$ | $\eta / m$ | $1$ |

<!--en:eb55f245-->
| Ausgabematrix ($n \times d_{\text{out}}$) | $1/n_{\textrm{base}}$ | $\eta$ | $1/m$ |

<!--en:47f4466b-->
*(Mit `base_width=8192` stimmen `MuMLP` und `MLP` bei der größten Breite $n = 8{,}192$ überein. Das erlaubt den direkten Vergleich mit der vorherigen Lernratensuche unter SP. In der Praxis setzt man `base_width` auf die Breite des kleinen Stellvertretermodells und skaliert mit $m > 1$ auf das Zielmodell.)*

<!--en:f9ef91fe-->
### Koordinatentest und Übertragung der Lernrate

<!--en:a1745e33-->
Vor der Lernratensuche können wir eine Implementierung von $\mu\text{P}$ mit dem **Koordinatentest** :cite:`Yang.Hu.Babuschkin.ea.2022,Dey.Anthony.Hestness.2024` überprüfen. Für jede Breite $n \in \{128, 256, \ldots, 8192\}$ machen wir einen einzigen Adam-Schritt und messen für jede Schicht den mittleren Betrag der Aktivierungen $\overline{|h_\ell|}$ mit `activations` und `coord_check`:

<!--en:e4399e60-->
Wir führen `coord_check` für das standardparametrisierte `MLP` bei sieben Breiten $n \in \{128, 256, \ldots, 8192\}$ aus. Mit der für Breite $128$ optimalen Lernrate `lr_128` zeigt sich, wie stark die Aktivierungen mit der Breite anwachsen:

<!--en:f73ce608-->
Denselben Test mit einem Schritt führen wir mit `coord_check` für `MuMLP` bei der Lernrate `lr_base` der Basisbreite aus. Er bestätigt, dass unter $\mu\text{P}$ die Aktivierungen aller drei versteckten Schichten und die Ausgabelogits unabhängig von der Breite ihre Größenordnung beibehalten:

<!--en:f0b14c52-->
Unter SP (`MLP`) wachsen die Aktivierungen in Schicht $2$ wie $O(n)$. In Schicht $3$ verstärkt sich das Wachstum auf $O(n^2)$, und die Ausgabelogits explodieren mit $O(n^3)$ (Faktor $1{,}000\times$ zwischen Breite $128$ und Breite $8{,}192$ nach nur einem Schritt). Unter $\mu\text{P}$ (`MuMLP`) bleiben die Größenordnungen der Aktivierungen jeder Schicht und der Logits über alle sieben Breiten konstant.

<!--en:331675f0-->
Wiederholen wir die Lernratensuche mit $400$ Schritten unter $\mu\text{P}$, so sehen wir, dass die optimale Lernrate über den gesamten Breitenbereich (Faktor $64\times$) nahezu gleich bleibt:

<!--en:31e73122-->
Zum Schluss vergleichen wir den Trainingsverlust bei Breite $8{,}192$ für zwei Fälle. Im ersten übertragen wir die am schmalen Stellvertretermodell der Breite $128$ abgestimmte Lernrate. Im zweiten stimmen wir sie direkt bei Breite $8{,}192$ ab. Wir vergleichen beide Fälle sowohl unter SP als auch unter $\mu\text{P}$:

<!--en:27fd3c74-->
Unter $\mu\text{P}$ liegt die Steigung von $\eta^\star(n)$ in doppelt logarithmischer Darstellung nahe null, während sie unter SP bei $-0.7$ oder noch tiefer liegt. Über den gesamten Breitenbereich (Faktor $64\times$) verschiebt sich das Optimum um höchstens einen Gitterschritt. Übertragen wir die bei Breite $128$ abgestimmte Lernrate direkt auf Breite $8{,}192$, liegt der Verlust nur $\le 0.003$ Nat über dem Optimum, weit unter den Schwankungen zwischen Zufallsstartwerten. Unter SP beträgt der zusätzliche Verlust $0.10\text{--}0.15$ Nat.

<!--en:e827a369-->
## Erweiterungen auf Transformer und spektrale Optimierer

<!--en:69e119cb-->
### $\mu\text{P}$-Regeln für einen Transformer

<!--en:268db817-->
Für $\mu\text{P}$ in einem Transformer wie `TinyLM` (:numref:`subsec_tinylm`) mit Modelldimension $d$ und Breitenfaktor $m = d / d_{\text{base}}$ gilt dieselbe Einteilung nach Fan-in und Fan-out. Für die Attention-Logits kommt eine Regel hinzu (:numref:`tab_mup-transformer`). Da sich die erlernten Abfrage- und Schlüsselvektoren im Training aneinander ausrichten, wächst ihr Skalarprodukt $\mathbf{q}^\top\mathbf{k}$ über die Kopfdimension $d_k \propto d$ wie $O(d_k)$. Deshalb müssen wir die Attention-Scores mit $1/d_k$ skalieren. Das entspricht einem zusätzlichen Faktor $1/\sqrt{m}$ gegenüber der üblichen Skalierung mit $1/\sqrt{d_k}$ :cite:`Yang.Hu.Babuschkin.ea.2022`.

<!--en:dce81039-->
:Regeln der Maximal Update Parametrization ($\mu\text{P}$) für einen Transformer mit Adam ($m = d/d_{\textrm{base}}$).

<!--en:5544c29e-->
| Komponente | Grund für die Skalierung | $\mu\text{P}$-Regel |

<!--en:b2ca103d-->
| Einbettungen, 1D-Bias-Terme, Skalierungs- und Verschiebungsparameter von LayerNorm | Fester Fan-in (Tabellenzugriff oder elementweise Operation) | Lernrate $\eta$ |

<!--en:b7eead8a-->
| Interne Blockmatrizen (`qkv`, `proj`, `mlp1`, `mlp2`) | Fan-in und Fan-out $\propto d$ | Lernrate $\eta/m$, Anfangsvarianz $1/\text{fan-in}$ |

<!--en:bacec1e1-->
| Ausgabeschicht `head` | Fan-in $\propto d$, feste Vokabulargröße $V$ | Anfangsvarianz $1/d_{\text{base}}$, Logits $\times 1/m$, Lernrate $\eta$ |

<!--en:9233479a-->
| Attention-Scores aus Skalarprodukten | Inneres Produkt gleichgerichteter Vektoren der Dimension $d_k$ | Scores mit $1/d_k$ skalieren (zusätzlicher Faktor $1/\sqrt{m}$) |

<!--en:0594f77c-->
| Residualzweige bei ebenfalls wachsender Tiefe $L$ | Die $L$ Zweigausgaben addieren sich im Residualstrom | Jeden Zweig mit $L_{\text{base}}/L$ multiplizieren (:numref:`subsec_scaling-depth`) |

<!--en:4f055d08-->
### Spektrale Skalierungsbedingung und Muon

<!--en:b6a05429-->
Wie :citet:`Yang.Simon.Bernstein.2023` zeigen, folgen alle $\mu\text{P}$-Regeln aus einer einzigen Bedingung an die Operatornorm. Betrachten wir eine Schicht, die $\mathbf{h} \in \mathbb{R}^{d_{\text{in}}}$ mit $\|\mathbf{h}\|_2 = \Theta(\sqrt{d_{\text{in}}})$ auf $\mathbf{z} \in \mathbb{R}^{d_{\text{out}}}$ mit $\|\mathbf{z}\|_2 = \Theta(\sqrt{d_{\text{out}}})$ abbildet. Damit die Aktivierungen und ihre Änderungen eintragsweise von der Größenordnung $\Theta(1)$ bleiben, brauchen sowohl die Gewichtsmatrix $\mathbf{W}$ als auch ihr Update $\Delta\mathbf{W}$ die Spektralnorm

<!--en:fa248568-->
Für eine quadratische $n \times n$-Gewichtsmatrix einer versteckten Schicht verlangt :eqref:`eq_spectral_condition`, dass $\|\Delta\mathbf{W}\|_2 = \Theta(1)$ gilt. Bei Adam hat ein Vorzeichenupdate vom Rang $1$ die Spektralnorm $\|\Delta\mathbf{W}\|_2 = \Theta(\eta n)$. Das erfordert $\eta \propto 1/n$. Bei Muon (:numref:`sec_muon`) hat das orthogonalisierte Update $\mathrm{NS}_5(\cdot)$ konstruktionsbedingt die Spektralnorm eins. Skalieren wir es mit $\sqrt{d_{\text{out}}/d_{\text{in}}}$ :cite:`Bernstein.2025`, lässt sich die Lernrate ohne Korrekturfaktor $1/m$ direkt auf andere Breiten übertragen. Verwenden wir hingegen den auf AdamW abgestimmten Faktor $0.2\sqrt{\max(d_{\text{out}}, d_{\text{in}})}$ aus :numref:`sec_muon`, hat das Update die Spektralnorm $\Theta(\eta\sqrt{n})$. Die Lernrate muss daher gemäß $\eta \propto 1/\sqrt{n}$ skalieren (Aufgabe :exref:`ex_scaling-spectral`).

<!--en:868a1b58-->
### Tiefe, Gewichtszerfall und endliche Modellgröße

<!--en:9e0d2250-->
Bislang haben wir allein die Breite variiert. In der Praxis kommen bei der Skalierung auch Schichten hinzu. Bei AdamW muss außerdem der Gewichtszerfall (engl. weight decay) jeder Änderung der Lernrate folgen. Für beides brauchen wir eine eigene Regel.

<!--en:dba6e1cd-->
Betrachten wir zunächst die Tiefe. In einem Netz mit $L$ Residualblöcken ist der Residualstrom die Summe von $L$ Zweigausgaben. Trägt jeder Zweig $\Theta(1)$ bei, wächst die Summe mit $L$. Auch die Wirkung eines einzelnen Optimierungsschritts wächst, denn jeder Block verändert die Ausgabe. Abhilfe schafft ein Faktor nach demselben Prinzip wie bei den Logits: Wir multiplizieren jeden Residualzweig mit $(L_{\text{base}}/L)^{\alpha}$. Dafür wurden zwei Exponenten vorgeschlagen. Bei $\alpha = 1/2$ addieren sich die Zweigausgaben wie unabhängige Zufallsterme, wie sie bei der Initialisierung vorliegen :cite:`Bordelon.Noci.Li.ea.2024,Yang.Yu.Zhu.ea.2024`. Bei $\alpha = 1$ addieren sie sich wie gleichgerichtete Terme, wie sie entstehen, sobald das Training die Ausgaben miteinander korreliert. Aus demselben Grund ergab sich für die Ausgabematrix der Faktor $1/n$ statt $1/\sqrt{n}$. Nach :citet:`Dey.Zhang.Noci.ea.2025` lernt in Transformern nur bei $\alpha = 1$ jeder Block auch mit wachsender Tiefe $L$ weiterhin Merkmale. Die Lernrate lässt sich dann ebenso auf andere Tiefen übertragen wie auf andere Breiten.

<!--en:b1564ca7-->
Betrachten wir nun den Gewichtszerfall. Bei AdamW bestimmt die Zeitskala des Gewichtszerfalls $\tau = 1/(\eta\lambda)$, wie schnell sich die Norm der Gewichte ihrem Gleichgewichtswert nähert (:numref:`sec_adam`). Teilen wir die Lernrate einer Schicht durch $m$ und lassen $\lambda$ unverändert, wächst die Zeitskala $\tau$ dieser Schicht auf das $m$-Fache. Das breite Modell wird dann anders regularisiert als das kleine Modell, an dem wir die Hyperparameter abgestimmt haben. Eine konsistente Wahl besteht darin, $\tau$ konstant zu halten: Wir multiplizieren $\lambda$ überall dort mit $m$, wo wir $\eta$ durch $m$ teilen :cite:`Wang.Aitchison.2025`.

<!--en:05fb4210-->
Alle diese Regeln gelten exakt erst im Grenzfall unendlicher Modellgröße. Bei den in der Praxis verwendeten Breiten verschiebt sich das Optimum noch ein wenig, in unserer Lernratensuche um einen Gitterschritt. Die übertragene Lernrate eignet sich deshalb am besten als Mittelpunkt einer eng begrenzten Lernratensuche bei der Zielgröße. Ersetzen kann sie diese Suche nicht.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:21fa2891-->
Bei der Standardparametrisierung (SP) ändert ein einziger Adam-Schritt die Ausgabe einer versteckten Schicht mit Fan-in $n$ um $\Theta(\eta n)$. Dadurch sinkt die optimale Lernrate $\eta^\star(n)$ mit wachsender Netzbreite. Die **Maximal Update Parametrization ($\mu\text{P}$)** skaliert die Lernraten der versteckten Gewichtsmatrizen mit $1/m$ (wobei $m = n/n_{\text{base}}$ gilt). Für die Ausgabeschicht setzt sie die Anfangsvarianz auf $1/n_{\text{base}}$ und skaliert die Logits mit $1/m$. In Transformern skaliert sie die Attention-Scores mit $1/d_k$. So bleiben Merkmalsänderungen von der Größenordnung $\Theta(1)$, und die Lernrate lässt sich ohne erneute Abstimmung auf andere Breiten übertragen. Mit einem **Koordinatentest** über einen einzigen Optimierungsschritt prüfen wir an einer geometrischen Folge von Breiten, ob die Beträge der Schichtaktivierungen über die Breiten hinweg konstant bleiben. Übertragen wir Hyperparameter auf Modelle, die auch tiefer werden, skalieren wir die Residualzweige zusätzlich mit $L_{\text{base}}/L$. Die Zeitskala des Gewichtszerfalls von AdamW, $\tau = 1/(\eta\lambda)$, halten wir bei Änderungen der schichtspezifischen Lernraten konstant. Damit erhalten wir einen asymptotischen Bezugspunkt für eine eng begrenzte Lernratensuche bei der Zielgröße.

<!--en:2294243a-->
## Übungen

<!--en:eb385a8a-->
1. [code] **Fehler in $\mu\text{P}$-Regeln mit dem Koordinatentest erkennen.** Setzen Sie `base_width=128` in `MuMLP` und testen Sie mit `coord_check` bei `lr_128` drei absichtlich eingebaute Fehler einzeln: (a) Lassen Sie den Logitfaktor $1/m$ weg. (b) Verzichten Sie auf die Skalierung der Lernrate versteckter Schichten auf $\eta/m$. (c) Setzen Sie die Lernrate versteckter Schichten auf $\eta/m^2$. Geben Sie an, welche Fehler `coord_check` sofort erkennt. Erklären Sie, warum bei zu stark verkleinerten Updates ($\eta/m^2$) erst eine Lernratensuche mit Verlustmessung über mehrere Schritte den Fehler sichtbar macht.
   :exlabel:`ex_scaling-breaking-mup`

<!--en:2edbcc08-->
1. [code] **Einfluss der Basisbreite.** Wiederholen Sie die Lernratensuche für $\mu\text{P}$ mit `base_width=128` bei den Breiten $\{128, 512, 2048, 8192\}$. Prüfen Sie, ob die Steigung von $\eta^\star(n)$ in doppelt logarithmischer Darstellung nahe null bleibt. Ermitteln Sie, welche konstante Relation den Verlustunterschied bei Breite $8{,}192$ erklärt: das Verhältnis der Lernraten von Eingabeschicht und versteckten Schichten oder die anfängliche Größenordnung der Logits.
   :exlabel:`ex_scaling-base`

<!--en:539b6328-->
1. **Spektralnormbedingung für verschiedene Schichten und Optimierer.** ● Prüfen Sie mit :eqref:`eq_spectral_condition`, ob die $\mu\text{P}$-Regeln aus :numref:`tab_mup-rules` in der Eingabeschicht, den versteckten Schichten und der Ausgabeschicht Updates mit Spektralnorm $\Theta(\sqrt{d_{\text{out}}/d_{\text{in}}})$ ergeben, wenn der Adam-Schritt Rang $1$ hat. Leiten Sie her, wie $\eta$ mit der Breite skalieren muss, wenn Sie Muon (:numref:`sec_muon`) auf die vier Blockmatrizen von `TinyLM` anwenden.
   :exlabel:`ex_scaling-spectral`

<!--en:8547f990-->
1. [code] **$\mu\text{P}$ und den Koordinatentest für `TinyLM` implementieren.** ● Wenden Sie :numref:`tab_mup-transformer` auf `d2l.TinyLM` mit `d_model` $\in \{64, 128, 256, 512, 1024\}$ an (`num_heads=2`, `base_width=64`). Prüfen Sie mit einem Koordinatentest nach einem Schritt bei $\eta = 3 \times 10^{-3}$, ob die Beträge der Ausgaben aller Teilschichten und der Logits über alle Breiten hinweg konstant bleiben.
   :exlabel:`ex_scaling-tinylm`

<!--en:599e7d9e-->
[Diskussion](https://d2l.discourse.group/t/36766)

<!--en:0d4ec1de-->
Parametrisierung nach Breite und μP-Skalierung<br>**Drift bei Standardparametrisierung · kohärente Summe im ersten Adam-Schritt · Maximal Update Parametrization (μP) · Koordinatentest · spektrale Skalierung**

<!--en:a9b4c0e7-->
::: {.slide title="Warum Lernraten schmaler Stellvertretermodelle bei breiten Netzen versagen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:355549f0-->
- **Drift bei Standardparametrisierung (SP):** Wer Adam an einem schmalen Stellvertretermodell abstimmt ($n = 128$, $\eta^\star \approx 3.9 \times 10^{-3}$) und die Lernrate auf ein um den Faktor $64\times$ breiteres Netz überträgt ($n = 8{,}192$), verliert den gesamten Vorteil der um den Faktor $1{,}000\times$ höheren Parameteranzahl (Verlust $\approx 0.33$ gegenüber $\approx 0.22$ nach erneuter Abstimmung). Denn $\eta^\star$ ist um den Faktor $16\times$ gesunken.

<!--en:cb5655ed-->
- **Maximal Update Parametrization ($\mu\text{P}$, Yang et al., 2022):** Skaliert die Lernraten der Schichten und die Ausgabefaktoren in Abhängigkeit von $m = n/n_{\text{base}}$. So lässt sich $\eta^\star$ ohne erneute Abstimmung auf andere Breiten übertragen.

<!--en:a92db96d-->
::: {.slide title="Wachstum von Fan-in und Fan-out mit der Netzbreite"}

<!--en:66082fdf-->
[Schichttypen]{.kicker}

<!--en:5fa92e00-->
Unser ReLU-MLP mit 3 versteckten Schichten hat die Architektur $784 \to n \to n \to n \to 10$. Für die Breiten $n \in \{128, 512, 2048, 8192\}$ lassen sich seine vier linearen Schichten nach ihrem Skalierungsverhalten in drei Klassen einteilen:

<!--en:06963a03-->
- **1. Eingabeschicht (`fc_in`, $784 \to n$):** Der Fan-in ist fest ($784$). Nur der Fan-out wächst mit $n$.

<!--en:c0cdd7e7-->
- **2. Versteckte Schichten (`fc_h1`, `fc_h2`, $n \to n$):** **Fan-in und Fan-out wachsen mit $n$**.

<!--en:ca2b2b6e-->
- **3. Ausgabeschicht (`fc_out`, $n \to 10$):** **Der Fan-in wächst mit $n$**. Der Fan-out ist fest ($10$).

<!--en:9e3df0b0-->
- **Standardparametrisierung (SP):** Die Einträge jeder Gewichtsmatrix stammen aus $\mathcal{N}(0, 1/\text{fan-in})$, die Bias-Terme starten bei $0$. Alle Schichten verwenden dieselbe globale Adam-Lernrate $\eta$.

<!--en:98d8af54-->
::: {.slide title="Unter SP sinkt die optimale Lernrate um den Faktor 16–32×"}

<!--en:fac34e8c-->
[Messung der Drift unter SP]{.kicker}

<!--en:efdd57f4-->
- Wir testen Adam für $10$ logarithmisch gleichmäßig verteilte Lernraten $\eta \in \{2^{-14}, \ldots, 2^{-5}\}$ über $400$ Schritte ($b = 512$, $2$ Zufallsstartwerte) bei den Breiten $n \in \{128, 512, 2048, 8192\}$:
  - Wächst die Breite $n$ um den Faktor $64\times$ ($128 \to 8{,}192$), verschieben sich die U-förmigen Verlustkurven stetig nach links.
  - Die optimale Lernrate sinkt um den Faktor **$16\text{--}32\times$**. Die Steigung in doppelt logarithmischer Darstellung beträgt **$-0.7$ bis $-0.8$**.
  - Bei Breite $8{,}192$ liegt das Optimum für Breite $128$ weit oben auf dem instabilen rechten Kurvenast.

<!--en:78770f42-->
::: {.slide title="Warum die Initialisierung Θ(1) ergibt und der erste Adam-Schritt Θ(ηn)"}

<!--en:dee534f1-->
[Kohärente und inkohärente Summen]{.kicker}

<!--en:405a434c-->
- **Bei der Initialisierung (unkorrelierte Zufallsvorzeichen $\to \Theta(1)$):** $W_{ij} \sim \mathcal{N}(0, 1/n)$ ist unabhängig von $h_j = \Theta(1)$. Daher addieren sich für $z_i = \sum_{j=1}^n W_{ij}h_j$ die Varianzen der Summanden zu $\operatorname{Var}(z_i) = \Theta(1)$.

<!--en:668c78db-->
- **Nach 1 Adam-Schritt (gleiche Vorzeichen in der Summe $\to \Theta(\eta n)$):** Bei $\Delta\mathbf{W} = -\eta\operatorname{sign}(\boldsymbol{\delta})\operatorname{sign}(\mathbf{h})^\top$ richten sich die Vorzeichen nach $\mathbf{h}$:
  $$(\Delta\mathbf{W}\,\mathbf{h})_i = -\eta\,\operatorname{sign}(\delta_i)\sum_{j=1}^n |h_j| = -\eta\,\operatorname{sign}(\delta_i)\,n\,\overline{|h|}$$

<!--en:abc1acbb-->
::: {.slide title="Fan-in-Gesetz: Ein Adam-Schritt verschiebt die Ausgaben um Θ(ηn)"}

<!--en:f63f6636-->
[Skalierung in Zahlen]{.kicker}

<!--en:c9a3266a-->
Alle $n$ Terme $|h_j| > 0$ gehen mit demselben Vorzeichen in die Summe ein. Ein einziger Adam-Schritt ändert daher die Präaktivierung einer versteckten Einheit um $\Theta(\eta n)$ bei einer ursprünglichen Größenordnung von $\Theta(1)$:

<!--en:d1d6637e-->
| Breite $n$ der versteckten Schichten | Fan-in von `fc_h1`, `fc_h2`, `fc_out` | Merkmalsänderung $\eta n$ im ersten Schritt bei $\eta = 2^{-8} \approx 0.0039$ | Verhalten bei festem $\eta = 0.0039$ |

<!--en:6babbe25-->
| **$n = 128$** | $128$ | $0.50\times$ Aktivierungsgröße | Stabile Verfeinerung der Merkmale |

<!--en:7dde032b-->
| **$n = 512$** | $512$ | $2.00\times$ Aktivierungsgröße | Mäßig großer Schritt |

<!--en:7433560c-->
| **$n = 2{,}048$** | $2{,}048$ | $8.00\times$ Aktivierungsgröße | Schritt überschießt die anfängliche Darstellung |

<!--en:652c441d-->
| **$n = 8{,}192$** | $8{,}192$ | **$32.0\times$ Aktivierungsgröße** | **Aktivierungen der Schicht explodieren** |

<!--en:c70c90f7-->
**Warum die Lernraten versteckter Schichten wie $1/n$ skalieren müssen:** Damit $(\Delta\mathbf{W}\,\mathbf{h})_i = \Theta(1)$ unabhängig von der Breite gilt, muss die Adam-Lernrate jeder versteckten Gewichtsmatrix mit Fan-in $n$ gemäß **$\eta_{\text{hidden}} \propto 1/n$** skalieren (Steigung $-1$ in doppelt logarithmischer Darstellung).

<!--en:8fd9dce5-->
::: {.slide title="Regeln der Maximal Update Parametrization (μP) für Adam"}

<!--en:9d201f65-->
[Drei Regeln nach Schichttyp]{.kicker}

<!--en:07670065-->
- Bezogen auf die Basisbreite $n_{\text{base}}$ ergibt sich der Breitenfaktor $m = n / n_{\text{base}}$. Mit $\mu\text{P}$ (Yang et al., 2022) bleiben Aktivierungen und Updates jeder Schicht von der Größenordnung $\Theta(1)$ für $m \to \infty$:
  - **1. Eingabegewichte und 1D-Bias-Terme:** Anfangsvarianz $1/\text{fan-in}$, Adam-Lernrate $\eta$.
  - **2. Versteckte Gewichtsmatrizen ($n \times n$):** Anfangsvarianz $1/n$, skalierte Adam-Lernrate **$\eta / m$**.
  - **3. Ausgabeschicht ($n \to d_{\text{out}}$):** Anfangsvarianz **$1/n_{\text{base}}$**, Adam-Lernrate $\eta$, Faktor im Vorwärtsdurchlauf **$1/m$**.

<!--en:d8939ca9-->
::: {.slide title="Mittelung in der Ausgabeschicht (1/n) und die Rolle der Basisbreite"}

<!--en:4d5ce1ee-->
[Ausgabeskalierung und Bezugspunkt]{.kicker}

<!--en:536ba6fc-->
- **1. Beitrag des Updates der Ausgabegewichte $\frac{1}{m}\Delta\mathbf{W}_{\text{out}}\,\mathbf{h}$:** Hier addieren sich $n$ Terme mit gleichem Vorzeichen. Division der Logits durch $m = n/n_{\text{base}}$ hält die Logitänderung in der Größenordnung $\Theta(1)$, wenn die Lernrate $\eta$ beträgt.

<!--en:1dcaade6-->
- **2. Beitrag der eingehenden Merkmalsänderung $\frac{1}{m}\mathbf{W}_{\text{out}}\,\Delta\mathbf{h}$:** Durch die Ausrichtung von $\Delta\mathbf{h}$ addieren sich $n$ Terme der Größe $|\frac{1}{m}W_{\text{out},ij}|$. Bei unskalierter Anfangsvarianz $1/n_{\text{base}}$ sind die effektiven Gewichte von der Größenordnung $O(1/n)$. So bleibt die Summe von der Größenordnung $\Theta(1)$.

<!--en:48fe6108-->
- **Basisbreite $n_{\text{base}}$:** Bei $n = n_{\text{base}}$ ($m = 1$) stimmt $\mu\text{P}$ genau mit SP überein.

<!--en:b975586f-->
::: {.slide title="μP in 15 Codezeilen"}

<!--en:7387d4d7-->
[Unterklasse des standardparametrisierten Modells]{.kicker}

<!--en:e89a0a70-->
- Mit `base_width=8192` ($m = 1$ bei $n = 8{,}192$) stimmt `MuMLP` beim breitesten Netz mit `MLP` überein. Beide Lernratensuchen haben somit exakt denselben Bezugspunkt bei $n = 8{,}192$:
  - **Regel 1 (`__init__`):** Die Gewichte von `fc_out` (Anfangsstandardabweichung $1/\sqrt{n}$) mit $\sqrt{m} = \sqrt{n / n_{\text{base}}}$ multiplizieren. Damit ergibt sich die Varianz $\frac{1}{n} \cdot \frac{n}{n_{\text{base}}} = \frac{1}{n_{\text{base}}}$.
  - **Regel 2 (`forward`):** Die Eingabemerkmale von `fc_out` durch $m$ teilen.
  - **Regel 3 (`configure_adam`):** Die Lernrate der versteckten Gewichtsmatrizen auf `lr / self.m` setzen.

<!--en:0cab36e2-->
::: {.slide title="Koordinatentest nach einem Schritt: Unter SP wachsen die Logits wie O(n³)"}

<!--en:52ccdc68-->
[Prüfung der Skalierung vor dem Training]{.kicker}

<!--en:18cd5ae7-->
- Beim **Koordinatentest** (Yang et al., 2022) machen wir **1 Adam-Schritt** mit einem einzigen Minibatch bei den Breiten $n \in \{128, 256, \ldots, 8192\}$. Für jede Schicht zeichnen wir den mittleren Betrag der Aktivierungen $\overline{|h_\ell|}$ auf.

<!--en:017ace6f-->
- **Bei Standardparametrisierung (`MLP`):**
  - **Schicht 1 (`fc_in`, Fan-in $784$):** Flache Kurve über alle Breiten ($\Theta(1)$).
  - **Schicht 2 (`fc_h1`, Fan-in $n$):** Lineares Wachstum mit der Breite gemäß **$O(n)$** (Steigung $+1$).
  - **Schicht 3 (`fc_h2`):** Das eigene Update der Größenordnung $O(n)$ verstärkt die bereits stark gewachsene Ausgabe von Schicht 2. Insgesamt ergibt sich **$O(n^2)$** (Steigung $+2$).
  - **Ausgabelogits (`fc_out`):** Anstieg um den Faktor $1{,}000\times$ über den Breitenbereich gemäß **$O(n^3)$** (Steigung $+3$).

<!--en:9bd53b1e-->
::: {.slide title="Unter μP verläuft die Kurve jeder Schicht im Koordinatentest flach"}

<!--en:286aad5d-->
[Breitenunabhängige Merkmalsänderungen]{.kicker}

<!--en:9e229c1a-->
- Derselbe Koordinatentest nach 1 Schritt ergibt für `MuMLP` bei den Breiten $n \in \{128, 256, \ldots, 8192\}$ **vier völlig flache Kurven** ($\Theta(1)$ für jede Schicht und jede Breite).

<!--en:f71d1e81-->
- **Unterschiedliche Aussagekraft steigender und flacher Kurven:**
  - Steigende Kurven **belegen** sofort einen Fehler in der Implementierung von $\mu\text{P}$ (z. B. fehlender Logitfaktor $1/m$ oder fehlende Skalierung der Lernrate versteckter Schichten auf $\eta/m$).
  - Flache Kurven schließen *explodierende* Updates ($\gg \Theta(1)$) aus. Auch eine zu stark verkleinerte Lernrate (etwa $\eta/m^2$) ergibt nach 1 Schritt eine flache Kurve, weil sich die Gewichte gegenüber der Initialisierung kaum ändern.

<!--en:3ebfccd2-->
::: {.slide title="Übertragung der Lernrate auf andere Breiten ohne erneute Abstimmung unter μP"}

<!--en:bafccee3-->
[Übereinstimmende Optima über einen Breitenfaktor von 64×]{.kicker}

<!--en:03e48a89-->
- Wir wiederholen die Lernratensuche mit $400$ Schritten für $n \in \{128, 512, 2048, 8192\}$ unter $\mu\text{P}$:
  - Die U-förmigen Kurven **liegen nun mit ihren Minima senkrecht übereinander**, und zwar über den gesamten Breitenbereich (Faktor $64\times$, 135 000 bis 141 Mio. Parameter).
  - Die Steigung von $\eta^\star(n)$ in doppelt logarithmischer Darstellung ändert sich von etwa $-0.7$ unter SP auf **nahezu $0.0$** (das Optimum liegt bei jeder Breite bei $2.4 \times 10^{-4}$ oder höchstens einen Gitterschritt davon entfernt).
  - Bei exakt derselben Lernrate erzielen breitere Netze einen strikt geringeren Trainingsverlust.

<!--en:857d3391-->
::: {.slide title="Zusätzlicher Verlust bei der Übertragung unter SP und μP"}

<!--en:a5e071c9-->
[Kosten der Übertragung vom Stellvertretermodell auf das Zielmodell]{.kicker}

<!--en:4c8de7a4-->
- Wie viel höher ist der Verlust bei Breite $8{,}192$, wenn wir die Lernrate vom um den Faktor $1{,}000\times$ kleineren Stellvertretermodell der Breite $128$ übernehmen, statt sie bei Breite $8{,}192$ neu abzustimmen?
  - **Unter SP:** **$0.09\text{--}0.15$ Nat** zusätzlicher Verlust ($\approx 0.33$ gegenüber $\approx 0.22$), kein Vorteil gegenüber dem schmalen Stellvertretermodell der Breite $128$.
  - **Unter $\mu\text{P}$:** **$\le 0.003$ Nat** zusätzlicher Verlust (in beiden Fällen $\approx 0.22$), von den Schwankungen zwischen Zufallsstartwerten nicht zu unterscheiden.

<!--en:5a39ad1e-->
- Unter $\mu\text{P}$ steigen die Verlustkurven schmalerer Stellvertretermodelle rechts von $\eta^\star$ früher an. Dort erreicht die Schrittweite $\eta/m$ die anfängliche Gewichtsgröße $1/\sqrt{n}$ bei $\eta = \sqrt{n}/n_{\text{base}}$. Eine auf dem schmalen Modell stabile Lernrate ist also auch auf dem breiten Zielmodell stabil.

<!--en:73610737-->
::: {.slide title="μP für Transformer: Die 1/d_k-Regel für Attention"}

<!--en:0ad718a6-->
[Parametrisierung von Transformern]{.kicker}

<!--en:b0391a2b-->
- **Einbettungen, eindimensionale Normierungsparameter und Bias-Terme (Fan-in $= 1$):** Standardinitialisierung, volle AdamW-Lernrate $\eta$.

<!--en:3d100eed-->
- **Interne 2D-Matrizen (`qkv`, `proj`, `mlp`):** Anfangsvarianz $1/\text{fan-in}$, skalierte AdamW-Lernrate **$\eta / m$** ($m = d/d_{\text{base}}$).

<!--en:ad107758-->
- **Ausgabeschicht `head` ($d \to V$):** Anfangsvarianz $1/d_{\text{base}}$, Logits im Vorwärtsdurchlauf **$\times 1/m$**, Lernrate $\eta$.

<!--en:182b8649-->
- **Attention-Scores ($\mathbf{q}^\top\mathbf{k}$ über $d_k \propto d$):** Skalierung mit **$1/d_k$** statt mit $1/\sqrt{d_k}$, weil die ausgerichteten Updates $\Delta\mathbf{q}, \Delta\mathbf{k}$ eine Summe aus $d_k$ kohärenten Produkten ergeben ($\Theta(d_k)$).

<!--en:95cd604b-->
::: {.slide title="Spektralbedingung: ||ΔW||₂ = Θ(√(d_out/d_in))"}

<!--en:578f5191-->
[Gemeinsame Regel für die Operatornorm]{.kicker}

<!--en:7fa0410b-->
- **Allgemeine Spektralbedingung (Yang, Simon und Bernstein, 2023):** Damit die Aktivierungen $\mathbf{h} \in \mathbb{R}^{d_{\text{in}}}$ und $\mathbf{z} = \mathbf{W}\mathbf{h} \in \mathbb{R}^{d_{\text{out}}}$ pro Eintrag von der Größenordnung $\Theta(1)$ bleiben, muss gelten:
  $$\|\mathbf{W}\|_2 = \Theta\!\left(\sqrt{\frac{d_{\text{out}}}{d_{\text{in}}}}\right), \quad \|\Delta\mathbf{W}\|_2 = \Theta\!\left(\sqrt{\frac{d_{\text{out}}}{d_{\text{in}}}}\right)$$

<!--en:f5699c24-->
- **Der erste Adam-Schritt ist ein äußeres Produkt zweier Vorzeichenvektoren und hat Rang 1:** $\|\Delta\mathbf{W}\|_2 = \eta\|\mathbf{s}\|_2\|\mathbf{t}\|_2 = \Theta(\eta\sqrt{d_{\text{out}}d_{\text{in}}})$. Dies entspricht genau dann $\Theta(\sqrt{d_{\text{out}}/d_{\text{in}}})$, wenn **$\eta \propto 1/d_{\text{in}}$** gilt.

<!--en:ee382d10-->
::: {.slide title="Skalierung mit der Breite bei spektralen Optimierern (Muon und AdamW)"}

<!--en:3afb9cbc-->
[Vergleich der Spektralnormen der Updates]{.kicker}

<!--en:7832438a-->
- Für eine versteckte $n \times n$-Gewichtsmatrix bestimmt die Bedingung $\|\Delta\mathbf{W}\|_2 = \Theta(\sqrt{n/n}) = \Theta(1)$, wie die Lernrate $\eta$ mit der Breite $n$ skaliert:
  - **AdamW:** $\|\Delta\mathbf{W}\|_2 = \Theta(\eta n) \implies$ **$\eta \propto 1/n$** ($\eta/m$).
  - **Muon ($0.2\sqrt{n}\,\mathrm{NS}_5$):** $\|\Delta\mathbf{W}\|_2 = \Theta(\eta\sqrt{n}) \implies$ **$\eta \propto 1/\sqrt{n}$** ($\eta/\sqrt{m}$).
  - **Spektral skaliertes Muon ($\sqrt{d_{\text{out}}/d_{\text{in}}}\,\mathrm{NS}_5$):** $\|\Delta\mathbf{W}\|_2 = \Theta(\eta) \implies$ **$\eta = \Theta(1)$** (Übertragung ohne den Faktor $1/m$).

<!--en:72d1c9ec-->
::: {.slide title="Verständnisfragen zu kohärenten Summen, Koordinatentests und μP"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:98667bb9-->
Prüfen Sie Ihr Verständnis der Parametrisierung nach Breite anhand dreier konkreter Aufgaben:

<!--en:c12733f2-->
1. **Verdopplung der Breite unter SP:** Sie verdoppeln die Breite $n$ der versteckten Schichten unseres MLP bei fester Adam-Lernrate $\eta$. Um welchen Faktor wächst dadurch die Ausgabeänderung im ersten Adam-Schritt für (a) die Eingabematrix `fc_in` ($784 \to n$), (b) die versteckte Gewichtsmatrix `fc_h1` ($n \to n$) und (c) die Ausgabematrix `fc_out` ($n \to 10$)?

<!--en:a870f9d8-->
2. **Fehler im Koordinatentest erkennen:** Sie implementieren $\mu\text{P}$ mit $m = n/n_{\text{base}} \ge 1$ und setzen die Lernrate der versteckten Gewichtsmatrizen korrekt auf $\eta/m$. Sie vergessen jedoch, die Ausgabelogits durch $m$ zu teilen. Wie sehen die 4 Kurven des Koordinatentests nach 1 Schritt aus?

<!--en:557b98b6-->
**Antworten:**

<!--en:b57ef8f1-->
1. Schritt 1 ändert die Schichtausgabe um $\Theta(\eta \cdot \text{fan-in})$: (a) `fc_in` hat den festen Fan-in $784$ $\to$ **$1\times$ (unverändert)**. (b) `fc_h1` und (c) `fc_out` haben beide den Fan-in $n$ $\to$ **$2\times$ so große Änderung**.

<!--en:c317d270-->
2. Für die Schichten 1, 2 und 3 (`h1`, `h2`, `h3`) verlaufen die Kurven **völlig flach ($\Theta(1)$)**. Die **Kurve der Ausgabelogits wächst linear mit der Breite gemäß $O(n)$ (Steigung $+1$)**, denn $\Delta\mathbf{W}_{\text{out}}\,\mathbf{h}$ summiert $n$ gleichgerichtete Terme ohne den Divisor $1/m$.

<!--en:aed1d2bd-->
::: {.slide title="Verständnisfrage zur Skalierung der Lernraten versteckter Schichten mit der Breite"}

<!--en:929b7671-2-->
[Wissensprüfung]{.kicker}

<!--en:11fd7a6d-->
3. **Verkleinerung der Lernraten von $n = 128$ auf $n = 2{,}048$ ($m = 16$):** Durch welchen Faktor muss man die Lernrate der versteckten Gewichtsmatrizen bei einer Verbreiterung um den Faktor $16\times$ teilen? Betrachten Sie (a) Adam mit $\mu\text{P}$, (b) Muon mit dem Faktor $0.2\sqrt{\max(m,n)}$ und (c) Muon mit dem Faktor $\sqrt{d_{\text{out}}/d_{\text{in}}}$.

<!--en:23db34b8-->
**Antwort:**

<!--en:67a49071-->
3. Bei einer Verbreiterung um den Faktor $16\times$ ($m = 16$) teilt man die Lernrate bei (a) Adam mit $\mu\text{P}$ durch **$m = 16\times$**, bei (b) Muon mit AdamW-Skalierung durch **$\sqrt{m} = 4\times$** und bei (c) Muon mit Skalierung nach der Operatornorm durch **$1\times$ (keine Änderung)**.

<!--en:74d2891e-->
::: {.slide title="Zusammenfassung: Parametrisierung nach Breite und μP-Skalierung"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:eaf9cd77-->
| Konzept oder Regel | Mathematischer Zusammenhang | Skalierung mit der Breite $n$ ($m = n/n_{\text{base}}$) | Praktische Bedeutung |

<!--en:5f7b6c9b-->
| **Drift im ersten SP-Schritt** | $(\Delta\mathbf{W}\,\mathbf{h})_i = -\eta\operatorname{sign}(\delta_i)\,n\,\overline{|h|}$ | Die kohärente Summe aus $n$ Termen mit gleichem Vorzeichen wächst wie $\Theta(\eta n)$ | $\eta^\star(n)$ sinkt um den Faktor $16\text{--}32\times$ beim Übergang $n=128 \to 8{,}192$ |

<!--en:8e8b183f-->
| **$\mu\text{P}$-Regel für versteckte Schichten** | Anfangsvarianz $1/n$, Adam-Lernrate $\eta / m$ | $(\eta/m)\,n\,\overline{|h|} = \eta\,n_{\text{base}}\,\overline{|h|} = \Theta(1)$ | Gleicht das Wachstum der Fan-in-Summe der Ordnung $O(n)$ bei internen $n \times n$-Matrizen aus |

<!--en:91432ff5-->
| **$\mu\text{P}$-Regel für die Ausgabeschicht** | Anfangsvarianz $1/n_{\text{base}}$, Logits im Vorwärtsdurchlauf $\times 1/m$, Lernrate $\eta$ | Effektive Gewichte und Updates skalieren wie $O(1/n)$ | Hält sowohl $\Delta\mathbf{W}_{\text{out}}\mathbf{h}$ als auch $\mathbf{W}_{\text{out}}\Delta\mathbf{h}$ in der Größenordnung $\Theta(1)$ |

<!--en:e70900c8-->
| **Koordinatentest nach 1 Schritt** | Nach 1 Schritt $\overline{|h_\ell|}$ gegen $n \in \{128, \ldots, 8192\}$ auftragen | Unter SP Wachstum wie $O(n), O(n^2), O(n^3)$, unter $\mu\text{P}$ flache Kurven | Erkennt fehlende $\mu\text{P}$-Faktoren vor dem Training in Sekunden |

<!--en:40a09169-->
| **Spektralbedingung** | $\|\Delta\mathbf{W}\|_2 = \Theta(\sqrt{d_{\text{out}}/d_{\text{in}}})$ | Vereint Adam ($\eta \propto 1/n$) und Muon ($\eta \propto 1/\sqrt{n}$ oder $1$) | Erlaubt die Übertragung von Hyperparametern kleiner Stellvertretermodelle ohne erneute Abstimmung |
