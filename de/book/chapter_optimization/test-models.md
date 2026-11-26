<!--en:dde63904-->
# Zwei Testmodelle

<!--en:c7ed911c-->
Analytische Modelle wie das zweidimensionale quadratische Tal erklären, wie Krümmung und Gradientenrauschen einfache Aktualisierungsregeln bestimmen. Um moderne Optimierer zu beurteilen, brauchen wir jedoch Messungen an realen neuronalen Netzen. Warum hat AdamW bei Transformern einen großen Vorsprung vor SGD, bei konvolutionären Netzen aber nur einen kleinen? Um solche Fragen zu beantworten, trainieren wir Modelle unter kontrollierten Versuchsbedingungen.

<!--en:520d4aa5-->
Bei stochastischem Training schwanken die Ergebnisse mit der zufälligen Initialisierung und der Auswahl der Minibatches. Ein Unterschied zwischen zwei Konfigurationen eines Optimierers ist daher nur aussagekräftig, wenn er die Schwankungen zwischen Läufen mit derselben Konfiguration übersteigt. In diesem Abschnitt stellen wir die beiden Benchmark-Architekturen für den Rest des Kapitels vor: das bytebasierte Transformer-Sprachmodell `TinyLM` ($470{,}784$ Parameter) und den konvolutionären Klassifikator `FashionCNN` ($421{,}642$ Parameter). Außerdem legen wir die gemeinsame Trainingsschleife, die Lernratensuche (engl. learning-rate sweep), die Referenzmessungen mit SGD und die Signifikanzschwellen fest.

<!--en:4bfce7ff-->
## Das Sprachmodell TinyLM

<!--en:92b2a721-->
`TinyLM` ist ein autoregressives Sprachmodell auf Byteebene. Es sagt das nächste UTF-8-Byte ($V = 256$) aus einem Kontextfenster von $64$ aufeinanderfolgenden Bytes vorher (:numref:`sec_text-sequence`). Ein typischer Lauf mit $2{,}000$ Schritten und Minibatches aus je $64$ Fenstern verarbeitet $2{,}000 \times 64 \times 64 \approx 8.2 \times 10^6$ Zielbytes. Wiederholte Trainingsbeispiele würden es erschweren, Optimierungsgeschwindigkeit und Auswendiglernen auseinanderzuhalten. Deshalb trainieren wir auf der ersten Teildatei von WikiText-103 :cite:`Merity.Xiong.Bradbury.ea.2016` ($\sim 2.7 \times 10^8$ UTF-8-Bytes aus Wikipedia-Artikeln). Für jeden Trainingsminibatch ziehen wir $64$ Fenster mit zufälligen Startpositionen im Bytestrom. Ein Lauf über $2{,}000$ Schritte sieht so nur $\sim 3\%$ des Korpus und verarbeitet in jedem Trainingsschritt neuen Text. Die Validierungsperplexität messen wir auf aufeinanderfolgenden, überlappungsfreien Fenstern der $1.1 \times 10^6$ Byte großen Validierungsmenge.

<!--en:85b3351e-->
Beim Aufruf von `WikiText103(batch_size=64, num_steps=64)` werden die Teildateien für Training und Validierung heruntergeladen. Anschließend dekodieren wir die ersten $300$ Bytes des Korpus:

<!--en:82de5a9a-->
### Architektur und Parametergruppen

<!--en:0e93031e-->
`TinyLM` ist ein reiner Decoder-Transformer mit zwei Blöcken (:numref:`chap_transformers`). Die Einbettungen haben die Breite $d_{\text{model}} = 128$. Jeder Block enthält $2$ kausale Attention-Köpfe und ein zweischichtiges MLP mit GELU und einer versteckten Schicht der Breite $4 d_{\text{model}} = 512$. Vor jeder Teilschicht steht eine LayerNorm (:numref:`subsec_layer-normalization-in-bn`), und eine Residualverbindung umspannt die Teilschicht. Auf die Blöcke folgen eine abschließende LayerNorm und ein linearer Ausgabekopf, der auf $256$ Byte-Logits projiziert. Die Zeilen der Einbettungsmatrizen initialisieren wir mit Werten aus $\mathcal{N}(0, 1)$. Alle Gewichte und Bias-Terme der linearen Schichten ziehen wir gleichverteilt aus $[-1/\sqrt{\text{fan-in}}, 1/\sqrt{\text{fan-in}}]$.

<!--en:1577e145-->
Für die Optimierung lassen sich die $470{,}784$ Parameter von `TinyLM` nach ihrer Struktur in drei Gruppen einteilen. In späteren Abschnitten behandeln wir sie unterschiedlich, etwa beim entkoppelten Gewichtszerfall in :numref:`sec_adamw` und bei der Orthogonalisierung von Matrizen in :numref:`sec_muon`:

<!--en:aea00d9d-->
* **Einbettungen ($9\%$):** Nachschlagetabellen (`token_emb`, `pos_emb`). Die Gradienten ihrer Zeilen sind dünn besetzt und proportional zur Tokenhäufigkeit.

<!--en:f8920a74-->
* **Gewichtsmatrizen ($90\%$):** dicht besetzte 2D-Projektionsmatrizen (`qkv`, `proj`, die beiden MLP-Schichten pro Block und `head`).

<!--en:ab5bee45-->
* **1D-Vektoren ($<1\%$):** Bias-Terme der linearen Schichten sowie Vektoren für Skalierung und Verschiebung in der LayerNorm.

<!--en:db57fa92-->
## Der Bildklassifikator FashionCNN

<!--en:7a6b8dd0-->
Die Rangfolge der Optimierer fällt bei Transformern deutlich anders aus als bei konvolutionären Netzen. Als zweiten Benchmark verwenden wir deshalb `FashionCNN`, einen Klassifikator für $28 \times 28$ Pixel große Bilder aus Fashion-MNIST (:numref:`sec_fashion_mnist`). Zwei Stufen aus $3 \times 3$-Faltung, ReLU und $2 \times 2$-Max-Pooling (mit $32$ und $64$ Kanälen) verringern die räumliche Auflösung. Das Ergebnis hat die Form $64 \times 7 \times 7$. Danach folgen zwei lineare Schichten ($3{,}136 \to 128 \to 10$). Alle Schichten verwenden eine gleichverteilte Initialisierung in den Grenzen $\pm 1/\sqrt{\text{fan-in}}$. Insgesamt sind es $421{,}642$ Parameter.

<!--en:3108b759-->
Mit `d2l.num_params` prüfen wir, ob `FashionCNN` tatsächlich $421{,}642$ trainierbare Parameter hat:

<!--en:a1705428-->
Bei einer Batchgröße von $256$ kommt ein Lauf mit $2{,}000$ Schritten auf $\sim 8.5$ Durchläufe durch die $60{,}000$ Trainingsbilder von Fashion-MNIST. Deshalb geben wir neben dem Trainingsverlust auch die mit `evaluate_accuracy` gemessene Genauigkeit auf separaten Testdaten an:

<!--en:b1886d7f-->
## Trainingsschleife und Hilfsfunktionen zur Auswertung

<!--en:022079bf-->
Beide Modelle bilden einen Minibatch auf Logits ab und minimieren den mittleren Kreuzentropieverlust bei einer festen Zahl von Schritten. :numref:`tab_test-models-helpers` gibt einen Überblick über die Trainingsschleife und die Hilfsfunktionen zur Auswertung, die wir in `d2l` speichern.

<!--en:0088b011-->
:Gemeinsame Funktionen für Training und Auswertung in `d2l`.

<!--en:ef8bc5c6-->
| Funktion | Zweck |

<!--en:5b24ab20-->
| `train_steps` | Trainiert für `num_steps` Schritte (oder bis `stop(step)` gilt) und gibt den Kreuzentropieverlust für jeden Schritt zurück |

<!--en:0ef63dc5-->
| `lm_perplexity` | Berechnet die Perplexität pro Token als $\exp(\text{mean cross-entropy})$ über einen Iterator von Batches (:numref:`sec_language-model`) |

<!--en:6b82c432-->
| `evaluate_accuracy` | Berechnet die Klassifikationsgenauigkeit über einen Iterator von Batches |

<!--en:bee000c7-->
| `final_loss` | Mittelt den Verlust über die letzten $k=100$ Schritte (`inf` bei Divergenz) |

<!--en:45cbf257-->
| `smooth` | Mittelt den Verlust für die grafische Darstellung blockweise über aufeinanderfolgende Fenster von je $k=25$ Werten |

<!--en:ac7179ca-->
| `steps_to_reach` | Liefert den ersten Schritt, in dem der über $25$ Schritte geglättete Verlust auf `target` sinkt (sonst `None`) |

<!--en:f3b83a5f-->
| `lr_sweep` | Trainiert für jede Lernrate einer Liste mit denselben Anfangsgewichten und gibt Verlustkurven und Bewertungsergebnisse zurück |

<!--en:13fddbea-->
Zuerst implementieren wir `train_steps`. Die Funktion minimiert die mittlere Kreuzentropie über `num_steps` Minibatches und ignoriert dabei das Padding-Label `-100`. Sie endet vorzeitig, wenn das optionale Abbruchkriterium `stop(step)` erfüllt ist:

<!--en:040ec40d-->
Als Nächstes werten wir mit `lm_perplexity` ein Sprachmodell im Inferenzmodus aus. Die Funktion durchläuft einen Iterator von Minibatches und gibt $\exp(\text{mean cross-entropy})$ pro Token zurück:

<!--en:e22848ec-->
Um den stark schwankenden Verlust pro Schritt zusammenzufassen und grafisch darzustellen, verwenden wir drei Funktionen. `final_loss` mittelt den Verlust über die letzten $k = 100$ Schritte, `smooth` bildet Blockmittelwerte über je $25$ Schritte und `steps_to_reach` liefert den ersten Schritt, in dem der geglättete Verlust den Zielwert `target` erreicht:

<!--en:c563d9f3-->
Schließlich trainiert `lr_sweep` für jede Lernrate aus der Liste ein neues Modell mit demselben Zufallsstartwert `seed`. Die Liste übergeben wir als `lrs`. Optional ruft die Funktion `evaluate(model)` alle `every` Schritte auf:

<!--en:9c701936-->
## Referenzläufe mit Minibatch-SGD und Versuchsprotokoll

<!--en:259b797c-->
Als Referenz verwenden wir Minibatch-SGD mit konstanter Lernrate ($2{,}000$ Schritte, ohne Momentum oder Gewichtszerfall). Der Anfangsverlust von `TinyLM` liegt etwas über der Kreuzentropie der Gleichverteilung von $\ln 256 \approx 5.55$ Nat/Byte (wegen der anfänglichen Varianz der Logits bei $\sim 5.71$ Nat/Byte, siehe Übung :exref:`ex_test-models-initial-loss`). Weitere Vergleichswerte auf WikiText-103 liefern Tabellen der Bytehäufigkeiten: Eine Unigrammtabelle erreicht $\sim 3.2$ Nat/Byte, eine Bigrammtabelle für Bytepaare $\sim 2.5$ Nat/Byte (Übung :exref:`ex_test-models-landmarks`). :numref:`fig_opt_loss_landmarks` zeigt diese Referenzwerte auf einer gemeinsamen Skala. Daran lässt sich jeder Verlustwert in diesem Kapitel messen.

<!--en:78c7f4d1-->
![Referenzwerte für den Verlust bei der Sprachmodellierung auf Byteebene mit WikiText-103. Grau: das untrainierte Modell `TinyLM` und die Vorhersage mit Gleichverteilung. Blau: Unigramm- und Bigrammtabellen für Bytes. Orange: `TinyLM` nach $2{,}000$ SGD-Schritten (Messung weiter unten) und nach den am besten abgestimmten Läufen im weiteren Verlauf des Kapitels.](../img/mdl-opt-loss-landmarks.svg)

<!--en:494e346f-->
Zuerst definieren wir drei Funktionen: `make_lm` erzeugt das Modell, `sgd` den einfachen SGD-Optimierer und `new_seed` einen neuen Zufallsstartwert:

<!--en:8eabc941-->
Wir führen `lr_sweep` für `TinyLM` mit den Lernraten $\eta \in \{0.1, 0.5, 2.5\}$ aus. Tragen wir den Verlust pro Schritt zusammen mit dem Referenzwert $\ln 256$ für gleichverteiltes Raten auf, erkennen wir drei typische Verläufe:

<!--en:6d311fc4-->
* **Zu hoch ($\eta = 2.5$):** Nach dem ersten Schritt überschreitet die Lernrate die lokale Stabilitätsgrenze $2/L$. Innerhalb von zehn Schritten divergiert der Verlust gegen `inf`.

<!--en:181111b7-->
* **Zu niedrig ($\eta = 0.1$):** Der Verlust sinkt gleichmäßig, aber langsam und erreicht nur $\sim 2.5$ Nat/Byte (Bigrammniveau) bei Schritt $2{,}000$.

<!--en:82af186b-->
* **Gut abgestimmt ($\eta = 0.5$):** Anfangs steigt der Verlust kurzzeitig stark an, während das Netz Bereiche hoher Krümmung durchläuft. Danach sinkt er auf etwa $2.1$ Nat/Byte.

<!--en:36867443-->
Auf `FashionCNN` erreicht einfacher SGD bei $\eta = 0.3$ einen Trainingsverlust von $\sim 0.22$ und eine Genauigkeit von $\sim 90\%$ auf separaten Testdaten:

<!--en:1853090e-->
### Vergleichsprotokoll und Signifikanzschwellen

<!--en:6343c31a-->
Für einen fairen Vergleich folgen wir dem Deep Learning Tuning Playbook :cite:`Godbole.Dahl.Gilmer.ea.2023` und behandeln die Lernrate $\eta$ als **Störhyperparameter** (engl. nuisance hyperparameter). Für jedes Verfahren suchen wir unabhängig nach der besten Lernrate (`lr_sweep` auf einem logarithmischen Gitter mit identischen Anfangsgewichten). Ein Vergleich bei einer willkürlich gewählten gemeinsamen Lernrate wäre unfair :cite:`Schmidt.Schneider.Hennig.2021`. Für jeden Optimierer erfassen wir den minimalen Verlust bei seiner besten Lernrate und den **nutzbaren Lernratenbereich**. Dieser umfasst die Gitterpunkte, bei denen der Endverlust höchstens $0.15$ Nat über dem Optimum liegt.

<!--en:30e270a5-->
Bei $5$ unabhängigen Zufallsstartwerten und gut abgestimmter Lernrate betrachten wir den über die letzten $100$ Schritte gemittelten Trainingsverlust (`final_loss`) und den Validierungsverlust von `TinyLM`. Ihre Spannweiten betragen jeweils $\sim 0.01\text{--}0.015$ Nat. Bei `FashionCNN` schwankt der Trainingsverlust um $\sim 0.02$ und die Testgenauigkeit um $\sim 1\%$. Als konservative Schwelle verwenden wir das $3\times$-Fache der Spannweite. Einen Unterschied zwischen Einzelläufen werten wir nur dann als statistisch aussagekräftig, wenn er **$0.05$ Nat** bei `TinyLM` oder **$0.06$ im Trainingsverlust ($3\%$ in der Testgenauigkeit)** bei `FashionCNN` übersteigt. Kleinere Unterschiede gelten als Gleichstand, sofern sie sich bei mehreren Zufallsstartwerten nicht bestätigen.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:470421c0-->
Für unsere empirischen Vergleiche verwenden wir zwei neuronale Netze, die sich als Benchmarks ergänzen. `TinyLM` hat $470{,}784$ Parameter: $90\%$ entfallen auf 2D-Projektionsmatrizen, $9\%$ auf Einbettungen und $<1\%$ auf 1D-Vektoren. Das Modell trainiert in einem Durchlauf auf zufällig gewählten $64$ Byte langen Fenstern aus WikiText-103. In einem Lauf über $2{,}000$ Schritte verarbeitet es so in jedem Schritt neuen Text. `FashionCNN` ($421{,}642$ Parameter) trainiert dagegen über $\sim 8.5$ Epochen mit Minibatches aus je $256$ Bildern von Fashion-MNIST. Wir messen sowohl den Trainingsverlust als auch die Genauigkeit auf separaten Testdaten. Die geschlossen berechenbaren Byteentropien für Gleichverteilung, Unigramme und Bigramme liefern schon vor dem Training Referenzwerte für den Verlust von `TinyLM`. Um echte algorithmische Verbesserungen von den Einflüssen der Hyperparameter und Zufallsstartwerte zu unterscheiden, testen wir die Lernrate $\eta$ für jeden Optimierer unabhängig und beginnen mit identischen Gewichten. Wir geben den optimalen Verlust und den nutzbaren Lernratenbereich an. Zur Beurteilung dient eine aus mehreren Zufallsstartwerten ermittelte Signifikanzschwelle von $0.05$ Nat bei `TinyLM` ($0.06$ im Verlust oder $3\%$ in der Genauigkeit bei `FashionCNN`).

<!--en:2294243a-->
## Übungen

<!--en:2ba63d7a-->
1. **Anfangsverlust vor dem Training.**
   :exlabel:`ex_test-models-initial-loss`
    1. Die Logits $z_1, \ldots, z_V$ seien unabhängig und identisch nach $\mathcal{N}(0, s^2)$ verteilt und unabhängig vom Zielwert. Zeigen Sie mithilfe von $\sum_{i=1}^V e^{z_i} \approx V \mathbb{E}[e^{z_i}] = V e^{s^2/2}$, dass die erwartete Kreuzentropie zu Beginn ungefähr $\ln V + s^2 / 2$ beträgt.
    1. Die Ausgaben der LayerNorm haben Mittelwert $0$ und Varianz $1$. Die Gewichte und Bias-Terme des Ausgabekopfs stammen aus $\mathcal{U}(-1/\sqrt{128}, 1/\sqrt{128})$. Zeigen Sie, dass $s^2 = 129/384 \approx 1/3$ gilt und sich ein Anfangsverlust von $\ln 256 + 1/6 \approx 5.71$ Nat ergibt. Dies entspricht dem Verlust in Schritt $1$ von `TinyLM`. Wie hoch ist der Anfangsverlust, wenn Sie die Gewichte und den Bias des Ausgabekopfs mit null initialisieren?

<!--en:b5b43baf-->
1. [code] **Verlustkurven von `FashionCNN` untersuchen.** Trainieren Sie `FashionCNN` mit einfachem SGD für $\eta \in \{0.003, 0.03, 0.3, 1.0\}$. Zeichnen Sie den Verlust pro Schritt auf einer logarithmischen Schrittachse ein, zusammen mit dem Referenzwert $\ln 10 \approx 2.3026$ für die Gleichverteilung. Untersuchen Sie für den Lauf, dessen Verlust bei etwa $\ln 10$ stagniert, den Anteil toter ReLU-Aktivierungen in einem Minibatch. Erklären Sie damit, warum die Gradienten verschwinden.
   :exlabel:`ex_test-models-diagnosis`

<!--en:055aec60-->
1. [code] **Divergenzgrenze bei `TinyLM`.** Testen Sie einfachen SGD auf `TinyLM` für $\eta \in \{0.7, 1.0, 1.4, 2.0\}$ mit jeweils $5$ Zufallsstartwerten (`lr_sweep(..., verbose=False)`). Halten Sie fest, welcher Anteil der Läufe stabil bleibt, bei welchem Schritt die divergenten Läufe scheitern und welchen minimalen Verlust die stabilen Läufe erreichen.
   :exlabel:`ex_test-models-edge`

<!--en:f4f8c6ab-->
1. [code] **Spannweite über mehrere Zufallsstartwerte und Standardfehler.** ● Trainieren Sie mit $20$ Zufallsstartwerten das Modell `TinyLM` mit AdamW (:numref:`sec_adam`) bei $\eta = 3 \times 10^{-3}$ (`weight_decay=0`). Vergleichen Sie die Stichprobenstandardabweichung $\sigma$ von `final_loss` mit dem Schätzer $\text{range}/3.7$, der bei Normalverteilung die Spannweite nutzt. Berechnen Sie, wie viele Zufallsstartwerte $k$ erforderlich sind, damit eine Differenz von $0.005$ Nat zwei Standardfehlern entspricht (Standardfehler der Differenz: $\sigma\sqrt{2/k}$).
   :exlabel:`ex_test-models-range`

<!--en:1ba2a463-->
1. [code] **Trainingsverlust, Validierungsverlust und Überanpassung an kleine Korpora.**
   :exlabel:`ex_test-models-held-out`
    1. Erklären Sie, warum `final_loss` (der Mittelwert über die letzten $100$ Trainingsschritte) etwas höher liegt (um $0.02\text{--}0.03$ Nat) als der Validierungsverlust nach Schritt $2000$ bei `TinyLM`. Überprüfen Sie Ihre Erklärung, indem Sie den letzten Checkpoint auf zurückgehaltenen Fenstern der Trainingsdaten auswerten.
    1. Beschränken Sie das Korpus auf $100{,}000$ Bytes. Verwenden Sie dazu `WikiText103(batch_size=64, num_steps=64, num_train=100000)` ($\sim 80$ Epochen in $2{,}000$ Schritten). Stellen Sie alle $250$ Schritte Trainings- und Validierungsverlust grafisch gegenüber (`every=250`).

<!--en:e0b83862-->
1. [code] **Referenzwerte aus $n$-Gramm-Häufigkeiten.** Bestimmen Sie die empirische Unigrammentropie der Bytes ($\sim 3.2$ Nat) sowie die Kreuzentropien für Bigramme ($\sim 2.5$ Nat) und Trigramme mit Add-eins-Glättung. Verwenden Sie dafür die ersten $10^7$ Trainingsbytes von `WikiText103` und werten Sie die Modelle auf der Validierungsmenge aus. Vergleichen Sie die Werte mit dem Verlust nach $2{,}000$ Schritten mit SGD (etwa $2.1$ Nat) und AdamW ($1.59$ Nat, :numref:`sec_adam`) bei `TinyLM`.
   :exlabel:`ex_test-models-landmarks`

<!--en:700b0975-->
[Diskussionen](https://d2l.discourse.group/t/36884)

<!--en:220a2215-->
Zwei Testmodelle: TinyLM und FashionCNN<br>**Bytebasierter Transformer · konvolutionärer Klassifikator · Referenzwerte aus Häufigkeiten · Protokoll zur Lernratensuche · Signifikanz bei mehreren Zufallsstartwerten**

<!--en:c4792d67-->
::: {.slide title="Von lösbaren quadratischen Problemen zu Benchmarks mit neuronalen Netzen"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:f31c97d2-->
- **Warum quadratische Funktionen nicht ausreichen:** Bei neuronalen Netzen ändert sich die Schärfe $\lambda_{\max}(\mathbf{x}_t)$, Gradienten verschiedener Schichten unterscheiden sich um einen Faktor von $40\times$ und die Ergebnisse schwanken mit dem Zufallsstartwert.

<!--en:3c69060d-->
- **Drei Grundlagen der Experimente:**
  1. **Zwei komplementäre Modelle:** `TinyLM` (Byte-Transformer mit einem Durchlauf) und `FashionCNN` (Bild-CNN mit mehreren Epochen).
  2. **Referenzwerte für den Verlust vor dem Training:** geschlossen berechenbare Byteentropien für Gleichverteilung, Unigramme und Bigramme.
  3. **Faires Vergleichsprotokoll:** unabhängige Lernratensuche pro Verfahren mit gleichem Zufallsstartwert. Signifikanzschwelle: das $3\times$-Fache der Spannweite über mehrere Zufallsstartwerte.

<!--en:dc37875f-->
::: {.slide title="Benchmark 1: TinyLM auf den Bytes von WikiText-103 in einem Durchlauf"}

<!--en:a94e4287-->
[Benchmark für Sprachmodellierung]{.kicker}

<!--en:758186fe-->
- **Autoregressive Bytevorhersage ($V = 256$):** Aus einem Kontextfenster von $64$ aufeinanderfolgenden Bytes sagt das Modell das nächste UTF-8-Byte vorher.

<!--en:1228b174-->
- **Ein Durchlauf durch den Datenstrom:**
  - Korpus: erste Teildatei von WikiText-103 ($\sim 2.7 \times 10^8$ UTF-8-Bytes aus Wikipedia-Artikeln).
  - Minibatch: $b = 64$ Fenster zu je $64$ Byte ($4{,}096$ Zielbytes pro Schritt) mit zufälligen Startpositionen.
  - Ein Lauf mit $2{,}000$ Schritten verarbeitet $8.2 \times 10^6$ Bytes ($\sim 3\%$ des Korpus). Jeder Trainingsschritt sieht neuen, unbekannten Text, ohne ihn auswendig zu lernen.

<!--en:6785d680-->
::: {.slide title="TinyLM von innen: Architektur eines reinen Decoder-Transformers"}

<!--en:e2318e9f-->
[Aufbau des Transformers]{.kicker}

<!--en:12d364fc-->
- **Einbettungen:** $\mathbf{H}_0 = \mathbf{E}_{\text{tok}}(\mathbf{X}) + \mathbf{E}_{\text{pos}}(0..63) \in \mathbb{R}^{B \times 64 \times 128}$, initialisiert mit Werten aus $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

<!--en:12fc287a-->
- **Zwei Residualblöcke mit vorangestellter Normalisierung ($d_{\text{model}} = 128, 2\text{ heads}$):**
  $$\mathbf{H} \leftarrow \mathbf{H} + \text{CausalMHA}(\text{LN}_1(\mathbf{H}))$$
  $$\mathbf{H} \leftarrow \mathbf{H} + \text{MLP}_{\text{GELU}}(\text{LN}_2(\mathbf{H}))$$

<!--en:ba0680d6-->
- **Ausgabekopf:** $\text{Linear}_{128 \to 256}(\text{LN}(\mathbf{H}))$ mit Anfangswerten aus $\mathcal{U}(-1/\sqrt{\text{fan-in}}, 1/\sqrt{\text{fan-in}})$.

<!--en:b988fbac-->
::: {.slide title="Die Parameter von TinyLM: drei strukturelle Gruppen"}

<!--en:02b290ea-->
[Verteilung der Parameter]{.kicker}

<!--en:ee465477-->
Für den Optimierer zerfallen die Parameter von `TinyLM` (insgesamt $470{,}784$) ihrer Struktur nach in drei Gruppen:

<!--en:162a45a9-->
- **1. Dicht besetzte 2D-Gewichtsmatrizen ($425{,}984$ Parameter, $90.5\%$):** Vier Matrizen pro Transformer-Block (`qkv` $128 \to 384$, `proj` $128 \to 128$, `mlp` $128 \to 512 \to 128$) und der Ausgabekopf `head` ($128 \to 256$). AdamW (§{sec_adam}) wendet Gewichtszerfall an, Muon (§{sec_muon}) orthogonalisiert die Aktualisierungen der internen Matrizen.

<!--en:c23d8e09-->
- **2. Nachschlagetabellen für Einbettungen ($40{,}960$ Parameter, $8.7\%$):** `token_emb` ($256 \times 128$) und `pos_emb` ($64 \times 128$). Die Zeilengradienten sind dünn besetzt und proportional zur Bytehäufigkeit.

<!--en:841fe259-->
- **3. 1D-Vektoren ($3{,}840$ Parameter, $0.8\%$):** Bias-Terme der linearen Schichten sowie Skalierungen und Verschiebungen der LayerNorm.

<!--en:7629fbcb-->
::: {.slide title="Die Parameter von TinyLM: drei strukturelle Gruppen (Fortsetzung)" layout="tall-output"}

<!--en:02b290ea-2-->
[Verteilung der Parameter]{.kicker}

<!--en:a3cf562e-->
Die Übersicht zeigt für jeden Parametertensor Form und Größe, anschließend die Gesamtzahl der Parameter in jeder der drei Gruppen:

<!--en:7ae2af0e-->
::: {.slide title="Benchmark 2: FashionCNN auf den Bildern von Fashion-MNIST"}

<!--en:8958999d-->
[Konvolutionärer Klassifikator]{.kicker}

<!--en:beb410b2-->
- **Warum eine zweite Architektur?** Große Unterschiede zwischen Optimierern bei Transformern fallen bei konvolutionären Netzen oft kleiner aus.

<!--en:d5aa8d00-->
- **Architektur ($421{,}642$ Parameter):**
  - `conv1` ($1 \to 32$, $3 \times 3$, ReLU, $2 \times 2$-Max-Pooling $\to 32 \times 14 \times 14$): $320$ Parameter.
  - `conv2` ($32 \to 64$, $3 \times 3$, ReLU, $2 \times 2$-Max-Pooling $\to 64 \times 7 \times 7$): $18{,}496$ Parameter.
  - `fc1` ($3{,}136 \to 128$, ReLU): $401{,}536$ Parameter (**$95\%$ aller Parameter**).
  - `fc2` ($128 \to 10$ Logits): $1{,}290$ Parameter.

<!--en:7dad6f1f-->
::: {.slide title="Unsere beiden neuronalen Benchmarks im direkten Vergleich"}

<!--en:fdeedfc1-->
[Zwei komplementäre Trainingsbedingungen]{.kicker}

<!--en:1d5a9879-->
- **Ein Durchlauf oder mehrere Epochen ($2{,}000$ Schritte):**
  - `TinyLM` sieht $0.03$ Epochen ($3\%$ von $270\text{M}$ Bytes): in jedem Schritt neuen, unbekannten Text, ohne ihn auswendig zu lernen.
  - `FashionCNN` sieht $8.5$ Epochen ($512\text{K}$ Auswertungen auf $60\text{K}$ Bildern). Um Überanpassung zu erkennen, messen wir die Genauigkeit auf separaten Testdaten.

<!--en:c1da6721-->
- **Unterschiedliche Parametergeometrie:**
  - Bei `TinyLM` verteilen sich $90.5\%$ der Gewichte auf $9$ Transformer-Matrizen. Hinzu kommen Einbettungen mit dünn besetzten Gradienten. Bei `FashionCNN` liegen $95\%$ der Gewichte in `fc1`.

<!--en:69e4e941-->
::: {.slide title="Gemeinsame Trainingsschleife und Hilfsfunktionen zur Auswertung"}

<!--en:f3b6b203-->
[Versuchsaufbau]{.kicker}

<!--en:307cf8f2-->
- **Kontrollierte Lernratensuche und Trainingsschleife (`lr_sweep`, `train_steps`):** Für jeden Wert des logarithmischen $\eta$-Gitters starten wir mit denselben Anfangsgewichten (Zufallsstartwert 0) und verarbeiten $T = 2{,}000$ Minibatches.

<!--en:366e23cb-->
- **Kennzahlen für den geglätteten Verlustverlauf (`final_loss`, `steps_to_reach`):** Mittelwert über die letzten $k = 100$ Schritte und erster Block von $25$ Schritten, dessen Mittelwert die Zielschwelle erreicht.

<!--en:2f8bb849-->
- **Auswertung auf separaten Daten (`lm_perplexity`, `evaluate_accuracy`):** Im Auswertungsmodus prüfen wir die Modelle auf $1.1\text{M}$ Validierungsbytes bzw. $10{,}000$ Testbildern.

<!--en:04419847-->
::: {.slide title="Informationstheoretische Referenzwerte vor dem Training"}

<!--en:897bf9d4-->
[Einordnung der Verlustwerte]{.kicker}

<!--en:2a46ed1d-->
- **Schritt 1 vor dem Training ($\approx 5.71$ Nat):** Zufällige Logits des Ausgabekopfs $z_i \sim \mathcal{N}(0, s^2)$ mit $s^2 \approx 1/3$ erhöhen den Verlust gegenüber der Gleichverteilung um den Varianzbeitrag $s^2/2 \approx 0.17$.

<!--en:18a5a3ee-->
- **Gleichverteiltes Raten ($5.55$ Nat):** $\ln 256$ ($\ln 10 \approx 2.30$ bei `FashionCNN`).

<!--en:5e0ace5b-->
- **Unigrammhäufigkeit der Bytes ($\approx 3.18$ Nat):** Randhäufigkeiten der UTF-8-Bytes ohne Kontext.

<!--en:99b50ddd-->
- **Bigrammtabelle ($\approx 2.50$ Nat):** Bedingte Wahrscheinlichkeit $p(b_t \mid b_{t-1})$. Ein Verlust unter $2.50$ Nat zeigt, dass das Modell mehr als ein Byte Kontext nutzt.

<!--en:36fd5241-->
::: {.slide title="Referenzläufe mit Minibatch-SGD auf TinyLM bei drei Lernraten"}

<!--en:72836ade-->
[Drei typische Verläufe]{.kicker}

<!--en:d89bfaad-->
- Einfacher Minibatch-SGD (ohne Momentum und Gewichtszerfall, $2{,}000$ Schritte) zeigt für $\eta \in \{0.1, 0.5, 2.5\}$ bei identischen Anfangsgewichten drei unterschiedliche Verlustverläufe:
  - **$\eta = 0.1$ (blau):** Der Verlust sinkt gleichmäßig und monoton auf **etwa $2.5$ Nat**. Damit erreicht er gerade den Referenzwert der Bigrammtabelle ($2.5$).
  - **$\eta = 0.5$ (orange):** Der Verlust steigt kurz um $>1$ Nat an, und zwar in den ersten $30$ Schritten, danach sinkt er auf **etwa $2.1$ Nat**.
  - **$\eta = 2.5$ (grün):** In Schritt 1 sinkt der Verlust, in Schritt 2 springt er auf über $6$ Nat. Er divergiert gegen **`inf`**, und das innerhalb von $10$ Schritten.

<!--en:03a3bafa-->
::: {.slide title="Ungeeignete Lernraten an Verlustkurven erkennen"}

<!--en:b6ed3036-->
[Trainingsprotokolle lesen]{.kicker}

<!--en:f7919ea4-->
- **$\eta = 0.1$ (zu kleines Schrittweitenbudget):** Der Verlust sinkt gleichmäßig, selbst noch bei Schritt $2{,}000$. Das Schrittweitenbudget $B_T = \eta T = 200$ ist zu klein $\to$ **$\eta$ erhöhen**.

<!--en:4e0effa8-->
- **$\eta = 0.5$ (nahe der anfänglichen Stabilitätsgrenze $2/L$):** Anstieg um $>1$ Nat in den Schritten $1\text{--}30$, danach rascher Abfall auf etwa $2.1$ Nat $\to$ **Aufwärmphase (§{sec_scheduler}) oder Gradientenbegrenzung (§{sec_practice}) einsetzen**.

<!--en:4dfe1837-->
- **$\eta = 2.5$ (über der Grenze $2/\lambda_{\max}$):** Schritt 1 schießt in ein steileres Becken hinein, der Verlust divergiert gegen `inf` $\to$ **$\eta$ senken**.

<!--en:9f254e8b-->
::: {.slide title="Referenzlauf mit Minibatch-SGD auf FashionCNN"}

<!--en:97e37b0e-->
[Referenzwerte für die Klassifikation]{.kicker}

<!--en:c1318232-->
- Dieselbe Schleife über $2{,}000$ Schritte mit `train_steps` liefert für `FashionCNN` ($b = 256$ Bilder) die Referenzwerte der Bildklassifikation mit einfachem SGD bei $\eta = 0.3$:
  - **Endgültiger Trainingsverlust:** $\approx 0.22\text{--}0.23$ Nat (gegenüber $\ln 10 \approx 2.30$ bei Gleichverteilung).
  - **Genauigkeit auf separaten Testdaten:** $\approx 89.5\%\text{--}90.5\%$.

<!--en:4db63b4c-->
- **`TinyLM` (etwa $2.1$ Nat mit einfachem SGD)** und **`FashionCNN` ($0.22$ Verlust / $90\%$ Testgenauigkeit)** liefern die Referenzwerte für §{sec_momentum} bis §{sec_practice}.

<!--en:9ff6b5fc-->
::: {.slide title="Faire Vergleiche: der Umgang mit Störhyperparametern"}

<!--en:0663da23-->
[Versuchsplanung]{.kicker}

<!--en:48de5ec1-->
- **Warum Vergleiche mit festem $\eta$ scheitern:** Einfacher SGD skaliert die unveränderten Gradienten ($\eta^\star \approx 0.5$), AdamW hingegen normalisiert die Schrittlängen ($\eta^\star \approx 3 \times 10^{-3}$).

<!--en:fbe9ac10-->
- **Abstimmung von Störhyperparametern (Godbole et al., 2023):** Für jeden Optimierer testen wir $\eta$ unabhängig auf einem logarithmischen Gitter mit denselben Anfangsgewichten (Zufallsstartwert 0).

<!--en:00643857-->
- **Bestwert und Breite angeben:** Ergebnis bei der besten Lernrate $\eta^\star$ *und* nutzbarer Lernratenbereich mit einem Endverlust von höchstens $+0.15$ Nat über dem Wert bei $\eta^\star$.

<!--en:344a5c0f-->
::: {.slide title="Bestes Ergebnis und nutzbaren Lernratenbereich angeben"}

<!--en:83d1b931-->
[Robustheit und Bestwert]{.kicker}

<!--en:b9a56cf1-->
Bei der Lernratensuche mit `lr_sweep` auf einem logarithmischen Gitter (z. B. mit einem Faktor von $\sim 3\times$ zwischen benachbarten Punkten) ermitteln wir für jeden Optimierer zwei Kennzahlen:

<!--en:665eaf1a-->
- **1. Bestwert bei der optimalen Lernrate $\eta^\star$ auf dem Gitter:** niedrigster Wert von `final_loss` (oder niedrigster Validierungsverlust) unter allen getesteten Gitterpunkten bei identischen Anfangsgewichten.

<!--en:c28894b8-->
- **2. Nutzbarer Lernratenbereich:** Bereich der Lernraten auf dem Gitter, deren Endverlust höchstens **$+0.15$ Nat** über dem Optimum liegt (das $>10\times$-Fache der Spannweite über mehrere Zufallsstartwerte).
  - *Beispiel für `TinyLM`:* Bei einfachem SGD mit $\{0.1, 0.5, 2.5\}$ ist nur ein Gitterpunkt nutzbar ($\eta = 0.5$ mit etwa $2.1$ Nat). Bei $\eta = 0.1$ ist der Verlust um etwa $0.35$ Nat höher, bei $\eta = 2.5$ divergiert er gegen `inf`.
  - Wie wir in §{sec_adam} bis §{sec_muon} sehen, erweitern adaptive und orthogonalisierte Optimierer den nutzbaren Bereich auf eine Größenordnung (von $10^{-3}$ bis $10^{-2}$ oder $3 \times 10^{-2}$).

<!--en:afa251ce-->
::: {.slide title="Spannweite über mehrere Zufallsstartwerte und die 3×-Signifikanzschwelle"}

<!--en:cc684e1e-->
[Statistische Absicherung]{.kicker}

<!--en:bce57f3d-->
- **Schwankungen zwischen Läufen:** Bei 5 Zufallsstartwerten führen unterschiedliche Anfangsgewichte und Minibatches zu Spannweiten von $0.010\text{--}0.015$ Nat bei `TinyLM` und $0.8\%\text{--}1.0\%$ in der Genauigkeit bei `FashionCNN`.

<!--en:c6f0da15-->
- **Entscheidungsschwelle für Einzelläufe ($3\times$-Faches der Spannweite):**
  - **`TinyLM`:** Verbesserung um **$\ge 0.05$ Nat/Byte** erforderlich.
  - **`FashionCNN`:** Verbesserung um **$\ge 0.06$ im Verlust** oder **$\ge 3.0\%$ in der Genauigkeit** erforderlich.

<!--en:557db4b6-->
- Kleinere Unterschiede zwischen Einzelläufen gelten als **statistischer Gleichstand**, sofern wir die Ergebnisse nicht über mehrere Zufallsstartwerte mitteln.

<!--en:07bfd2db-->
::: {.slide title="Verständnisfragen: Referenzwerte, Kurvenverläufe und Zufallsstartwerte"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:8a74d102-->
Wenden Sie das Versuchsprotokoll dieses Abschnitts auf drei konkrete Fragen an:

<!--en:be732bd0-->
1. **Mit null initialisierter Ausgabekopf:** In Übung 1 beginnt das untrainierte Modell `TinyLM` mit den üblichen zufälligen Gewichten im Ausgabekopf ($s^2 \approx 1/3$) bei $\ln 256 + s^2/2 \approx 5.71$ Nat. Wie hoch ist der Verlust in Schritt 1, wenn wir die Gewichte und Bias-Terme des Ausgabekopfs alle auf null setzen?

<!--en:88ea756d-->
2. **Sieger oder statistischer Gleichstand?** In je einem Lauf über $2{,}000$ Schritte mit identischen Anfangsgewichten erreicht Optimierer A $1.57$ Nat auf `TinyLM` und $90.4\%$ Genauigkeit auf `FashionCNN`. Optimierer B erreicht $1.59$ Nat und $89.6\%$ Genauigkeit. Welcher Optimierer gewinnt?

<!--en:525fd026-->
3. **Trainings- und Validierungsverlust bei `TinyLM`:** Warum liegt der Validierungsverlust von `TinyLM` nach Schritt $2{,}000$ typischerweise um $0.02\text{--}0.03$ Nat *unter* `final_loss`?

<!--en:6fe2cdb5-->
::: {.slide title="Verständnisfragen: Referenzwerte, Kurvenverläufe und Zufallsstartwerte (Antworten)"}

<!--en:929b7671-2-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:557b98b6-->
**Antworten:**

<!--en:055c1cea-->
1. Sind alle Gewichte und Bias-Terme im Ausgabekopf null, sind auch die Logits null ($\mathbf{z} = \mathbf{0}$). Das ergibt exakt die Gleichverteilung $p_i = 1/256$ und einen Anfangsverlust von **$\ln 256 \approx 5.545$ Nat** (schon in Schritt 1 $0.17$ Nat weniger).

<!--en:4bb73f1f-->
2. **Beide Vergleiche enden unentschieden.** Bei `TinyLM` beträgt die Differenz $|1.57 - 1.59| = 0.02 < 0.05$ Nat, bei `FashionCNN` gilt $|90.4\% - 89.6\%| = 0.8\% < 3.0\%$. Um einen Unterschied von $0.02$ Nat nachzuweisen, müssen wir über mehrere Zufallsstartwerte mitteln.

<!--en:dab91ca2-->
3. Da `TinyLM` in jedem Schritt neuen Text sieht (keine Überanpassung), verbessert sich das Modell zwischen Schritt $1{,}900$ und Schritt $2{,}000$ weiter. `final_loss` mittelt über die Schritte $1{,}901\text{--}2{,}000$, während die Validierung die endgültigen Gewichte aus Schritt $2{,}000$ bewertet.

<!--en:92bbb21b-->
::: {.slide title="Zusammenfassung: Testmodelle, Referenzwerte und Protokoll zur Lernratensuche"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:346214f9-->
| Komponente | Ausführung | Wichtige Zahlen | Was der Vergleich zeigt |

<!--en:744a2a8b-->
| **`TinyLM` auf WikiText-103** | Transformer mit 2 Blöcken, $d=128$, $2$ Köpfe, Batches zu $64 \times 64$ Byte | $470{,}784$ Parameter ($90\%$ Matrizen, $9\%$ Einbettungen, $1\%$ Vektoren). $3\%$ des Korpus in $2{,}000$ Schritten | Benchmark für Pre-Training in einem Durchlauf ohne Auswendiglernen |

<!--en:1a782340-->
| **`FashionCNN` auf Fashion-MNIST** | 2 Stufen aus Faltung, ReLU und Pooling, dann 2 vollständig verbundene Schichten, $b = 256$ Bilder | $421{,}642$ Parameter ($95\%$ in `fc1`). $8.5$ Epochen in $2{,}000$ Schritten | Klassifikation über mehrere Epochen mit Genauigkeit auf separaten Testdaten |

<!--en:54410060-->
| **Verlustreferenzwerte vor dem Training** | Bytevorhersage mit geschlossener Formel oder anhand von Häufigkeiten | Untrainiert $5.71 \to$ Gleichverteilung $5.55 \to$ Unigramm $3.2 \to$ Bigramm $2.5$ Nat | Ordnet den Verlust in Nat/Byte anhand der Vorhersagequalität ein |

<!--en:8dd29da3-->
| **Referenzwerte für einfachen SGD** | $2{,}000$ Schritte, konstantes $\eta$, auf dem Gitter abgestimmt | `TinyLM` bei $\eta=0.5$: **$\approx 2.1$ Nat**. `FashionCNN` bei $\eta=0.3$: **$0.22$ Verlust / $90\%$ Genauigkeit** | Referenzwerte, die Momentum, AdamW und Muon übertreffen müssen |

<!--en:845f8f4b-->
| **Vergleich und Signifikanz** | Unabhängige Lernratensuche pro Verfahren mit gleichem Zufallsstartwert | Schwelle für Einzelläufe: **$0.05$ Nat** (`TinyLM`), **$0.06$ Verlust / $3\%$ Genauigkeit** (`FashionCNN`) | Trennt algorithmische Verbesserungen von Effekten der Lernratensuche und zufälligen Schwankungen |
