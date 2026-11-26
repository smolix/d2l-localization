<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:73a3f808-->
In diesem Kapitel haben Sie die rekurrenten Schichten implementiert, die moderne Modelle für lange Kontexte mit Attention kombinieren. Sie haben ihre Ursprünge bis zum LSTM zurückverfolgt und gemessen, wo ein Zustand fester Größe an seine Kapazitätsgrenze stößt. Die meisten dieser Arbeiten sind weniger als fünf Jahre alt. Nur ein Teil davon wird Bestand haben. Dieser Abschnitt stellt Erweiterungen und offene Fragen vor, die im Kapitel noch nicht zur Sprache kamen. Außerdem verweist er auf Referenzimplementierungen und Tutorials.

<!--en:612fedb7-->
## Rekurrenz und Attention im Zusammenspiel

<!--en:1ef2b51a-->
### Hybridmodelle in der Praxis

<!--en:21a26c16-->
:numref:`sec_hybrids` endete mit drei Fragen. Wie wenige Attention-Schichten genügen? An welche Stellen zwischen den übrigen Schichten gehören sie? Kann Attention mit gleitendem Fenster (engl. sliding-window attention) globale Attention ersetzen? Für ihren Vergleich von Modellen mit jeweils 8 Milliarden Parametern kombinieren :citet:`Waleffe.Byeon.Riach.ea.2024` 24 Mamba-2-Schichten mit 4 Attention-Schichten. Zählt man die MLPs mit, machen die Attention-Schichten etwa 8% aller Schichten aus. Die erste Schicht ist eine Mamba-2-Schicht, sodass das Modell ohne Positionseinbettungen auskommt. Die Attention-Schichten folgen in gleichmäßigen Abständen. Mit diesem Verhältnis schnitt das Hybridmodell auf allen zwölf Standardbenchmarks mindestens so gut ab wie ein Transformer. Reine Mamba-Modelle blieben dagegen bei MMLU mit fünf Beispielen und beim Nachschlagen im Telefonbuch zurück.

<!--en:bf4f3be1-->
Bei Modellen im produktiven Einsatz variiert die Mischung. Qwen3-Next :cite:`Qwen.Team.2025` und Kimi Linear :cite:`Kimi.Team.2025b` verwenden Rekurrenzen mit Delta-Regel (:numref:`sec_deltanet`). Jede vierte Schicht ist eine Attention-Schicht. Bei Jamba :cite:`Lieber.Lenz.Bata.ea.2024,Jamba.Team.2024` ist es jede achte, und Nemotron-H :cite:`NVIDIA.2025` behält Mamba-2-Schichten bei. Griffin :cite:`De.Smith.Fernando.ea.2024` verzichtet ganz auf globale Attention. Auf je zwei Schichten mit diagonaler Rekurrenz und Gattern folgt eine Attention-Schicht mit gleitendem Fenster (:numref:`sec_windowed-linear-attention`). Wie :numref:`sec_hybrids` vorhersagt, wird der exakte Abruf unzuverlässiger, sobald der Kontext die Fensterbreite überschreitet.

<!--en:7c7b05c9-->
### Destillation aus Transformern

<!--en:0d2fecc1-->
Für ein rekurrentes Modell oder ein Hybridmodell muss das Pre-Training nicht bei null beginnen. Man kann einen vortrainierten Transformer umwandeln. Dazu initialisiert man Schichten mit Matrixzustand aus seinen Attention-Projektionen (:numref:`sec_matrix-state`) und trainiert sie durch Destillation. :citet:`Wang.Paliotta.May.ea.2024` behalten einen Teil der Attention-Schichten bei und erhalten so ein Hybridmodell. :citet:`Bick.Li.Xing.ea.2024` gleichen schrittweise die Attention-Karten, die versteckten Zustände und die endgültigen Ausgaben an. Dafür verwenden sie 3 bis 5 Milliarden Token, weniger als 1% des ursprünglichen Budgets für das Pre-Training.

<!--en:c931a33c-->
### Zwischen einem Zustand fester Größe und einem vollständigen Cache

<!--en:b73440ae-->
Zwei Ansätze liegen zwischen einem Zustand fester Größe und einem vollständigen KV-Cache. Based :cite:`Arora.Eyuboglu.Zhang.ea.2024` kombiniert eine Taylor-Näherung zweiter Ordnung für $\exp(\mathbf{q}^\top\mathbf{k})$ mit exakter Attention mit gleitendem Fenster. Dabei lässt sich die Zustandsgröße unmittelbar gegen die Zuverlässigkeit des Abrufs abwägen. Bei der log-linearen Attention :cite:`Guo.Yang.Goel.ea.2025` entfällt auf jeden dyadischen Block der bisherigen Sequenz ein Matrixzustand. Je weiter ein Block zurückliegt, desto länger ist er: Von Block zu Block verdoppelt sich die Länge. Der Speicherbedarf wächst daher wie $O(\log T)$. Das Verfahren lässt sich direkt auf Mamba-2 und Gated DeltaNet anwenden.

<!--en:7cbafdb7-->
## Schreiben in einen Matrixspeicher

<!--en:94200efd-->
### Ausdrucksstärke von Modellen mit Zustand fester Größe

<!--en:2546ce6c-->
Manche Sequenzaufgaben können diagonale oder zeitinvariante lineare Rekurrenzen unabhängig von der Zustandsgröße nicht lösen. Verfolgen wir etwa eine Kugel unter fünf Bechern, die wiederholt paarweise vertauscht werden. Dazu müssen wir Permutationen in der symmetrischen Gruppe $S_5$ verketten. :citet:`Merrill.Petty.Sabharwal.2024` beweisen, dass Zustandsraummodelle (engl. state space models, SSMs) mit diagonalen oder festen Übergängen, darunter S4 und Mamba, derselben Einschränkung unterliegen wie Transformer. Sie berechnen nur Funktionen aus der Klasse $\mathsf{TC}^0$ der Schwellwertschaltungen konstanter Tiefe. Die Verkettung von Permutationen auf fünf oder mehr Elementen ist vollständig für $\mathsf{NC}^1$ (Schaltungen logarithmischer Tiefe). Sie liegt daher außerhalb von $\mathsf{TC}^0$, sofern nicht $\mathsf{TC}^0 = \mathsf{NC}^1$ gilt. (Bei drei oder vier Bechern ist die symmetrische Gruppe auflösbar, und die Verkettung bleibt in $\mathsf{TC}^0$.)

<!--en:f8e23517-->
Der Übergang $\mathbf{I} - \beta_t \mathbf{k}_t\mathbf{k}_t^\top$ der Delta-Regel ist eingabeabhängig und nichtdiagonal. Die Schranke für $\mathsf{TC}^0$ schließt also nicht aus, dass die Delta-Regel solche Vertauschungen verfolgen kann. :citet:`Grazzi.Siems.Franke.ea.2025` zeigen jedoch eine Grenze für das Rechnen mit endlicher Genauigkeit auf. Liegen die Eigenwerte der Übergangsmatrix einer linearen Rekurrenz ausschließlich im Intervall $[0, 1]$, so kann die Rekurrenz für beliebige Sequenzlängen weder die Parität berechnen noch die Vertauschungen verfolgen. In der Standardversion von `GatedDeltaNet` liefert eine Sigmoidfunktion den Wert $\beta_t \in (0, 1)$. Damit gilt diese Einschränkung. Lässt man $\beta_t \in (0, 2)$ zu, kann die Übergangsmatrix auch negative Eigenwerte haben. Für $\beta_t = 2$ und den Schlüssel $(\mathbf{e}_i - \mathbf{e}_j)/\sqrt{2}$ ist $\mathbf{I} - 2\mathbf{k}_t\mathbf{k}_t^\top$ eine Spiegelung, die die Koordinaten $i$ und $j$ vertauscht. In :numref:`subsec_dn-state-tracking` trainieren wir diese Variante auf der Paritätsaufgabe.

<!--en:3f6b1545-->
### Verallgemeinerte Delta-Regeln, exponentielle Gatter und neuronale Netze als Speicher

<!--en:c04474bb-->
- **Verallgemeinerte Delta-Regeln.** RWKV-7 :cite:`Peng.Zhang.Goldstein.ea.2025` lockert die Kopplung in der Rang-eins-Projektion der Delta-Regel. Jede Koordinate klingt mit ihrer eigenen Rate ab. Zum Löschen und Schreiben dienen zwei verschiedene Schlüsselvektoren, die jeweils die Richtung vorgeben. Der so entstehende asymmetrische Übergang kann jede reguläre Sprache erkennen. DeltaProduct :cite:`Siems.Carstensen.Zela.ea.2025` verkettet für jedes Token mehrere Spiegelungen $\prod_i (\mathbf{I} - \beta_i\mathbf{k}_i\mathbf{k}_i^\top)$. So kann ein einziger Schritt Permutationen höheren Ranges darstellen.

<!--en:bb0b5dc5-->
- **Exponentielle Gatter.** Ein Eingabegatter (engl. input gate) mit Sigmoidfunktion begrenzt das Schreibgewicht jedes einzelnen Tokens auf $1$. Das mLSTM von xLSTM :cite:`Beck.Poppel.Spanring.ea.2024,Beck.Poppel.Lippe.ea.2025` kombiniert exponentielle Eingabegatter mit einem laufend aktualisierten Zustand für die Normierung und nutzt die Online-Log-Max-Stabilisierung aus :numref:`sec_attention-at-scale`.

<!--en:2ccd6ad0-->
- **Neuronale Netze als Speicher.** TTT :cite:`Sun.Li.Dalal.ea.2024` und Titans :cite:`Behrouz.Zhong.Mirrokni.2025` verallgemeinern die Sichtweise aus :numref:`sec_deltanet`, nach der sich Schreibvorgänge als Online-Gradientenschritte verstehen lassen. An die Stelle der Matrix $\mathbf{S}_t$ tritt eine lineare Schicht oder ein kleines MLP. Bei jedem Token aktualisiert ein Gradientenschritt die Gewichte, bei Titans mit Momentum und Gewichtszerfall. Offen bleibt, ob sich mit nichtlinearen schnellen Gewichten (engl. fast weights) pro Byte Zustand mehr Assoziationen abrufen lassen als mit einer Matrix gleicher Parameteranzahl.

<!--en:4a2e0640-->
## Zustandsraummodelle und lange Faltungen

<!--en:39f0ce63-->
### Gedächtnis durch Projektion auf Polynome

<!--en:7f7109f7-->
In :numref:`sec_ssm` haben wir die Abklingraten von S4D mit $\lambda_n = -n$ initialisiert. Diese Wahl geht auf HiPPO :cite:`Gu.Dao.Ermon.ea.2020` zurück. HiPPO speichert die Koeffizienten der besten Approximation vom Grad $(N-1)$, die den bisherigen Eingabeverlauf $x(s)$ ($s \le t$) durch orthogonale Polynome darstellt. Diese Koeffizienten erfüllen eine lineare gewöhnliche Differentialgleichung mit einer festen unteren Dreiecksmatrix, der HiPPO-Matrix $\mathbf{A}$. S4 :cite:`Gu.Goel.Re.2022` machte das System zeitinvariant. Das entspricht Legendre-Polynomen auf einer exponentiell verzerrten Zeitachse für die bisherigen Eingaben :cite:`Gu.Johnson.Timalsina.ea.2023`. Um den Faltungskern schnell auszuwerten, zerlegte S4 die Matrix $\mathbf{A}$ in eine normale Matrix und einen additiven Korrekturterm niedrigen Ranges. S4D :cite:`Gu.Gupta.Goel.Re.2022` zeigte, dass mit einem Bruchteil des Aufwands fast dieselbe Güte erreichbar ist. Dafür genügen die Diagonaleinträge, also die komplexen Eigenwerte oder die reelle Diagonale $-1, \ldots, -N$. S5 :cite:`Smith.Warrington.Linderman.2023` ersetzte die skalaren SSMs der einzelnen Kanäle durch ein einziges diagonales SSM mit mehreren Ein- und Ausgängen. Es lässt sich mit einem parallelen Scan auswerten (:numref:`subsec_parallel-scans`). Die Dissertation von Gu :cite:`Gu.2023` fasst diese Forschungsrichtung zusammen.

<!--en:b85a1f39-->
### Nachfolger von Mamba, direkte lineare Rekurrenzen und lange Faltungen

<!--en:663c5fb6-->
- **Mamba-2 und Mamba-3.** Bei Mamba-2 :cite:`Dao.Gu.2024` gilt innerhalb jedes Kopfes ein gemeinsamer skalarer Abklingfaktor. Damit lässt sich das Training wie in :numref:`sec_matrix-state` blockweise mit Matrixmultiplikationen berechnen. Mamba-3 :cite:`Lahoti.Li.Chen.ea.2026` führt die Diskretisierung mit der Trapezregel und komplexe Zustände zum Verfolgen von Drehungen ein. Aktualisierungen mit mehreren Eingaben erhöhen die arithmetische Intensität beim autoregressiven Dekodieren.

<!--en:41f26e23-->
- **Linear Recurrent Unit (LRU).** :citet:`Orvieto.Smith.Gu.ea.2023` parametrisieren eine diskrete diagonale Rekurrenz direkt mit komplexen Eigenwerten im Inneren der Einheitskreisscheibe und einer Eingabenormierung $\sqrt{1 - |\lambda|^2}$. In der Long Range Arena erzielt die LRU damit ebenso gute Ergebnisse wie zeitkontinuierliche SSMs. Dieser Ansatz gab den Anstoß zur reellwertigen RG-LRU in Griffin.

<!--en:f884f3ad-->
- **Lange Faltungen.** H3 :cite:`Fu.Dao.Saab.ea.2023` schaltete ein SSM für Verschiebungen und ein multiplikatives assoziatives SSM hintereinander. Hyena :cite:`Poli.Massaroli.Nguyen.ea.2023` ersetzte den Faltungskern des SSM vollständig durch einen implizit parametrisierten Filter über die gesamte Sequenzlänge. Mit der schnellen Fourier-Transformation (FFT) lässt er sich in $O(T \log T)$ auswerten. Bei allgemeinen langen Faltungen (engl. long convolutions) fehlt ein kompakter rekurrenter Zustand, der das Dekodieren in $O(1)$ ermöglichen würde. Daher kommen in Sprachmodellen überwiegend rekurrente SSMs zum Einsatz. In der Genomik sind lange Faltungen weiterhin verbreitet.

<!--en:0a293908-->
## Anwendungen außerhalb der Sprachmodellierung

<!--en:93ad1fb9-->
Auch außerhalb der Sprachmodellierung kommen diese Verfahren zum Einsatz. Vision Mamba :cite:`Zhu.Liao.Zhang.ea.2024` und VMamba :cite:`Liu.Tian.Zhao.ea.2024` verarbeiten Bild-Patches mit bidirektionalen Scans. In SaShiMi :cite:`Goel.Gu.Donahue.ea.2022` modellieren S4-Schichten auf mehreren Skalen die Wellenform von Audiosignalen direkt. HyenaDNA :cite:`Nguyen.Poli.Faizi.ea.2023` modelliert genomische Kontexte von einer Million Nukleotiden mit langen Faltungen. Caduceus :cite:`Schiff.Kao.Gokaslan.ea.2024` verwendet dafür bidirektionale Mamba-Schichten, die gegenüber der Bildung des reversen Komplements äquivariant sind.

<!--en:30616fcb-->
## Referenzimplementierungen und Tutorials

<!--en:471ed40c-->
- **Implementierungen:** [S4 mit Erläuterungen](https://srush.github.io/annotated-s4/) (Rush & Karamcheti, 2022) entwickelt das vollständige S4 Zeile für Zeile. [state-spaces/mamba](https://github.com/state-spaces/mamba) (Gu & Dao) enthält die Referenzimplementierung von Mamba sowie `ssd_minimal` für Mamba-2. [mamba-minimal](https://github.com/johnma2006/mamba-minimal) implementiert Mamba in PyTorch in einer einzigen Datei (:numref:`sec_mamba`). [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) bietet Triton-Kernel für GLA, DeltaNet und Gated DeltaNet (:numref:`sec_matrix-state` und :numref:`sec_deltanet`). [Gated DeltaNet für lineare Attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/08_deltanet) (Raschka, 2025) zeigt, wie man Hybridmodelle mit Gated DeltaNet von Grund auf implementiert. [zoology](https://github.com/HazyResearch/zoology) enthält die synthetischen Aufgaben zum assoziativen Abruf aus :numref:`sec_matrix-state` und :numref:`sec_hybrids`.

<!--en:46cc938d-->
- **Einführungen und Übersichten:** Gu und Dao (2024–2025) vergleichen in [Vor- und Nachteile von SSMs und Transformern](https://goombalab.github.io/blog/2025/tradeoffs/) die beiden Modellfamilien und erläutern die [Zustandsraumdualität (Mamba-2), Teile I–IV](https://goombalab.github.io/blog/2024/mamba2-part1-model/). Weitere Einführungen sind [Eine visuelle Einführung in Mamba und Zustandsraummodelle](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) (Grootendorst, 2024) und [DeltaNet erklärt, Teile I–III](https://sustcsonglin.github.io/blog/2024/deltanet-1/) (Yang, 2024). [Regression zur Testzeit](https://arxiv.org/abs/2501.12352) (Wang, Shi & Fox, 2025) behandelt die einheitliche Sicht auf Speicher als Regressionsmodelle am Ende von :numref:`sec_deltanet`.

<!--en:3ed01eb7-->
[Diskussion](https://d2l.discourse.group/t/36786)
