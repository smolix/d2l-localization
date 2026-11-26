<!--en:a7b5ef60-->
# Gauß-Pfade: Diffusion und Flow-Matching

<!--en:e59b98e2-->
Wie wir in :numref:`sec_diffusion-denoising` gesehen haben, reicht ein einziger Rauschpegel (engl. noise level) $\sigma$ nicht aus, um die Verteilung gut zu durchmischen und zugleich scharfe Samples zu erzeugen. Bei kleinem $\sigma$ bleiben Langevin-Ketten in ihren anfänglichen Modi gefangen. Bei großem $\sigma$ wechseln sie zwar zwischen den Modi, ziehen ihre Samples aber aus einer stark geglätteten Dichte. Als Abhilfe verbinden wir die Datenverteilung bei $t = 0$ mit einer Standardnormalverteilung bei $t = 1$. Dazu wählen wir einen stetigen *Gauß-Pfad* (engl. Gaussian path) $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$. Wir durchlaufen ihn vom hohen Rauschpegel bis hinunter zum rauschfreien Zustand.

<!--en:43450d3c-->
Probabilistische Diffusionsmodelle zur Entrauschung (engl. denoising diffusion probabilistic models, DDPM) :cite:`ho2020denoising`, scorebasierte Modelle :cite:`song2019generative,song2021score` und Flow-Matching :cite:`Lipman.Chen.BenHamu.ea.2022,Liu.Gong.Liu.2022,Albergo.Boffi.VandenEijnden.2023` trainieren jeweils ein einziges zeitabhängiges Netz mittels Kleinste-Quadrate-Regression über alle Stufen $t \in [0, 1]$ hinweg. Die üblichen Trainingsziele sind der rauschfreie Punkt $\mathbf{x}_0$, das Rauschen $\boldsymbol\epsilon$, der Score $\nabla \log p_t(\mathbf{x}_t)$, die Geschwindigkeit $\dot\alpha_t \mathbf{x}_0 + \dot\sigma_t \boldsymbol\epsilon$ und die Winkelgeschwindigkeit $\mathbf{v} = \alpha_t \boldsymbol\epsilon - \sigma_t \mathbf{x}_0$. Jedes Ziel ist eine affine Funktion von $\mathbf{x}_t$ und dem A-posteriori-Mittelwert (engl. posterior mean) $\hat{\mathbf{x}}_0(\mathbf{x}_t) = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$. Ein Netz, das auf eines dieser Ziele trainiert ist, bestimmt daher auch alle anderen. In diesem Abschnitt definieren wir die gängigen Gauß-Pfade und ihre gemeinsame Zeitskala, das logarithmische Signal-Rausch-Verhältnis (Log-SNR). Wir prüfen das Langevin-Sampling mit sinkendem Rauschpegel anhand exakter Scores und leiten das bedingte Flow-Matching sowie die Umrechnungsregeln für die Zielgrößen her. An der Mischung mit drei Modi prüfen wir unsere trainierten Netze anhand der exakten Verlustuntergrenze (engl. loss floor).

<!--en:117b7e12-->
## Pfade von den Daten zum Rauschen

<!--en:3c6efd83-->
Um alle Rauschpegel zu durchlaufen, ordnen wir sie der Reihe nach an und weisen jedem einen Zeitpunkt zu.

<!--en:253436b9-->
### Signal, Rauschen und Log-SNR

<!--en:befcb7b0-->
Der verrauschte Punkt aus dem vorigen Abschnitt, $\tilde{\mathbf{x}} = \mathbf{x}_0 + \sigma\boldsymbol\epsilon$, beschreibt einen Pfad, sobald $\sigma$ mit der Zeit wächst. Auf diesem Pfad behalten die Daten ihre ursprüngliche Größe. Das Rauschen wächst, bis die Daten vernachlässigbar sind. Die Varianz wächst unbegrenzt, daher die Bezeichnung *varianzexplodierender* Pfad (engl. variance-exploding path, VE-Pfad). Ein Netz sieht hier Eingaben, deren Größe mit dem Rauschen wächst. Dieselben Gewichte müssen Eingaben der Größe eins verarbeiten und im Modell von Song und Ermon für CIFAR-10-Bilder auch fünfzigmal so große Eingaben. Die meisten Modelle verkleinern deshalb mit zunehmendem Rauschen den Datenanteil. So bleibt jede Eingabe ungefähr von der Größe eins. Ein *Gauß-Pfad* ist ein Paar von Funktionen $\alpha_t$ und $\sigma_t$ auf $t \in [0, 1]$. Sie definieren den verrauschten Punkt

<!--en:cef8d225-->
mit den Anfangswerten $(\alpha_0, \sigma_0) = (1, 0)$. Anders als der VE-Pfad enden die beiden gängigen Pfade bei reinem Rauschen, $(\alpha_1, \sigma_1) = (0, 1)$, oder sehr nahe daran.

<!--en:8c9c9e7e-->
Für den *varianzerhaltenden* Pfad (engl. variance-preserving path, VP-Pfad) von DDPM :cite:`ho2020denoising` gilt stets $\alpha_t^2 + \sigma_t^2 = 1$. Daten mit Varianz eins behalten also diese Varianz. Die Rate $\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min})\, t$ wächst linear von $\beta_{\min} = 0.1$ bis $\beta_{\max} = 20$ und bestimmt, wie schnell der Pfad die Daten vergisst. In jedem Zeitintervall $dt$ verliert das Signal den Anteil $\beta(t)\, dt$ seiner Varianz. Den verbleibenden Anteil erhalten wir daher, indem wir die Rate integrieren, das Vorzeichen wechseln und die Exponentialfunktion anwenden:

<!--en:ab10fa46-->
Die Werte $0.1$ und $20$ sind nicht willkürlich gewählt. Sie ergeben sich aus der ersten und letzten Schrittvarianz von DDPM, $10^{-4}$ und $0.02$, multipliziert mit der Anzahl seiner $1{,}000$ Schritte. Im nächsten Unterabschnitt sehen wir, wie aus den Schritten eine Rate wird. Am Ende des Pfades gilt $\bar\alpha_1 = e^{-10.05} \approx 4 \cdot 10^{-5}$, für alle praktischen Zwecke also reines Rauschen. Verfolgen wir einen Punkt auf dem VP-Pfad, so beschreibt er einen Ornstein-Uhlenbeck-Prozess, eine Irrfahrt mit einer Rückstellkraft zum Ursprung (:numref:`sec_mdl-ornstein-uhlenbeck`). Dabei läuft die Zeit immer schneller: Sowohl die Rückstellkraft als auch die Rauschvarianz sind proportional zu $\beta(t)$, das am Ende zweihundertmal so groß ist wie zu Beginn.

<!--en:163cbdaa-->
Der *lineare* Pfad von Flow-Matching und Rectified Flow :cite:`Lipman.Chen.BenHamu.ea.2022,Liu.Gong.Liu.2022,Albergo.Boffi.VandenEijnden.2023` verbindet jeden rauschfreien Punkt durch eine Gerade mit seinem Rauschen:

<!--en:8c23b723-->
Er ist der einfachste Pfad, der genau bei reinem Rauschen endet. Jedes Beispiel bewegt sich mit konstanter Geschwindigkeit auf seiner Geraden. Wegen dieser Einfachheit verwenden ihn das Standardmodell dieses Kapitels und seit 2024 auch viele Bildgeneratoren :cite:`Esser.Kulal.Blattmann.ea.2024`.

<!--en:f286d3e4-->
Wann liefern zwei Pfade einem Entrauscher (engl. denoiser) zu unterschiedlichen Zeiten dieselbe Information? Was der Entrauscher aus $\mathbf{x}_t$ erschließen kann, hängt allein vom Verhältnis von Signal zu Rauschen ab. Wir messen es als *Log-SNR* $\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$. Um das zu sehen, multiplizieren wir einen verrauschten Punkt mit $c = 1/(\alpha_t + \sigma_t)$. Die Koeffizienten von $c\alpha_t \mathbf{x}_0 + c\sigma_t \boldsymbol\epsilon$ summieren sich zu eins. Das Ergebnis ist also ein Punkt auf dem linearen Pfad zum Zeitpunkt $\sigma_t/(\alpha_t + \sigma_t)$ mit demselben Signal-Rausch-Verhältnis. Bei der Multiplikation mit einer bekannten Zahl geht keine Information verloren. Der umskalierte Punkt enthält daher dieselbe Information wie $\mathbf{x}_t$.

<!--en:3f7ff6f1-->
Betrachten wir zwei Pfade zu Zeiten mit gleichem $\lambda$, so enthalten sie dieselbe Information über den rauschfreien Punkt. Die Log-SNR gibt also eine für alle Pfade gemeinsame Zeitskala vor. Sie fällt von $+\infty$ bei den Daten bis auf $-\infty$ bei reinem Rauschen. Der lineare Pfad durchläuft die ganze Skala, der VP-Pfad endet nahe $\lambda = -10$. Mit dem Pfad wählen wir also, wie schnell diese Uhr läuft und in welchen Bereichen Training und Sampling wie viel Zeit verbringen.

<!--en:ae0b988f-->
`GaussianPath` speichert $\alpha_t$ und $\sigma_t$, berechnet die Log-SNR und zieht $\mathbf{x}_t$. Die Konstruktoren erzeugen den VP-Pfad, den linearen Pfad und einen dritten Pfad, den Kosinuspfad, den wir weiter unten kennenlernen.

<!--en:fb556c3a-->
Bei $t = 0.5$ sind Signal und Rauschen auf dem linearen Pfad gleich stark, also $\lambda = 0$. Wo steht zu diesem Zeitpunkt der VP-Pfad?

<!--en:8dc7e12e-->
Nach der Hälfte der Zeit überwiegt auf dem VP-Pfad bereits das Rauschen, mit $\alpha = 0.28$ und $\lambda \approx -2.5$. Bei $t = 0.9$ bleiben zwei Prozent des Signals übrig, auf dem linearen Pfad dagegen zehn. Der VP-Pfad erreicht $\lambda = 0$ schon bei etwa $t = 0.26$ und verbringt somit drei Viertel der Zeit im Bereich, in dem das Rauschen überwiegt (:numref:`fig_diffusion-paths`).

<!--en:247b1a8c-->
![Zwei Gauß-Pfade auf derselben Zeitskala. (a) Signalkoeffizient $\alpha_t$ (durchgezogen) und Rauschkoeffizient $\sigma_t$ (gestrichelt) für den varianzerhaltenden Pfad, auf dem $\beta(t)$ von $0.1$ auf $20$ steigt, sowie für den linearen Pfad. (b) Die Log-SNR $\lambda_t$ beider Pfade, begrenzt auf $[-10, 10]$. Auf dem varianzerhaltenden Pfad überwiegt das Rauschen viel früher.](../img/mdl-diffusion-paths.svg)

<!--en:0828cb12-->
Bei kleinen Bildern ist das eine Zeitverschwendung. :citet:`Nichol.Dhariwal.2021` beobachteten, dass bei Bildern mit $32$ und $64$ Pixeln Kantenlänge der letzte Teil des DDPM-Zeitplans wenig beiträgt. Das Bild ist schon lange vor $t = 1$ verschwunden. Ihr *Kosinuspfad* setzt $\alpha_t = \cos\frac{\pi t}{2}$, bis auf einen kleinen Versatz nahe den Daten, und $\sigma_t = \sin\frac{\pi t}{2}$. Auch dies ist ein VP-Pfad. Sein $\bar\alpha_t$ fällt jedoch im mittleren Bereich fast linear und ändert sich an beiden Enden langsam. So verbringt der Pfad weniger Zeit in dem Bereich, in dem das Bild bereits verschwunden ist.

<!--en:d3d09605-->
::: {.callout-important title="Die Zeitskala"}

<!--en:3405696a-->
Im ganzen Buch liegen die Daten bei $t = 0$ und das Rauschen bei $t = 1$. Jeder Sampler läuft von $t = 1$ zurück bis $0$. Arbeiten zum Flow-Matching kehren die Zeitrichtung um (:numref:`fig_mdl-dyn-time-conventions`). Setzen Sie $1 - t$ für das dortige $t$ ein, lesen Sie das Rauschen $\mathbf{x}_0$ als unser $\boldsymbol\epsilon$ und die Daten $\mathbf{x}_1$ als unser $\mathbf{x}_0$. Wechseln Sie außerdem bei jeder Geschwindigkeit das Vorzeichen.

<!--en:d41b94e3-->
Arbeiten über Diffusion verwenden unterschiedliche Schreibweisen für dieselben Größen. :numref:`tab_diffusion-notation` dient zum Nachschlagen. Sie stellt die Symbole, die wir ab hier verwenden, den Schreibweisen in der Literatur gegenüber. Bei der ersten Lektüre können Sie die Tabelle überspringen und später darauf zurückgreifen, wenn die Notation einer Arbeit Schwierigkeiten bereitet.

<!--en:4c3b244e-->
:Notation im Diffusionskapitel. In späteren Abschnitten kommen bei Bedarf einzelne Symbole hinzu.

<!--en:2b476dec-->
| Symbol | Bedeutung | Schreibweise in der Literatur |

<!--en:def4f6de-->
| $t \in [0, 1]$ | Zeit auf dem Pfad: Daten bei $0$, Rauschen bei $1$ | Daten bei $t = 1$ im Flow-Matching, ganzzahliger Schrittindex bei DDPM |

<!--en:d072ade3-->
| $\mathbf{x}_0$, $\boldsymbol\epsilon$, $\mathbf{x}_t$ | rauschfreier Punkt, standardnormalverteiltes Rauschen, verrauschter Punkt | $\mathbf{x}_1$ für Daten, $\mathbf{x}_0$ für Rauschen im Flow-Matching |

<!--en:088543ae-->
| $\alpha_t$, $\sigma_t$ | Signal- und Rauschkoeffizient des Pfades | $\sqrt{\bar\alpha_t}$, $\sqrt{1 - \bar\alpha_t}$ bei DDPM, dort bezeichnet $\alpha_t$ die Größe $1 - \beta_t$ |

<!--en:18f265d1-->
| $\bar\alpha_t = \alpha_t^2$, $\beta(t)$ | quadrierter Signalkoeffizient des VP-Pfades und seine Rate | |

<!--en:14ea50d1-->
| $k$, $T$, $\beta_k$ | Schritt der DDPM-Kette, Anzahl der Schritte, im Schritt $k$ hinzugefügte Varianz | $t$ für $k$ |

<!--en:da3d49e5-->
| $\lambda_t$ | Log-SNR, $\log(\alpha_t^2 / \sigma_t^2)$ | $-\gamma(t)$ in VDM |

<!--en:8a878b57-->
| $\sigma$, $\tilde\sigma_i$ | Rauschpegel des VE-Pfades, Pegel $i$ einer Stufenfolge | $\sigma_i$ für $\tilde\sigma_i$ |

<!--en:18056bbe-->
| $p_t$, $\nabla \log p_t$ | Dichte von $\mathbf{x}_t$ und ihr Score | |

<!--en:15afdbac-->
| $\hat{\mathbf{x}}_0$, $\hat{\boldsymbol\epsilon}$ | A-posteriori-Mittelwerte von $\mathbf{x}_0$ und $\boldsymbol\epsilon$ bei gegebenem $\mathbf{x}_t$, mit Index $\boldsymbol\theta$ die Vorhersagen eines Netzes für diese Größen | $D_{\boldsymbol\theta}$ (ein Entrauscher), $\boldsymbol\epsilon_{\boldsymbol\theta}$ |

<!--en:4b025bf8-->
| $\mathbf{u}_t$, $\mathbf{u}_{\boldsymbol\theta}$ | marginale Geschwindigkeit des Pfades, Netz zu ihrer Vorhersage | $u_t$, $v_t$ im Flow-Matching, mit umgekehrtem Vorzeichen |

<!--en:f4675ca1-->
| $\mathbf{v}$ | $\alpha_t \boldsymbol\epsilon - \sigma_t \mathbf{x}_0$, Zielgröße der $\mathbf{v}$-Vorhersage | |

<!--en:37344ce8-->
| $\pi_k$, $\boldsymbol\mu_k$, $s$ | Gewichte, Mittelwerte und Standardabweichung der Mischungskomponenten ($k$ indiziert hier die Komponenten) | |

<!--en:a14e6867-->
### Die DDPM-Kette

<!--en:ffadd2bb-->
Ursprünglich war der VP-Pfad anders formuliert. :citet:`ho2020denoising` definierten eine Markov-Kette aus $T$ kleinen Schritten. Jeder verkleinert den Punkt ein wenig und fügt etwas Rauschen hinzu:

<!--en:e0b3ac0d-->
Dabei sind die $\boldsymbol\epsilon_k$ unabhängige standardnormalverteilte Zufallsgrößen. Für die Schrittvarianzen gilt $\beta_k \in (0, 1)$. Jeder Schritt erhält die Varianz eins. Da die Rauschterme unabhängig und normalverteilt sind, können wir $k$ Schritte zu einer einzigen Ziehung zusammenfassen:

<!--en:e5d22fe0-->
Der verbleibende Signalanteil ergibt sich aus dem Produkt der Faktoren aller bisherigen Schritte (den Induktionsbeweis liefert :numref:`sec_mdl-ddpm-discretized-sde`). Dank dieser geschlossenen Form müssen wir die Kette beim Training nie simulieren: Für jedes $\mathbf{x}_k$ genügt ausgehend von $\mathbf{x}_0$ eine einzige Ziehung aus einer Normalverteilung.

<!--en:212aee69-->
Legen wir die Schritte auf die Zeitpunkte $t_k = k/T$ und wählen wir $\beta_k = 1 - \bar\alpha_{t_k} / \bar\alpha_{t_{k-1}}$. Im Produkt kürzen sich alle Zwischenfaktoren, sodass $\bar\alpha_{t_k}$ übrig bleibt. Die Kette entspricht also dem VP-Pfad auf einem Gitter. Auf einem feinen Gitter gilt $\beta_k \approx \beta(t_k)/T$. Schrittvarianzen und Rate beschreiben somit dieselbe Größe, einmal pro Schritt und einmal pro Zeiteinheit. :numref:`fig_diffusion-forward-reverse` zeigt die Vorwärtskette und die Rückwärtskette, die wir bei der Generierung durchlaufen.

<!--en:be27e92e-->
![Das grafische Modell eines Diffusionsmodells. Der Vorwärtsprozess (oben) ist eine feste Markov-Kette mit gaußschen Übergängen von den Daten $\mathbf{x}_0$ zum Rauschen $\mathbf{x}_T$. Der Rückwärtsprozess (unten) ist eine gelernte Kette mit gaußschen Übergängen, deren Mittelwerte ein Netz vorhersagt.](../img/mdl-diffusion-forward-reverse.svg)

<!--en:403f6572-->
`grid` gibt $\alpha$ und $\sigma$ für gleichmäßig verteilte Zeitpunkte vom Rauschen bis zu den Daten an. Dies ist das Standardgitter für die Sampler dieses Kapitels.

<!--en:b92badcf-->
Welche Schrittvarianzen liefert unser Pfad für $T = 1000$? Aus $\beta_k \approx \beta(t_k)/T$ erwarten wir etwa $10^{-4}$ für den ersten Schritt und $0.02$ für den letzten. Das Gitter beginnt am Rauschende. Seine ersten Einträge gehören daher zum letzten Schritt.

<!--en:ab9e594f-->
Die Schrittvarianzen steigen von $1.1 \cdot 10^{-4}$ auf $0.0198$. Das entspricht annähernd dem linearen Zeitplan von Ho et al., der von $10^{-4}$ bis $0.02$ reicht. Nach tausend Schritten bleibt von den Daten ein Anteil von $\sqrt{\bar\alpha_T} \approx 0.007$ erhalten.

<!--en:4d881b37-->
### Schrittweises Absenken des Rauschpegels

<!--en:465970f9-->
Gelingt mit mehreren Rauschpegeln, was mit einem einzigen nicht möglich war? Wir kennen die Scores der geglätteten Mischung exakt und können das vor jedem Netztraining testen. Dazu durchlaufen wir den VE-Pfad aus dem vorigen Abschnitt in Richtung abnehmenden Rauschens. Wir verwenden den Langevin-Schritt $\mathbf{x} \leftarrow \mathbf{x} + \tfrac{h}{2}\nabla \log p(\mathbf{x}) + \sqrt{h}\,\boldsymbol\xi$ aus :numref:`sec_diffusion-denoising` mit jeweils frischem gaußschem Rauschen $\boldsymbol\xi$. Dabei steigen wir eine *Stufenfolge* von zehn Rauschpegeln $\tilde\sigma_i$ hinab, die auf einer logarithmischen Skala gleichmäßig von $3$ bis $0.05$ verteilt sind. Auf jeder Stufe gehen wir $100$ Schritte mit dem exakten Score, ausgehend vom Endpunkt der vorigen Stufe. Die Schrittweite $h_i = 0.1\, \tilde\sigma_i^2$ nimmt quadratisch mit dem Pegel ab. Dadurch bleibt das Rauschen pro Schritt, $\sqrt{h_i} \approx 0.3\, \tilde\sigma_i$, ein fester Bruchteil der aktuellen Glättungsbreite.

<!--en:0ba4115e-->
Im Kontrollversuch gehen wir dieselben $1{,}000$ Schritte auf der feinsten Stufe mit der Schrittweite $h = 0.05$ aus dem vorigen Abschnitt. Diese Schrittweite ist zweihundertmal so groß wie die der Stufenfolge beim selben Pegel. Die Kontrolle hat damit beste Chancen, ein Tal zu überwinden. Beide Versuche beginnen mit denselben $2{,}000$ Punkten aus $\mathcal{N}(\mathbf{0}, 9\mathbf{I})$. Stimmt unsere Überlegung, so sollten die Gewichte bei der Stufenfolge nahe bei $0.5$, $0.3$ und $0.2$ liegen, im Kontrollversuch dagegen bei jeweils etwa einem Drittel.

<!--en:01b045a9-->
Bei der Stufenfolge weichen die Gewichte um höchstens etwa $0.03$ von den richtigen Werten ab. Ein bis zwei Prozent der Punkte liegen weit von jedem Modus entfernt, gegenüber etwa einem Prozent in den Daten selbst. Im Kontrollversuch stimmt die Streuung um jeden Modus, aber die Gewichte bleiben wie bei den Startpunkten bei jeweils etwa einem Drittel. Die Anteile legen sich auf den groben Stufen fest, wo die geglättete Dichte keine Täler hat. Die feinen Stufen schärfen nur noch die einzelnen Modi. Ein Diffusionsmodell teilt die Arbeit auf seinem Pfad ebenso auf. Da kein Netz beteiligt war, prüft dieser Versuch allein die Idee. Was später schiefgeht, liegt also am Netz.

<!--en:ecf581e1-->
Diese *Annealed Langevin Dynamics* dienten schon beim ersten Generator dieser Art zum Sampling, dem auf den Rauschpegel bedingten Score-Netz von :citet:`song2019generative`. Ein einziges Netz erhält den Rauschpegel als Eingabe. Wir trainieren es auf jeder Stufe durch Denoising-Score-Matching mit dem Gewicht $\tilde\sigma_i^2$ (:numref:`sec_mdl-score-based-generative-modeling`). Die Pegel, Schrittweiten und Schrittzahlen wurden von Hand festgelegt. In hohen Dimensionen erfordert das große Sorgfalt (Aufgabe :exref:`ex_gaussian-paths-ladder-spacing`). Ein Pfad ersetzt die zehn Stufen durch ein Kontinuum, das ein einziges Netz vollständig lernen muss.

<!--en:11f4f331-->
## Regression auf das marginale Feld

<!--en:ff244abe-->
Ein einziges Netz soll für jeden Punkt des Pfades ausreichen. Zuerst legen wir fest, was es dort lernen soll. Dann suchen wir eine Regression, mit der es diese Größe lernen kann, ohne dass wir sie bereits kennen.

<!--en:784ef711-->
### Bedingte Zielgrößen und marginale Felder

<!--en:846890c0-->
Das Netz im vorigen Abschnitt lernte für einen Rauschpegel den Score und damit über die Tweedie-Formel den A-posteriori-Mittelwert des rauschfreien Punktes, $\hat{\mathbf{x}}_0 = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$. Diese eine Größe liefert zu jedem Zeitpunkt den Score und gibt damit die Bewegungsrichtung eines verrauschten Punktes vor. Sie hängt von der gesamten Datenverteilung ab und ist unbekannt. Konditionieren wir zusätzlich auf den rauschfreien Punkt, wird das Problem einfach: Kennen wir $\mathbf{x}_0$, so ist die beste Schätzung für $\mathbf{x}_0$ der Punkt $\mathbf{x}_0$ selbst.

<!--en:700fa94a-->
Ein Sampler muss wissen, in welche Richtung er einen verrauschten Punkt bewegen soll. Lassen wir das Netz diese Richtung also direkt vorhersagen. Ein rauschfreier Punkt und eine Realisierung des Rauschens ergeben die *bedingte Geschwindigkeit* (engl. conditional velocity) $\dot{\mathbf{x}}_t = \dot\alpha_t \mathbf{x}_0 + \dot\sigma_t \boldsymbol\epsilon$. Auf dem linearen Pfad hat sie den konstanten Wert $\boldsymbol\epsilon - \mathbf{x}_0$. Viele solcher Punkte durchlaufen dieselbe Position $\mathbf{x}$ zur Zeit $t$, jeder in seiner eigenen Richtung. Die Punktwolke als Ganzes bewegt sich dort mit ihrer Durchschnittsgeschwindigkeit,

<!--en:8ea71d08-->
der *marginalen Geschwindigkeit* (engl. marginal velocity). Kann nur ein einziger rauschfreier Punkt $\mathbf{x}$ erzeugt haben, so ist sie die Geschwindigkeit dieses Punktes. Kommen mehrere Punkte infrage, mitteln wir ihre Geschwindigkeiten, gewichtet mit den jeweiligen Wahrscheinlichkeiten.

<!--en:c50f192d-->
Für die Mischung kennen wir dieses Feld exakt. Bei gegebenem Modus sind $\mathbf{x}_0$ und $\boldsymbol\epsilon$ jeweils gemeinsam mit $\mathbf{x}_t$ normalverteilt. Ihre bedingten Mittelwerte sind daher linear in $\mathbf{x}_t$. Die Methode `velocity` mittelt sie über die Modi, gewichtet mit deren A-posteriori-Wahrscheinlichkeiten.

<!--en:c7c18f28-->
Für reale Daten können wir dieses Feld nicht berechnen, wohl aber lernen. Die Methode der kleinsten Quadrate liefert einen bedingten Mittelwert. Den bedingten Erwartungswert lernen wir also durch Regression auf die Größe, über die wir den Erwartungswert bilden. Auf diese Weise hat das Denoising-Score-Matching den Score gelernt. Für die Geschwindigkeit ergibt die Regression den Verlust für das *bedingte Flow-Matching* (engl. conditional flow matching). Er unterscheidet sich vom Regressionsverlust für das unbekannte marginale Feld nur um eine Konstante $C$, die nicht vom Netz abhängt:

<!--en:83f84269-->
Beide Seiten haben denselben Minimierer. Ein Netz, das wir auf die Geschwindigkeiten einzelner Beispiele trainieren, lernt also die marginale Geschwindigkeit (den Beweis liefert :numref:`sec_mdl-flow-matching`). Um $\mathbf{u}_t$ zu lernen, müssen wir das Feld ebenso wenig kennen, wie der Entrauscher den Score kennen musste. :numref:`fig_diffusion-blueprint` zeigt beide Sichtweisen für die Mischung. Bei festem $\mathbf{x}_t = \mathbf{x}$ ist die bedingte Geschwindigkeit $\boldsymbol\epsilon - \mathbf{x}_0$ auf dem linearen Pfad gleich $(\mathbf{x} - \mathbf{x}_0)/t$. Die Abbildung zeigt ihren negativen Wert, also die Richtung, in die sich ein Sampler bewegt.

<!--en:42e2bbc2-->
![Bedingte und marginale Sicht auf den linearen Pfad der Mischung, exakt berechnet. (a) Bedingter Pfad eines rauschfreien Punktes $\mathbf{x}_0$ aus dem Modus mit dem größten Gewicht: Kreise mit Radius $2\sigma_t$ um $\alpha_t\mathbf{x}_0$ zu fünf Zeitpunkten von $0$ bis $1$. (b) Marginaler Pfad: Höhenlinien von $p_t$ zu denselben Zeitpunkten. (c) Das bedingte Feld $-(\mathbf{x} - \mathbf{x}_0)/t$ bei $t = 0.5$ zeigt auf diesen einen Punkt. (d) Das marginale Feld $-\mathbf{u}_{0.5}(\mathbf{x})$ ist der A-posteriori-Mittelwert der bedingten Felder.](../img/mdl-diffusion-blueprint.svg)

<!--en:9dfdc739-->
`target` berechnet für jedes Beispiel die Zielgröße auf einem beliebigen Pfad. Im folgenden Training rufen wir damit die Geschwindigkeit ab. Die Methode `rates` liefert dafür $\dot\alpha_t$ und $\dot\sigma_t$ durch automatische Differentiation im Vorwärtsmodus. Wir müssen also für keinen Pfad die Ableitungen von Hand angeben. Die anderen Zielgrößen betrachten wir im nächsten Unterabschnitt. Dort verwenden wir auch `x0_eps`, das eine Vorhersage wieder in $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ umrechnet.

<!--en:66bb77c0-->
Das ganze Verfahren passt nun in eine Codezelle. In jedem Schritt ziehen wir $512$ neue Punkte aus der Mischung und für jeden Punkt einen Zeitpunkt und eine Realisierung des Rauschens. Wir trainieren `TimeMLP` durch Regression auf die bedingte Geschwindigkeit des linearen Pfades. Dieses kleine Netz aus dem vorigen Abschnitt erhält den Punkt und sinusförmige Merkmale der Zeit als Eingabe. Das Training umfasst $40{,}000$ Adam-Schritte mit einer Lernrate, die einer Kosinuskurve folgend sinkt. Wir bilden einen gleitenden Mittelwert der Gewichte mit einer Halbwertszeit von $700$ Schritten. Die gemittelten Gewichte glätten das Rauschen der letzten Schritte. Sie verwenden wir unten für die Auswertung. Da jeder Batch neu ist, gibt es keinen Trainingsdatensatz, den das Netz auswendig lernen könnte. Im vorigen Abschnitt waren die $2{,}000$ Punkte dagegen fest vorgegeben (Aufgabe :exref:`ex_gaussian-paths-memorization` untersucht diesen Fall). Das Training jedes Netzes dauert auf einer GPU ein bis zwei Minuten.

<!--en:7f25a388-->
Jeder Punkt der Kurve ist ein Mittelwert über $400$ Schritte. Nach wenigen tausend Schritten pendelt sich der Verlust nahe $5$ ein, weit über null. Steckt das Netz fest? Wir werden sehen, dass kein Netz wesentlich besser abschneiden kann.

<!--en:af092958-->
### Äquivalente Zielgrößen

<!--en:787aa193-->
Das Netz von DDPM sagt das Rauschen vorher, andere Netze den rauschfreien Punkt. Ändert diese Wahl, was das Netz lernt? Bei festem $\mathbf{x}_t$ sind das zwei Darstellungen derselben Größe. Bilden wir auf beiden Seiten von :eqref:`eq_diffusion-gaussian-path` den A-posteriori-Mittelwert, erhalten wir $\mathbf{x}_t = \alpha_t \hat{\mathbf{x}}_0 + \sigma_t \hat{\boldsymbol\epsilon}$ mit $\hat{\boldsymbol\epsilon} = \mathbb{E}[\boldsymbol\epsilon \mid \mathbf{x}_t]$. Jeder der beiden Mittelwerte bestimmt also den anderen.

<!--en:174daa88-->
Der Score ist eine dritte Darstellung derselben Größe. Nach der Tweedie-Formel aus dem vorigen Abschnitt gilt $\mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}] = \tilde{\mathbf{x}} + \sigma^2 \nabla \log p_\sigma(\tilde{\mathbf{x}})$ für $\tilde{\mathbf{x}} = \mathbf{x}_0 + \sigma\boldsymbol\epsilon$. Wir wenden sie mit $\alpha_t \mathbf{x}_0$ als rauschfreiem Punkt und $\sigma_t$ als Rauschpegel an. Dann gilt $\alpha_t \hat{\mathbf{x}}_0 = \mathbf{x}_t + \sigma_t^2 \nabla \log p_t(\mathbf{x}_t)$, und der Score lautet

<!--en:a1bb298f-->
Der Score zeigt von $\mathbf{x}_t$ in Richtung des skalierten A-posteriori-Mittelwerts. Er ist gleich der Rauschvorhersage dividiert durch $-\sigma_t$.

<!--en:bc3c1057-->
Die marginale Geschwindigkeit aus :eqref:`eq_diffusion-marginal-velocity` ist $\dot\alpha_t \hat{\mathbf{x}}_0 + \dot\sigma_t \hat{\boldsymbol\epsilon}$. Eliminieren wir $\hat{\mathbf{x}}_0$ aus diesem Ausdruck und der Formel für den Score, erhalten wir die Identität zwischen Score und Geschwindigkeit:

<!--en:dfb44afe-->
Auf dem linearen Pfad mit $\dot\alpha_t = -1$ und $\dot\sigma_t = 1$ lautet sie

<!--en:ccc4dfbd-->
Die Geschwindigkeit ist die Summe eines bekannten Vielfachen des Punktes und eines bekannten Vielfachen des Scores. Beide Faktoren stehen mit dem Pfad fest. Ein Geschwindigkeitsmodell ist somit eine andere Darstellung eines Score-Modells.

<!--en:0d89d092-->
Viele Bildmodelle verwenden eine fünfte Zielgröße, die *$\mathbf{v}$-Vorhersage* :cite:`Salimans.Ho.2022` mit $\mathbf{v} = \alpha_t \boldsymbol\epsilon - \sigma_t \mathbf{x}_0$. Auf einem VP-Pfad können wir $\alpha_t = \cos\phi_t$ und $\sigma_t = \sin\phi_t$ schreiben. Dann ist $\mathbf{v} = d\mathbf{x}_t/d\phi_t$ die Geschwindigkeit pro Winkeleinheit statt pro Zeiteinheit. Auf dem Kosinuspfad ohne den kleinen Versatz gilt $\phi_t = \pi t/2$, und $\mathbf{v}$ ist das $2/\pi$-Fache der Geschwindigkeit (Aufgabe :exref:`ex_gaussian-paths-v-to-velocity`). Salimans und Ho führten diese Zielgröße für Sampler mit sehr wenigen Schritten ein. Diese fragen das Netz vorwiegend bei niedrigem Signal-Rausch-Verhältnis ab. Dort erhalten wir den rauschfreien Punkt aus einer Rauschvorhersage erst durch Division durch ein kleines $\alpha_t$. Mit $\mathbf{v}$ benötigen wir keine Division.

<!--en:dabe2726-->
Alle fünf Zielgrößen sind affine Funktionen von $\mathbf{x}_t$ und einem einzigen A-posteriori-Mittelwert. Ihre Koeffizienten hängen nur vom Pfad und vom Zeitpunkt ab. Den exakten A-posteriori-Mittelwert nennen wir das *Orakel* (engl. oracle). Er ist der Prädiktor, den ein perfektes Netz unabhängig von seiner Zielgröße lernen würde. Die Formeln für den Wechsel zwischen den Zielgrößen nennen wir die *Umrechnungsregeln*. :numref:`tab_mdl-score-noise-velocity` führt sie vollständig auf. Die Methode `target` verwendet sie in einer Richtung, von $(\mathbf{x}_0, \boldsymbol\epsilon)$ zu einer beliebigen Zielgröße. Die oben daneben definierte Methode `x0_eps` geht in die Gegenrichtung, von einer beliebigen Vorhersage zu $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$. Trainieren wir ein Netz auf eine Zielgröße, so bestimmt es also auch alle anderen. Das gilt selbst für ein auf einem anderen Pfad trainiertes Netz, wenn wir es zum Zeitpunkt mit gleicher Log-SNR auswerten.

<!--en:43947d35-->
Sehen wir uns die Umrechnung an einem Punkt an. Für die Mischung liefert `oracle` die exakten Werte $\hat{\mathbf{x}}_0$ und $\hat{\boldsymbol\epsilon}$. Damit können wir das Orakel in jeder Verlustfunktion und jedem Sampler anstelle eines Netzes verwenden. Für den Punkt $(0.5, -0.5)$ auf dem VP-Pfad zur Zeit $t = 0.5$ mit $\alpha = 0.28$ und $\sigma = 0.96$ geben wir alle Zielgrößen aus. Die letzte Zeile skaliert den Punkt mit $1/(\alpha + \sigma)$ auf den linearen Pfad bei gleicher Log-SNR um, also auf $t = 0.773$. Dort wertet sie das exakte Feld `velocity` aus, eine andere Formel an einem anderen Punkt. Das Ergebnis sollte $\hat{\boldsymbol\epsilon} - \hat{\mathbf{x}}_0$ sein. Diesen Wert können wir aus den ersten beiden Zeilen im Voraus berechnen.

<!--en:7d695d89-->
Die Vorhersage für den rauschfreien Punkt liegt bei $(0.29, -0.98)$ im Inneren des Dreiecks mit den drei Mittelwerten $(-2.5, -1.5)$, $(2.5, -1.5)$ und $(0, 2.5)$ als Ecken, weit von allen dreien entfernt. Denn bei $\alpha = 0.28$ erstreckt sich die A-posteriori-Verteilung noch über alle drei Modi. Der Score ist die Rauschvorhersage dividiert durch $-0.96$. Das Feld auf dem linearen Pfad liefert $(0.15, 0.75) = \hat{\boldsymbol\epsilon} - \hat{\mathbf{x}}_0$. Fünf Zielgrößen auf zwei Pfaden ergeben sich aus einem einzigen A-posteriori-Mittelwert.

<!--en:3f0cfe1d-->
Auf welche Zielgröße sollten wir also trainieren? Für ein Netz, das das Orakel erreicht, wäre die Wahl gleichgültig. Ein reales Netz erreicht es nicht. In der Praxis entscheiden zwei Gesichtspunkte. Erstens sagt ein Rauschprädiktor bei reinem Rauschen nichts über den rauschfreien Punkt aus, weil $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sigma_t \hat{\boldsymbol\epsilon})/\alpha_t$ dann durch null dividiert. Er gehört also auf einen Pfad, dessen $\alpha_t$ positiv bleibt, etwa den VP-Pfad. Zweitens gewichten die Zielgrößen die Zeitpunkte unterschiedlich. Auf dem linearen Pfad entspricht der Geschwindigkeitsverlust dem mit $1/(1 - t)^2$ gewichteten Rauschverlust und dem mit $1/t^2$ gewichteten Verlust für den rauschfreien Punkt (Aufgabe :exref:`ex_gaussian-paths-prediction-targets`). Bei gleichem quadratischem Geschwindigkeitsfehler ist der Fehler der Rauschvorhersage am Rauschende klein und der Fehler für den rauschfreien Punkt nahe den Daten. In der Praxis überwiegen drei Kombinationen: die Geschwindigkeit auf dem linearen Pfad, $\mathbf{v}$ auf einem Kosinuspfad und das Rauschen auf einem VP-Pfad. Alle drei sind zulässig und lassen sich mit unseren Regeln ineinander umrechnen.

<!--en:364e2a34-->
Der *einfache Verlust* (engl. simple loss) von DDPM :cite:`ho2020denoising` verwendet das Rauschen als Regressionsziel:

<!--en:e3539b04-->
Für jedes Beispiel ist der Score der bedingten Dichte $\mathcal{N}(\alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$ von $\mathbf{x}_t$ bei gegebenem $\mathbf{x}_0$ gleich $-\boldsymbol\epsilon/\sigma_t$. Der einfache Verlust entspricht also dem Denoising-Score-Matching auf jedem Rauschpegel mit dem Gewicht $\sigma_t^2$ (:numref:`sec_mdl-ddpm-discretized-sde`). Ho et al. leiteten ihn aus einer Variationsschranke für die Likelihood her. Anschließend ließen sie die Gewichtung der Schranke über $t$ weg, weil die Samples ohne sie besser aussahen. Unser zweites Modell folgt DDPM und verwendet das Rauschen auf dem VP-Pfad als Zielgröße. Alles andere bleibt gleich.

<!--en:8b6bf4ab-->
Sind beide Netze gut trainiert, stellen sie nach den Umrechnungsregeln trotz unterschiedlicher Pfade und Zielgrößen denselben A-posteriori-Mittelwert dar. Anhand der Verlustuntergrenze können wir das prüfen.

<!--en:576df726-->
### Die Verlustuntergrenze

<!--en:43d1e1e1-->
Warum hat sich der Geschwindigkeitsverlust nahe $5$ eingependelt? Jede Zielgröße enthält das Rauschen ihres eigenen Beispiels. Kein Netz kann dieses Rauschen aus $\mathbf{x}_t$ vorhersagen. Selbst beim Orakel verbleibt als Verlust die über den Pfad gemittelte bedingte Varianz der Zielgröße. Diese *Verlustuntergrenze* haben wir im vorigen Abschnitt für einen einzelnen Rauschpegel kennengelernt. Sie ist die Konstante $C$ aus :eqref:`eq_diffusion-cfm-loss`. Bei festem $\mathbf{x}_t$ ist das Rauschen gleich $(\mathbf{x}_t - \alpha_t \mathbf{x}_0)/\sigma_t$. Jede Zielgröße ist also die Summe aus $c_t \mathbf{x}_0$ und einer Funktion von $\mathbf{x}_t$. Ihre Untergrenze zum Zeitpunkt $t$ ist

<!--en:d848d5fb-->
Für den rauschfreien Punkt gilt $c_t = 1$, für das Rauschen $c_t = -\alpha_t/\sigma_t$ und für die Geschwindigkeit des linearen Pfades $c_t = -1/t$. Die Zielgrößen unterscheiden sich nur darin, wie stark sie dieselbe A-posteriori-Varianz skalieren.

<!--en:7112c213-->
Nahe den Daten können wir den Zahlenwert vorhersagen. Bei $t = 0.1$ liegen die Modi im Vergleich zum Rauschpegel noch weit auseinander. Daher entspricht die A-posteriori-Verteilung der eines einzigen gaußschen Modus mit der Varianz $s^2 \sigma_t^2 / (\alpha_t^2 s^2 + \sigma_t^2)$ pro Koordinate. Multiplizieren wir diese Varianz mit $c_t^2$ und summieren über beide Koordinaten, erhalten wir die Untergrenze $2\alpha_t^2 s^2/(\alpha_t^2 s^2 + \sigma_t^2)$ für die Rauschvorhersage und die Untergrenze $2 s^2/(\alpha_t^2 s^2 + \sigma_t^2)$ für die Geschwindigkeit auf dem linearen Pfad, wo $\sigma_t = t$ gilt. Für $s = 0.5$ ergibt sich auf dem VP-Pfad ($\alpha = 0.95$, $\sigma = 0.32$) eine Untergrenze von etwa $1.37$ für die Rauschvorhersage. Auf dem linearen Pfad ($\alpha = 0.9$, $\sigma = 0.1$) beträgt die Untergrenze für die Geschwindigkeit $2.35$.

<!--en:e2a625ea-->
Die Codezelle misst jede Untergrenze anhand einer Million Ziehungen, sowohl zu drei Zeitpunkten als auch gemittelt über $t$, jeweils mit Standardfehler. Der Verlust jedes Netzes zerfällt in die Untergrenze und den mittleren quadratischen Abstand seiner Vorhersage von der des Orakels. Denn der Fehler des Orakels ist mit jeder Funktion von $\mathbf{x}_t$ unkorreliert. Die Zelle misst den Überschussverlust (engl. excess loss) des Netzes mit gemittelten Gewichten daher direkt über diesen Abstand (Spalte `excess`). Daneben gibt sie den Trainingsverlust der letzten $1{,}000$ Schritte aus (Spalte `train loss`). Die letzte Zeile verwendet das Geschwindigkeitsnetz als Rauschprädiktor auf dem VP-Pfad. Dazu verschiebt `eps_from_velocity` jeden Punkt bei gleicher Log-SNR auf den linearen Pfad, fragt dort das Geschwindigkeitsnetz ab und rechnet seine Antwort in $\hat{\boldsymbol\epsilon}$ um, mithilfe von `x0_eps`.

<!--en:7f80245f-->
Die Untergrenzen bei $t = 0.1$ stimmen bis auf $0.01$ mit unserer analytischen Schätzung für einen einzelnen Modus überein. Entlang des Pfades bewegen sie sich in entgegengesetzte Richtungen. Die Untergrenze für die Rauschvorhersage ist nahe $t = 0$ am höchsten, weil sich schwaches Rauschen kaum von der Streuung innerhalb eines Modus unterscheiden lässt. Sie fällt auf $0.002$ bei $t = 0.9$. Die Untergrenze für die Geschwindigkeit hingegen steigt auf $\approx 9$ bei $t = 0.9$ und geht dann gegen $\operatorname{tr}\textrm{Cov}[\mathbf{x}_0]$ bei $t = 1$ (Aufgabe :exref:`ex_gaussian-paths-loss-floor`). Im Mittel über $t \sim U(0, 1)$ unterscheiden sich die Untergrenzen um eine Größenordnung ($0.54$ für Rauschen gegenüber $4.9$ für Geschwindigkeit). Das liegt am Skalierungsfaktor $c_t^2$ der Zielgrößen und an den unterschiedlichen Zeitskalen der beiden Pfade.

<!--en:da58ebf7-->
Beide trainierten Netze erreichen ihre theoretischen Untergrenzen: Der mittlere quadratische Fehler gegenüber dem exakten Orakel (`excess`) beträgt $0.0002$ für das Rauschnetz und $0.001$ für das Geschwindigkeitsnetz. Beide Werte liegen deutlich unter dem Monte-Carlo-Standardfehler der Untergrenze selbst. Verwenden wir das Geschwindigkeitsnetz des linearen Pfades über `eps_from_velocity` als VP-Rauschprädiktor, so erhalten wir sogar genau denselben Verlust ($0.540$) und Überschussverlust ($0.0002$) wie beim direkt auf VP-Rauschen trainierten Netz. Beide Netze haben denselben zugrunde liegenden A-posteriori-Mittelwert gelernt. (Für die ungewichtete Score-Zielgröße $-\boldsymbol\epsilon/\sigma_t$ müssen wir die Untergrenze der Rauschvorhersage durch $\sigma_t^2$ dividieren. Sie divergiert deshalb für $t \to 0$. Score-Netze gewichten das Trainingsziel des Denoising-Score-Matchings daher mit $\sigma_t^2$ und führen es damit auf die Rauschvorhersage zurück.) Wie wir aus den gelernten Feldern Samples gewinnen, behandelt :numref:`sec_diffusion-integration`.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:759327f9-->
Ein Gauß-Pfad $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$ interpoliert zwischen rauschfreien Daten bei $t = 0$ und gaußschem Rauschen bei $t = 1$. Alle solchen Pfade haben eine gemeinsame natürliche Zeitskala, das logarithmische Signal-Rausch-Verhältnis $\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$. Das schrittweise Absenken des Rauschpegels löst das Dilemma des einzelnen Pegels: Bei starkem Rauschen können die Punkte die geglätteten Täler überwinden, sodass sich die globalen Gewichte der Modi einstellen. Bei schwachem Rauschen schärft sich jeder einzelne Modus. Durch Regression auf die bedingte Zielgröße jedes Beispiels ($\mathbf{x}_0$, $\boldsymbol\epsilon$, $-\boldsymbol\epsilon/\sigma_t$, $\mathbf{v}$ oder $\dot\alpha_t \mathbf{x}_0 + \dot\sigma_t \boldsymbol\epsilon$) lernt ein Netz, den zugehörigen bedingten Erwartungswert bei gegebenem $\mathbf{x}_t$ auszugeben. Alle fünf Zielgrößen sind affine Funktionen von $\mathbf{x}_t$ und $\hat{\mathbf{x}}_0 = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$. Ein einziger trainierter Prädiktor bestimmt daher alle fünf auf jedem Gauß-Pfad. Jedes dieser Regressionsziele hat eine positive Verlustuntergrenze $c_t^2\, \mathbb{E}\operatorname{tr}\textrm{Cov}[\mathbf{x}_0 \mid \mathbf{x}_t]$, da die jeweilige Realisierung des Rauschens unbeobachtet bleibt. Die Genauigkeit eines Prädiktors messen wir deshalb am Überschussverlust über dieser Untergrenze und nicht am reinen Trainingsverlust.

<!--en:2294243a-->
## Übungen

<!--en:30930e21-->
1. **Von $\mathbf{v}$ zur Geschwindigkeit.** Für den Punkt $\mathbf{x} = (0.5, -0.5)$ auf dem VP-Pfad zur Zeit $t = 0.5$ mit $\alpha_t = 0.28$ und $\sigma_t = 0.96$ haben wir im Text die exakte Vorhersage $\hat{\mathbf{v}} = (-0.15, 0.88)$ ausgegeben.
    1. Auf einem VP-Pfad gilt $\alpha_t^2 + \sigma_t^2 = 1$. Zeigen Sie, dass $\hat{\mathbf{x}}_0 = \alpha_t \mathbf{x} - \sigma_t \hat{\mathbf{v}}$ und $\hat{\boldsymbol\epsilon} = \sigma_t \mathbf{x} + \alpha_t \hat{\mathbf{v}}$ gelten, und berechnen Sie die ausgegebenen Werte von $\hat{\mathbf{x}}_0$ und $\hat{\boldsymbol\epsilon}$ von Hand. Das Runden der Eingaben auf zwei Stellen führt zu einer Abweichung von etwa $0.01$.
    1. Schreiben Sie $\alpha_t = \cos\phi_t$ und $\sigma_t = \sin\phi_t$. Zeigen Sie, dass die Geschwindigkeit $\mathbf{u}_t = \dot\phi_t\, \mathbf{v}$ beträgt und $\mathbf{v}$ somit die Geschwindigkeit pro Winkeleinheit ist. Leiten Sie daraus den Faktor $2/\pi$ für den Kosinuspfad ab.
    1. Zeigen Sie für den VP-Pfad dieses Abschnitts, dass $\dot\phi_t = \beta(t)\, \alpha_t / (2 \sigma_t)$ gilt. Berechnen Sie diesen Wert bei $t = 0.5$ und bestimmen Sie die ausgegebene Geschwindigkeit $(-0.22, 1.29)$ aus $\hat{\mathbf{v}}$ bis auf die Rundungsabweichung aus dem ersten Teil.
   :exlabel:`ex_gaussian-paths-v-to-velocity`

<!--en:78362d2f-->
1. **Geschwindigkeit, Rauschen und rauschfreier Punkt als Zielgrößen.** Bilden Sie auf dem linearen Pfad den A-posteriori-Mittelwert beider Seiten von $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\boldsymbol\epsilon$.
    1. Zeigen Sie, dass die marginale Geschwindigkeit sowohl $(\mathbf{x} - \hat{\mathbf{x}}_0)/t$ als auch $(\hat{\boldsymbol\epsilon} - \mathbf{x})/(1 - t)$ entspricht.
    1. Nach dem ersten Teil geht ein Fehler in $\hat{\mathbf{x}}_0$ dividiert durch $t$ in die Geschwindigkeit ein, ein Fehler in $\hat{\boldsymbol\epsilon}$ dividiert durch $1 - t$. An welchem Ende des Pfades lässt sich die Geschwindigkeit aus der jeweiligen Vorhersage nur unzuverlässig bestimmen? Wie erklärt das, warum man ein Rauschmodell nie bei $\alpha = 0$ abfragt? (Aufgabe :exref:`ex_mdl-flow-matching-transport-target-conditioning` in :numref:`sec_mdl-flow-matching-transport` untersucht die Fehlerfortpflanzung zwischen allen Zielgrößen auf dem Kosinuspfad.)
    1. Zeigen Sie, dass der Geschwindigkeitsverlust für $\mathbf{u}_{\boldsymbol\theta} = (\mathbf{x} - \hat{\mathbf{x}}_{0,\boldsymbol\theta})/t$ gleich dem Verlust $\mathbb{E}\|\hat{\mathbf{x}}_{0,\boldsymbol\theta} - \mathbf{x}_0\|^2$ für den rauschfreien Punkt mit dem Gewicht $1/t^2$ ist. Zeigen Sie ebenso, dass der Geschwindigkeitsverlust für $\mathbf{u}_{\boldsymbol\theta} = (\hat{\boldsymbol\epsilon}_{\boldsymbol\theta} - \mathbf{x})/(1 - t)$ dem Rauschverlust mit dem Gewicht $1/(1 - t)^2$ entspricht. Welche Zeiten gewichtet das Flow-Matching im Vergleich zu den beiden ungewichteten Verlusten jeweils stärker?
   :exlabel:`ex_gaussian-paths-prediction-targets`

<!--en:ee1080ed-->
1. [code] **Die Verlustuntergrenze.** Verwenden Sie `oracle` und `floor_and_excess`.
    1. Zeichnen Sie die Untergrenzen für die Geschwindigkeit auf dem linearen Pfad und für das Rauschen auf dem VP-Pfad als Funktionen von $t$ auf einem Gitter in $(0, 1)$. Wo ist die jeweilige Untergrenze am höchsten? Der ausgegebene Trainingsverlust ist ein Mittelwert über $1{,}000$ Minibatches zu je $512$ Punkten. Schätzen Sie seinen Standardfehler anhand von `losses_v` und vergleichen Sie ihn mit dem Abstand zwischen ausgegebenem Trainingsverlust und Untergrenze.
    1. Sei $\boldsymbol\Sigma$ die Kovarianzmatrix der Mischung mit den Eigenwerten $\kappa_1, \kappa_2$. Zeigen Sie für gaußsche Daten mit dieser Kovarianz, dass die Untergrenze für die Geschwindigkeit auf dem linearen Pfad zur Zeit $t$ gleich $\sum_i \kappa_i / \big((1 - t)^2 \kappa_i + t^2\big)$ ist. Ergänzen Sie das Diagramm um diese Kurve und erklären Sie, warum sie nicht unter der Untergrenze der Mischung liegen kann. Warum stimmen beide Untergrenzen an beiden Enden des Pfades überein?
    1. Die Untergrenze für die Score-Zielgröße $-\boldsymbol\epsilon/\sigma_t$ ist die Untergrenze der Rauschvorhersage dividiert durch $\sigma_t^2$. Berechnen Sie sie auf dem VP-Pfad bei $t = 0.01$, $0.1$ und $0.5$. Erklären Sie, warum man kein Netz auf diese Zielgröße trainiert, wenn $t$ auf $(0, 1)$ gleichverteilt ist.
    1. Der Trainingsverlust des Rauschmodells beträgt etwa ein Zehntel des Trainingsverlusts des Geschwindigkeitsmodells. Ist es das bessere Modell? Messen Sie zu den VP-Zeitpunkten $t \in \{0.1, 0.5, 0.9\}$ den Fehler jedes Netzes gegenüber dem Orakel in einer gemeinsamen Einheit, $\mathbb{E}\|\hat{\mathbf{x}}_0 - \hat{\mathbf{x}}_0^\star\|^2$. Dabei stammt $\hat{\mathbf{x}}_0^\star$ aus `oracle`. Rechnen Sie `ema_eps` mit `vp.x0_eps` um. Verwenden Sie für `ema_v` zuerst `eps_from_velocity` und dann $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sigma_t \hat{\boldsymbol\epsilon})/\alpha_t$. Erklären Sie anhand der Faktoren $c_t$ aus dem Text, warum sich die Rohverluste so stark unterscheiden. Welcher Vergleich ergibt eine Rangfolge der beiden Modelle: die Rohverluste, die Überschussverluste über den jeweiligen Untergrenzen oder die Fehler in einer gemeinsamen Einheit?

    *Teil (d) ist angelehnt an CMU 10-799, [Hausaufgabe 2](https://kellyyutonghe.github.io/10799S26/assets/pdf/hw2.pdf) (Frühjahr 2026), Frage 3(a).*
   :exlabel:`ex_gaussian-paths-loss-floor`

<!--en:c888bbb3-->
1. **Abstand der Rauschstufen in hohen Dimensionen.** ● Die Daten sollen aus einem einzigen Punkt im Ursprung von $\mathbb{R}^d$ bestehen, sodass $p_{\tilde\sigma} = \mathcal{N}(\mathbf{0}, \tilde\sigma^2 \mathbf{I})$ gilt. Die Rauschpegel sollen eine geometrische Folge mit dem Quotienten $q = \tilde\sigma_{i-1}/\tilde\sigma_i > 1$ bilden.
    1. Zeigen Sie, dass für $r = \|\mathbf{x}\|$ mit $\mathbf{x} \sim p_{\tilde\sigma}$ die Beziehungen $\mathbb{E}[r^2] = d\tilde\sigma^2$ und $\textrm{Var}[r^2] = 2d\tilde\sigma^4$ gelten. Folgern Sie, dass $r$ näherungsweise die Verteilung $\mathcal{N}(\tilde\sigma\sqrt d, \tilde\sigma^2/2)$ hat, wenn $d$ groß ist.
    1. Verlässt eine Kette die Stufe $i - 1$ mit einem Sample aus $p_{\tilde\sigma_{i-1}}$, so beginnt sie die Stufe $i$ mit einem Radius, der mit hoher Wahrscheinlichkeit in $\tilde\sigma_{i-1}(\sqrt d \pm 3/\sqrt 2)$ liegt. Zeigen Sie, dass $p_{\tilde\sigma_i}$ diesem Intervall die Masse $\Phi\big(\sqrt{2d}(q - 1) + 3q\big) - \Phi\big(\sqrt{2d}(q - 1) - 3q\big)$ zuweist. Dabei bezeichnet $\Phi$ die Verteilungsfunktion der Standardnormalverteilung. Warum bedeutet eine Masse nahe null, dass die Kette diese Stufe in einem Bereich beginnt, in dem das Netz kaum Trainingseingaben gesehen hat?
    1. :citet:`Song.Ermon.2020` wählen $q$ so, dass diese Masse etwa ein Halb beträgt. Bestimmen Sie $q$ für $d = 3072$, die Dimension von CIFAR-10-Bildern. Zählen Sie die Stufen von $\tilde\sigma = 50$ bis hinunter zu $0.01$. Vergleichen Sie das Ergebnis mit den $232$ Stufen ihres CIFAR-10-Modells.
    1. Die oberste Stufe muss außerdem grob genug sein, damit die Modi verschmelzen. Zwei gleiche gaußsche Komponenten bei $\pm\mu$ mit Varianz $v$ haben genau dann einen einzigen Modus, wenn $\mu^2 \le v$ gilt (Aufgabe :exref:`ex_mdl-fokker-planck-probability-flow-mode-merging` in :numref:`sec_mdl-fokker-planck-probability-flow`). Schätzen Sie mit diesem Kriterium für die beiden unteren Modi der Mischung den Rauschpegel, bei dem sie verschmelzen. Vergleichen Sie ihn mit dem höchsten Pegel $3$ der Stufenfolge.
   :exlabel:`ex_gaussian-paths-ladder-spacing`

<!--en:827c8cfd-->
5. [code] **Das exakte Optimum lernt auswendig.** Ein endlicher Trainingsdatensatz ist selbst eine Verteilung, nämlich eine Mischung aus Punktmassen. Ihr exaktes Geschwindigkeitsfeld ist bekannt. :citet:`Bertrand.Gagneux.Massias.ea.2025` gehen von der Beobachtung aus, dass wir die Trainingspunkte reproduzieren, wenn wir diesem Feld vom Rauschen aus folgen. Sie fragen, warum trainierte Netze stattdessen generalisieren.
    1. Ziehen Sie $64$ Punkte aus der Mischung und trainieren Sie `TimeMLP(256, 3)` mit `regression_loss(linear, 'velocity')` auf Batches, die Sie ausschließlich aus diesen Punkten ziehen.
    1. `d2l.GaussianMixture(points.tolist(), [1/64] * 64, 0.0)` ist die empirische Verteilung (übergeben Sie auch `device`). Ihre Methode `velocity` liefert das exakte Feld für $t > 0$. Folgen Sie diesem Feld und dem trainierten Netz, jeweils ausgehend von denselben $2{,}000$ standardnormalverteilten Startpunkten. Gehen Sie $100$ geradlinige Schritte $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{u}_t(\mathbf{x})/100$ zu den Zeitpunkten $t = 1, 0.99, \ldots, 0.01$. Zeichnen Sie für beide Mengen von Samples je ein Histogramm der Abstände zum nächstgelegenen Trainingspunkt.
    1. Welches Feld reproduziert den Trainingsdatensatz und welches erzeugt neue Punkte? Bertrand et al. argumentieren, dass die Generalisierung des Netzes nicht auf das Rauschen in den Regressionszielen zurückzuführen ist. Wiederholen Sie die ersten beiden Teile mit $1{,}024$ Trainingspunkten und beschreiben Sie, wie sich die beiden Histogramme ändern.
   :exlabel:`ex_gaussian-paths-memorization`

<!--en:fca5f435-->
5. [code] **Das exakte Optimum lernt auswendig.** Ein endlicher Trainingsdatensatz ist selbst eine Verteilung, nämlich eine Mischung aus Punktmassen. Ihr exaktes Geschwindigkeitsfeld ist bekannt. :citet:`Bertrand.Gagneux.Massias.ea.2025` gehen von der Beobachtung aus, dass wir die Trainingspunkte reproduzieren, wenn wir diesem Feld vom Rauschen aus folgen. Sie fragen, warum trainierte Netze stattdessen generalisieren.
    1. Ziehen Sie $64$ Punkte aus der Mischung und trainieren Sie `TimeMLP(256, 3)` mit `regression_loss(linear, 'velocity')` auf Batches, die Sie ausschließlich aus diesen Punkten ziehen.
    1. `d2l.GaussianMixture(points.tolist(), [1/64] * 64, 0.0)` ist die empirische Verteilung. Ihre Methode `velocity` liefert das exakte Feld für $t > 0$. Folgen Sie diesem Feld und dem trainierten Netz, jeweils ausgehend von denselben $2{,}000$ standardnormalverteilten Startpunkten. Gehen Sie $100$ geradlinige Schritte $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{u}_t(\mathbf{x})/100$ zu den Zeitpunkten $t = 1, 0.99, \ldots, 0.01$. Zeichnen Sie für beide Mengen von Samples je ein Histogramm der Abstände zum nächstgelegenen Trainingspunkt.
    1. Welches Feld reproduziert den Trainingsdatensatz und welches erzeugt neue Punkte? Bertrand et al. argumentieren, dass die Generalisierung des Netzes nicht auf das Rauschen in den Regressionszielen zurückzuführen ist. Wiederholen Sie die ersten beiden Teile mit $1{,}024$ Trainingspunkten und beschreiben Sie, wie sich die beiden Histogramme ändern.
   :exlabel:`ex_gaussian-paths-memorization`

<!--en:d430f574-->
[Diskussion](https://d2l.discourse.group/t/36888)

<!--en:790b9f7c-->
Gauß-Pfade für Wahrscheinlichkeitsverteilungen: Diffusion und Flow-Matching<br>**VP-, lineare und Kosinus-Zeitpläne · Zeitskala der Log-SNR · Langevin-Sampling mit sinkendem Rauschpegel · bedingtes Flow-Matching · Umrechnungsregeln für Zielgrößen · Verlustuntergrenzen**

<!--en:e9c23ed7-->
::: {.slide title="Warum ein Kontinuum von Rauschpegeln nötig ist"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:c9abeec0-->
- **Starkes Rauschen ($\sigma \gg s$) bestimmt die globalen Gewichte:** Füllt die gaußsche Glättung die $10\text{--}12$ Nat tiefen Täler zwischen den Modi, stellen sich entlang der Trajektorien die tatsächlichen Anteile $0.5 / 0.3 / 0.2$ ein.

<!--en:2a165f12-->
- **Schwaches Rauschen ($\sigma \ll s$) macht die lokale Geometrie sichtbar:** Für $\sigma \to 0$ gilt $\nabla\log p_\sigma \to \nabla\log p$. Dadurch zieht es die Punkte dicht an das Zentrum des jeweiligen Modus.

<!--en:fd62e17e-->
- **Ein gemeinsamer stetiger Pfad:** $\mathbf{x}_t = \alpha_t\mathbf{x}_0 + \sigma_t\boldsymbol\epsilon$ über $t \in [0, 1]$ interpolieren und **ein einziges zeitabhängiges Netz** über alle Skalen trainieren

<!--en:0f8fd40c-->
::: {.slide title="Annealed Langevin Dynamics mit exakten Scores"}

<!--en:ab836e82-->
[Prüfung der Idee vor dem Training]{.kicker}

<!--en:54c11b0e-->
Vor dem Netztraining lässt sich das schrittweise Absenken des Rauschpegels auf mehreren Skalen testen. Dazu dienen die exakten Scores $\nabla \log p_{\tilde\sigma}$ der Mischung auf einer geometrischen Folge von 10 Rauschpegeln $\tilde\sigma_i \in [3.0, 0.05]$ ($100$ Schritte pro Stufe, Schrittweite $h_i = 0.1\tilde\sigma_i^2$):

<!--en:9adaa085-->
- **Eine feine Stufe ($1{,}000$ Schritte bei $\tilde\sigma = 0.05$):** Die Ketten bleiben in ihren anfänglichen Modi gefangen und liefern gleiche Gewichte $\approx [0.33, 0.33, 0.33]$.

<!--en:f82645c9-->
- **Zehn absteigende Stufen ($10 \times 100$ Schritte):** Auf den groben Stufen wechseln die Ketten über geglättete Täler hinweg zwischen den Modi. Die feinen Stufen schärfen jeden Modus. Die Gewichte entsprechen $[0.50, 0.30, 0.20]$ bis auf $\pm 0.03$, mit $\approx 1.5\%$ weit abseits liegenden Punkten.

<!--en:8f8e15ee-->
- **Warum ein stetiger Pfad die Stufenfolge ablöst:** In hohen Dimensionen reagieren die Ergebnisse empfindlich auf die von Hand gewählten Stufen $\tilde\sigma_i$ und Schrittweiten $h_i$. Ein stetiger Pfad $t \in [0, 1]$ ersetzt die Stufenfolge durch eine glatte gewöhnliche oder stochastische Differentialgleichung.

<!--en:b19a953c-->
::: {.slide title="Gauß-Pfade für Wahrscheinlichkeitsverteilungen: VE, VP und linearer Pfad"}

<!--en:c6d70f98-->
[Zeitpläne der Vorwärtsinterpolation]{.kicker}

<!--en:124fd6f7-->
Auf dem varianzexplodierenden Pfad (VE) $\tilde{\mathbf{x}}_t = \mathbf{x}_0 + \sigma_t\boldsymbol\epsilon$ wachsen die Normen der Eingaben auf das Fünfzigfache. Moderne Zeitpläne verkleinern den Signalkoeffizienten $\alpha_t$ mit zunehmendem Rauschen $\sigma_t$. So bleiben die Eingaben von der Größenordnung $O(1)$:

<!--en:c9cd8648-->
- **Varianzerhaltend (VP / DDPM):** Die Bedingung $\alpha_t^2 + \sigma_t^2 = 1$ gilt bei linearer Rauschrate $\beta(t) = 0.1 + 19.9t$:
  $$\bar\alpha_t = \alpha_t^2 = \exp\!\Big(-\int_0^t \beta(r)\,dr\Big)$$

<!--en:b689d64e-->
- **Linearer Pfad (Flow-Matching / Rectified Flow):** Verbindet $\mathbf{x}_0$ mit $\boldsymbol\epsilon$ entlang einer Geraden mit konstanter Geschwindigkeit für $t \in [0, 1]$: $\alpha_t = 1 - t$ und $\sigma_t = t$.

<!--en:187b8cae-->
::: {.slide title="Die gemeinsame Zeitskala: logarithmisches Signal-Rausch-Verhältnis"}

<!--en:7a28b62e-->
[Äquivalenz der Zeitpläne]{.kicker}

<!--en:810385e5-->
- **Exakte Umskalierung auf den linearen Pfad:** Nach Division von $\mathbf{x}_t = \alpha_t\mathbf{x}_0 + \sigma_t\boldsymbol\epsilon$ durch $\alpha_t + \sigma_t$ ($c = 1/(\alpha_t + \sigma_t)$) liegt das Ergebnis auf dem linearen Pfad bei $t_{\textrm{lin}} = \sigma_t/(\alpha_t + \sigma_t)$ mit derselben Log-SNR $\lambda_t$.

<!--en:c8d4a14b-->
- **Zeitplan = Umparametrisierung der Zeit:** Alle Gauß-Pfade durchlaufen dieselbe durch $\lambda \in (-\infty, +\infty)$ indizierte Familie von Verteilungen. Nur die Geschwindigkeit auf dieser Zeitskala unterscheidet sich.

<!--en:5603f1a9-->
::: {.slide title="Zeitskalen von VP-, linearem und Kosinuspfad im Codevergleich"}

<!--en:27488ad1-->
[Zeitliche Verteilung]{.kicker}

<!--en:5387826d-->
Die Auswertung von `GaussianPath.vp()` und `GaussianPath.linear()` bei $t \in \{0.1, 0.5, 0.9\}$ zeigt, wie unterschiedlich sich die Zeit über die Log-SNR verteilt:

<!--en:ebf2f702-->
**Warum Nichol und Dhariwal (2021) den Kosinus-Zeitplan einführten:** Der VP-Pfad mit linearem $\beta$ erreicht $\lambda_t = 0$ (gleich starkes Signal und Rauschen) bei $t \approx 0.26$. Während $74\%$ seiner Zeit überwiegt das Rauschen. Auf dem **Kosinuspfad** $\alpha_t \approx \cos(\pi t / 2)$ sinkt die Log-SNR über $[0, 1]$ viel gleichmäßiger.

<!--en:6d0cc269-->
::: {.slide title="Die DDPM-Markov-Kette entspricht dem VP-Pfad auf einem Gitter"}

<!--en:afbfa960-->
[Zeitdiskreter Vorwärtsprozess]{.kicker}

<!--en:f064a467-->
Ho et al. (2020) definierten DDPM als Markov-Kette mit $T$ Schritten. Sie skaliert den Zustand mit $\sqrt{1 - \beta_k}$ und addiert unabhängiges gaußsches Rauschen $\sqrt{\beta_k}\,\boldsymbol\epsilon_k$ in jedem Schritt $k = 1, \ldots, T$:

<!--en:68ddb002-->
- **Randverteilung nach $k$ Schritten in geschlossener Form:** Bei unabhängigen normalverteilten Größen addieren sich die Varianzen. Daher reduzieren sich $k$ Schritte ab $\mathbf{x}_0$ auf eine einzige Ziehung mit $\bar\alpha_k = \prod_{j=1}^k (1 - \beta_j)$:
  $$\mathbf{x}_k = \sqrt{\bar\alpha_k}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_k}\,\boldsymbol\epsilon$$

<!--en:b44a98ac-->
- **Training ohne Simulation der Kette:** Das Training durchläuft nie $k$ Schritte nacheinander. Jedes $\mathbf{x}_k$ lässt sich in $O(1)$ aus $(\mathbf{x}_0, \boldsymbol\epsilon)$ ziehen.

<!--en:fc190140-->
::: {.slide title="Bedingte Geschwindigkeit und marginales Geschwindigkeitsfeld"}

<!--en:49a0efea-->
[Formulierung des Flow-Matchings]{.kicker}

<!--en:b6a4afd3-->
- **Bedingte Geschwindigkeit $\dot{\mathbf{x}}_t = \dot\alpha_t\mathbf{x}_0 + \dot\sigma_t\boldsymbol\epsilon$:** Geschwindigkeit eines einzelnen Paares (auf dem linearen Pfad $\boldsymbol\epsilon - \mathbf{x}_0$). Die Verbindungsstrecken verschiedener Paare kreuzen sich überall im Raum.

<!--en:e743313d-->
- **Marginales Geschwindigkeitsfeld $\mathbf{u}_t(\mathbf{x})$:** A-posteriori-Mittelwert der Geschwindigkeiten aller Trajektorien durch $(\mathbf{x}, t)$. So entsteht ein eindeutiges Vektorfeld, das $p_t$ transportiert.

<!--en:e02863a8-->
::: {.slide title="Bedingtes und marginales Feld im Bild"}

<!--en:b511f6e4-->
[Geometrie der zweidimensionalen Mischung]{.kicker}

<!--en:4a7ceddf-->
Für die Mischung mit drei Modi lassen sich sowohl der bedingte Pfad $p_t(\mathbf{x} \mid \mathbf{x}_0)$ eines einzelnen Punktes als auch das gesamte marginale Geschwindigkeitsfeld $\mathbf{u}_t(\mathbf{x})$ in geschlossener Form darstellen:

<!--en:6cf2eeb2-->
- **(a) Bedingter Pfad:** Wachsende Kreise mit Radius $2\sigma_t$ um $\alpha_t\mathbf{x}_0$ für einen einzelnen Datenpunkt $\mathbf{x}_0$.

<!--en:25d7c4dd-->
- **(b) Marginale Dichte $p_t$:** Läuft $t$ von $0$ bis $1$, verschmelzen die drei scharf ausgeprägten Modi zu einer einzigen Standardnormalverteilung $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

<!--en:621b6423-->
- **(c) Bedingtes Feld $-(\mathbf{x} - \mathbf{x}_0)/t$:** Zeigt direkt auf das einzelne Trainingsbeispiel $\mathbf{x}_0$.

<!--en:c18b19b5-->
- **(d) Marginales Feld $-\mathbf{u}_t(\mathbf{x})$:** Mittelt über alle möglichen rauschfreien Punkte mit Gewichten $p(\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x})$. Das glatte Feld zeigt in die drei Modi hinein.

<!--en:e0c02b4c-->
::: {.slide title="Regression beim bedingten Flow-Matching (CFM)"}

<!--en:fd47286a-->
[Training ohne Kenntnis des marginalen Feldes]{.kicker}

<!--en:d360e3a3-->
- **Identität des bedingten Flow-Matchings (Lipman et al., 2022):** Konditionieren auf $\mathbf{x}_t$ zerlegt den Verlust in den Fehler gegenüber dem marginalen Feld und eine von $\boldsymbol\theta$ unabhängige, varianzbedingte Untergrenze $C$.

<!--en:77d1a4a2-->
- **Identische Gradienten:** Wegen $\nabla_{\boldsymbol\theta} C = \mathbf{0}$ hat der stochastische Gradientenabstieg mit zufälligen Paaren $(\mathbf{x}_0, \boldsymbol\epsilon)$ im Erwartungswert exakt denselben Gradienten wie die Regression auf das unbekannte Orakel $\mathbf{u}_t(\mathbf{x}_t)$.

<!--en:aa14f0c8-->
::: {.slide title="Training eines Geschwindigkeitsnetzes in fünf Codezeilen"}

<!--en:5e1f4461-->
[Implementierung und Konvergenz]{.kicker}

<!--en:7ec8f50e-->
Beim Training von `TimeMLP(256, 3)` auf dem linearen Pfad besteht jeder neue Batch aus $512$ Punkten der Mischung, gleichverteilten Zeitpunkten $t \sim U(0, 1)$ und Rauschen $\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Das Training umfasst $40{,}000$ Adam-Schritte:

<!--en:0075f2ce-->
- **Zielgröße pro Beispiel:** Auf dem linearen Pfad ergibt `path.target('velocity', x0, eps, t)` direkt `eps - x0`.

<!--en:6797a842-->
- **Sinusförmige Zeitmerkmale:** `TimeMLP` bettet $t \in [0, 1]$ mit geometrisch gestaffelten Frequenzen von $1$ bis $64\,\text{rad}$ ein. So bleiben benachbarte Rauschpegel unterscheidbar.

<!--en:f07a8405-->
- **Warum erreicht der Verlust nahe $4.9$ ein Plateau?** Die Trainingskurve fällt rasch und flacht weit über null ab. Das liegt an der unvermeidbaren, varianzbedingten Untergrenze $C$. Das Training ist nicht ins Stocken geraten.

<!--en:6734bd4c-->
::: {.slide title="Umrechnungsregeln: fünf Sichtweisen auf ein Orakel"}

<!--en:5c6ea3ae-->
[Score, Rauschen, rauschfreier Punkt, v und Geschwindigkeit]{.kicker}

<!--en:3cd3cdc1-->
- **Ein Orakel für die A-posteriori-Verteilung, fünf Koordinaten:** Da der bedingte Erwartungswert linear ist, bestimmen die A-posteriori-Mittelwerte $\hat{\mathbf{x}}_0 = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$ und $\hat{\boldsymbol\epsilon} = \mathbb{E}[\boldsymbol\epsilon \mid \mathbf{x}_t]$ alle fünf Zielgrößen:
  - **Score:** $\nabla\log p_t(\mathbf{x}_t) = -\hat{\boldsymbol\epsilon}/\sigma_t$
  - **Winkelgeschwindigkeit:** $\hat{\mathbf{v}} = \alpha_t\hat{\boldsymbol\epsilon} - \sigma_t\hat{\mathbf{x}}_0$
  - **Marginale Geschwindigkeit:** $\mathbf{u}_t = \dot\alpha_t\hat{\mathbf{x}}_0 + \dot\sigma_t\hat{\boldsymbol\epsilon}$

<!--en:c80048fb-->
::: {.slide title="Die Identität zwischen Score und Geschwindigkeit auf verschiedenen Pfaden"}

<!--en:7ce64d8a-->
[Zusammenhang zwischen Score-Modellen und Flow-Matching]{.kicker}

<!--en:884f5937-->
- **Linearer Pfad ($\alpha_t = 1 - t, \sigma_t = t$):** $\mathbf{u}_t(\mathbf{x}) = \hat{\boldsymbol\epsilon} - \hat{\mathbf{x}}_0 = -\frac{\mathbf{x} + t\nabla\log p_t(\mathbf{x})}{1 - t}$.

<!--en:7fcb7149-->
- **VP-Pfad ($\alpha_t^2 + \sigma_t^2 = 1$):** $\mathbf{u}_t(\mathbf{x}) = -\frac{1}{2}\beta(t)\big(\mathbf{x} + \nabla\log p_t(\mathbf{x})\big)$.

<!--en:1d837a58-->
- **Ergebnis:** Geschwindigkeitsnetze beim Flow-Matching und scorebasierte Diffusionsmodelle lernen exakt dieselbe Größe in unterschiedlichen Koordinaten.

<!--en:67e46e69-->
::: {.slide title="Prüfung der Umrechnungsregeln an einem einzelnen Punkt"}

<!--en:8b60e5c1-->
[Umrechnung zwischen Pfaden im Code]{.kicker}

<!--en:83f3e74a-->
Am Punkt $\mathbf{x} = (0.5, -0.5)$ des VP-Pfades zur Zeit $t = 0.5$ ($\alpha_t = 0.28, \sigma_t = 0.96$) liefert `oracle` alle fünf Zielgrößen der Mischung exakt. Anschließend folgt die Umrechnung auf den linearen Pfad:

<!--en:547993ad-->
::: {.slide title="Konditionierung in der Praxis: Welche Zielgröße passt zu welchem Pfad?"}

<!--en:9124a48a-->
[Unterschiede in Numerik und Gewichtung]{.kicker}

<!--en:e1aacb23-->
- **Division durch null bei reinem Rauschen ($\alpha_1 = 0$):** Die Rauschvorhersage (`'eps'`) rekonstruiert $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sigma_t\hat{\boldsymbol\epsilon})/\alpha_t$. Erreicht der Zeitplan reines Rauschen, steht im Nenner $\alpha_t \to 0$. Daher benötigt `'eps'` den VP-Pfad mit $\alpha_1 \approx 0.007 > 0$.

<!--en:45fc0d25-->
- **Gut konditionierte Geschwindigkeits- und $\mathbf{v}$-Vorhersage:** Die Geschwindigkeit auf dem linearen Pfad ($\hat{\mathbf{x}}_0 = \mathbf{x}_t - t\hat{\mathbf{u}}$) und die $\mathbf{v}$-Vorhersage auf dem Kosinuspfad ($\hat{\mathbf{x}}_0 = \alpha_t\mathbf{x}_t - \sigma_t\hat{\mathbf{v}}$) liefern $\hat{\mathbf{x}}_0$ ohne Division durch $\alpha_t$.

<!--en:3f84af8c-->
::: {.slide title="Der einfache Verlust von DDPM und die Gewichtung beim Score-Matching"}

<!--en:e80f6bb1-->
[Historische Ansätze in gemeinsamer Darstellung]{.kicker}

<!--en:06b82d3c-->
Jede wegweisende Familie generativer Diffusionsmodelle entspricht einer Wahl von Pfad, Zielgröße aus den Umrechnungsregeln und zeitlicher Gewichtung:

<!--en:c8778e0b-->
| Familie generativer Modelle | Vorwärtspfad | Regressionsziel | Zusammenhang mit Denoising-Score-Matching |

<!--en:9e9beb07-->
| **NCSN** (Song und Ermon, 2019) | VE ($\alpha_t = 1, \tilde\sigma_i$) | Score $-\boldsymbol\epsilon/\tilde\sigma_i$ | Mit $\tilde\sigma_i^2$ gewichtet ergibt sich der Verlust $\mathbb{E}\|\tilde\sigma_i\mathbf{s}_{\boldsymbol\theta} + \boldsymbol\epsilon\|^2$ |

<!--en:8ef92214-->
| **DDPM** (Ho et al., 2020) | VP ($\alpha_t^2 + \sigma_t^2 = 1$) | Rauschen $\boldsymbol\epsilon$ | $L_{\textrm{simple}} = \mathbb{E}\|\boldsymbol\epsilon - \boldsymbol\epsilon_{\boldsymbol\theta}(\mathbf{x}_t, t)\|^2 = \mathbb{E}\big[\sigma_t^2\|\mathbf{s}_{\boldsymbol\theta} + \boldsymbol\epsilon/\sigma_t\|^2\big]$ |

<!--en:9d0564c8-->
| **$\mathbf{v}$-Vorhersage** (Salimans und Ho, 2022) | VP-Kosinuspfad ($\cos\frac{\pi t}{2}$) | Winkelgeschwindigkeit $\mathbf{v} = \alpha_t\boldsymbol\epsilon - \sigma_t\mathbf{x}_0$ | Gleich $(2/\pi)\,\mathbf{u}_t$, gut konditioniert bei $\alpha_1 = 0$ |

<!--en:b0307b42-->
| **Flow-Matching** (Lipman et al., 2022) | Linear ($1 - t, t$) | Geschwindigkeit $\boldsymbol\epsilon - \mathbf{x}_0$ | Verlust gleich $\frac{1}{(1-t)^2}\|\hat{\boldsymbol\epsilon} - \boldsymbol\epsilon\|^2$, konstante Geschwindigkeit auf geraden Pfaden |

<!--en:36de75c4-->
::: {.slide title="Analytische Verlustuntergrenzen für verschiedene Zielgrößen und Zeitpunkte"}

<!--en:4517f4a7-->
[Warum Trainingsverluste nie null erreichen]{.kicker}

<!--en:780cff5e-->
- **Jede Zielgröße hat die Form $c_t\mathbf{x}_0 + f(\mathbf{x}_t)$ bei festem $\mathbf{x}_t$:** Ihre unvermeidbare Verlustuntergrenze ist daher $c_t^2$ mal die Spur der A-posteriori-Kovarianz.

<!--en:0d58c2e7-->
- **Warum sich die Plateaus um den Faktor $9\times$ unterscheiden ($4.91$ gegenüber $0.54$):** Für die Geschwindigkeit auf dem linearen Pfad ($c_t = -1/t$) liegt das Plateau bei $\approx 4.91$, für VP-Rauschen ($c_t = -\alpha_t/\sigma_t$) bei $\approx 0.54$. Die Faktoren $c_t^2$ skalieren die Zielgrößen unterschiedlich, und die beiden Pfade haben unterschiedliche Zeitskalen.

<!--en:e362f24d-->
::: {.slide title="Messung des Überschussverlusts über der exakten Orakeluntergrenze"}

<!--en:00cefcbf-->
[Prüfung der Äquivalenz über Pfade hinweg]{.kicker}

<!--en:5f5fdec9-->
Mit $10^6$ Monte-Carlo-Ziehungen lässt sich der Verlust der trainierten Netze für Geschwindigkeit auf dem linearen Pfad (`ema_v`) und für VP-Rauschen (`ema_eps`) in die exakte Orakeluntergrenze und den Überschussverlust $\mathbb{E}\|\text{net} - \text{oracle}\|^2$ des jeweiligen Netzes zerlegen:

<!--en:fdec55ee-->
**Beide Netze lernen denselben A-posteriori-Mittelwert:** Ihre Überschussverluste gegenüber dem Orakel sind $\le 0.001$ und damit kleiner als der Monte-Carlo-Standardfehler. Das Geschwindigkeitsnetz des linearen Pfades dient mit `eps_from_velocity` als VP-Rauschprädiktor. Sein Ergebnis stimmt auf vier Nachkommastellen mit `ema_eps` überein (Verlust $0.540$, Überschussverlust $0.0002$).

<!--en:c479528d-->
::: {.slide title="Fehlerquelle: direkter Vergleich der Trainingsverluste verschiedener Zielgrößen"}

<!--en:3761e17b-->
[Diagnose in der Praxis]{.kicker}

<!--en:826bb660-->
Da die Verlustuntergrenze mit $c_t^2$ skaliert, kann der reine Trainingsverlust bei Diffusion und Flow-Matching zu drei überraschenden Fehlschlüssen verleiten:

<!--en:ea419047-->
- **Fehlschluss 1: Hohes Plateau $\neq$ Unteranpassung:** Ein Geschwindigkeitsverlust mit Plateau bei $4.9$ im Flow-Matching und ein DDPM-Rauschverlust mit Plateau bei $0.54$ liegen *beide* höchstens $0.001$ über dem bayesoptimalen Orakel.

<!--en:3ece2ef1-->
- **Fehlschluss 2: Geringerer Rohverlust $\neq$ besseres Modell:** Der Faktor $9\times$ zwischen $4.9$ und $0.54$ ergibt sich aus dem Skalierungsfaktor $c_t^2$ ($1/t^2$ gegenüber $\alpha_t^2/\sigma_t^2$) und den unterschiedlichen Zeitskalen der beiden Pfade. Modelle mit unterschiedlichen Trainingszielen oder Zeitplänen sind in einer **gemeinsamen Zieleinheit** zu vergleichen (etwa $\mathbb{E}\|\hat{\mathbf{x}}_0 - \mathbf{x}_0\|^2$ bei gleicher Log-SNR $\lambda$) oder anhand der Sample-Qualität.

<!--en:7811cdc7-->
- **Fehlschluss 3: Auswendiglernen bei endlichem Datensatz:** Für einen kleinen Datensatz aus $N$ Punkten ist das empirische Optimum der exakte A-posteriori-Mittelwert von $N$ Dirac-Punktmassen. Er gibt die Trainingsbeispiele exakt wieder (Aufgabe 17.2.5). Neuronale Netze generalisieren, weil der induktive Bias ihrer Architektur das Geschwindigkeitsfeld räumlich glättet.

<!--en:16e1ee8b-->
::: {.slide title="Verständnisfragen: Zeitskala der Log-SNR und Umrechnung der Zielgrößen"}

<!--en:929b7671-->
[Selbsttest]{.kicker}

<!--en:9db8c264-->
Fragen zu Gauß-Pfaden für Wahrscheinlichkeitsverteilungen und zu den Umrechnungsregeln für Zielgrößen:

<!--en:be4d8ffe-->
1. **Zeitpunkt mit gleich starkem Signal und Rauschen ($\lambda_t = 0$):** Zu welchem Zeitpunkt $t \in [0, 1]$ ist die Signalvarianz gleich der Rauschvarianz ($\alpha_t^2 = \sigma_t^2$, also $\lambda_t = 0$), einmal auf dem linearen Pfad und einmal auf dem VP-Pfad mit $\log \bar\alpha_t = -(0.1t + 9.95t^2)$?

<!--en:93edb066-->
2. **Umskalierung zwischen Pfaden:** Bei $t = 0.5$ gilt auf dem VP-Pfad $\alpha_{0.5} = 0.28$ und $\sigma_{0.5} = 0.96$. Zu welcher Zeit $t_{\textrm{lin}}$ hat der lineare Pfad dieselbe Log-SNR? Mit welchem Skalar $c$ muss ein VP-Punkt $\mathbf{x}_{0.5}$ multipliziert werden, damit er auf dem linearen Pfad liegt?

<!--en:d6e71d06-->
3. **Untergrenzen der Rauschvorhersage an den Endpunkten:** Warum geht bei der zweidimensionalen Mischung ($d = 2$) die Untergrenze der VP-Rauschvorhersage gegen $2$ für $t \to 0$ und gegen $0$ für $t \to 1$?

<!--en:557b98b6-->
**Lösungen:**

<!--en:52fd2ea3-->
1. **Linear:** $1 - t = t \implies \mathbf{t = 0.50}$. **VP:** $\bar\alpha_t = 0.5 \implies 0.1t + 9.95t^2 = \log 2 \approx 0.693 \implies \mathbf{t \approx 0.26}$ (während $74\%$ der Zeit überwiegt auf dem VP-Pfad das Rauschen).

<!--en:df941d4a-->
2. **Entsprechende Zeit auf dem linearen Pfad:** $t_{\textrm{lin}} = \frac{\sigma}{\alpha + \sigma} = \frac{0.96}{0.28 + 0.96} \approx \mathbf{0.773}$ mit dem Skalierungsfaktor $c = \frac{1}{\alpha + \sigma} = \frac{1}{1.24} \approx \mathbf{0.806}$.

<!--en:4bd08c7b-->
3. Für $t \to 1$ gilt $\mathbf{x}_t \approx \boldsymbol\epsilon$. Damit ist $\boldsymbol\epsilon$ fast exakt bekannt ($\text{floor} \to \mathbf{0}$). Für $t \to 0$ ist das schwache Rauschen $\sigma_t\boldsymbol\epsilon$ von der Streuung innerhalb eines Modus nicht zu unterscheiden. Also gilt $\hat{\boldsymbol\epsilon} \to \mathbf{0}$ und $\mathbb{E}\|\boldsymbol\epsilon - \mathbf{0}\|^2 = d = \mathbf{2}$.

<!--en:ceceb553-->
::: {.slide title="Zusammenfassung: Gauß-Pfade und äquivalente Zielgrößen"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:d35f4855-->
| Begriff / Pfad | Mathematische Definition | Zentrale Eigenschaft / Invariante | Bedeutung für Diffusionsmodelle |

<!--en:5d475d12-->
| **Varianzerhaltend (VP)** | $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\boldsymbol\epsilon$ | $\alpha_t^2 + \sigma_t^2 = 1$, auf einem Gitter die diskrete DDPM-Kette | Erhält die Varianz eins der Eingaben, $\alpha_1 \approx 0.007 > 0$ |

<!--en:37badb4d-->
| **Linearer Pfad (Flow-Matching)** | $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\boldsymbol\epsilon$ | Gerade bedingte Pfade, Geschwindigkeit $\boldsymbol\epsilon - \mathbf{x}_0$ | Erreicht mit konstanter Geschwindigkeit exakt reines Rauschen $(\alpha_1 = 0, \sigma_1 = 1)$ |

<!--en:3a3111df-->
| **Zeitskala der Log-SNR** | $\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$ | Fällt monoton von $+\infty$ auf $-\infty$ | Gemeinsame Zeitskala für alle Gauß-Pfade durch Umskalierung mit $c = 1/(\alpha_t + \sigma_t)$ |

<!--en:8b54fe1f-->
| **Bedingtes Flow-Matching** | $\mathbb{E}\big[\|\mathbf{u}_{\boldsymbol\theta}(\mathbf{x}_t, t) - (\dot\alpha_t\mathbf{x}_0 + \dot\sigma_t\boldsymbol\epsilon)\|^2\big]$ | Entspricht dem marginalen Fehler plus einer Konstanten: $\mathbb{E}\|\mathbf{u}_{\boldsymbol\theta} - \mathbf{u}_t\|^2 + C$ | Lernt die marginale Geschwindigkeit $\mathbf{u}_t(\mathbf{x})$ aus den Paaren einzelner Beispiele |

<!--en:d2b0ed5f-->
| **Umrechnungsregeln für Zielgrößen** | $\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon}, \nabla \log p_t, \hat{\mathbf{v}}, \mathbf{u}_t$ | Alle affin in $\mathbf{x}_t$ und $\hat{\mathbf{x}}_0 = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$ | Rechnen jeden trainierten Prädiktor zwischen Zielgrößen und Zeitplänen um |

<!--en:20419ccb-->
| **Unvermeidbare Verlustuntergrenze** | $c_t^2\,\mathbb{E}[\operatorname{tr}\mathrm{Cov}(\mathbf{x}_0 \mid \mathbf{x}_t)]$ | $\approx 0.54$ (VP-Rauschen) gegenüber $\approx 4.9$ (Geschwindigkeit auf dem linearen Pfad) | Erklärt hohe Verlustplateaus. Zum Modellvergleich dient der Überschuss über der Untergrenze |
