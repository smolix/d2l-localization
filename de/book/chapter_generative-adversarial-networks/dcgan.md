<!--en:49b70a11-->
# Adversariale Bilderzeugung

<!--en:0ce6b095-->
Das mehrschichtige Perzeptron aus :numref:`sec_basic_gan` erzeugte auf einem Raster von $28 \times 28$ Pixeln erkennbare Fashion-MNIST-Silhouetten. Die Texturen wirkten jedoch körnig, weil vollständig verbundene Schichten räumliche Nachbarschaften ignorieren. Für ein RGB-Bild der Größe $64 \times 64$ muss der Generator aus einem 100-dimensionalen latenten Vektor $12{,}288$ Pixelwerte erzeugen. Daraus ergeben sich zwei Fragen:

<!--en:c7caa376-->
1. Mit welchen konvolutionären Architekturen lässt sich die adversariale Bilderzeugung stabil trainieren?

<!--en:a86b90d2-->
2. Wie können wir Bildgeneratoren quantitativ bewerten und vergleichen, wenn keine Likelihood verfügbar ist?

<!--en:dda3900e-->
Mit dem Deep Convolutional GAN (DCGAN) von :citet:`Radford.Metz.Chintala.2015` ließen sich GANs dank konvolutionärer Generatoren und Diskriminatoren erstmals auf natürliche Bilder skalieren. Inzwischen sind Diffusionsmodelle (:numref:`chap_diffusion`) das wichtigste Verfahren zur allgemeinen Bilderzeugung. Für moderne Bild-Tokenizer und destillierte Einschrittgeneratoren spielen konvolutionäre Diskriminatoren und Bewertungsmaße im Merkmalsraum aber weiterhin eine zentrale Rolle (:numref:`sec_gan_beyond`).

<!--en:223f61a6-->
In diesem Abschnitt trainieren wir mit einem Datensatz aus 40 597 Pokémon-Sprites der Größe $64 \times 64$. Zunächst implementieren wir die klassische DCGAN-Architektur von 2015. Anschließend bauen wir nach dem Vorbild von R3GAN :cite:`Huang.Gokaslan.Kuleshov.ea.2024` ein minimalistisches konvolutionäres Basisnetz (engl. backbone) ohne Normalisierung. Wir trainieren es mit drei Zielfunktionen: dem klassischen logistischen Verlust ohne Strafterme, demselben Verlust mit den Straftermen $R_1 + R_2$ und dem Verlust für relativistische Paarbildung mit den Straftermen $R_1 + R_2$. So prüfen wir, ob um null zentrierte Gradientenstrafterme (engl. zero-centered gradient penalties, :numref:`sec_gan_convergence`) die Batchnormalisierung (engl. batch normalization) und andere Architekturheuristiken ersetzen. Danach implementieren wir die gängigen Bewertungsmaße im Merkmalsraum: Fréchet Inception Distance (FID), Kernel Inception Distance (KID) sowie $k$-NN-Präzision und -Trefferquote (engl. precision and recall). Abschließend untersuchen wir die Interpolation im latenten Raum und die Überanpassung (engl. overfitting) des Diskriminators.

<!--en:f34a8962-->
## Die DCGAN-Architektur

<!--en:9a2ac32a-->
DCGAN :cite:`Radford.Metz.Chintala.2015` behält die klassische nicht sättigende logistische Zielfunktion aus :numref:`sec_basic_gan` bei. Mehrere Architekturkonventionen stabilisieren das Training der konvolutionären Netze:

<!--en:5de4940a-->
- Für das räumliche Upsampling nutzt der Generator *transponierte Faltungen* (engl. transposed convolutions, auch Faltungen mit gebrochenem Stride). Für das Downsampling nutzt der Diskriminator Faltungen mit Stride anstelle von Pooling-Schichten.

<!--en:1f14b42f-->
- Auf jede Faltung folgt eine *Batchnormalisierung*, außer in der Ausgabeschicht des Generators und der Eingabeschicht des Diskriminators.

<!--en:fcfb6174-->
- Der Generator verwendet ReLU in den versteckten Schichten und $\tanh$ als Ausgabeaktivierung, passend zu Pixelwerten in $[-1, 1]$. Der Diskriminator verwendet durchgehend Leaky ReLU ($\text{slope} = 0.2$), damit keine Bereiche mit verschwindendem Gradienten entstehen.

<!--en:416195ca-->
- Die Anfangsgewichte stammen aus $\mathcal{N}(0, 0.02^2)$. Adam verwendet ein reduziertes Momentum $\beta_1 = 0.5$.

<!--en:2c3850f5-->
### Der Datensatz mit Pokémon-Sprites

<!--en:1f2c9537-->
Unser Datensatz enthält 40 597 RGB-Sprites von 721 Pokémon-Arten aus [pokemondb.net](https://pokemondb.net/sprites). Wir bringen sie auf die Größe $64 \times 64$ und skalieren die Pixelwerte auf $[-1, 1]$. Da der gesamte Datensatz in den Arbeitsspeicher passt, dekodiert ihn `load_pokemon` in einen einzigen Tensor.

<!--en:5e89459f-->
Mit einer festen Permutation reservieren wir ein Zehntel der Bilder und verwenden sie nie für das Training. Diese echten Bilder dienen als Vergleichsstichprobe für die erzeugten Bilder. An ihnen erkennen wir auch, ob der Kritiker seinen Trainingsbildern höhere Scores gibt als ungesehenen Bildern. Wir teilen nach Bildern auf, nicht nach Arten. Varianten einer Art können also in beiden Teilmengen vorkommen. Kopiert ein Generator ein Trainings-Sprite, kann die Kopie einer zurückgehaltenen Variante sehr nahekommen.

<!--en:53d12cd4-->
### Transponierte Faltungen und Batchnormalisierung

<!--en:7944a414-->
Für das Upsampling einer $1 \times 1$-Merkmalskarte (engl. feature map) mit 100 Kanälen zu einem $64 \times 64$-Bild verwendet DCGAN *transponierte Faltungen* (:numref:`sec_transposed_conv`). Eine transponierte Faltung mit Kerngröße $k$, Stride $s$ und Padding $p$ bildet ein $n \times n$-Gitter auf eine Ausgabe mit folgender Seitenlänge ab:

<!--en:699a7d9e-->
Mit $k = 4, s = 2, p = 1$ verdoppelt jede Schicht die räumliche Auflösung ($n \to 2n$). In der ersten Schicht ($k = 4, s = 1, p = 0$) wächst die Eingabe von $1 \times 1$ auf $4 \times 4$. Umgekehrt halbiert eine Faltung mit Stride bei $k = 4, s = 2, p = 1$ die räumliche Auflösung.

<!--en:4f1346b0-->
Wie :numref:`fig_gan_dcgan` zeigt, schaltet der DCGAN-Generator fünf transponierte Faltungen mit den Auflösungen $1 \to 4 \to 8 \to 16 \to 32 \to 64$ hintereinander. Dabei halbiert er auf jeder Stufe die Kanalzahl und gibt schließlich 3 RGB-Kanäle aus. Der Diskriminator durchläuft diese Hierarchie in umgekehrter Richtung: Faltungen mit Stride reduzieren die Eingabe auf ein einziges $1 \times 1$-Logit.

<!--en:f3bfbf48-->
![Die DCGAN-Architektur :cite:`Radford.Metz.Chintala.2015` (mit 512 Basiskanälen und $4 \times 4$-Faltungskernen). Der Generator bildet einen 100-dimensionalen latenten Vektor durch transponierte Faltungen auf ein $64 \times 64$-RGB-Bild ab. Der Kritiker durchläuft die Hierarchie mit Faltungen mit Stride in umgekehrter Richtung. Jeder Block verwendet Batchnormalisierung, außer der Ausgabeschicht des Generators und dem Eingabeblock des Kritikers.](../img/mdl-gan-dcgan.svg)

<!--en:261594fc-->
Vier `G_block`-Stufen mit einer abschließenden transponierten Faltung und `tanh` bilden `net_G`. Vier `D_block`-Stufen (ohne Batchnormalisierung im ersten Block) und eine $4 \times 4$-Faltung bilden `net_D`. Mit einem Vorwärtsdurchlauf für zwei 100-dimensionale latente Vektoren prüfen wir die Ausgabeformen:

<!--en:d589dd62-->
### Training mit dem klassischen Verlust

<!--en:6065b2da-->
Die Schleife `d2l.train_gan` aus :numref:`sec_basic_gan` funktioniert für Bilder unverändert. Sie wechselt zwischen einem Schritt der logistischen Regression für den Kritiker und einem nicht sättigenden Schritt für den Generator. Wir verwenden die DCGAN-Einstellungen mit einem größeren Minibatch: 256 Bilder anstelle der 128 aus der Veröffentlichung. Die Lernrate beträgt $2 \cdot 10^{-4}$, es gilt $\beta_1 = 0.5$, und die Faltungskerne ziehen wir aus $\mathcal{N}(0, 0.02^2)$. Wie im veröffentlichten Code der Autoren initialisieren wir die Skalierungsfaktoren der Batchnormalisierung aus $\mathcal{N}(1, 0.02^2)$ und die Verschiebungen mit null.

<!--en:46765edb-->
Zum Erzeugen von Samples verwenden wir einen Mittelwert der jüngsten Generatorgewichte statt der letzten Gewichte. `EMA` berechnet einen exponentiellen gleitenden Mittelwert (engl. exponential moving average) der Parameter. Nach jedem Schritt aktualisieren wir ihn nach der Regel

<!--en:df4704de-->
sodass sich der Beitrag eines früheren Schritts alle $n_{1/2}$ Schritte halbiert. Diesen Zeitraum nennen wir *Halbwertszeit*. Die Iterierten des Spiels kreisen um das Gleichgewicht (:numref:`sec_gan_convergence`). Ihr Mittelwert liegt deshalb näher am Gleichgewicht als die einzelnen Iterierten. Die Statistiken der Batchnormalisierung gehören nicht zu den Parametern. Das gemittelte Netz übernimmt sie vom aktuellen Netz. Da die kopierten Statistiken zu den aktuellen Gewichten gehören, schätzen wir sie nach dem Training für die gemittelten Gewichte neu.

<!--en:55993539-->
In den späteren Trainingsläufen dieses Abschnitts mitteln wir über 32 000 Bilder. Bei Batchgröße 64 entspricht das einer Halbwertszeit von 500 Schritten. Bei Batchgröße 256 genügen für dieselben 32 000 Bilder 125 Schritte. Das ist weniger als ein Zwanzigstel der 2840 Schritte dieses Laufs (20 Epochen zu je 142 Schritten).

<!--en:fb1ae5b0-->
Der ausgegebene Kritikerverlust ist die Summe der beiden Batch-Mittelwerte aus :numref:`sec_basic_gan`. Im Gleichgewicht beträgt sie $2 \log 2 \approx 1.39$. Der Generatorverlust läge dort bei $\log 2 \approx 0.69$.

<!--en:1b036a95-->
In der nächsten Zelle schätzen wir die Statistiken der Batchnormalisierung für den gemittelten Generator neu. Dazu verarbeiten wir im Trainingsmodus 100 neue Batches latenter Codes. Pro Batch sinkt das Gewicht früherer Beobachtungen im gleitenden Mittel um den Anteil 0.1. Den kopierten Werten bleibt damit nur das Gewicht $0.9^{100} \approx 3 \cdot 10^{-5}$, also praktisch null.

<!--en:058a2619-->
Wir erzeugen Samples im Auswertungsmodus, damit jedes Bild nur von seinem eigenen latenten Code abhängt.

<!--en:29851e2c-->
Die erzeugten Sprites zeigen vielfältige Silhouetten und Farbpaletten auf weißem Hintergrund, ihre Texturen wirken allerdings etwas fleckig. Der Diskriminatorverlust ($\sim 0.25\text{--}0.45$) bleibt deutlich unter $2\log 2 \approx 1.39$, ohne einen Moduskollaps (engl. mode collapse) auszulösen.

<!--en:b69ac394-->
## Ein Basisnetz ohne Normalisierung

<!--en:e7113128-->
In den Jahren nach DCGAN kamen bei großen Bild-GANs zahlreiche stabilisierende Architekturelemente hinzu. StyleGAN2 :cite:`Karras.Laine.Aittala.ea.2020` kombinierte zehn verschiedene Techniken. R3GAN :cite:`Huang.Gokaslan.Kuleshov.ea.2024` zeigte, dass diese Ad-hoc-Lösungen entfallen können, wenn man die Zielfunktion durch die Gradientenstrafen $R_1 + R_2$ und den Verlust für relativistische Paarbildung regularisiert. Insbesondere muss man bei $R_1$ und $R_2$ auf die Batchnormalisierung im Diskriminator verzichten: Sie koppelt die Samples eines Minibatches, sodass sich die Eingabegradienten nicht mehr unabhängig pro Sample berechnen lassen.

<!--en:bdd2d929-->
### Architektur und Fan-in-Initialisierung

<!--en:075c9db5-->
Wir bauen ein minimalistisches konvolutionäres Basisnetz (:numref:`fig_gan_backbone`) ohne Normalisierungsschichten und ohne $\tanh$ am Ausgang:

<!--en:9d929955-->
- Der Generator verkettet einen erlernten konstanten $4 \times 4 \times 128$-Tensor mit einer linearen Projektion von $z$ ($4 \times 4 \times 100$, optional mit zusätzlichen Kanälen für die Klassenkonditionierung in :numref:`sec_gan_conditional`). Dann erhöht er die Auflösung viermal ($4 \to 8 \to 16 \to 32 \to 64$), jeweils durch bilineare Interpolation mit anschließender $3 \times 3$-Faltung und Leaky ReLU ($\text{slope} = 0.2$). Bilineares Upsampling mit anschließender $3 \times 3$-Faltung vermeidet die Schachbrettartefakte, die bei ungleichmäßiger Überlappung in transponierten Faltungen entstehen (:numref:`fig_gan_transposed_overlap`) :cite:`Odena.Dumoulin.Olah.2016`.

<!--en:5fc2d4d5-->
- Der Kritiker kehrt diese Struktur um. Mit $3 \times 3$-Faltungen, Leaky ReLU und bilinearem Downsampling reduziert er die Auflösung bis auf eine $4 \times 4$-Merkmalskarte $\phi(x)$. Darauf folgt ein linearer Kopf.

<!--en:cfc66be1-->
- Damit die Varianz der Aktivierungen auch ohne Batchnormalisierung über die Schichten hinweg erhalten bleibt, verwenden wir für alle Gewichte die Fan-in-Initialisierung nach He :cite:`He.Zhang.Ren.ea.2015`. Für Leaky ReLU mit Steigung $0.2$ skalieren wir die Gewichte auf die Varianz $2 / ((1 + 0.2^2)\,n_{\mathrm{in}})$.

<!--en:871cea05-->
![Das konvolutionäre Basisnetz ohne Normalisierung. Der Generator kombiniert eine erlernte $4 \times 4 \times 128$-Konstante mit einem projizierten latenten Vektor. Darauf folgen vier Stufen aus bilinearem Upsampling, $3 \times 3$-Faltung und Leaky ReLU. Der Kritiker durchläuft diese Hierarchie in umgekehrter Richtung mit bilinearem Downsampling bis zu einem linearen Kopf.](../img/mdl-gan-backbone.svg)

<!--en:d0e859ce-->
![Anzahl der Überlappungen bei einer transponierten Faltung mit Stride 2. Jedes Eingabepixel prägt der Ausgabe eine Kopie des Faltungskerns auf. Jede Zahl gibt an, wie viele Kopien das jeweilige Ausgabepixel abseits der Ränder überdecken. Zwei benachbarte Kopien sind umrandet. Bei einem $3 \times 3$-Faltungskern (links) wechseln die Anzahlen zwischen 4, 2 und 1. So entsteht selbst bei konstanten Gewichten ein Schachbrettmuster. Bei einem $4 \times 4$-Faltungskern (rechts) ist die Kerngröße ein Vielfaches des Strides. Dann überdecken genau vier Kopien jedes Pixel.](../img/mdl-gan-transposed-overlap.svg)

<!--en:16d92ecf-->
Der Kritiker kehrt den Aufbau des Generators um. Vier Stufen aus Faltung, Leaky ReLU und bilinearem Downsampling reduzieren ein $64 \times 64$-Bild auf $4 \times 4$. Eine Faltung mischt die Kanäle, ein linearer Kopf liefert einen einzigen Score.

<!--en:7267de97-->
Der Generator hat $\sim 2.8\text{M}$ Parameter, der Kritiker $\sim 1.6\text{M}$ Parameter.

<!--en:33e620a8-->
## Kontrollierter Vergleich der Verlustfunktionen

<!--en:ec05deb8-->
### Drei Versuchsvarianten

<!--en:f7aa79a4-->
Wir trainieren das Basisnetz ohne Normalisierung dreimal mit identischen Anfangsgewichten. In allen Läufen verwenden wir Adam ($\beta_1 = 0, \beta_2 = 0.99, \text{lr} = 2 \times 10^{-4}$), Batchgröße 64, zufällige horizontale Spiegelungen und EMA ($\text{half-life} = 500$ Schritte). Nur die Zielfunktion variiert:

<!--en:07a5188f-->
Dabei bezeichnet $L(D)$ entweder den klassischen logistischen Verlust (`d2l.gan_loss_D`) oder den Verlust für relativistische Paarbildung (`d2l.rpgan_loss_D`). Der Generator verwendet den entsprechenden nicht sättigenden Verlust:

<!--en:8e72bae3-->
| Variante | Kritikerverlust | Gradientenstrafe | Schritte |

<!--en:c298424f-->
| `NS` | klassisch logistisch | $\gamma = 0$ (keine) | 6000 |

<!--en:1896ad0d-->
| `NS + R1 + R2` | klassisch logistisch | $\gamma = 10$ | 12 000 |

<!--en:5e99d714-->
| `RpGAN + R1 + R2` | relativistische Paarbildung | $\gamma = 10$ | 12 000 |

<!--en:61fe43a0-->
Der Vergleich der ersten beiden Varianten isoliert den Einfluss der Gradientenstrafe $R_1 + R_2$. Die Variante ohne Strafterme beenden wir nach 6000 Schritten, da der Kollaps schon deutlich früher eintritt. Der Vergleich der zweiten und dritten Variante isoliert den Einfluss des paarweisen Rankings. In Aufgabe :exref:`ex_dcgan-penalty-weight-resolution` untersuchen wir, wie $\gamma$ mit der Bildauflösung skaliert. Zwei gespeicherte Hilfsfunktionen unterstützen die Trainingsschleife: `sample_real` zieht zufällige Minibatches mit horizontalen Spiegelungen. `pair_stats` berechnet die Genauigkeit des paarweisen Rankings $\hat P(D(x) > D(x'))$ und das mittlere relativistische Ranggewicht $w_{\mathrm{Rp}} = \mathbb{E}[\sigma(D(x') - D(x))]$.

<!--en:1360b763-->
In JAX kompilieren wir zunächst `r3gan_step`. Damit fassen wir `d2l.critic_step` und `d2l.generator_step` für jede Trainingsiteration in einem einzigen JIT-kompilierten Aufruf zusammen:

<!--en:5ac5a855-->
Die gespeicherte Schleife `train_r3gan` führt abwechselnd `d2l.critic_step` und `d2l.generator_step` aus :numref:`sec_gan_convergence` auf Batches echter Bilder mit zufälliger Spiegelung aus. Nach jedem Schritt aktualisiert sie den EMA-Generator. Der Kritikerschritt liefert die Batch-Mittelwerte von $R_1$ und $R_2$. So protokollieren wir für jede Variante die Steilheit des Kritikers auf echten Bildern, $\mathbb{E}_x\|\nabla_x D(x)\|^2$ (in der ersten Variante mit Gewicht $\gamma = 0$). Alle 250 Schritte wendet die Schleife außerdem `pair_stats` auf 256 neue Paare an:

<!--en:38e55f80-->
`train_arm` baut die beiden Netze in jeder Variante mit demselben Zufallsstartwert auf, trainiert sie und liefert die Score-Differenz des Kritikers zwischen Trainingsdaten und zurückgehaltenen Daten: Dazu ziehen wir den mittleren Score für 1024 zurückgehaltene Bilder vom mittleren Score für 1024 Trainingsbilder ab.

<!--en:31fa408d-->
Ein Schritt mit Straftermen kostet mehr Rechenzeit als einer ohne, weil wir für die Strafe den Eingabegradienten des Kritikers ein weiteres Mal differenzieren. Nach jedem Lauf geben wir den Kritikerverlust und die Strafe aus, jeweils gemittelt über die acht Messungen der letzten 2000 Schritte, außerdem die Score-Differenz zwischen Trainingsdaten und zurückgehaltenen Daten.

<!--en:5c279c0f-->
Die Verluste beider Kritiker mit Straftermen pendeln sich etwas unter ihren Gleichgewichtswerten ein ($1.04$ gegenüber $2\log 2 \approx 1.39$ für den klassischen logistischen Verlust, $0.47\text{--}0.51$ gegenüber $\log 2 \approx 0.69$ für die Paarbildung). Ohne Strafterme sinkt der Kritikerverlust auf $0.57\text{--}0.58$.

<!--en:690cf827-->
### Trainingsverläufe und erzeugte Samples

<!--en:72bd2f80-->
Wir zeichnen alle 250 Schritte die quadrierte Norm des Eingabegradienten des Kritikers $\mathbb{E}_x[\|\nabla_x D(x)\|^2]$, die Genauigkeit des paarweisen Rankings $\hat P(D(x) > D(x'))$ und das relativistische Ranggewicht $w_{\mathrm{Rp}} = \mathbb{E}[\sigma(D(x') - D(x))]$ auf. Die Kurven zeigen, wie die Gradientenstrafe das Training stabilisiert:

<!--en:2c9526bf-->
Ohne $R_1 + R_2$ steigt die quadrierte Gradientennorm des Kritikers um zwei Größenordnungen (auf $1\text{--}8$ gegenüber $\sim 0.01$ für die Kritiker mit Straftermen). Alle drei Kritiker ordnen $> 90\%$ der Paare richtig. Die Kritiker mit Straftermen halten jedoch den Abstand zwischen den Scores klein ($w_{\mathrm{Rp}} \in [0.25, 0.40]$, entsprechend einer medianen Logit-Differenz von etwa $0.7$). Ohne Strafterme drückt der Kritiker $w_{\mathrm{Rp}}$ unter $0.15$.

<!--en:a759f42e-->
Für den visuellen Vergleich der Sample-Qualität definieren wir zunächst die Hilfsfunktion `tile_images`. Sie ordnet einen Batch von Bildern mit Werten in $[-1, 1]$ zu einem einzigen Raster mit Werten in $[0, 1]$ an:

<!--en:a9dec851-->
Nun werten wir alle vier EMA-Generatoren – DCGAN und die drei Varianten ohne Normalisierung – mit denselben 64 latenten Vektoren `z_show` aus und ordnen ihre Ausgaben in $8 \times 8$-Rastern an:

<!--en:ee7a24ce-->
Ohne Batchnormalisierung oder Gradientenstrafen erleidet die Variante `NS` einen vollständigen Moduskollaps: Für alle 64 latenten Codes entsteht eine einheitlich graue Textur (siehe Aufgabe :exref:`ex_dcgan-rerun-arm-classic-loss`). Die zusätzliche Gradientenstrafe $R_1 + R_2$ verhindert den Kollaps in den Varianten `NS + R1 + R2` und `RpGAN + R1 + R2`. Sie erzeugen vielfältige, zentrierte Sprite-Silhouetten auf reinweißem Hintergrund.

<!--en:ae892e27-->
## Quantitative Bewertungsmaße

<!--en:c70106dc-->
Einen vollständigen Kollaps erkennen wir leicht mit bloßem Auge. Um zwei funktionierende Generatoren zu vergleichen, brauchen wir jedoch quantitative Maße. Bei der üblichen Bewertung von GANs verarbeitet man jeweils $n$ echte und $n$ erzeugte Bilder mit einem vortrainierten Merkmalsextraktor $\phi(x) \in \mathbb{R}^d$. Die Abweichung zwischen den beiden empirischen Merkmalsverteilungen messen wir mit den Abstandsmaßen aus :numref:`sec_gan_objectives`, für die geschlossene Formeln vorliegen.

<!--en:cbb2d481-->
### FID und KID

<!--en:81b55e3e-->
1. Für die **Fréchet Inception Distance (FID)** :cite:`Heusel.Ramsauer.Unterthiner.ea.2017` passen wir die multivariaten Gauß-Verteilungen $\mathcal{N}(\mu_p, \Sigma_p)$ und $\mathcal{N}(\mu_q, \Sigma_q)$ an die Punktwolken der echten und der erzeugten Merkmalsvektoren an. Dann berechnen wir den quadrierten Wasserstein-2-Abstand zwischen ihnen :cite:`Dowson.Landau.1982,Givens.Shortt.1984`:

   $$
   \begin{aligned}
   \mathrm{FID}
   &= \big\| \mu_p - \mu_q \big\|^2 \\
   &\quad + \operatorname{tr}\!\Big( \Sigma_p + \Sigma_q
   - 2 \big( \Sigma_p^{1/2}\, \Sigma_q\, \Sigma_p^{1/2} \big)^{1/2} \Big).
   \end{aligned}
   $$
   :eqlabel:`eq_gan_fid`

   Die FID hängt nur von den ersten beiden Momenten der Merkmalsverteilung ab. Setzen wir die Stichprobenkovarianz in die Formel ein, hat der FID-Schätzer bei endlichem Stichprobenumfang eine positive Verzerrung der Ordnung $O(1/n)$ (Aufgabe :exref:`ex_dcgan-measure-finite-sample-bias`).

<!--en:1143fbfe-->
2. Die **Kernel Inception Distance (KID)** :cite:`Binkowski.Sutherland.Arbel.ea.2018` schätzt das Quadrat der Maximum Mean Discrepancy erwartungstreu (`d2l.mmd2_unbiased`, :numref:`sec_gan_objectives`). Dazu verwendet sie den kubischen Polynom-Kernel $k(u, v) = (u^\top v / d + 1)^3$ und vergleicht damit die Merkmalsmomente bis zur dritten Ordnung (Aufgabe :exref:`ex_dcgan-moments-fid-kid`):

   $$
   \begin{aligned}
   \widehat{\mathrm{MMD}}^2
   &= \frac{1}{n_p(n_p-1)} \sum_{i \neq i'} k\big(\phi(x_i), \phi(x_{i'})\big) \\
   &\quad + \frac{1}{n_q(n_q-1)} \sum_{j \neq j'} k\big(\phi(x'_j), \phi(x'_{j'})\big) \\
   &\quad - \frac{2}{n_p n_q} \sum_{i,j} k\big(\phi(x_i), \phi(x'_j)\big).
   \end{aligned}
   $$
   :eqlabel:`eq_gan_kid`

<!--en:cd64a997-->
### Ein Netz zur Merkmalsextraktion

<!--en:ccb9c3b6-->
Gängige Benchmarks berechnen FID und KID mit 2048-dimensionalen Merkmalen eines auf ImageNet vortrainierten Inception-v3-Netzes oder mit selbstüberwacht erlernten DINOv2-Merkmalen :cite:`Stein.Cresswell.Hosseinzadeh.ea.2023,Oquab.Darcet.Moutakanni.ea.2024`. Damit dieses Notebook für sich allein lauffähig bleibt, trainieren wir einen kleinen CIFAR-10-Klassifikator aus 3 Blöcken, `d2l.FeatureCNN` (:numref:`subsec_utils_feature_cnn`), bei einer Auflösung von $32 \times 32$ für 15 Epochen. Seine vorletzte, 128-dimensionale Schicht dient als $\phi$. Dazu verkleinern wir die $64 \times 64$-Sprites auf $32 \times 32$ durch $2 \times 2$-Average-Pooling.

<!--en:450419b5-->
Mit `batched` begrenzen wir den Speicherbedarf der Auswertung. Wir laden CIFAR-10, trainieren `d2l.FeatureCNN` mit der Auflösung $32 \times 32$ und zufälligen Verschiebungen um 4 Pixel und prüfen die Klassifikationsgenauigkeit auf dem Testdatensatz:

<!--en:49a641c9-->
### Präzision und Trefferquote

<!--en:64be1072-->
Ein einzelner Abstand wie FID oder KID unterscheidet zwei Fehlerarten nicht: Bei *geringer Bildtreue* erzeugt der Generator unrealistische Bilder außerhalb des Trägers von $p$. Bei *geringer Abdeckung* fehlen Teile des Trägers von $p$ (Modusausfall). :citet:`Kynkaanniemi.Karras.Laine.ea.2019` trennen die beiden Fälle mithilfe von Kugeln im Merkmalsraum, deren Radius durch die $K$ nächsten Nachbarn bestimmt ist. Um jeden echten Merkmalsvektor $\phi(x_i)$ sei $B(\phi(x_i), r_i)$ die euklidische Kugel, die bis zum $K$-ten nächsten echten Nachbarn reicht. Analog definieren wir $B(\phi(x'_j), r'_j)$ für die erzeugten Merkmalsvektoren (wir verwenden $K = 3$). Dann gilt:

<!--en:fe8ec8eb-->
Die *Präzision* ist der Anteil der erzeugten Samples, die im geschätzten Träger der echten Merkmalsverteilung liegen. Die *Trefferquote* ist der Anteil der echten Samples, die der geschätzte Träger der erzeugten Merkmalsverteilung abdeckt. Kollabiert ein Generator auf eine einzige Ausgabe, schrumpfen seine $K$-NN-Radien $r'_j$ auf null. Damit sinkt die Trefferquote auf $0$.

<!--en:a6739e5e-->
### Bewertung der Trainingsläufe

<!--en:7ed020bf-->
Wir vergleichen $n = 4{,}059$ Samples jedes Generators mit allen 4059 zurückgehaltenen Sprites. Die Samples erzeugen wir mit den gleitend gemittelten Gewichten. Für jeden Generator ziehen wir zweimal latente Codes. Die zweite Zeile zeigt also, wie stark sich ein Score ändert, wenn nur die Samples wechseln. Als Referenzwert vergleichen wir zwei disjunkte Mengen von je 4059 Trainingsbildern miteinander. Jeder gemessene Abstand beruht hier auf Schätzrauschen und bei der FD zusätzlich auf der Verzerrung durch den endlichen Stichprobenumfang. Da wir ein eigenes Netz zur Merkmalsextraktion verwenden, nennen wir die Maße FD und $\mathrm{MMD}^2$ statt FID und KID. Die Ergebnisse sind mit veröffentlichten Werten nicht vergleichbar, für die zudem zwölfmal so viele Samples verwendet werden. Das Basisnetz hat keine tanh-Ausgabeaktivierung. Deshalb begrenzen wir seine Ausgabewerte vor der Auswertung. Die letzten beiden Spalten geben an, welche Anteile der ursprünglichen Pixelwerte über $+1$ bzw. unter $-1$ liegen.

<!--en:d3c3cc72-->
Für jeden Generator vergleichen wir die Merkmalsvektoren seiner Bilder mit denen der zurückgehaltenen Sprites. Die erste Zeile enthält den Referenzwert aus zwei echten Stichproben.

<!--en:97919362-->
Aus der Tabelle lassen sich mehrere Schlüsse ziehen:

<!--en:4e846ab4-->
1. **Referenzwert für zwei echte Stichproben:** Für zwei disjunkte Mengen von je $4{,}059$ echten Sprites erhalten wir $\text{FD} \approx 0.2$ (aufgrund der Verzerrung durch die Kovarianzschätzung bei endlicher Stichprobe), $\widehat{\mathrm{MMD}}^2 \approx 0.000$ sowie eine Präzision und eine Trefferquote von jeweils $0.95$. Eine zweite Ziehung der latenten Codes verändert die FD um $\leq 0.2$.

<!--en:2caff9c3-->
2. **Kollaps und Training mit Straftermen:** Die kollabierte Variante `NS` ohne Strafterme erreicht $\text{FD} \approx 64\text{--}72$ bei $\text{recall} = 0.00$ und $\text{precision} \leq 0.07$. Mit $R_1 + R_2$ sinkt die FD um eine Größenordnung (auf $\sim 3.5\text{--}5.5$), die Präzision steigt auf $\sim 0.35\text{--}0.4$ und die Trefferquote auf $\sim 0.4$.

<!--en:9afe1c5e-->
3. **Einfluss des Merkmalsraums (DCGAN und Basisnetz mit Straftermen):** Bei $64 \times 64$ erzeugt das Basisnetz mit Straftermen sauberere Konturen und weiße Hintergründe. Dennoch erreicht das DCGAN von 2015 eine niedrigere FD ($\sim 2$) und einen niedrigeren Wert für $\widehat{\mathrm{MMD}}^2$. Der Grund liegt in den Merkmalen unseres CIFAR-10-Klassifikators bei $32 \times 32$: Das $2 \times 2$-Downsampling glättet die hochfrequenten Flecken der DCGAN-Bilder. Maße im Merkmalsraum spiegeln immer die Invarianz und Auflösung des Netzes wider, das die Merkmale extrahiert.

<!--en:ee96e2fa-->
## Interpolation im latenten Raum und Überanpassung des Kritikers

<!--en:33782492-->
### Interpolation im latenten Raum

<!--en:b788f25a-->
Hat der Generator eine glatte Abbildung von latenten Codes auf Bilder gelernt oder einzelne Vorlagen auswendig gelernt? Um das zu prüfen, interpolieren wir linear für 64 Paare latenter Vektoren jeweils zwischen den beiden Vektoren, gemäß $z_t = (1 - t)z_0 + t z_1$ für $t \in [0, 1]$. Die erzeugten Bilder $G(z_t)$ vergleichen wir mit den pixelweisen Überblendungen $(1 - t)G(z_0) + t G(z_1)$.

<!--en:2955f868-->
Die Bilder aus den interpolierten latenten Codes erreichen dieselbe Präzision im Merkmalsraum ($\sim 0.35\text{--}0.38$) wie zufällige Samples des Generators. Form und Farbe gehen fließend ineinander über. Pixelweise Überblendungen erzeugen dagegen Doppelbelichtungen mit Geisterkonturen. Ihre Präzision sinkt auf die Hälfte ($\sim 0.18\text{--}0.21$).

<!--en:603b81a4-->
### Überanpassung des Kritikers und Augmentierung ohne Durchschlagen auf den Generator

<!--en:42f4e159-->
Mit $1.6\text{M}$ Parametern und $36{,}538$ Trainings-Sprites sieht der Kritiker jedes Trainingsbild in 12 000 Schritten etwa 20-mal. Bei den Kritikern mit Straftermen unterscheiden sich die Scores für Trainingsbilder und zurückgehaltene Bilder nur wenig ($+0.02$ bis $+0.05$ gegenüber einem Abstand von $\sim 0.7$ zwischen echten und erzeugten Bildern, siehe Aufgabe :exref:`ex_dcgan-scale-free-overfitting-statistic`). Bei kleineren Datensätzen oder längeren Läufen kann sich die FID jedoch verschlechtern, wenn der Diskriminator Trainingsbilder auswendig lernt :cite:`Karras.Aittala.Hellsten.ea.2020`. Die gängigen Gegenmaßnahmen, Adaptive Discriminator Augmentation (ADA, :citet:`Karras.Aittala.Hellsten.ea.2020`) und DiffAugment :cite:`Zhao.Liu.Lin.ea.2020`, wenden innerhalb des Gradientenpfads invertierbare Augmentierungen auf echte und erzeugte Bilder an. Diese Augmentierungen *schlagen nicht auf den Generator durch* (engl. non-leaking). Augmentieren wir hingegen nur echte Bilder, wie bei unseren horizontalen Spiegelungen, lernt der Generator die augmentierte Verteilung (Aufgabe :exref:`ex_dcgan-augmentation-leaks`).

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:c913f398-->
DCGAN stabilisiert das adversariale Training konvolutionärer Netze mit dem klassischen logistischen Verlust durch transponierte Faltungen und Faltungen mit Stride, Batchnormalisierung, kleine Anfangsgewichte ($\sigma = 0.02$) und Adam mit $\beta_1 = 0.5$. Ohne Batchnormalisierung und stabilisierende Architekturelemente kollabiert der Generator beim Training mit dem logistischen Verlust ohne Strafterme vollständig. Um null zentrierte Gradientenstrafterme ($R_1 + R_2$) dämpfen die Gradienten des Kritikers um zwei Größenordnungen. Damit lässt sich ein minimalistisches Basisnetz mit bilinearem Resampling zuverlässig trainieren. Der exponentielle gleitende Mittelwert der Generatorgewichte dämpft die Minimax-Oszillationen zusätzlich, sofern wir bei vorhandener Batchnormalisierung deren gleitende Statistiken neu kalibrieren. Für die quantitative Bildbewertung vergleichen wir schließlich die Punktwolken echter und erzeugter Merkmalsvektoren mithilfe von Fréchet-Distanz (:eqref:`eq_gan_fid`), Kernel Inception Distance (:eqref:`eq_gan_kid`) sowie $K$-NN-Präzision und -Trefferquote. Als Referenz für den endlichen Stichprobenumfang dient dabei stets der Vergleich zweier echter Stichproben.

<!--en:2294243a-->
## Übungen

<!--en:079b218a-->
1. **Gewicht der Gradientenstrafe und Bildauflösung.** Da $\nabla_x D(x)$ die Dimension einer inversen Länge hat, hängt das Gewicht $\gamma$ der Gradientenstrafe von der Bildauflösung und der Skalierung der Pixelwerte ab.
    1. Der Operator $U$ vergrößere ein $64 \times 64$-Bild $x$ auf $128 \times 128$, indem er jedes Pixel in einen $2 \times 2$-Block kopiert. Der Operator $A$ mittele jeden $2 \times 2$-Block, sodass für $D' = D \circ A$ die Gleichung $D'(Ux) = D(x)$ gilt. Drücken Sie $\|\nabla D'(Ux)\|^2$ durch $\|\nabla D(x)\|^2$ aus. Bestimmen Sie das Gewicht $\gamma'$, bei dem die Strafe für $128 \times 128$ genauso groß ist wie die für $64 \times 64$.
    1. Wie muss $\gamma$ skaliert werden, wenn die Pixelwerte auf $[0, 1]$ statt auf $[-1, 1]$ normalisiert werden?
    1. Setzen Sie Ihr Ergebnis aus (a) in Beziehung zur Heuristik $\gamma \propto r^2 / B$ (für die Auflösung $r$ und die Batchgröße $B$), die StyleGAN2-ADA verwendet :cite:`Karras.Aittala.Hellsten.ea.2020`.
   :exlabel:`ex_dcgan-penalty-weight-resolution`

<!--en:25d1f16f-->
1. [code] **Skalenunabhängige Statistiken zur Überanpassung des Kritikers.** Um für Zielfunktionen mit unterschiedlichen Score-Skalen zu vergleichen, wie stark der Kritiker die Trainingsdaten auswendig lernt, definieren :citet:`Karras.Aittala.Hellsten.ea.2020` die Größen

   $$
   \begin{aligned}
   r_v &= \frac{\mathbb{E}[D_{\textrm{train}}] - \mathbb{E}[D_{\textrm{holdout}}]}{\mathbb{E}[D_{\textrm{train}}] - \mathbb{E}[D_{\textrm{generated}}]}, \\
   r_t &= \mathbb{E}\big[\operatorname{sign}(D_{\textrm{train}})\big].
   \end{aligned}
   $$

    1. Zeigen Sie, dass $r_v$ unter positiven affinen Transformationen $D \mapsto aD + c$ ($a > 0$) invariant ist. Zeigen Sie außerdem, dass die unnormierte Differenz $\mathbb{E}[D_{\textrm{train}}] - \mathbb{E}[D_{\textrm{holdout}}]$ mit $a$ skaliert und $r_t$ von der Verschiebung $c$ abhängt.
    1. Betrachten Sie einen verschiebungsinvarianten Paarkritiker. Unter welchen Transformationen von $D$ bleibt $\mathbb{E}[\operatorname{sign}(D_{\textrm{train}} - D_{\textrm{holdout}})]$ invariant? Welchen Wert nimmt diese Größe an, wenn die Scores für Trainingsbilder und zurückgehaltene Bilder identisch verteilt sind?
    1. Ändern Sie `train_arm` so, dass die Funktion `net_D` zurückgibt. Berechnen Sie $r_v$ und $\mathbb{E}[\operatorname{sign}(D_{\textrm{train}} - D_{\textrm{holdout}})]$ für jede der drei Varianten anhand von jeweils 1024 Bildern.
   :exlabel:`ex_dcgan-scale-free-overfitting-statistic`

<!--en:9cc0a1eb-->
1. [code] **Verzerrung bei endlichen Stichproben: FID und KID im Vergleich.**
    1. Berechnen Sie die FD und $\widehat{\mathrm{MMD}}^2$ zwischen zwei disjunkten Mengen von je $n$ echten Trainings-Sprites. Verwenden Sie für $n \in \{250, 500, 1000, 2000, 4059\}$ jeweils fünf zufällige Aufteilungen.
    1. Passen Sie eine lineare Regression der FD auf $1/n$ an und extrapolieren Sie auf $1/n \to 0$ (:citet:`Chong.Forsyth.2020`). Wie groß ist die Verzerrung durch die endliche Stichprobe bei $n = 500$ und $n = 4{,}059$?
    1. Wiederholen Sie die lineare Anpassung in Abhängigkeit von $1/n$ für den Generator `RpGAN + R1 + R2` im Vergleich mit zurückgehaltenen Sprites. Die Steigung bezüglich $1/n$ hängt von der Merkmalskovarianz des Generators ab. Erklären Sie damit, wie die Verzerrung bei kleinem $n$ die Rangfolge zweier Generatoren umkehren kann.
   :exlabel:`ex_dcgan-measure-finite-sample-bias`

<!--en:8021cfc2-->
1. [code] **Gewichtsinitialisierung und Moduskollaps.** ● Definieren Sie die Sample-Vielfalt als Quotienten zweier mittlerer paarweiser Abstände im Merkmalsraum: Im Zähler stehen die Abstände der aus `z_show` erzeugten Bilder, im Nenner die Abstände von 64 zurückgehaltenen Sprites.
    1. Protokollieren Sie die Sample-Vielfalt und $\mathbb{E}_x[\|\nabla_x D(x)\|^2]$ im Abstand von 500 Schritten über insgesamt 5000 Schritte für die Variante `NS` ohne Strafterme. Verwenden Sie einmal die Fan-in-Initialisierung nach He und dann eine Initialisierung aus $\mathcal{N}(0, \sigma^2)$ mit $\sigma \in \{0.01, 0.02, 0.05\}$. Steigt die Gradientennorm des Kritikers sprunghaft an, bevor die Vielfalt einbricht?
    1. Wiederholen Sie den Lauf mit Fan-in-Initialisierung für drei Zufallsstartwerte. Messen Sie, wie stark der Schritt, bei dem der Kollaps eintritt, zwischen den Läufen variiert.
   :exlabel:`ex_dcgan-rerun-arm-classic-loss`

<!--en:abbb7a5b-->
1. **Von FID und KID erfasste Momente.** ●
    1. Entwickeln Sie den kubischen KID-Kernel $k(u, v) = (u^\top v / d + 1)^3$ mit dem binomischen Lehrsatz. Zeigen Sie damit, dass $\mathrm{MMD}^2(p, q)$ für die Grundgesamtheit gleich

       $$\sum_{j=1}^{3} \binom{3}{j} d^{-j}\, \big\| \mathbb{E}_p[u^{\otimes j}] - \mathbb{E}_q[u^{\otimes j}] \big\|_F^2,$$

       ist und somit die ersten drei Momente der Merkmalsverteilung vergleicht.
    1. Berechnen Sie in einer Dimension ($d = 1$) die FID und die $\mathrm{MMD}^2$ mit kubischem Kernel jeweils für die Grundgesamtheit. Vergleichen Sie $p = \mathcal{N}(0, 1)$ mit (i) einer verschobenen Exponentialverteilung $X - 1$ mit $X \sim \textrm{Exp}(1)$ (Mittelwert $0$, Varianz $1$, Schiefe $2$) und (ii) der symmetrischen Mischverteilung $\tfrac12\mathcal{N}(-a, 1 - a^2) + \tfrac12\mathcal{N}(a, 1 - a^2)$ ($0 < a < 1$).
   :exlabel:`ex_dcgan-moments-fid-kid`

<!--en:bce0717a-->
1. **Datenaugmentierungen ohne Durchschlagen auf den Generator.** ● Sei $F$ der Operator für die horizontale Spiegelung und $T_\rho r = (1 - \rho)r + \rho\,F_\# r$ die Verteilung, die entsteht, wenn Samples aus $r$ mit Wahrscheinlichkeit $\rho$ gespiegelt werden.
    1. Wenn nur echte Bilder mit $\rho = \tfrac12$ gespiegelt werden (wie in `sample_real`), lernt der Generator die Verteilung $T_{1/2} p$. Ein Anteil $\alpha$ der echten Sprites blickt nach links, ohne ein rechtsgewandtes Gegenstück zu haben. Welcher Anteil der Samples eines optimalen Generators blickt nach rechts?
    1. Die Transformation $T_\rho$ werde im Kritiker auf echte und erzeugte Bilder angewandt. Zeigen Sie mit $F^2 = I$, dass $T_\rho q = T_\rho p \implies q = p$ gilt, sofern $\rho \neq \tfrac12$ ist (die Augmentierung *schlägt nicht auf den Generator durch*, :citet:`Karras.Aittala.Hellsten.ea.2020`). Zeigen Sie außerdem, dass bei $\rho = \tfrac12$ jede Verteilung mit derselben durch Spiegelung symmetrisierten Dichte wie $p$ von $p$ ununterscheidbar ist.
   :exlabel:`ex_dcgan-augmentation-leaks`

<!--en:0168d7f7-->
[Diskussionen](https://d2l.discourse.group/t/36812)

<!--en:2f3e9583-->
Tiefe konvolutionäre GANs und Stabilisierung durch die Architektur<br>**transponierte Faltungen · DCGAN-Konventionen · EMA und Neukalibrierung von BatchNorm · R3GAN-Basisnetz ohne Normalisierung · FID, KID, Präzision und Trefferquote**

<!--en:35b9cdb9-->
::: {.slide title="Von flachen Vektoren zu 64×64-RGB-Bildern"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:a3111a77-->
Vollständig verbundene MLP-Generatoren (§{sec_basic_gan}) ignorieren räumliche Nachbarschaften und lieferten körnige $28 \times 28$-Silhouetten. RGB-Bilder der Größe $64 \times 64$ sind anspruchsvoller:

<!--en:e10682ad-->
- **Hochdimensionale räumliche Synthese:** Ein $64 \times 64$-RGB-Bild enthält $3 \times 64 \times 64 = 12{,}288$ korrelierte Pixelwerte, die aus einem zufälligen $100$-dimensionalen latenten Vektor $z \sim \mathcal{N}(0, I_{100})$ entstehen.

<!--en:53247227-->
- **Benchmark-Datensatz ($40{,}597$ Pokémon-Sprites):** Bildgröße $64 \times 64$, Pixelwerte auf $[-1, 1]$ skaliert, $36{,}538$ Trainings-Sprites und $4{,}059$ zurückgehaltene Test-Sprites für die quantitative Bewertung

<!--en:fc02691c-->
- **Zwei Leitfragen:** (1) Ersetzen um null zentrierte Gradientenstrafterme ($R_1 + R_2$) BatchNorm und stabilisierende Architekturelemente? (2) Wie lassen sich Bildtreue und Modusabdeckung ohne Likelihood bewerten?

<!--en:a24e26c4-->
::: {.slide title="Räumliches Upsampling durch transponierte Faltungen"}

<!--en:b6eae6dc-->
[Verdopplung der Auflösung]{.kicker}

<!--en:ad1ec1ba-->
- **Stufenweise Verdopplung der Auflösung:** Stufe 1 ($k = 4, s = 1, p = 0$) projiziert gemäß $1 \times 1 \to 4 \times 4$. Die Stufen 2–5 ($k = 4, s = 2, p = 1$) verdoppeln jeweils die Seitenlänge: $4 \to 8 \to 16 \to 32 \to 64$.

<!--en:b76a100f-->
- **Schachbrettmuster durch Überlappung (Odena et al., 2016):** Ist die Kerngröße $k$ nicht durch den Stride $s$ teilbar (z. B. $k = 3, s = 2$), überlappen sich die Beiträge an benachbarten Ausgabepixeln abwechselnd unterschiedlich oft ($2, 1, 2, 1$). Dadurch entstehen Schachbrettartefakte.

<!--en:e453b079-->
::: {.slide title="Die DCGAN-Architektur von 2015 und ihre Stabilisierung"}

<!--en:fc0cea51-->
[Architekturkonventionen]{.kicker}

<!--en:9fa67d90-->
Radford, Metz und Chintala (2015) stabilisierten den nicht sättigenden logistischen Verlust ohne Strafterme durch fünf Architekturkonventionen:

<!--en:2bac0b35-->
1. **Hierarchie ausschließlich aus Faltungen:** Alle Pooling-Schichten und vollständig verbundenen Schichten durch transponierte Faltungen in $G$ ($512 \to 256 \to 128 \to 64 \to 3$) und Faltungen mit Stride in $D$ ersetzen.

<!--en:1a348546-->
2. **Batchnormalisierung:** Aktivierungen nach jeder Faltung normalisieren, *außer* in der Ausgabeschicht des Generators und im Eingabeblock des Diskriminators.

<!--en:fc065211-->
3. **Aktivierungen, Initialisierung und Adam:** ReLU mit $\tanh$ am Ausgang in $G$, durchgehend Leaky ReLU ($\text{slope} = 0.2$) in $D$, Gewichte aus $\mathcal{N}(0, 0.02^2)$, Adam mit reduziertem Momentum $\beta_1 = 0.5$

<!--en:e8e90be1-->
::: {.slide title="Implementierung der Generator- und Kritikerblöcke von DCGAN" only="pytorch"}

<!--en:5a2cf839-->
[Aufbau des Codes]{.kicker}

<!--en:bcce77dd-->
::: {.slide title="Implementierung der Generator- und Kritikerblöcke von DCGAN" only="jax"}

<!--en:5a2cf839-2-->
[Aufbau des Codes]{.kicker}

<!--en:2b40cbf8-->
Initialisierung bei DCGAN: Faltungskerne aus $\mathcal{N}(0, 0.02^2)$, BatchNorm-Skalierungsfaktoren aus $\mathcal{N}(1, 0.02^2)$ und gleitende Mittelwerte mit Abklingfaktor $0.9$:

<!--en:0a270180-->
::: {.slide title="Implementierung der Generator- und Kritikerblöcke von DCGAN (Forts.)" only="jax"}

<!--en:5a2cf839-3-->
[Aufbau des Codes]{.kicker}

<!--en:53de967e-->
Der Generatorblock erhöht die Auflösung durch eine transponierte Faltung. Der Kritikerblock verringert sie durch eine Faltung mit Stride:

<!--en:6859f9b8-->
::: {.slide title="Implementierung der Generator- und Kritikerblöcke von DCGAN (Netze)" only="jax"}

<!--en:5a2cf839-4-->
[Aufbau des Codes]{.kicker}

<!--en:ad26929f-->
Die verketteten Blöcke bilden einen 100-dimensionalen latenten Vektor auf ein $64 \times 64$-RGB-Bild und dieses zurück auf einen einzigen Score ab:

<!--en:19d942e0-->
::: {.slide title="EMA der Gewichte und Neukalibrierung von BatchNorm" only="pytorch"}

<!--en:b5f00f5b-->
[Dämpfung der Kreisbewegung]{.kicker}

<!--en:2bc5bedd-->
Die Iterierten des Minimax-Gradientenverfahrens kreisen um das Gleichgewicht (§{sec_gan_convergence}). Ihr Schwerpunkt liegt näher an der Lösung als die letzte Iterierte. Zur Erzeugung von Samples dient deshalb der **exponentielle gleitende Mittelwert (EMA)** der Generatorparameter:

<!--en:c4f2d78e-->
- **Parametrisierung über die Halbwertszeit ($n_{1/2}$ Schritte, $\beta = 2^{-1/n_{1/2}}$):**
  $$\bar{\theta} \leftarrow \beta\,\bar{\theta} + (1 - \beta)\,\theta$$
  Die Halbwertszeit beträgt $32{,}000$ Bilder ($n_{1/2} = 125$ Schritte bei Batchgröße $256$, $500$ Schritte bei Batchgröße $64$).

<!--en:9d2a1a2b-->
- **Fehlerquelle: veraltete BatchNorm-Statistiken:** Die vom aktuellen Netz kopierten Puffer für gleitende Mittelwerte und Varianzen passen nicht zu den gemittelten Gewichten $\bar{\theta}$. Nach dem Training kalibrieren $100$ neue Vorwärtsdurchläufe mit je einem Batch im Modus `train()` ($0.9^{100} \approx 3 \times 10^{-5}$) die BatchNorm-Puffer für $\bar{\theta}$ neu.

<!--en:a2c4041d-->
::: {.slide title="EMA der Gewichte und Neukalibrierung von BatchNorm" only="jax"}

<!--en:b5f00f5b-2-->
[Dämpfung der Kreisbewegung]{.kicker}

<!--en:2bc5bedd-2-->
Die Iterierten des Minimax-Gradientenverfahrens kreisen um das Gleichgewicht (§{sec_gan_convergence}). Ihr Schwerpunkt liegt näher an der Lösung als die letzte Iterierte. Zur Erzeugung von Samples dient deshalb der **exponentielle gleitende Mittelwert (EMA)** der Generatorparameter:

<!--en:c4f2d78e-2-->
- **Parametrisierung über die Halbwertszeit ($n_{1/2}$ Schritte, $\beta = 2^{-1/n_{1/2}}$):**
  $$\bar{\theta} \leftarrow \beta\,\bar{\theta} + (1 - \beta)\,\theta$$
  Die Halbwertszeit beträgt $32{,}000$ Bilder ($n_{1/2} = 125$ Schritte bei Batchgröße $256$, $500$ Schritte bei Batchgröße $64$).

<!--en:9d2a1a2b-2-->
- **Fehlerquelle: veraltete BatchNorm-Statistiken:** Die vom aktuellen Netz kopierten Puffer für gleitende Mittelwerte und Varianzen passen nicht zu den gemittelten Gewichten $\bar{\theta}$. Nach dem Training kalibrieren $100$ neue Vorwärtsdurchläufe mit je einem Batch im Modus `train()` ($0.9^{100} \approx 3 \times 10^{-5}$) die BatchNorm-Puffer für $\bar{\theta}$ neu.

<!--en:b432368a-->
::: {.slide title="EMA der Gewichte und Neukalibrierung von BatchNorm (Forts.)" only="jax"}

<!--en:b5f00f5b-3-->
[Dämpfung der Kreisbewegung]{.kicker}

<!--en:d80c2dfa-->
`average` ist ein einziger kompilierter Aktualisierungsschritt. Er verschiebt jeden Parameter der Kopie in Richtung des aktuellen Modells und kopiert den übrigen Zustand, etwa die BatchNorm-Statistiken:

<!--en:93e63b94-->
::: {.slide title="DCGAN-Trainingsergebnisse für Pokémon-Sprites"}

<!--en:14bf9eab-->
[Referenzwert nach 20 Epochen]{.kicker}

<!--en:7beed333-->
Nach einem Training über $20$ Epochen ($2{,}840$ Schritte bei Batchgröße $256$, $\text{lr} = 2 \times 10^{-4}, \beta_1 = 0.5$) mit dem klassischen nicht sättigenden logistischen Verlust erzeugt DCGAN erkennbare, vielfältige Sprites:

<!--en:3169b1fd-->
- **Sichtprüfung:** Die erzeugten Sprites zeigen vielfältige Silhouetten und Farbpaletten auf weißem Hintergrund, allerdings mit feinen, fleckigen Texturen.

<!--en:d0193b27-->
- **Kritikerverlust unter dem Gleichgewichtswert:** Obwohl `loss_D` ($\sim 0.25\text{--}0.45$) deutlich unter dem Wert für zufälliges Raten $2\log 2 \approx 1.39$ bleibt, verhindern Batchnormalisierung und Initialisierung aus $\mathcal{N}(0, 0.02^2)$ einen Moduskollaps.

<!--en:ae811728-->
::: {.slide title="Warum ohne BatchNorm? Das minimalistische R3GAN-Basisnetz"}

<!--en:9ca574c2-->
[Von Heuristiken zu Zielfunktionen mit Straftermen]{.kicker}

<!--en:3f445209-->
- **Warum BatchNorm für $R_1 + R_2$ entfernen?** Batchnormalisierung koppelt alle Samples eines Minibatches. Damit entfällt die Unabhängigkeit der Eingabegradienten $\nabla_{x_i} D(x_i)$ pro Sample, die $R_1$ und $R_2$ bestrafen.

<!--en:c1c99b0b-->
- **Minimalistische Architektur ohne Normalisierung (nach R3GAN, Huang et al., 2024):**
  - **Bilineares Resampling und $3 \times 3$-Faltungen:** Sie ersetzen transponierte Faltungen und Faltungen mit Stride mit Kerngröße $4 \times 4$ und vermeiden so Schachbrettartefakte.
  - **Keine Normalisierungsschichten und kein $\tanh$ am Ausgang:** Sowohl $G$ ($2.8\text{M}$ Parameter, beginnend mit einer erlernten $4 \times 4 \times 128$-Konstante und dem projizierten $z$) als auch $D$ ($1.6\text{M}$ Parameter) verwenden Leaky ReLU ($\text{slope} = 0.2$) mit Fan-in-Initialisierung nach He, $\mathrm{Var}(w) = \frac{2}{(1 + 0.2^2)\,n_{\mathrm{in}}}$.

<!--en:b461d260-->
::: {.slide title="Implementierung des Generators ohne Normalisierung" layout="tight"}

<!--en:ccb020e5-->
[Codeaufbau]{.kicker}

<!--en:49cc584e-->
- **4×4-Eingangsteil und 4 bilineare Stufen:** Einen erlernten konstanten $4 \times 4 \times 128$-Tensor mit einem projizierten latenten Vektor verketten, dann die Auflösung mit bilinearem Resampling, $3 \times 3$-Faltungen und Leaky ReLU erhöhen:

<!--en:dc0074ae-->
::: {.slide title="Implementierung des Generators ohne Normalisierung (Code)" layout="tight"}

<!--en:ccb020e5-2-->
[Codeaufbau]{.kicker}

<!--en:2940236e-->
::: {.slide title="Implementierung des Kritikers ohne Normalisierung" layout="tight"}

<!--en:ccb020e5-3-->
[Codeaufbau]{.kicker}

<!--en:66aaa1ab-->
- **Unabhängige Batch-Zeilen:** Der Kritiker kehrt den Aufbau von $G$ um. Vier Stufen aus $3 \times 3$-Faltung, Leaky ReLU und bilinearem Downsampling führen ohne BatchNorm zu einer $4 \times 4$-Merkmalskarte $\phi(x)$:

<!--en:6259aa2e-->
::: {.slide title="Implementierung des Kritikers ohne Normalisierung (Code)" layout="tight"}

<!--en:ccb020e5-4-->
[Codeaufbau]{.kicker}

<!--en:5fcdb130-->
::: {.slide title="Kontrollierte Ablationsstudie mit drei Zielfunktionen"}

<!--en:0663da23-->
[Versuchsaufbau]{.kicker}

<!--en:32e84cac-->
Drei Trainingsläufe für das Basisnetz ohne Normalisierung beginnen mit identischen Gewichten (Adam mit $\beta_1 = 0$, $\beta_2 = 0.99$ und $\text{lr} = 2 \times 10^{-4}$, Batchgröße $64$, horizontale Spiegelungen, EMA-Halbwertszeit $500$ Schritte). Jeweils ein Bestandteil der Zielfunktion ändert sich:

<!--en:8162d4c0-->
| Versuchsvariante | Kritikerverlust $L(D)$ | Gradientenstrafe $\frac{\gamma}{2}(R_1 + R_2)$ | Trainingsschritte | Geprüfte Hypothese |

<!--en:fa0fcd32-->
| **`NS`** | Klassisch logistisch (`gan_loss_D`) | $\gamma = 0$ (keine) | $6{,}000$ | Lässt sich das Basisnetz ohne BatchNorm und Strafterme trainieren? |

<!--en:fdf2ee61-->
| **`NS + R1 + R2`** | Klassisch logistisch (`gan_loss_D`) | $\gamma = 10$ | $12{,}000$ | Ersetzen die um null zentrierten Gradientenstrafterme $R_1 + R_2$ die Batchnormalisierung? |

<!--en:35dbe314-->
| **`RpGAN + R1 + R2`** | Relativistische Paarbildung (`rpgan_loss_D`) | $\gamma = 10$ | $12{,}000$ | Verbessert das paarweise Ranking die Ergebnisse gegenüber dem klassischen logistischen Verlust mit Straftermen? |

<!--en:01dba6a4-->
::: {.slide title="Steilheit des Kritikers, Ranking-Genauigkeit und Ranggewichte" layout="tight"}

<!--en:2b547e79-->
[Verlauf der Trainingsdynamik]{.kicker}

<!--en:b0b9dd59-->
Die Messwerte für $\mathbb{E}_x[\|\nabla_x D(x)\|^2]$, die Genauigkeit des paarweisen Rankings $\hat{P}(D(x) > D(x'))$ und das relativistische Ranggewicht $w_{\mathrm{Rp}} = \mathbb{E}[\sigma(D(x') - D(x))]$ im Abstand von $250$ Schritten erklären den Kollaps ohne Strafterme:

<!--en:dd7363ec-->
- **Steilheit des Kritikers ($\mathbb{E}_x\|\nabla_x D\|^2$, links):** Ohne $R_1 + R_2$ steigt die quadrierte Gradientennorm des Kritikers auf einen $100\times$ so hohen Wert ($1\text{--}8$ gegenüber $\sim 0.01$ in beiden Varianten mit Straftermen).

<!--en:273f7849-->
- **Genauigkeit und Ranggewicht (Mitte und rechts):** Alle drei Kritiker ordnen $> 90\%$ der Paare richtig. Die Ranking-Genauigkeit allein warnt also nicht vor dem Kollaps. Ohne Strafterme drückt der Kritiker jedoch $w_{\mathrm{Rp}}$ unter $0.15$. Mit Straftermen bleibt der Logit-Abstand klein ($w_{\mathrm{Rp}} \in [0.25, 0.40]$, mediane Differenz $\approx 0.7$).

<!--en:b8b95539-->
::: {.slide title="Visueller Vergleich: Strafterme verhindern den Kollaps"}

<!--en:2112c738-->
[Raster erzeugter Sprites]{.kicker}

<!--en:21457778-->
Alle vier EMA-Generatoren erhalten dieselben $64$ latenten Vektoren `z_show`:

<!--en:89c9a9fc-->
- Ohne BatchNorm oder $R_1 + R_2$ kollabiert die Variante `NS` ohne Strafterme vollständig zu einer einheitlich grauen Textur. Beide Varianten mit Straftermen (`NS + R1 + R2` und `RpGAN + R1 + R2`) erzeugen dagegen scharf umrissene, vielfältige Sprite-Silhouetten auf reinweißem Hintergrund.

<!--en:c6e73c89-->
::: {.slide title="Bewertung im Merkmalsraum: FID und KID" layout="tight"}

<!--en:f0159241-->
[Quantitative Qualitätsmaße]{.kicker}

<!--en:54199155-->
Für einen quantitativen Vergleich der Generatoren bildet ein vortrainierter Merkmalsextraktor $\phi(x) \in \mathbb{R}^d$ echte und erzeugte Bilder in den Merkmalsraum ab. Die Abstände zwischen den Merkmalsverteilungen lassen sich in geschlossener Form berechnen:

<!--en:24f70f07-->
- **Fréchet Inception Distance (FID, Heusel et al., 2017):** An die echten und erzeugten Merkmalsvektoren je eine Gauß-Verteilung $\mathcal{N}(\mu_p, \Sigma_p)$ bzw. $\mathcal{N}(\mu_q, \Sigma_q)$ anpassen und deren quadrierten $W_2$-Abstand berechnen:
  $$\mathrm{FID} = \|\mu_p - \mu_q\|^2 + \operatorname{tr}\!\Big(\Sigma_p + \Sigma_q - 2\big(\Sigma_p^{1/2}\Sigma_q\Sigma_p^{1/2}\big)^{1/2}\Big)$$
  Erfasst die ersten beiden Momente. Bei endlicher Stichprobe verursacht die Kovarianzschätzung eine positive Verzerrung der Ordnung $O(1/n)$.

<!--en:cd71c20c-->
- **Kernel Inception Distance (KID, Bińkowski et al., 2018):** Erwartungstreuer Schätzer $\widehat{\mathrm{MMD}}^2$ mit kubischem Polynom-Kernel $k(u, v) = (u^\top v / d + 1)^3$ (vergleicht Merkmalsmomente bis zur dritten Ordnung, erwartungstreu für jeden Stichprobenumfang $n$)

<!--en:1b1c670f-->
::: {.slide title="Bewertung im Merkmalsraum: FID und KID (Code)" layout="tight"}

<!--en:f0159241-2-->
[Quantitative Qualitätsmaße]{.kicker}

<!--en:787c5433-->
::: {.slide title="Bildtreue und Abdeckung: k-NN-Präzision und -Trefferquote" layout="tight"}

<!--en:20d5e8e3-->
[Trägerschätzung mit k-NN]{.kicker}

<!--en:bd718e29-->
Ein einzelner Abstand wie FID oder KID unterscheidet **geringe Bildtreue der Samples** (unrealistische Artefakte) und **geringe Modusabdeckung** (Modusausfall) nicht. Kynkäänniemi et al. (2019) trennen beides durch Kugeln im Merkmalsraum, deren Radius die $K$ nächsten Nachbarn bestimmen ($K = 3$):

<!--en:0b363998-->
- Um jeden echten Merkmalsvektor $\phi(x_i)$ die euklidische Kugel $B(\phi(x_i), r_i)$ bis zu seinem $K$-ten nächsten echten Nachbarn legen. Für die erzeugten Merkmalsvektoren $B(\phi(x'_j), r'_j)$ analog definieren:
  $$\mathrm{precision} = \frac{1}{n_q}\sum_{j=1}^{n_q}\mathbb{1}\!\left[\phi(x'_j) \in \bigcup_i B\big(\phi(x_i), r_i\big)\right]$$
  $$\mathrm{recall} = \frac{1}{n_p}\sum_{i=1}^{n_p}\mathbb{1}\!\left[\phi(x_i) \in \bigcup_j B\big(\phi(x'_j), r'_j\big)\right]$$

<!--en:9b53153e-->
- Kollabiert ein Generator auf eine einzige Ausgabe, gilt für seine Radien $r'_j \to 0$ und damit **$\mathrm{recall} \to 0$**.

<!--en:b718d984-->
::: {.slide title="Bildtreue und Abdeckung: k-NN-Präzision und -Trefferquote (Code)" layout="tight"}

<!--en:20d5e8e3-2-->
[Trägerschätzung mit k-NN]{.kicker}

<!--en:e5215078-->
::: {.slide title="Bewertung der Läufe im Vergleich mit zwei echten Stichproben" layout="tall-output"}

<!--en:8ac029fe-->
[Bewertung von 4059 Sprites]{.kicker}

<!--en:5bed6823-->
Grundlage sind die 128-dimensionalen Merkmale des CIFAR-10-Klassifikators `FeatureCNN` aus 3 Blöcken (Average-Pooling verkleinert $64 \times 64$-Sprites auf $32 \times 32$). Für zwei unabhängige Ziehungen latenter Codes jeweils $n = 4{,}059$ erzeugte Samples mit den $4{,}059$ zurückgehaltenen Sprites vergleichen:

<!--en:03af1577-->
- **1. Referenzwert für zwei echte Stichproben:** Zwei disjunkte Mengen echter Bilder ergeben $\mathrm{FD} \approx 0.18$ (Verzerrung durch die Kovarianzschätzung bei endlicher Stichprobe), $\widehat{\mathrm{MMD}}^2 \approx 0.000$ sowie Präzision und Trefferquote von jeweils $0.95$.

<!--en:b563e641-->
::: {.slide title="Bewertung der Läufe im Vergleich mit zwei echten Stichproben (Forts.)"}

<!--en:8ac029fe-2-->
[Bewertung von 4059 Sprites]{.kicker}

<!--en:335de6ba-->
- **2. Kollaps und Varianten mit Straftermen:** Ohne Strafterme erreicht `NS` die Werte $\mathrm{FD} \approx 64\text{--}72$ und $\mathrm{recall} = 0.00$. Mit $R_1 + R_2$ sinkt die FD auf ein Zehntel ($\sim 3.5\text{--}5.5$), die Präzision steigt wieder auf $\sim 0.35\text{--}0.4$ und die Trefferquote auf $\sim 0.4$.

<!--en:2da0588a-->
- **3. Invarianz des Netzes zur Merkmalsextraktion:** Das $2 \times 2$-Pooling auf die Auflösung $32 \times 32$ für `FeatureCNN` glättet die hochfrequenten Flecken von DCGAN. Daher erreicht DCGAN die niedrigste FD ($\sim 2.0$), obwohl die Bilder bei $64 \times 64$ fleckiger aussehen.

<!--en:db5473ce-->
::: {.slide title="Interpolation im latenten Raum und pixelweise Überblendungen"}

<!--en:76209c44-->
[Prüfung der Glattheit der Mannigfaltigkeit]{.kicker}

<!--en:59b3813b-->
Lernt `RpGAN + R1 + R2` eine glatte Bildmannigfaltigkeit oder lernt der Generator einzelne Vorlagen auswendig? Dazu zwischen den Codes von $64$ latenten Codepaaren linear interpolieren, $z_t = (1 - t)z_0 + t z_1$ ($t \in [0, 1]$), und $G(z_t)$ mit pixelweisen Überblendungen $(1 - t)G(z_0) + t G(z_1)$ vergleichen:

<!--en:97326296-->
- **Übergänge auf der Mannigfaltigkeit:** Die latenten Zwischenschritte bewahren die volle 3-NN-Präzision des Generators im Merkmalsraum ($\sim 0.35\text{--}0.38$). Form und Farbe ändern sich fließend ohne Sprünge.

<!--en:542afbe6-->
- **Überblendungen außerhalb der Mannigfaltigkeit:** Pixelweise Mischungen erzeugen Doppelbelichtungen mit Geisterkonturen. Ihre Präzision im Merkmalsraum sinkt auf die Hälfte ($\sim 0.18\text{--}0.21$).

<!--en:f14e1a83-->
::: {.slide title="Überanpassung des Kritikers und Augmentierung ohne Durchschlagen auf den Generator"}

<!--en:3573544c-->
[Generalisierung bei wenigen Daten]{.kicker}

<!--en:d16ea3fa-->
- **Überanpassung bei kleinen Datensätzen:** Mit $R_1 + R_2$ bleibt die Score-Differenz des Kritikers zwischen Trainingsdaten und zurückgehaltenen Daten sehr klein ($+0.02\text{--}0.05$ bei $36{,}538$ Sprites). Kleinere Datensätze erfordern Datenaugmentierung, damit der Kritiker die Bilder nicht auswendig lernt.

<!--en:d5350e55-->
- **Warum die Augmentierung nur echter Bilder auf den Generator durchschlägt:** Eine zufällige Spiegelung $T_\rho$ nur der echten Bilder trainiert $G$ auf die Verteilung $T_\rho p$. Dadurch entstehen rechtsgewandte Kopien von Sprites, die im Original ausschließlich nach links blicken.

<!--en:791eb4be-->
- **Augmentierung ohne Durchschlagen (ADA / DiffAugment):** Die invertierbare Transformation $T_\rho$ ($\rho \neq \frac{1}{2}$) auf **beide** Bildgruppen (echte und erzeugte Bilder) innerhalb von $D$ anwenden. Dann gilt $T_\rho q = T_\rho p \implies q = p$.

<!--en:1d3250fe-->
::: {.slide title="Verständnisfragen: Faltungen, Bewertungsmaße und Spiegelungen"}

<!--en:929b7671-->
[Wissensprüfung]{.kicker}

<!--en:8bc0297f-->
Fragen zu konvolutionären GAN-Architekturen und Bewertungsmaßen:

<!--en:9738d455-->
1. **Skalierung von $\gamma$ mit der Auflösung:** Ein $64 \times 64$-Bild $x$ wird auf $128 \times 128$ vergrößert. Jedes Pixel wird dazu in einen $2 \times 2$-Block kopiert ($Ux$). Ein $128 \times 128$-Kritiker $D'(Ux) = D(x)$ mittelt jeden $2 \times 2$-Block wieder zu einem Pixel. Wie verhält sich $\|\nabla D'(Ux)\|^2$ zu $\|\nabla D(x)\|^2$, und wie muss $\gamma'$ mit der Auflösung $r$ skalieren, damit die Strafe gleich bleibt?

<!--en:1fc8339e-->
2. **Fehlerdiagnose anhand von Präzision und Trefferquote:** Generator A erreicht $(\text{precision}, \text{recall}) = (0.78, 0.01)$, Generator B dagegen $(\text{precision}, \text{recall}) = (0.12, 0.68)$. Welcher Fehler liegt jeweils vor?

<!--en:557b98b6-->
**Lösungen:**

<!--en:b2c4dafe-->
1. Jedes Pixel von $x$ verteilt sich auf $4$ Pixel von $Ux$ mit Gewicht $\frac{1}{4}$ pro Pixel. Daher gilt $\|\nabla D'(Ux)\|^2 = 4 \cdot \frac{1}{16}\|\nabla D(x)\|^2$ $= \mathbf{\frac{1}{4}\|\nabla D(x)\|^2}$. Damit die Strafe bei doppelter Auflösung gleich bleibt, muss sich das Gewicht vervierfachen: **$\gamma' = 4\gamma \propto r^2$**.

<!--en:8e071ade-->
2. **Generator A** hat eine hohe Präzision, aber eine Trefferquote nahe null. Er leidet unter **Moduskollaps** und erzeugt realistische Samples aus nur einem oder wenigen Modi. **Generator B** hat eine geringe Präzision, aber eine hohe Trefferquote. Er deckt den Träger der Datenverteilung breit ab, erzeugt jedoch viele **Ausreißerbilder mit geringer Bildtreue oder starkem Rauschen**.

<!--en:ffd482b1-->
::: {.slide title="Verständnisfragen: Faltungen, Bewertungsmaße und Spiegelungen (Forts.)"}

<!--en:929b7671-2-->
[Wissensprüfung]{.kicker}

<!--en:1d3a1a7c-->
3. **Horizontale Spiegelungen, die auf den Generator durchschlagen:** Ein Anteil $\alpha$ der echten Sprites blickt nach links und hat kein rechtsgewandtes Gegenstück. Alle übrigen Sprites sind spiegelsymmetrisch. Nur echte Trainingsbilder werden mit Wahrscheinlichkeit $\frac{1}{2}$ gespiegelt. Welcher Anteil der Samples eines optimalen Generators blickt dann nach rechts?

<!--en:23db34b8-->
**Lösung:**

<!--en:9731ed3c-->
3. Der Generator bildet die durch Spiegelung symmetrisierte Verteilung $T_{1/2}p$ nach. Darin blickt die Hälfte der ursprünglich linksgewandten asymmetrischen Masse $\alpha$ nach der Spiegelung nach rechts. Also blickt ein Anteil von **$\alpha / 2$** der erzeugten Samples nach rechts.

<!--en:4ae564a6-->
::: {.slide title="Zusammenfassung: Konvolutionäre GANs und Bildbewertung"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:c1d616dd-->
- **Von DCGAN-Heuristiken zu R3GAN:** Bilineares Resampling und $3 \times 3$-Faltungen beseitigen die Schachbrettmuster transponierter Faltungen. Ohne BatchNorm bleiben die Batch-Zeilen unabhängig. So lassen sich die Gradientenstrafen $R_1 + R_2$ exakt pro Sample berechnen.

<!--en:98821b57-->
- **EMA der Gewichte:** Glättet die Kreisbewegung um das Minimax-Gleichgewicht ($\bar{\theta} \leftarrow \beta\bar{\theta} + (1-\beta)\theta$). Bei vorhandener Batchnormalisierung sind $100$ neue Durchläufe im Modus `train()` nötig, um die BatchNorm-Puffer neu zu kalibrieren.

<!--en:abe10f69-->
- **Sich ergänzende Maße im Merkmalsraum:** **FID** am Referenzwert $+O(1/n)$ für zwei echte Stichproben kalibrieren. **KID** für erwartungstreue Vergleiche bei endlichem $n$ verwenden. **$K$-NN-Präzision und -Trefferquote** angeben, um Bildtreue und Modusabdeckung zu unterscheiden.
