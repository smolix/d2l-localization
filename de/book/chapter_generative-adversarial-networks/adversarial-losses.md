<!--en:459023a5-->
# Adversariale Verluste jenseits von GANs

<!--en:406dbd51-->
Bei der Vorhersage eines einzelnen Skalars, etwa eines Hauspreises, minimiert man üblicherweise den mittleren quadratischen Fehler (engl. mean squared error, MSE), also den $L_2$-Verlust. Bei der bedingten Bilderzeugung führen pixelweise Regressionsverluste dagegen zu unscharfen Ausgaben. Beispiele sind die Superauflösung (engl. super-resolution), die ein niedrig aufgelöstes Vorschaubild $c$ auf ein hochauflösendes Bild $x$ abbildet, und die Umwandlung von Skizzen in Fotos (pix2pix, :citet:`Isola.Zhu.Zhou.ea.2017`). In beiden Fällen tritt die Unschärfe sowohl beim $L_2$- als auch beim $L_1$-Verlust auf. Zu derselben niedrig aufgelösten Eingabe passen viele verschiedene hochauflösende Texturen. Minimiert das Netz den pixelweisen quadratischen Fehler, muss es ihren bedingten Mittelwert $\mathbb{E}[x \mid c]$ ausgeben. Dieses unscharfe Bild entspricht keinem der plausiblen scharfen Bilder. Kombinieren wir den Regressionsverlust mit einem Diskriminator $D(x, c)$, der beurteilt, ob die Ausgabe für die Bedingung $c$ ein realistisches Bild ist, erhalten wir wieder scharfe, hochfrequente Details.

<!--en:dcf38e64-->
Dieser Mechanismus erklärt, warum adversariale Verluste weiterhin verbreitet sind, obwohl Diffusions- und Flow-Matching-Modelle (:numref:`chap_diffusion`) eigenständige GANs bei der allgemeinen Bilderzeugung abgelöst haben. Ein erlernter Diskriminator dient nach wie vor zum Training von Einschrittgeneratoren, die aus mehrstufigen Diffusionsmodellen destilliert werden. Er kommt auch beim Training der Bild-Autoencoder (Tokenizer) für latente Diffusionsmodelle zum Einsatz und bei neuronalen Audio-Vocodern, die Sprachwellenformen aus Spektrogrammen rekonstruieren. Wir untersuchen zunächst an einem minimalen bimodalen Regressionsproblem, wie die bedingte Mittelung wirkt. Dann zeigen wir, warum die Einschrittdestillation bei hohem Rauschpegel des entrauschenden Lehrers ohne Diskriminator auf den bedingten Mittelwert zusammenfällt. Schließlich betrachten wir die Einsatzgebiete adversarialer Verluste in modernen generativen Pipelines.

<!--en:0682ddb2-->
## Bedingte Mittelwerte und multimodale Zielverteilungen

<!--en:b39714c3-->
### Zerlegung mithilfe des bedingten Mittelwerts

<!--en:2feda223-->
Sei $g(c)$ ein deterministischer Prädiktor für $x$ unter der Bedingung $c$, und sei $\mu(c) = \mathbb{E}[x \mid c]$ der bedingte Mittelwert. Addieren und subtrahieren wir $\mu(c)$ innerhalb von $\mathbb{E}[\|x - g(c)\|^2]$, so verschwindet der Kreuzterm. Es gilt $2\,\mathbb{E}[\langle x - \mu(c), \mu(c) - g(c) \rangle] = 0$ nach dem Satz vom iterierten Erwartungswert bei gegebenem $c$. Damit erhalten wir die orthogonale Zerlegung:

<!--en:f8046c22-->
Der erste Term ist die irreduzible bedingte Varianz $\operatorname{Var}(x \mid c)$, der zweite der quadrierte Abstand zwischen $g(c)$ und $\mu(c)$. Der bedingte Mittelwert $g^\star(c) = \mu(c)$ minimiert also als einziger den quadratischen Fehler. Ist $p(x \mid c)$ unimodal und stark konzentriert, so ist $\mu(c)$ ein repräsentativer Zielwert. Ist $p(x \mid c)$ multimodal, so fällt $\mu(c)$ in einen Bereich geringer Dichte zwischen den Modi. Gehen wir vom $L_2$-Verlust zum $L_1$-Verlust $\mathbb{E}[|x - g(c)|]$ über, tritt lediglich der koordinatenweise bedingte Median an die Stelle des bedingten Mittelwerts (Übung :exref:`ex_adversarial-losses-mean-and-median`). Auch dieser Median kann bei einer symmetrischen bimodalen Verteilung in der Lücke zwischen den Modi liegen.

<!--en:edaf0e92-->
### Ein bimodaler Benchmark mit zwei Datenbändern

<!--en:2ee4f377-->
Betrachten wir eine eindimensionale bedingte Verteilung mit $c \sim \textrm{Uniform}[-1, 1]$ und zwei parallelen Datenbändern, zwischen denen ein leerer Korridor liegt:

<!--en:e642bce1-->
Der bedingte Mittelwert $\mu(c) = c$ verläuft mitten durch den leeren Korridor zwischen den Bändern $x = c \pm 1$ und erreicht $\mathbb{E}[(x - \mu(c))^2] = 1 + 0.05^2 = 1.0025$. Legt sich ein Prädiktor dagegen auf das obere Band $g(c) = c + 1$ fest, beträgt sein quadratischer Fehler $\tfrac12(0) + \tfrac12(2^2) + 0.05^2 = 2.0025$ und ist damit doppelt so groß. Der quadratische Fehler bestraft also die Festlegung auf einen gültigen Modus der Datenverteilung.

<!--en:0c1f6341-->
Ein bedingter Diskriminator $D(x, c)$ schätzt hingegen das logarithmierte Dichteverhältnis (:numref:`sec_gan_conditional`):

<!--en:9f2cb398-->
Im leeren Korridor gilt $p(x, c) \approx 0$, und $D^\star(x, c)$ ist stark negativ. Die Minimierung des nicht sättigenden adversarialen Verlusts $-\log \sigma(D(\hat x, c))$ zwingt den Prädiktor $\hat x = g(c)$ daher auf eines der Datenbänder mit hoher Dichte.

<!--en:4b6ff38c-->
Wir prüfen das an zwei identischen MLPs mit jeweils 2 versteckten Schichten und $\tanh$-Aktivierung (`g_mse` und `g_adv`). Das Training beginnt mit denselben Gewichten. Dabei minimiert `g_mse` den quadratischen Fehler, während `g_adv` gegen einen Diskriminator $D(x, c)$ mit Leaky ReLU trainiert wird. Dazu verwenden wir `d2l.gan_loss_D` und `d2l.gan_loss_G` (:numref:`sec_gan_convergence`).

<!--en:72917b8c-->
Der erste Schüler minimiert den quadratischen Fehler mit Adam. Seine Lernrate sinkt über 2000 Schritte entlang einer Kosinuskurve von 0.01 auf null. Das Ergebnis kennen wir schon ohne Experiment: Nach :eqref:`eq_beyond_condmean` sollte der Schüler der Geraden $x = c$ folgen, die von jedem Band eine Einheit entfernt ist. Er dient als Kontrolle für den zweiten Schüler.

<!--en:d37088af-->
Der zweite Schüler spielt das Spiel mit logarithmischem Verlust. Dazu verwenden wir die Verluste `d2l.gan_loss_D` und `d2l.gan_loss_G` aus :numref:`sec_gan_convergence`. In jedem Schritt führt der Kritiker einen Schritt der logistischen Regression aus, um echte Paare $(x, c)$ von vorhergesagten Paaren $(\hat x, c)$ zu unterscheiden. Anschließend macht der Schüler einen Schritt zur Minimierung von $-\log \sigma\big(D(\hat x, c)\big)$ gegen den aktualisierten Kritiker. Die Vorhersage geht mit demselben Beobachtungsrauschen wie die Daten in das Spiel ein: Es gilt $\hat x = g(c) + 0.05\,\xi'$ mit einem jeweils neu gezogenen $\xi' \sim \mathcal{N}(0, 1)$. Die beiden Verteilungen sind damit gleich glatt, unterscheiden sich aber darin, wo ihre Masse liegt. Da ein Spiel langsamer vorankommt als eine Regression, erhält dieser Schüler 8000 Schritte. Der Schüler mit quadratischem Fehler erreicht sein Optimum nach 2000 Schritten und würde sich bei weiteren Schritten nicht mehr bewegen.

<!--en:9e6ab311-->
Drei Kennzahlen beschreiben jeden Schüler auf einem Gitter aus 201 Eingaben: der mittlere Abstand seiner Kurve zum näheren Band, der mittlere Abstand zum bedingten Mittelwert $x = c$ und der Anteil der Eingaben, bei denen die Kurve von beiden Bändern mehr als 0.25 entfernt ist.

<!--en:b4bb506a-->
Wir zeichnen die beiden erlernten Kurven aus `curves` über die Trainingsstichprobe `(c_data, x_data)` und tragen den bedingten Mittelwert $x = c$ als gestrichelte Linie ein. So wird der Unterschied zwischen Regression und adversarialem Training unmittelbar sichtbar:

<!--en:c38bd3ff-->
Wie vorhergesagt folgt das mit quadratischem Fehler trainierte Modell `g_mse` dem bedingten Mittelwert $\mu(c) = c$ durch den leeren Korridor. Es bleibt etwa eine Einheit von jedem Band entfernt, und $100\%$ der Vorhersagen liegen außerhalb der Bänder. Das adversariale Modell `g_adv` legt sich jeweils auf ein gültiges Datenband fest. Der Abstand zum nächsten Band beträgt im Mittel wenige Hundertstel, und höchstens $2\%$ der Punkte liegen beim Wechsel durch den Korridor zwischen den Bändern.

<!--en:331c5c23-->
Hier zeigt sich der grundlegende Zielkonflikt zwischen Wahrnehmungsqualität und Verzerrung (engl. perception–distortion trade-off) :cite:`Blau.Michaeli.2017`. Selbst ein idealer stochastischer Sampler $\hat x \sim p(x \mid c)$ hat den erwarteten quadratischen Fehler $2\,\operatorname{Var}(x \mid c)$. Sein MSE ist damit doppelt so groß (das PSNR um $3\text{ dB}$ geringer) wie beim unscharfen bedingten Mittelwert $\mu(c)$ (Übung :exref:`ex_adversarial-losses-perfect-sampler`). Zudem kann eine deterministische Funktion $g(c)$ für jede Eingabe $c$ nur einen Wert ausgeben. Um *beide* Bänder gleichzeitig abzudecken, müssen wir zusätzlich eine stochastische latente Variable $z$ zusammen mit $c$ übergeben: $\hat x = g(c, z)$ (Übung :exref:`ex_adversarial-losses-latent-bit`).

<!--en:e69310a1-->
## Diskriminatoren bei der Diffusionsdestillation

<!--en:938213a4-->
### Entrauscher als Schätzer bedingter Mittelwerte

<!--en:7d7322b2-->
Diffusions- und Flow-Matching-Modelle (:numref:`chap_diffusion`) trainieren einen Entrauscher (engl. denoiser) mit dem quadratischen Fehler. Er soll die rauschfreien Daten $x_0$ aus einer verrauschten Eingabe $x_\tau = x_0 + \tau\xi$ vorhersagen ($\xi \sim \mathcal{N}(0, I)$). Nach :eqref:`eq_beyond_condmean` berechnet der optimale Entrauscher den A-posteriori-Mittelwert $\mathbb{E}[x_0 \mid x_\tau]$ (:numref:`sec_diffusion-denoising`). Bei hohen Rauschpegeln $\tau$ sind viele rauschfreie Bilder mit $x_\tau$ vereinbar. Daher ist $\mathbb{E}[x_0 \mid x_\tau]$ ein unscharfer Mittelwert. Mehrstufige Sampler vermeiden diese Ausgabe, indem sie den Rauschpegel in vielen kleinen Schritten von einem hohen Wert bis auf $\tau \to 0$ senken.

<!--en:0b297d0a-->
Bei der *Destillation* eines mehrstufigen Diffusionslehrers zu einem schnellen Einschrittgenerator als Schüler (:numref:`sec_diffusion-few-step`) ist folgendes Vorgehen verbreitet: Wir addieren Rauschen $\tau\xi$ zur Ausgabe $\hat x$ des Schülers und passen $\hat x$ durch Regression an den entrauschten Schätzwert $\mathbb{E}[x_0 \mid \hat x + \tau\xi]$ des eingefrorenen Lehrers an. Bei hohem Rauschpegel $\tau$ ist der entrauschte Zielwert des Lehrers jedoch selbst ein bedingter Mittelwert. Ohne einen adversarialen Kritiker zieht er den Einschrittschüler zurück zum unscharfen Mittelwert.

<!--en:e4f571a9-->
### Destillation in geschlossener Form für den Benchmark mit zwei Datenbändern

<!--en:99930088-->
Beim Problem mit zwei Datenbändern :eqref:`eq_beyond_twoband` lässt sich der Bayes-optimale Lehrer für die Entrauschung beim Rauschpegel $\tau$ exakt in geschlossener Form angeben. Für eine verrauschte Beobachtung $\tilde x = x + \tau\xi$ eines Punkts auf dem Band $b \in \{-1, +1\}$ gilt $\tilde x \mid c, b \sim \mathcal{N}(c + b, \tau^2 + 0.05^2)$. Nach der Bayes-Regel betragen die A-posteriori-Log-Odds für $b = +1$ gegenüber $b = -1$ genau $2(\tilde x - c)/(\tau^2 + 0.05^2)$. Damit ergibt sich für das Bandzentrum folgender A-posteriori-Mittelwert:

<!--en:4e6b54b8-->
Jeder Schüler $g(c, z)$ erhält nun eine eindimensionale gaußverteilte latente Variable $z \sim \mathcal{N}(0, 1)$, sodass er beide Bänder abdecken kann. Wir vergleichen drei Trainingsziele beim Rauschpegel $\tau = 2$ des Lehrers:

<!--en:6d009697-->
1. *Nur Regression*: Wir minimieren $\|\hat x - \operatorname{stop\_grad}(\text{teacher}(\hat x + \tau\xi, c, \tau))\|^2$. Die Steigung von :eqref:`eq_beyond_teacher` bezüglich $\tilde x$ beträgt höchstens $(\tau^2 + 0.05^2)^{-1} \approx \tfrac14 < 1$ für $\tau = 2$. Die rechte Seite der Fixpunktgleichung $u = \mathbb{E}_\xi[\tanh((u + \sqrt{\tau^2 + 0.05^2}\,\xi)/(\tau^2 + 0.05^2))]$ für den Versatz $u = g(c, z) - c$ ist daher eine Kontraktion mit dem einzigen Fixpunkt $u = 0$ (im leeren Korridor).

<!--en:0df62ab1-->
2. *Nur Kritiker*: Wir trainieren $g(c, z)$ gegen einen bedingten Diskriminator $D(x, c)$ mit `d2l.gan_loss_D` und `d2l.gan_loss_G`.

<!--en:6cfad6c8-->
3. *Kritiker + Regression*: Wir addieren den adversarialen Verlust und den Regressionsverlust gegenüber dem Lehrer mit gleichem Gewicht (wie bei Adversarial Diffusion Distillation, ADD, :citet:`Sauer.Lorenz.Blattmann.ea.2023`).

<!--en:f689ab33-->
Ein Schüler liefert nun für jedes $c$ eine Verteilung von Ausgaben. Die Funktion `band_shares` wertet ihn für die 201 Eingaben des ersten Experiments und für 64 gleichmäßig abgestufte Quantile der latenten Variablen aus. Sie gibt den mittleren Abstand zum näheren Band zurück sowie die Anteile der Ausgaben, die höchstens 0.25 vom oberen beziehungsweise vom unteren Band entfernt sind.

<!--en:caa1cc83-->
Wir trainieren jeden Schüler mit drei Zufallsstartwerten. Bei gleichem Zufallsstartwert beginnen die drei Schüler mit denselben Gewichten und sehen dieselben Daten.

<!--en:b2ba4445-->
Die Ergebnisse für die drei Zufallsstartwerte bestätigen die Fixpunktanalyse:

<!--en:7663dd9c-->
- Bei der *Regression auf den Lehrer mit hohem Rauschpegel* fallen $100\%$ der Ausgaben in den leeren Korridor, etwa eine Einheit von jedem Band entfernt. Der Schüler ignoriert die latente Variable $z$.

<!--en:2cb0e85b-->
- Sowohl bei *Nur Kritiker* als auch bei *Kritiker + Regression* liegen $98\text{--}99\%$ der Ausgaben direkt auf den beiden Datenbändern. Der mittlere Abstand beträgt wenige Hundertstel und liegt stets unter $0.1$. Die Wahrscheinlichkeitsmasse verteilt sich gleichmäßig ($\sim 49\% / 49\%$) über die latente Variable $z$ auf das obere und das untere Band. Weil die Bandbreite mit $\sigma = 0.05$ so gering ist, wirkt die Steigung des optimalen Kritikers ($\delta / 0.05^2 = 400\delta$) stark rückstellend. Sie überwiegt den schwachen Zug des Lehrers mit hohem Rauschpegel zur Mitte deutlich.

<!--en:5b6daf8c-->
### Destillation, Autoencoder und Vocoder im großen Maßstab

<!--en:ae959f06-->
Drei wichtige Klassen moderner generativer Systeme kombinieren einen Regressionsverlust oder einen wahrnehmungsbasierten Verlust mit einem adversarialen Diskriminator:

<!--en:f2960c5f-->
1. **Adversarial Diffusion Distillation (ADD und DMD2):** ADD :cite:`Sauer.Lorenz.Blattmann.ea.2023` trainiert einen Einschrittgenerator für Bilder. Das Verfahren kombiniert den Regressionsverlust gegenüber einem entrauschenden Lehrer mit einem Diskriminator auf Basis eingefrorener DINOv2-Merkmale. Das Ergebnis entspricht unserem Experiment: Die Regression auf den Lehrer allein liefert unscharfe Bilder, der zusätzliche Diskriminator ermöglicht scharfe Samples in einem Schritt. Distribution Matching Distillation (DMD, :citet:`Yin.Gharbi.Zhang.ea.2024`) bildet die Differenz zwischen dem Score $\nabla_x \log p_\tau$ eines eingefrorenen Lehrers und dem Score $\nabla_x \log q_\tau$ eines Diffusionsmodells für die Verteilung der generierten Daten. Damit verringert das Verfahren $D_{\textrm{KL}}(q_\tau \,\|\, p_\tau)$. Diese Differenz kennen wir bereits: $\nabla_x \log p_\tau - \nabla_x \log q_\tau = \nabla_x \lambda_\tau$ ist der Eingabegradient des optimalen Kritikers $\lambda_\tau = \log(p_\tau/q_\tau)$ für die beiden verrauschten Verteilungen. Das Generatorupdate von DMD entspricht also dem ungewichteten ($w = 1$) Update für die umgekehrte KL-Divergenz aus Übung :exref:`ex_gan-generator-weight-divergences`. Die Steigung des Kritikers liefern hier zwei Score-Netze anstelle eines Diskriminators. Da wir beiden Verteilungen Rauschen hinzufügen, überlappen ihre Träger bei jedem Rauschpegel $\tau > 0$. Das entspricht dem Instance Noise aus :numref:`subsec_gan_disjoint_supports`. DMD2 :cite:`Yin.Gharbi.Park.ea.2024` ergänzt einen adversarialen Diskriminator, der echte Bilder zum Vergleich heranzieht. So kann der Einschrittschüler die Qualität des Lehrers übertreffen.

<!--en:f4fd6c32-->
2. **Latente Bild-Autoencoder (Tokenizer):** Latente Diffusionsmodelle (:numref:`sec_diffusion-latent`) komprimieren Bilder mithilfe eines Autoencoders in einen niedrigerdimensionalen latenten Raum (etwa mit VQGAN oder dem VAE von Stable Diffusion, :citet:`Esser.Rombach.Ommer.2021,Rombach.Blattmann.Lorenz.ea.2022`). Der komprimierte latente Code legt feine lokale Texturen nur unvollständig fest. Deshalb ergänzen wir beim Training des Decoders die pixelweisen $L_1/L_2$-Verluste und den wahrnehmungsbasierten Verlust (LPIPS) um einen Diskriminator für lokale Bildausschnitte (PatchGAN). Er sorgt für scharfe, hochfrequente Texturen.

<!--en:5984b9f1-->
3. **Neuronale Sprachvocoder:** Systeme zur Sprachsynthese aus Text sagen typischerweise ein Mel-Spektrogramm vorher. Es enthält für jeden Zeitrahmen die Beträge der Frequenzkomponenten, verwirft aber die Phase. Ein Vocoder wie HiFi-GAN :cite:`Kong.Kim.Bae.2020` rekonstruiert daraus die Audiowellenform. Mitteln wir eine Sinusschwingung $\sin(\omega n + \vartheta)$ über eine unbekannte Phase $\vartheta \sim \textrm{Uniform}[-a, a]$, sinkt ihre Amplitude um den Faktor $\sin(a)/a$. Für $\vartheta \sim \textrm{Uniform}[-\pi, \pi]$ verschwindet sie ganz (Übung :exref:`ex_adversarial-losses-phase`). Die Regression der Wellenform führt dann zu Stille. HiFi-GAN kombiniert deshalb einen $L_1$-Verlust für das Mel-Spektrogramm und einen Verlust für den Abgleich der Diskriminatoraktivierungen mit Wellenformdiskriminatoren für mehrere Perioden und Skalen.

<!--en:75054aad-->
## Vergleich der adversarialen Zielfunktionen dieses Kapitels

<!--en:6f0e4e44-->
:numref:`tab_gan_games` vergleicht die sechs adversarialen Formulierungen aus diesem Kapitel anhand dreier Fragen: Was schätzt der optimale Kritiker? Wie bewertet er Eingaben? Wie wird seine Funktionenklasse regularisiert?

<!--en:f65f7f1c-->
:Überblick über die adversarialen Zielfunktionen dieses Kapitels.

<!--en:8d2e8696-->
| Zielfunktion | Schätzwert des optimalen Kritikers | Bewertungsregel | Nebenbedingung / Regularisierung des Kritikers | Verhalten bei disjunkten Trägern (optimaler Kritiker) |

<!--en:3cf54432-->
| Logarithmischer Verlust / $f$-GAN (:numref:`sec_basic_gan`, :numref:`sec_gan_objectives`) | $\log(p/q)$ oder $f'(p/q)$ | Klassifikation einzelner Samples | Ohne Nebenbedingung | Sättigt |

<!--en:46b71a03-->
| Hinge-Verlust (:numref:`sec_gan_objectives`) | $\operatorname{sign}\log(p/q)$ | Klassifikation mit Margin | Ohne Nebenbedingung (oder mit Spektralnorm) | Sättigt (linear bis $2/\kappa$ bei einem $\kappa$-Lipschitz-stetigen Kritiker) |

<!--en:77c406fa-->
| MMD (:numref:`sec_gan_objectives`) | Mit einem Kernel geglättete Differenz $p - q$ (Zeugenfunktion) | Differenz der Mittelwerte | Einheitskugel im RKHS | Gradient ungleich null innerhalb der Kernel-Bandbreite $s$ |

<!--en:b5b7d8b0-->
| Wasserstein-1 (:numref:`sec_gan_objectives`) | Kantorovich-Potential | Differenz der Mittelwerte | Kugel der 1-Lipschitz-Funktionen (z. B. WGAN-GP) | Steigung eins bei jedem Abstand |

<!--en:6bbbfc5b-->
| RpGAN (:numref:`sec_gan_relativistic`) | $\log(p/q) + c$ | Paarweises Bradley-Terry-Ranking | Ohne Nebenbedingung | Sättigt |

<!--en:b5da4a4c-->
| R3GAN: RpGAN + $R_1 + R_2$ (:numref:`sec_gan_convergence`) | $\log(p/q)$, durch Tiefpassfilterung auf Längenskalen unterhalb von $2\sqrt{\gamma}$ geglättet | Paarweises Bradley-Terry-Ranking | Um null zentrierter Strafterm für den Eingabegradienten auf $p$ und $q$ | Glatte Interpolation über Lücken durch das regularisierte Netz |

<!--en:e6bc5681-->
Im Vergleich zu iterativen Diffusions- und Flow-Modellen (:numref:`chap_diffusion`) benötigen adversariale Generatoren nur einen Durchlauf für die Inferenz und liefern bei unterbestimmten inversen Problemen scharfe Ausgaben. Dem stehen vier grundsätzliche Schwierigkeiten gegenüber: (i) Die Likelihood lässt sich nicht explizit auswerten. (ii) Die gekoppelte Minimax-Dynamik der beiden Netze erfordert sorgfältig abgestimmte Gradientenstrafterme. (iii) Das Trainingsziel bestraft Modusausfall von sich aus nur schwach (:numref:`subsec_gan_mode_dropping`). (iv) Beim Training im großen Maßstab können Instabilitäten auftreten :cite:`Brock.Donahue.Simonyan.2019`. Im nächsten Kapitel (:numref:`chap_diffusion`) wenden wir uns Diffusions- und Flow-Matching-Modellen zu. Sie ersetzen das Minimax-Training durch iterative Entrauschung in mehreren Schritten.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:1481ad4d-->
Minimiert ein Prädiktor den quadratischen Fehler $\mathbb{E}[\|x - g(c)\|^2]$, nähert er sich dem bedingten Mittelwert $\mu(c) = \mathbb{E}[x \mid c]$ (:eqref:`eq_beyond_condmean`). Dieser mittelt über verschiedene Modi, wenn der Zielwert $x$ bei gegebenem $c$ mehrdeutig ist. Kombinieren wir den Prädiktor mit einem bedingten Diskriminator $D(x, c)$, der $\log(p(x, c)/q(x, c))$ schätzt, so bestraft dieser Vorhersagen in den Korridoren geringer Dichte zwischen den Modi. Er zwingt die Ausgaben auf die Datenmannigfaltigkeit. Die Ergebnisse wirken realistischer, allerdings steigt der MSE dabei auf bis zu das Doppelte (Einbuße von $3\text{ dB}$ beim PSNR) :cite:`Blau.Michaeli.2017`. Auch ein Diffusionsentrauscher berechnet den A-posteriori-Mittelwert $\mathbb{E}[x_0 \mid x_\tau]$. Destillieren wir einen Einschrittgenerator durch direkte Regression auf einen Entrauscher mit hohem Rauschpegel, zieht dies die Vorhersagen zum unscharfen Mittelwert, sofern kein adversarialer Kritiker mitwirkt. Aus demselben Grund bleiben adversariale Diskriminatoren wesentliche Bestandteile latenter Bild-Autoencoder, neuronaler Sprachvocoder und destillierter Diffusionsgeneratoren, die nur einen Schritt benötigen.

<!--en:2294243a-->
## Übungen

<!--en:83b06ae0-->
1. **Zielkonflikt zwischen Wahrnehmungsqualität und Verzerrung bei einem idealen Sampler.** Wir ziehen $\hat x \sim p(x \mid c)$ unabhängig von $x$ bei gegebenem $c$, sodass der Prädiktor die wahre bedingte Verteilung $p(x \mid c)$ exakt trifft :cite:`Blau.Michaeli.2017`.
    1. Beweisen Sie mit :eqref:`eq_beyond_condmean`, dass $\mathbb{E}\big[(x - \hat x)^2\big] = 2\, \mathbb{E}\big[(x - \mu(c))^2\big]$ gilt. Wie viele Dezibel verliert ein exakter Sampler beim PSNR ($10 \log_{10}(x_{\max}^2 / \text{MSE})$) gegenüber dem bedingten Mittelwert $\mu(c)$?
    1. Vergleichen Sie für die Verteilung mit zwei Datenbändern :eqref:`eq_beyond_twoband` den erwarteten quadratischen Fehler des exakten Samplers mit dem des bedingten Mittelwerts $\mu(c) = c$ ($1.0025$) und dem des Prädiktors für ein einzelnes Band $g(c) = c + 1$ ($2.0025$). Bei welchem der drei Verfahren kann ein optimaler Kritiker $D(x, c)$ die Ausgaben nicht von den echten Daten unterscheiden?
    1. Es seien $x \sim \mathcal{N}(0, 1)$ und $c = x + \tau\xi$ mit unabhängigem Rauschen $\xi \sim \mathcal{N}(0, 1)$ gegeben. Zeigen Sie, dass der deterministische lineare Prädiktor $\hat x = c / \sqrt{1 + \tau^2}$ dieselbe Randverteilung $\mathcal{N}(0, 1)$ wie $x$ hat. Berechnen Sie das Verhältnis seines MSE zum MSE des bedingten Mittelwerts $\mathbb{E}[x \mid c]$ für $\tau \in \{0.1, 1, 10\}$.
   :exlabel:`ex_adversarial-losses-perfect-sampler`

<!--en:b3aa28dc-->
1. [code] **Bedingter Mittelwert und bedingter Median ($L_2$ gegenüber $L_1$).** Die Minimierung des absoluten Fehlers $\mathbb{E}[|x - g(c)|]$ liefert einen bedingten Median $x_{\textrm{med}}(c)$ mit $P(x \leq x_{\textrm{med}}(c) \mid c) = \tfrac12$.
    1. Der Versatz $b$ in :eqref:`eq_beyond_twoband` nehme die Werte $(-1, 0, +1)$ mit den Wahrscheinlichkeiten $(\tfrac13, \tfrac13, \tfrac13)$ oder $(\tfrac14, \tfrac14, \tfrac12)$ an. Bestimmen Sie für beide Fälle den bedingten Mittelwert und den bedingten Median. Geben Sie jeweils an, ob sie auf einem Datenband liegen.
    1. Berechnen Sie für das Problem mit zwei Datenbändern und asymmetrischer A-priori-Wahrscheinlichkeit $P(b = +1) = r$ den bedingten Mittelwert als Funktion von $r$. Zeigen Sie, dass für $r = \tfrac34$ der bedingte Median auf dem oberen Band liegt. Beschreiben Sie den Verlauf des $L_1$-Risikos im Korridor $g(c) - c \in (-1, 1)$ für $r = \tfrac12$.
    1. Trainieren Sie eine Kopie von `g_mse` mit dem $L_1$-Verlust für die Fälle aus (a) und (b). Vergleichen Sie die erlernten Kurven mit Ihren analytischen Vorhersagen.
   :exlabel:`ex_adversarial-losses-mean-and-median`

<!--en:66b6e921-->
1. [code] **Erweiterung um eine binäre latente Variable.** ● Ergänzen Sie den Schüler aus :numref:`subsec_gan_beyond_capacity` um eine unabhängige, gleichverteilte binäre latente Variable $z \in \{-1, +1\}$, sodass $\hat x = g(c, z)$ gilt. Der Kritiker bewertet weiterhin Paare $(x, c)$.
    1. Trainieren Sie einen Schüler mit quadratischem Fehler und einen weiteren adversarial. Prüfen Sie, auf welche Bänder die Schüler $z = -1$ und $z = +1$ abbilden. Warum ignoriert der Schüler mit quadratischem Fehler die Variable $z$?
    1. Implicit Maximum Likelihood Estimation (IMLE, :citet:`Li.Malik.2018`) minimiert $\mathbb{E}_{(c, x)}[\min_{z \in \{-1, +1\}} (g(c, z) - x)^2]$ ohne Kritiker. Trainieren Sie einen Schüler mit diesem Verlust. Erklären Sie, warum das Minimum über $z$ für jeden Datenpunkt erzwingt, dass die beiden latenten Zustände unterschiedliche Bänder abdecken.
   :exlabel:`ex_adversarial-losses-latent-bit`

<!--en:8ff341bf-->
1. **Direct Discriminative Optimization (DDO).** DDO :cite:`Zheng.Chen.Chen.ea.2025` nutzt beim Fine-Tuning eines Likelihood-Modells $q_\theta$ eine eingefrorene Referenzkopie $q_{\textrm{ref}}$ als Gegenspieler. Das Verfahren parametrisiert den Kritiker direkt als logarithmiertes Likelihood-Verhältnis $D_\theta(x) = \log(q_\theta(x) / q_{\textrm{ref}}(x))$ und minimiert den logistischen Diskriminatorverlust $J(\theta) = -\mathbb{E}_{x \sim p}[\log \sigma(D_\theta(x))] - \mathbb{E}_{x \sim q_{\textrm{ref}}}[\log \sigma(-D_\theta(x))]$ ausgehend von $\theta = \theta_{\textrm{ref}}$.
    1. Zeigen Sie: Gilt $q_\theta = p$, so stimmt $D_\theta(x)$ mit dem optimalen Kritiker $D^\star(x)$ für $p$ und $q_{\textrm{ref}}$ überein.
    1. Zeigen Sie mithilfe von $\mathbb{E}_{x \sim q_\theta}[\nabla_\theta \log q_\theta(x)] = 0$, dass bei der Initialisierung ($\theta = \theta_{\textrm{ref}}$, also $D_\theta(x) = 0$) der Gradient $\nabla_\theta J(\theta)$ gleich dem $\tfrac12$-Fachen des Gradienten der negativen Log-Likelihood $-\nabla_\theta \mathbb{E}_{x \sim p}[\log q_\theta(x)]$ ist.
   :exlabel:`ex_adversarial-losses-ddo-plays-log-loss`

<!--en:0463b050-->
1. **Mehrdeutige Phase und Mittelung von Wellenformen bei Sprachvocodern.** Modellieren Sie einen einzelnen Audiozeitrahmen der Länge $N$ als Sinusschwingung $x_n = \sin(\omega n + \vartheta)$ für $n = 0, \ldots, N - 1$. Die Frequenz $\omega$ ist aus dem Spektrogramm bekannt, die Phase $\vartheta \sim \textrm{Uniform}[-a, a]$ ($0 < a \leq \pi$) bleibt unbeobachtet.
    1. Zeigen Sie, dass bei der bedingten mittleren Wellenform $\mathbb{E}_\vartheta[x_n] = \frac{\sin a}{a}\sin(\omega n)$ die Amplitude um den Faktor $\frac{\sin a}{a}$ abgeschwächt ist (für $a = \pi$ entsteht Stille).
    1. Zeigen Sie für $\omega = 2\pi k / N$ ($0 < k < N/2$), dass die Beträge $|\hat x_m|$ der diskreten Fourier-Transformation unabhängig von $\vartheta$ sind.
    1. Erzeugt man zwei benachbarte Zeitrahmen mit unabhängigen Phasen $\vartheta_1 \neq \vartheta_2$, so ist der Verlust für die DFT-Beträge in jedem einzelnen Zeitrahmen null. Welches Artefakt entsteht an der Rahmengrenze, und warum erkennt es ein Wellenformdiskriminator im Zeitbereich?
   :exlabel:`ex_adversarial-losses-phase`

<!--en:58234517-->
[Diskussion](https://d2l.discourse.group/t/36814)

<!--en:17e6e862-->
Adversariale Verluste jenseits eigenständiger GANs<br>**Unschärfe durch bedingte Mittelung · Zielkonflikt zwischen Wahrnehmungsqualität und Verzerrung · Diffusionsdestillation (ADD und DMD2) · Patch-Kritiker in VQGAN · Sprachvocoder in HiFi-GAN**

<!--en:6090b2ea-->
::: {.slide title="Warum pixelweise Regression bei multimodalen Zielen unscharfe Bilder liefert"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:4cb40cad-->
- **Warum punktweise Verluste zu stark mitteln:** Gibt es mehrere plausible Ausgaben $x$ pro Bedingung $c$ (Superauflösung, Inpainting, Spektrogramminversion), erzwingt der pixelweise $L_2$- oder $L_1$-Verlust beim Prädiktor $g(c)$ den **bedingten Mittelwert** bzw. Median. Das unscharfe Ergebnis entspricht keinem der Modi.

<!--en:bf221b7d-->
- **Wo adversariale Kritiker heute unverzichtbar sind:** (1) Einschrittdestillation von Diffusionsmodellen (ADD / SDXL-Turbo, DMD2), (2) latente Bild-Autoencoder (VQGAN, Stable Diffusion VAE) und (3) neuronale Sprachvocoder (HiFi-GAN).

<!--en:06099f1c-->
::: {.slide title="Orthogonale Zerlegung mithilfe des bedingten Mittelwerts"}

<!--en:b0b2c77d-->
[Warum L₂ bei mehrdeutigen Zielen mittelt]{.kicker}

<!--en:d2dd9303-->
- **Eindeutiger $L_2$-Minimierer:** Durch Addition und Subtraktion von $\mu(c) = \mathbb{E}[x \mid c]$ verschwindet der Kreuzterm nach dem Satz vom iterierten Erwartungswert. Damit gilt für jedes Pixel $g^\star(c) = \mu(c)$.

<!--en:450633a9-->
- **Lücke zwischen den Modi:** Ist $p(x \mid c)$ bimodal und symmetrisch, liegen sowohl der bedingte Mittelwert $\mu(c)$ ($L_2$) als auch der zentrale bedingte Median ($L_1$) im Korridor geringer Dichte zwischen den Modi.

<!--en:a1298380-->
::: {.slide title="Der bimodale Benchmark mit zwei Datenbändern"}

<!--en:ee6baa3f-->
[Ein minimaler analytischer Benchmark]{.kicker}

<!--en:9fe79ca6-->
- **Bedingter Mittelwert $\mu(c) = c$ (leerer Korridor):** Sein Abstand zu den Bandzentren beträgt $20\sigma$. Dennoch erreicht er den kleinstmöglichen MSE von **$1.0025$**.

<!--en:233bdf6d-->
- **Prädiktor auf einem Band $g(c) = c + 1$:** Liefert stets gültige Daten, sein MSE beträgt jedoch **$2.0025$** ($2\times$ so groß).

<!--en:c4dcf445-->
- **Adversarialer Kritiker $D(x, c)$:** Bestraft den leeren Korridor ($D^\star = \log(p/q) \ll 0$) und drängt Vorhersagen auf ein gültiges Band.

<!--en:4d33cce4-->
::: {.slide title="Training mit quadratischem Fehler und adversariales Training" only="pytorch"}

<!--en:ab269957-->
[Kontrollierter Vergleich im Code]{.kicker}

<!--en:1b7a5132-->
Zwei identische MLPs mit 2 versteckten Schichten und $\tanh$-Aktivierung (`g_mse` und `g_adv`, $1 \to 32 \to 32 \to 1$) beginnen das Training mit denselben Gewichten. `g_adv` addiert das Beobachtungsrauschen $0.05\,\xi'$ der Daten und trainiert gegen einen Kritiker $D(x, c)$ mit Leaky ReLU. Dazu dienen `gan_loss_D` und `gan_loss_G`:

<!--en:54069185-->
::: {.slide title="Training mit quadratischem Fehler und adversariales Training" only="jax"}

<!--en:ab269957-2-->
[Kontrollierter Vergleich im Code]{.kicker}

<!--en:1b7a5132-2-->
Zwei identische MLPs mit 2 versteckten Schichten und $\tanh$-Aktivierung (`g_mse` und `g_adv`, $1 \to 32 \to 32 \to 1$) beginnen das Training mit denselben Gewichten. `g_adv` addiert das Beobachtungsrauschen $0.05\,\xi'$ der Daten und trainiert gegen einen Kritiker $D(x, c)$ mit Leaky ReLU. Dazu dienen `gan_loss_D` und `gan_loss_G`:

<!--en:dd8e650d-->
::: {.slide title="Training mit quadratischem Fehler und adversariales Training (Forts.)" only="jax"}

<!--en:ab269957-3-->
[Kontrollierter Vergleich im Code]{.kicker}

<!--en:22c8e2db-->
`g_adv` trainiert hingegen gegen den Kritiker, mit je einem `gan_loss_D`- und einem `gan_loss_G`-Schritt pro Batch:

<!--en:fbd0cff8-->
::: {.slide title="Ergebnisse für zwei Bänder: Korridor oder Festlegung auf einen Modus"}

<!--en:63718f6d-->
[Empirische Überprüfung]{.kicker}

<!--en:14420d0a-->
Die Auswertung beider trainierter Prädiktoren für $201$ Eingaben $c \in [-1, 1]$ bestätigt die analytische Vorhersage:

<!--en:3b845f42-->
- **Schüler mit quadratischem Fehler (`g_mse`):** Folgt dem bedingten Mittelwert $x = c$ durch den leeren Korridor (etwa eine Einheit Abstand zu jedem Band, $100\%$ der Vorhersagen außerhalb der Bänder).

<!--en:0a78c00c-->
- **Adversarial trainierter Schüler (`g_adv`):** Legt sich jeweils auf ein gültiges Datenband fest (im Mittel wenige Hundertstel Abstand zum näheren Band). Höchstens $2\%$ der Punkte liegen außerhalb der Bänder, wo die stetige Kurve den Zweig wechselt.

<!--en:8539f679-->
::: {.slide title="Der Zielkonflikt zwischen Wahrnehmungsqualität und Verzerrung"}

<!--en:01e6b853-->
[Satz von Blau und Michaeli]{.kicker}

<!--en:a85af4e7-->
- **Exakter Faktor 2 beim MSE:** Für einen idealen stochastischen Sampler $\hat{x} \sim p(x \mid c)$, der die bedingte Datenverteilung exakt trifft ($q = p$), ist der erwartete quadratische Fehler **doppelt** so groß wie für den bedingten Mittelwert $\mu(c)$ ($2.0050$ gegenüber $1.0025$).

<!--en:602c76fe-->
- **3 dB Einbuße beim PSNR (Blau & Michaeli, 2018):** Vollkommen realistische Bilder kosten gegenüber dem unscharfen bedingten Mittelwert zwangsläufig bis zu $10\log_{10} 2 \approx 3.01\text{ dB}$ beim PSNR.

<!--en:f1f165ff-->
::: {.slide title="Mehrdeutige Ziele und begrenzte Modellkapazität"}

<!--en:f59cc544-->
[Warum eine latente Variable nötig ist]{.kicker}

<!--en:b3045020-->
- **1. Mehrdeutiges Ziel (aleatorische Unsicherheit):** Da $c$ null Bit Information über das Vorzeichen des Bands $b \in \{-1, +1\}$ enthält, kann *keine* deterministische Funktion $g(c)$ bei demselben $c$ beide Bänder abdecken, unabhängig von ihrer Breite oder Tiefe.

<!--en:201aae48-->
- **2. Begrenzte Modellkapazität:** Selbst wenn eine deterministische Abbildung existiert (etwa durch eine Diffusions-ODE), mittelt ein kleines 1-Schritt-Netz bei der $L_2$-Regression über abrupte Sprünge hinweg.

<!--en:c75a42b1-->
- **Bandwahl über eine Zufallsvariable:** Erhält der Schüler $z \sim \mathcal{N}(0, 1)$, kann $\hat{x} = g(c, z)$ die Fälle $z > 0$ und $z < 0$ gleichmäßig auf die beiden Datenbänder verteilen.

<!--en:949d1c4f-->
::: {.slide title="Diffusionsentrauscher und Einschrittdestillation"}

<!--en:1df56169-->
[Warum Lehrer bei hohem Rauschpegel unscharfe Bilder liefern]{.kicker}

<!--en:c4d150e9-->
- **Niedriger ($\tau \ll 1$) und hoher Rauschpegel ($\tau \gg 1$):** Bei wenig Rauschen legt $x_\tau = x_0 + \tau\xi$ den Wert $x_0$ nahezu fest, sodass $\mathbb{E}[x_0 \mid x_\tau]$ scharf ist. Bei viel Rauschen passen viele rauschfreie Bilder zu $x_\tau$, sodass $\mathbb{E}[x_0 \mid x_\tau]$ ein unscharfer Mittelwert ist.

<!--en:266be1fa-->
- **Gefahr bei der Einschrittdestillation:** Die direkte Regression eines 1-Schritt-Schülers $\hat{x} = g(c, z)$ auf die Schätzung $\mathbb{E}[x_0 \mid \hat{x} + \tau\xi]$ eines eingefrorenen Lehrers mit hohem Rauschpegel zieht den Schüler zum bedingten Mittelwert.

<!--en:feb446c3-->
::: {.slide title="Ein entrauschender Lehrer in geschlossener Form für die beiden Bänder"}

<!--en:f3d2431d-->
[Analytisches Modell der Destillation]{.kicker}

<!--en:e29d8ef3-->
- **Exakter A-posteriori-Schätzer als Lehrer:** Nach der Bayes-Regel für $\tilde{x} = x + \tau\xi$ verwendet der optimale Entrauscher die Funktion $\tanh$ mit maximaler Steigung $(\tau^2 + 0.05^2)^{-1}$.

<!--en:672be39e-->
- **Kontraktion in den leeren Korridor bei $\tau = 2$:** Überall gilt für die Steigung $\le (4.0025)^{-1} \approx 0.25 < 1$. Daher führt die Iteration der Fixpunktgleichung $u = \mathbb{E}_\xi[\tanh(\frac{u + \sqrt{\tau^2 + 0.05^2}\xi}{\tau^2 + 0.05^2})]$ für jeden latenten Wert $z$ zum einzigen Fixpunkt $u^\star = 0$ (im leeren Korridor).

<!--en:c430154b-->
::: {.slide title="Implementierung der Einschrittdestillation mit Kritiker" only="pytorch"}

<!--en:29b545ed-->
[Experiment im Code]{.kicker}

<!--en:c3a96431-->
`train_student` trainiert einen stochastischen Schüler $g(c, z)$ ($z \sim \mathcal{N}(0, 1)$). Dazu dient entweder der Regressionsverlust gegenüber dem eingefrorenen entrauschenden Lehrer ($\tau = 2.0$), ein adversarialer Kritiker $D(x, c)$ oder beides zusammen (`critic + regression`):

<!--en:b59b0c70-->
::: {.slide title="Implementierung der Einschrittdestillation mit Kritiker (Forts.)" only="pytorch"}

<!--en:29b545ed-2-->
[Experiment im Code]{.kicker}

<!--en:d9e5a56b-->
Jeder Schritt aktualisiert zuerst den Kritiker, sofern vorhanden. Danach passt er den Schüler an die Bewertung des Kritikers und den festen Schätzwert des Lehrers an:

<!--en:1d14a85a-->
::: {.slide title="Implementierung der Einschrittdestillation mit Kritiker" only="jax"}

<!--en:29b545ed-3-->
[Experiment im Code]{.kicker}

<!--en:c3a96431-2-->
`train_student` trainiert einen stochastischen Schüler $g(c, z)$ ($z \sim \mathcal{N}(0, 1)$). Dazu dient entweder der Regressionsverlust gegenüber dem eingefrorenen entrauschenden Lehrer ($\tau = 2.0$), ein adversarialer Kritiker $D(x, c)$ oder beides zusammen (`critic + regression`):

<!--en:4f28f33a-->
::: {.slide title="Implementierung der Einschrittdestillation mit Kritiker (Forts.)" only="jax"}

<!--en:29b545ed-4-->
[Experiment im Code]{.kicker}

<!--en:020c80e4-->
Der Verlust des Schülers addiert zum Term des Kritikers das $\lambda$-Fache des quadratischen Fehlers gegenüber dem festen Schätzwert des Lehrers:

<!--en:584ea3e5-->
::: {.slide title="Ergebnisse der Destillation: Mit Kritiker sind beide Bänder wieder abgedeckt"}

<!--en:e0a190f8-->
[Vergleich mit drei Zufallsstartwerten]{.kicker}

<!--en:843bf78e-->
Alle drei Schülervarianten durchlaufen den Test für $201$ Eingaben $c$ und $64$ Quantile der latenten Variablen $z$. Die Ergebnisse für drei Zufallsstartwerte bestätigen unsere Fixpunktanalyse:

<!--en:a45a6e16-->
- **1. Nur Regression auf den Lehrer:** $100\%$ der Ausgaben fallen in den leeren Korridor (etwa eine Einheit Abstand zu jedem Band). Der Schüler ignoriert $z$.

<!--en:85381cda-->
- **2. Nur Kritiker und Kritiker + Regression:** Bei beiden Varianten liegen **$98\text{--}99\%$** der Ausgaben direkt auf den Datenbändern (wenige Hundertstel Abstand). Die Wahrscheinlichkeitsmasse verteilt sich gleichmäßig ($\sim 49\% / 49\%$) über $z$ auf das obere und das untere Band. Bei der geringen Bandbreite $\sigma = 0.05$ überwiegt die starke Rückstellkraft des Kritikers ($\sim \delta / 0.05^2 = 400\delta$) den schwachen Zug des Lehrers mit hohem Rauschpegel zur Mitte ($\delta \approx 0.008$).

<!--en:afb0ad76-->
::: {.slide title="Adversarial Diffusion Distillation (ADD und DMD2)"}

<!--en:687b319b-->
[Destillation von Basismodellen auf einen Schritt]{.kicker}

<!--en:931595da-->
- **ADD / SDXL-Turbo (Sauer et al., 2023):** Kombiniert die Regression mit quadratischem Fehler auf den entrauschten Schätzwert eines eingefrorenen Diffusionslehrers mit einem bedingten Diskriminator auf Basis eingefrorener **DINOv2**-Bildmerkmale.

<!--en:8b8239cf-->
- **DMD und DMD2 (Yin et al., 2024):** DMD bildet die Differenz zwischen dem Score $\nabla_x \log p_\tau$ des eingefrorenen Lehrers und dem Score $\nabla_x \log q_\tau$ für generierte Daten. So bewegt es die Samples des Schülers entlang von $\nabla_x \log(p_\tau/q_\tau)$, dem exakten Eingabegradienten eines optimalen Kritikers mit logarithmischem Verlust. DMD2 ergänzt einen GAN-Kritiker für echte Bilder, um Modusausfall zu verhindern.

<!--en:db65c1d0-->
::: {.slide title="Latente Bild-Tokenizer (VQGAN) und PatchGAN"}

<!--en:8ca7697f-->
[Autoencoder für latente Diffusion]{.kicker}

<!--en:9edad0bb-->
- **8× räumliche Kompression im latenten Raum:** Latente Diffusionsmodelle (Stable Diffusion, Flux) komprimieren Bilder der Größe $H \times W \times 3$ mit einem Autoencoder zu einem kompakten latenten Gitter (VQGAN oder KL-VAE nach Esser et al., 2021, bzw. Rombach et al., 2022).

<!--en:c36c9bf1-->
- **Drei einander ergänzende Verluste für den Decoder:**
  1. **Pixelweiser $L_1$-/$L_2$-Verlust** erhält niederfrequente Farbverläufe und die räumliche Anordnung.
  2. **Der wahrnehmungsbasierte LPIPS-Verlust** gleicht die Merkmalsstatistiken mittlerer VGG-Schichten ab.
  3. **Der PatchGAN-Kritiker** bewertet lokale Bildausschnitte der Größe $N \times N$ und sorgt für scharfe, hochfrequente Texturen.

<!--en:7218a351-->
::: {.slide title="Sprachvocoder (HiFi-GAN) und Mehrdeutigkeit der Phase"}

<!--en:d4eded27-->
[Warum die Regression der Wellenform zu Stille führt]{.kicker}

<!--en:23e647dd-->
- **Warum die $L_2$-Regression der Wellenform im Zeitbereich scheitert:** Ein Mel-Spektrogramm enthält die Beträge der Frequenzkomponenten bei gegebener Frequenz $\omega$, verwirft aber die Phase $\vartheta \sim \textrm{Uniform}[-\pi, \pi]$ ($a = \pi$). Die bedingte mittlere Wellenform ergibt daher **völlige Stille** ($\frac{\sin\pi}{\pi} = 0$).

<!--en:27d20fce-->
- **HiFi-GAN (Kong et al., 2020):** Kombiniert den $L_1$-Verlust für das Mel-Spektrogramm mit Wellenformkritikern für mehrere Perioden und Skalen im Zeitbereich. So bleiben die Phasen über Zeitrahmen hinweg kohärent.

<!--en:ae8c7044-->
::: {.slide title="Die adversarialen Spiele aus §{num:chap_gans} im Überblick"}

<!--en:cb8a9aab-->
[Vergleich der sechs Formulierungen]{.kicker}

<!--en:6de1bfa1-->
| Adversariale Formulierung | Schätzwert des optimalen Kritikers | Bewertungsregel | Nebenbedingung / Strafterm für den Kritiker | Gradient bei disjunkten Trägern (bei $D^\star$) |

<!--en:df74f260-->
| **Logistischer Verlust / $f$-GAN (§{sec_basic_gan}–§{num:sec_gan_objectives})** | $\log(p/q)$ oder $f'(p/q)$ | Klassifikation einzelner Samples | Ohne Nebenbedingung | Sättigt ($\mathrm{JS} = \log 2$) |

<!--en:d4917dfe-->
| **Hinge-Verlust (§{sec_gan_objectives})** | $\operatorname{sign}(\log(p/q))$ | Klassifikation mit Margin | Ohne Nebenbedingung oder mit Spektralnorm | Sättigt (linear bis $2/\kappa$ bei $\kappa$-Lipschitz-Stetigkeit) |

<!--en:0f68cdd7-->
| **MMD (§{sec_gan_objectives})** | Mit einem Kernel geglättete Differenz $p - q$ | Differenz der Mittelwerte | Einheitskugel im RKHS (kein Kritikernetz) | Ungleich null innerhalb der Kernel-Bandbreite $s$ |

<!--en:bab1d83b-->
| **Wasserstein-1 (§{sec_gan_objectives})** | Kantorovich-Potential | Differenz der Mittelwerte | Kugel der 1-Lipschitz-Funktionen (WGAN-GP) | Konstante Steigung eins bei jedem Abstand |

<!--en:4b844a0e-->
| **RpGAN (§{sec_gan_relativistic})** | $\log(p/q) + c$ | Paarweises Bradley-Terry-Ranking | Ohne Nebenbedingung | Sättigt ($d_{\mathrm{Rp}} = \log 2$) |

<!--en:59df14aa-->
| **R3GAN (§{sec_gan_convergence}–§{num:sec_gan_conditional})** | $\log(p/q)$, gefiltert oberhalb der Grenzfrequenz zur Längenskala $2\sqrt{\gamma}$ | Paarweises Bradley-Terry-Ranking | Um null zentrierter Strafterm $R_1 + R_2$ auf $p, q$ | Glatte Interpolation durch gedämpften Kritiker |

<!--en:e7534d5d-->
::: {.slide title="Verständnisfragen zu Unschärfe, Destillation und Phase"}

<!--en:929b7671-->
[Wissenskontrolle]{.kicker}

<!--en:2456dca4-->
Fragen zur Unschärfe durch bedingte Mittelung und zur Rolle adversarialer Verluste in modernen generativen Pipelines:

<!--en:bb6e08c1-->
1. **PSNR-Einbuße bei exaktem Sampling:** Für einen idealen Sampler gilt die Identität $\mathbb{E}[\|x - \hat{x}\|^2] = 2\,\mathbb{E}[\|x - \mu(c)\|^2]$ von Blau und Michaeli mit $\hat{x} \sim p(x \mid c)$. Wie viele Dezibel verliert ein vollkommen realistischer Generator beim Spitzen-Signal-Rausch-Verhältnis ($\mathrm{PSNR} = 10\log_{10}(x_{\max}^2 / \mathrm{MSE})$) gegenüber dem unscharfen bedingten Mittelwert $\mu(c)$?

<!--en:dcc5edb1-->
2. **Teilweise unbekannte Phase bei Vocodern:** Das Modell beschreibt einen Audiozeitrahmen durch die Sinusschwingung $x_n = \sin(\omega n + \vartheta)$ mit der Phase $\vartheta \sim \textrm{Uniform}[-\pi/2, \pi/2]$. Auf welchen Bruchteil der wahren Amplitude sinkt das Signal bei der $L_2$-Regression der Wellenform? Welches Artefakt entsteht, wenn ein Vocoder ohne Wellenformkritiker im Zeitbereich nur den Fehler der DFT-Beträge pro Zeitrahmen minimiert?

<!--en:557b98b6-->
**Lösungen:**

<!--en:b8ddbd7a-->
1. Verdoppelt sich der MSE, so sinkt $\mathrm{PSNR} = 10\log_{10}(x_{\max}^2 / \mathrm{MSE})$ um $10\log_{10}(2) \approx \mathbf{3.01\text{ dB}}$.

<!--en:23bd1023-->
2. Bei $a = \pi / 2$ sinkt die Amplitude auf den Faktor $\frac{\sin(\pi/2)}{\pi/2} = \mathbf{\frac{2}{\pi} \approx 0.637}$ ihres ursprünglichen Werts. Minimiert der Vocoder nur den Fehler der DFT-Beträge pro Zeitrahmen, bleibt die Phase $\vartheta_k$ jedes Zeitrahmens unabhängig. Die **Phasensprünge erzeugen Knackgeräusche** an jeder Rahmengrenze, die ein Wellenformdiskriminator im Zeitbereich sofort bestraft.

<!--en:ef20a051-->
::: {.slide title="Verständnisfragen zu Unschärfe, Destillation und Phase (Forts.)"}

<!--en:929b7671-2-->
[Wissenskontrolle]{.kicker}

<!--en:ac121b30-->
3. **Wahl des Trainingsverfahrens:** Welche dieser Aufgaben nutzen heute einen adversarialen Kritiker? (a) Pre-Training eines allgemeinen Text-zu-Bild-Basismodells von Grund auf, (b) Destillation eines Diffusionsmodells mit 50 Schritten zu einem Generator mit 1 Schritt oder (c) Training des Decoders eines Bild-Tokenizers für latente Diffusion?

<!--en:23db34b8-->
**Lösung:**

<!--en:e49cd810-->
3. **(a)** nutzt mehrstufige Diffusions- oder Flow-Regression für eine breite Abdeckung und stabile Skalierung. **(b)** Die Einschrittdestillation von Diffusionsmodellen (ADD/DMD2) und **(c)** die Decoder latenter Autoencoder (VQGAN/SD-VAE) kombinieren hingegen Regressions- oder wahrnehmungsbasierte Verluste mit einem **adversarialen Kritiker**, um Unschärfe durch bedingte Mittelung zu vermeiden.

<!--en:ef74c836-->
::: {.slide title="Zusammenfassung: Adversariale Verluste jenseits von GANs"}

<!--en:c5a222f8-->
[Kernaussagen]{.kicker}

<!--en:1c43dba2-->
| Anwendung / Konzept | Zentrale mathematische Identität | Warum Regression allein scheitert | Rolle des adversarialen Kritikers |

<!--en:116309ae-->
| **Zerlegung mithilfe des bedingten Mittelwerts** | $\mathbb{E}[(x - g(c))^2] = \mathrm{Var}(x \mid c)$ $+ \mathbb{E}[(\mu(c) - g(c))^2]$ | Der eindeutige $L_2$-Minimierer $\mu(c) = \mathbb{E}[x \mid c]$ mittelt über verschiedene Modi | Bestraft Korridore geringer Dichte ($D^\star = \log\frac{p}{q} \ll 0$) und erzwingt die Festlegung auf einen Modus |

<!--en:7e9007a9-->
| **Zielkonflikt zwischen Wahrnehmungsqualität und Verzerrung** | $\mathrm{MSE}(\hat{x} \sim p(\cdot \mid c)) = 2\,\mathrm{MSE}(\mu(c))$ | Die Optimierung von PSNR/MSE ($-3\text{ dB}$ für exakte Samples) begünstigt unscharfe Mittelwerte | Nimmt punktweise Verzerrung für eine realistische Verteilung in Kauf ($q(x \mid c) = p(x \mid c)$) |

<!--en:e4173a6e-->
| **Einschrittdestillation von Diffusionsmodellen** | $D_\tau^\star(x_\tau) = \mathbb{E}[x_0 \mid x_\tau]$ (A-posteriori-Mittelwert des Lehrers) | Bei hohem Rauschpegel ziehen die Zielwerte des Lehrers den Einschrittschüler zum bedingten Mittelwert | ADD und DMD2 kombinieren die Regression auf Lehrer oder Score mit einem merkmalsbasierten Kritiker für echte Bilder |

<!--en:7be03784-->
| **Bild-Tokenizer (VQGAN / VAE)** | $\mathcal{L}_{L_1} + \lambda_{\textrm{perc}}\mathcal{L}_{\textrm{LPIPS}}$ $+ \lambda_{\textrm{adv}}\mathcal{L}_{\textrm{PatchGAN}}$ | Der $8\times$-Flaschenhals im latenten Raum lässt feine lokale Texturen unter $L_1$/LPIPS mehrdeutig | Der PatchGAN-Kritiker erzwingt scharfe, hochfrequente Texturen in lokalen Bildausschnitten |

<!--en:8bc862a1-->
| **Sprachvocoder (HiFi-GAN)** | $\mathbb{E}_\vartheta[\sin(\omega n + \vartheta)] = \frac{\sin a}{a}\sin(\omega n)$ | Bei unbekannter Phase ($a = \pi$) führen $L_2$-Vorhersagen der Wellenform zu Stille ($0$) | Wellenformkritiker für mehrere Perioden und Skalen erzwingen die Stetigkeit von Amplitude und Phase |
