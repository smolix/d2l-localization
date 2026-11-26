<!--en:4c69064d-->
# Generative adversariale Netze

<!--en:ff27bff3-->
Angenommen, wir ziehen einen 100-dimensionalen Zufallsvektor $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ und geben ihn in ein neuronales Netz $G(\mathbf{z})$ ein, um ein Bild zu erzeugen. Ein solches Generatornetz lässt sich in wenigen Zeilen Code schreiben. Die Schwierigkeit liegt im Training. Klassifikatoren und autoregressive Sprachmodelle maximieren die Log-Likelihood der beobachteten Labels oder Token. Ein direkter Generator definiert dagegen eine implizite Verteilung über Bilder, ohne eine explizite Dichte $q(\mathbf{x})$ anzugeben, die wir maximieren könnten. Wie trainieren wir ein generatives Modell, aus dem wir Samples ziehen können, dessen Likelihood wir aber nicht auswerten können?

<!--en:9cc971e5-->
Wir haben bereits gesehen, wie ein Klassifikator zwei Verteilungen allein anhand von Samples vergleichen kann. Zur Korrektur der Kovariatenverschiebung in :numref:`subsec_covariate-shift-correction` haben wir einen binären Klassifikator trainiert, der Trainingseingaben von Testeingaben unterscheidet. Sein optimaler Logit ergab den Logarithmus des Dichteverhältnisses der beiden Verteilungen. :citet:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` machten aus diesem Vergleich zweier Stichproben ein Trainingsprinzip für generative Modelle. In einem *generativen adversarialen Netz* (engl. generative adversarial network, GAN) lernt ein *Diskriminator* (oder *Kritiker*) $D$, echte Daten von erzeugten Samples zu unterscheiden. Zugleich lernt der *Generator* $G$, Samples zu erzeugen, die der Diskriminator nicht von echten Daten unterscheiden kann. An den Gradienten des Diskriminators erkennt der Generator, wo seine Samples im Vergleich zur Datenverteilung zu häufig oder zu selten vorkommen. Dazu müssen wir keine der beiden Dichten explizit angeben.

<!--en:3a0ca300-->
Diffusionsmodelle und Flow-Matching-Modelle (:numref:`chap_diffusion`) haben eigenständige GANs bei der Bilderzeugung für allgemeine Zwecke weitgehend abgelöst. Dennoch bleiben adversariale Zielfunktionen ein zentraler Bestandteil moderner generativer Systeme. Mit ihnen trainiert man die latenten Autoencoder, auf denen Modelle für latente Diffusion beruhen, ebenso wie neuronale Audio-Vocoder und Einschrittgeneratoren, die durch Destillation aus mehrstufigen Diffusionsmodellen entstehen.

<!--en:97299268-->
In diesem Kapitel entwickeln wir die adversariale Modellierung in drei Schritten. Wir beginnen mit Zielfunktionen und Divergenzen. In :numref:`sec_basic_gan` leiten wir den optimalen Diskriminator für den logarithmischen Verlust, die zugehörige Jensen-Shannon-Divergenz und den nicht sättigenden Generatorverlust her. In :numref:`sec_gan_objectives` verallgemeinern wir diesen Ansatz auf $f$-Divergenzen und integrale Wahrscheinlichkeitsmetriken (Maximum Mean Discrepancy und Wasserstein-1-Distanz). In :numref:`sec_gan_relativistic` führen wir relativistische Kritiker für paarweise Vergleiche (RpGAN) auf Grundlage des Bradley-Terry-Rankings ein. Anschließend untersuchen wir in :numref:`sec_gan_convergence` die Minimax-Trainingsdynamik am Dirac-GAN. Wir zeigen, warum unregularisierte Updates auf Kreisbahnen oder spiralförmig nach außen führen und wie um null zentrierte Gradientenstrafterme ($R_1$ und $R_2$) das Trainingsverfahren von R3GAN stabilisieren :cite:`Huang.Gokaslan.Kuleshov.ea.2024`. In :numref:`sec_dcgan` und :numref:`sec_gan_conditional` skalieren wir konvolutionäre und durch Projektion konditionierte Architekturen für die Bilderzeugung. Zur Bewertung verwenden wir Metriken im Merkmalsraum (FID, KID sowie $k$-NN-Präzision und -Trefferquote). Schließlich erklärt :numref:`sec_gan_beyond`, wie adversariale Kritiker bei der Diffusionsdestillation, bei Bildtokenisierern und bei Sprach-Vocodern die Unschärfe verhindern, die durch den bedingten Mittelwert entsteht.

<!--en:5f1d3268-->
## Notation {.unnumbered}

<!--en:12d973c1-->
:numref:`tab_gan_notation` fasst die Notation dieses Kapitels zusammen. Wie in :numref:`chap_mdl-information-theory` messen wir Divergenzen und Entropien in Nat. Dabei bezeichnet $D_{\textrm{KL}}$ die Kullback-Leibler-Divergenz, $\mathrm{JS}$ die Jensen-Shannon-Divergenz, $H$ die differentielle oder diskrete Entropie und $H_b(t)$ die binäre Entropie einer Bernoulli-Verteilung mit Parameter $t$.

<!--en:bdf56af1-->
:Notation in diesem Kapitel.

<!--en:e31ebed0-->
| Symbol | Bedeutung | Erste Verwendung | Übliche Alternative |

<!--en:800c3ddf-->
| $p$, $q$, $m = \tfrac12(p + q)$ | Dichten der Datenverteilung, der Generatorverteilung und ihrer gleichgewichteten Mischung | :numref:`sec_basic_gan` | $p_{\textrm{data}}$, $p_g$ |

<!--en:3f5823cd-->
| $x \sim p$, $x' \sim q$; $y \in \{0, 1\}$ | echtes Sample, erzeugtes Sample und Label für die Herkunft ($1$ = echt) | :numref:`sec_basic_gan` | |

<!--en:a1a1f280-->
| $z \in \mathbb{R}^{d_z}$, $G$ | latenter Vektor und Generatornetz ($x' = G(z)$) | :numref:`sec_basic_gan` | |

<!--en:86426113-->
| $D$; $V(D)$ | Logit des Kritikers bzw. Diskriminators für die Klasse „echt“; Minimax-Wertfunktion | :numref:`sec_basic_gan` | |

<!--en:4ad45273-->
| $\rho = p/q$, $\lambda = \log(p/q)$ | Dichteverhältnis und logarithmiertes Dichteverhältnis | :numref:`sec_basic_gan` | |

<!--en:b3a5ee6c-->
| $w_{\textrm{sat}}$, $w_{\textrm{ns}}$, $w_{\textrm{Rp}}$ | Gewichte des Generatorgradienten je Sample (sättigend, nicht sättigend, relativistisch) | :numref:`sec_basic_gan`, :numref:`sec_gan_relativistic` | |

<!--en:36ed181b-->
| $\eta = p/(p + q)$ | A-posteriori-Wahrscheinlichkeit der Klasse nach Bayes $P(y = 1 \mid x)$ | :numref:`sec_gan_objectives` | |

<!--en:6dccf4fe-->
| $\ell$, $L(\eta)$, $\Delta_\ell(p, q)$ | Auszahlung in Abhängigkeit von der Margin, bedingtes Bayes-Risiko und Differenz der Bayes-Risiken | :numref:`sec_gan_objectives` | |

<!--en:c0a0882c-->
| $D_f(p \,\|\, q)$; $k(x, x')$ | $f$-Divergenz mit konvexer Funktion $f$; positiv definiter Kern | :numref:`sec_gan_objectives` | |

<!--en:a28ded29-->
| $\Phi(D)$, $d_{\mathrm{Rp}}(p, q)$ | relativistische Zielfunktion und Divergenz für paarweise Vergleiche | :numref:`sec_gan_relativistic` | |

<!--en:c7f6845e-->
| $R_1$, $R_2$; $\gamma$ | um null zentrierte Gradientenstrafterme an echten und erzeugten Samples; Gewicht der Strafterme | :numref:`sec_gan_convergence` | |

<!--en:27a93164-->
| $c$, $e_c$, $\phi(x)$ | Konditionierungsvariable, Einbettung der Bedingung und Merkmalsdarstellung | :numref:`sec_dcgan`, :numref:`sec_gan_conditional` | |

<!--en:61f230be-->
Generative adversariale Netze<br>**Implizite Dichteschätzung · Minimax-Spiele mit zwei Stichproben · $f$-Divergenzen und IPMs · Rotationsdynamik und Gradientenstrafterme · bedingte Generierung und perzeptuelle Kritiker**

<!--en:84462dec-->
::: {.slide title="Training impliziter Generatoren ohne Likelihood"}

<!--en:86e68ef5-->
[Das Grundproblem]{.kicker}

<!--en:42ff4ccf-->
Ein **impliziter Generator** zieht einen latenten Vektor $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_z})$ ($d_z = 100$) und erzeugt in einem einzigen Vorwärtsdurchlauf ein Bild $\mathbf{x}' = G(\mathbf{z}) \in \mathbb{R}^{d_x}$. Damit ist ein Sampling-Verfahren gegeben, aber keine praktisch berechenbare Dichte $q(\mathbf{x})$, die wir maximieren könnten.

<!--en:65e2309c-->
- **Niedrigdimensionale Mannigfaltigkeit als Träger:** Bilden wir $100$ latente Koordinaten auf $784$ Pixel (Fashion-MNIST) oder $12{,}288$ Pixel ($64 \times 64$ RGB) ab, konzentrieren sich die erzeugten Samples auf einer dünnen Fläche. Ihre Verteilung hat keine Lebesgue-Dichte.

<!--en:885abde9-->
- **Warum die Likelihood täuschen kann (Theis et al., 2016):** Mischen wir die wahre Dichte mit $99\%$ Rauschen, so erfüllt die Mischung $\tilde{p} = 0.01\,p_{\textrm{data}} + 0.99\,p_{\textrm{noise}}$ die Ungleichung $\log \tilde{p}(\mathbf{x}) \ge \log p_{\textrm{data}}(\mathbf{x}) - \log 100$. Die Log-Likelihood liegt damit höchstens $4.6$ Nat unter dem Optimum (bei Bildern geht es um Tausende Nat), obwohl der Generator zu $99\%$ reines Rauschen liefert.

<!--en:c68320d4-->
- **Grundidee:** An die Stelle der Likelihood-Auswertung tritt ein erlernter **Zweistichproben-Kritiker** $D$, der die erzeugten Samples direkt beurteilt.

<!--en:94639566-->
::: {.slide title="Schätzung des Dichteverhältnisses durch Zweistichproben-Klassifikation"}

<!--en:fa562a5e-->
[Das Funktionsprinzip]{.kicker}

<!--en:7d022792-->
Goodfellow et al. (2014) machten die binäre Klassifikation zweier Stichproben zum Trainingsprinzip für generative Modelle. Dazu lassen sie einen **Generator** $G$ gegen einen **Diskriminator (Kritiker)** $D$ antreten.

<!--en:f1fa3993-->
- **Ausgewogenes Zweistichprobenspiel:** Echte Samples $x \sim p$ erhalten das Label $y = 1$, erzeugte Samples $x' = G(z) \sim q$ das Label $y = 0$. Beide Klassen haben die A-priori-Wahrscheinlichkeit $\frac{1}{2}$.

<!--en:798e5da2-->
- **Der optimale Kritiker liefert das logarithmierte Dichteverhältnis:** Minimieren wir den binären logarithmischen Verlust, erhalten wir die A-posteriori-Wahrscheinlichkeit nach Bayes $\sigma(D^\star(x)) = \frac{p(x)}{p(x) + q(x)}$. Damit ergibt sich der optimale Logit als
  $$D^\star(x) = \log \frac{p(x)}{q(x)} = \lambda(x)$$

<!--en:88707ffb-->
- **Gradienten ohne Dichten:** Backpropagation durch $D$ verschiebt erzeugte Samples $x' = G(z)$ in Richtung des steilsten Anstiegs $\nabla_x D(x')$. So wandert Wahrscheinlichkeitsmasse aus überrepräsentierten Bereichen ($q > p \implies D^\star < 0$) in unterrepräsentierte Bereiche ($p > q \implies D^\star > 0$).

<!--en:c2d36363-->
::: {.slide title="Drei Gestaltungsfragen für jedes adversariale Spiel"}

<!--en:69ffac88-->
[Gemeinsamer Rahmen]{.kicker}

<!--en:657ede69-->
- **Warum das klassische Minimax-Verfahren sättigt:** Liegen die erzeugten Samples außerhalb des Trägers der Datenverteilung, erreicht $\max_D V(D) = 2\,\mathrm{JS}(p, q) - 2\log 2$ bei $\mathrm{JS}(p, q) = \log 2 \approx 0.6931$ Nat ein Plateau. Der Gradient verschwindet.

<!--en:5dfbaa42-->
- **Drei unabhängige Änderungen:** Eine GAN-Zielfunktion legt fest, was der Kritiker schätzt ($D^\star(\lambda)$ oder die Zeugenfunktion $T$), wie er Samples bewertet (einzeln oder paarweise nach Bradley-Terry) oder wie seine Gradienten regularisiert werden ($R_1 + R_2$).

<!--en:90fe55c9-->
- **Das modernisierte R3GAN-Verfahren (Huang et al., 2024):** Paarweises Ranking (`RpGAN`) und um null zentrierte Strafterme $R_1 + R_2$ dämpfen gemeinsam die Rotationsdynamik ohne BatchNorm.

<!--en:adf10ab6-->
::: {.slide title="Überblick über Kapitel §{num:chap_gans}: Theorie, Dynamik und Systeme"}

<!--en:88710d83-->
[Architekturen und mathematische Grundlagen]{.kicker}

<!--en:f8aad7b5-->
- **Teil I: Zielfunktionen und Divergenzen (§§{sec_basic_gan}–§{num:sec_gan_relativistic}):** Von der Schätzung des Dichteverhältnisses über den logarithmischen Verlust ($D^\star = \log(p/q)$) und nicht sättigenden Gewichten ($2\,D_{\textrm{KL}}(m \,\|\, p)$) zu $f$-Divergenzen, integralen Wahrscheinlichkeitsmetriken ($W_1$, MMD) und relativistischen Paarvergleichen ($d_{\mathrm{Rp}}$).

<!--en:5d265cc3-->
- **Teil II: Dynamik und Konvergenz (§{sec_gan_convergence}):** Warum Minimax-Schritte ohne Strafterm beim Dirac-GAN auf Kreisbahnen oder spiralförmig nach außen führen und wie um null zentrierte Strafterme $R_1 + R_2$ die Eigenwerte der Jacobi-Matrix in die offene linke Halbebene verschieben.

<!--en:03c5e235-->
- **Teil III: Bilder, Konditionierung und Anwendungen (§§{sec_dcgan}–§{num:sec_gan_beyond}):** DCGAN im Vergleich zu Grundnetzen ohne Normalisierung, FID/KID und $K$-NN-Präzision und -Trefferquote, Projektionskritiker und Diffusionsdestillation (ADD/DMD2).

<!--en:72b72fbb-->
::: {.slide title="Durchgängige Benchmarks und Leitfaden zum Kapitel"}

<!--en:05adf916-->
[Roter Faden]{.kicker}

<!--en:2d86ef20-->
- **Zweidimensionale Gauß-Verteilungen und Mischungen mit bekannter Dichte (§{sec_basic_gan}–§{num:sec_gan_relativistic}):** Vergleichen Sie die Logits des Kritikers mit dem analytischen Wert $\lambda(x) = \log(p(x)/q(x))$. Messen Sie anhand der exakten Werte von $D_{\textrm{KL}}(q \,\|\, p)$ und $\mathrm{JS}(p, q)$ in Nat, wie gut der Generator die Datenverteilung trifft.

<!--en:ea8a9616-->
- **25 Gauß-Verteilungen im $5 \times 5$-Raster (§{sec_gan_convergence}):** Prüfen Sie getrennt, ob paarweises Ranking oder die Gradientenstrafterme $R_1 + R_2$ den Modusausfall verhindern.

<!--en:4e8f8ee1-->
- **Bilderzeugung und bedingte Bilderzeugung (§{sec_basic_gan}, §{sec_dcgan}–§{num:sec_gan_conditional}):** Fashion-MNIST ($28 \times 28$), Pokémon-Sprites ($64 \times 64$) und CIFAR-10 mit 10 Klassen ($32 \times 32$).

<!--en:9c0b1285-->
- **Bimodale Regression und Destillation (§{sec_gan_beyond}):** Benchmark mit zwei Bändern $x = c \pm 1 + 0.05\,\xi$ und einem Lehrermodell, dessen Entrauschung sich exakt in geschlossener Form angeben lässt.

<!--en:e9d832ba-->
::: {.slide title="Verständnisfragen zur adversarialen Modellierung"}

<!--en:929b7671-->
[Prüfen Sie Ihr Verständnis]{.kicker}

<!--en:d36a1bfa-->
Prüfen Sie Ihre Intuition anhand dreier grundlegender Szenarien aus dem Kapitelüberblick.

<!--en:ba1c76c2-->
1. **Den Logit eines optimalen Kritikers deuten:** Ein optimaler Kritiker für den logarithmischen Verlust liefert $D^\star(x_1) = 1.2$ am Punkt $x_1$ und $D^\star(x_2) = -2.0$ am Punkt $x_2$. Wie groß ist das Dichteverhältnis $p(x)/q(x)$ an den beiden Punkten? In der Nähe welches Punktes sollte der Generator die Dichte seiner Samples verringern?

<!--en:97370dc0-->
2. **Drei Fehlerbilder erkennen:** Beim Training beobachten Sie Folgendes. (a) Die erzeugten Samples liegen weit von den Daten entfernt, und die Generatorgradienten sind fast null. (b) Beide Verluste bleiben nahe ihren Gleichgewichtswerten, während die erzeugten Samples auf einer geschlossenen Bahn kreisen. (c) Fordern Sie einen *„Vogel“* an, erhalten Sie jedes Mal genau denselben typischen Vogel. Welcher Mechanismus aus diesem Kapitel hilft bei welchem Symptom?

<!--en:557b98b6-->
**Antworten:**

<!--en:0ece88d0-->
1. Wegen $D^\star(x) = \log(p(x)/q(x))$ gilt $p(x_1)/q(x_1) = e^{1.2} \approx \mathbf{3.32}$ (die Dichte der echten Daten ist $3.3\times$ so hoch wie die des Generators) und $p(x_2)/q(x_2) = e^{-2.0} \approx \mathbf{0.135}$ (die Dichte des Generators ist $\approx 7.4\times$ so hoch wie die der echten Daten). Der Generator muss Wahrscheinlichkeitsmasse von **$x_2$** in Richtung **$x_1$** verschieben.

<!--en:11e01bae-->
2. **(a)** Verwenden Sie einen **nicht sättigenden Generatorverlust** (§{sec_basic_gan}) oder einen glatten bzw. durch Strafterme regularisierten Kritiker (§{sec_gan_objectives}). So behalten verworfene Samples ($D(x') \ll 0$) das Gradientengewicht eins. **(b)** Ergänzen Sie **um null zentrierte Gradientenstrafterme $R_1 + R_2$** (§{sec_gan_convergence}) und einen exponentiellen gleitenden Mittelwert (**EMA**, §{sec_dcgan}), um die rotierende Minimax-Dynamik zu dämpfen. **(c)** Messen Sie die **$K$-NN-Trefferquote je Klasse** (§{sec_gan_conditional}). So erkennen Sie einen Moduskollaps innerhalb einer Klasse, selbst wenn der Klassifikator die erzeugten Samples fast immer der gewünschten Klasse zuordnet.

<!--en:a8dec3da-->
::: {.slide title="Zusammenfassung: Notation und Zusammenhänge"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:203ebc5e-->
- **Zwei Stichproben und logarithmiertes Dichteverhältnis:** Echte Samples $x \sim p$ und erzeugte Samples $x' = G(z) \sim q$ bilden die gleichgewichtete Mischung $m = \frac{1}{2}(p + q)$. Der optimale Zielwert für den Kritiker ist $\lambda(x) = \log(p(x)/q(x))$, die A-posteriori-Wahrscheinlichkeit nach Bayes beträgt $\eta(x) = \sigma(\lambda(x))$.

<!--en:6a8cae73-->
- **Generatorgewichte je Sample:** Multiplizieren Sie $\nabla_\theta D(G_\theta(z))$ mit $w_{\textrm{sat}} = \sigma(D(x'))$, mit $w_{\textrm{ns}} = \sigma(-D(x'))$ oder mit dem verschiebungsinvarianten Gewicht $1 - w_{\textrm{Rp}} = \mathbb{E}_p[\sigma(D(x) - D(x'))]$.

<!--en:be63adec-->
- **Dämpfung am Gleichgewicht:** Die um null zentrierten Strafterme $R_1 + R_2 = \frac{\gamma}{2}(\mathbb{E}_p\|\nabla_x D\|^2 + \mathbb{E}_q\|\nabla_{x'} D\|^2)$ verschwinden bei $q = p$ ($\nabla_x D^\star = \mathbf{0}$) und stabilisieren zugleich die Minimax-Trajektorien.
