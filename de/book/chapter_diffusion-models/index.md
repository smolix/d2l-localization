<!--en:98cdb6cb-->
# Diffusionsmodelle

<!--en:ae2dba21-->
Die adversarialen Generatoren aus :numref:`chap_gans` brauchten ein zweites Netz, einen Diskriminator oder Kritiker. Denn ein Generator, der Rauschen auf Daten abbildet, hat kein direktes Regressionsziel (engl. regression target). Diffusionsmodelle und Flow-Matching-Modelle konstruieren ein Regressionsziel, indem sie jedes Trainingsbeispiel schrittweise mit gaußschem Rauschen überlagern. Um einen kleinen Rauschschritt umzukehren, müssen wir aus der verrauschten Beobachtung den bedingten Erwartungswert (engl. conditional expectation) der unverrauschten Daten schätzen.

<!--en:0904a9bc-->
Diese Sichtweise verbindet drei Forschungsrichtungen: die Nichtgleichgewichtsthermodynamik und probabilistische Diffusionsmodelle zur Entrauschung (engl. denoising diffusion probabilistic models, DDPM) :cite:`sohl2015deep,ho2020denoising`, die Score-basierte generative Modellierung :cite:`song2019generative,song2021score` und das Flow-Matching beziehungsweise die stochastischen Interpolanten :cite:`Lipman.Chen.BenHamu.ea.2022,Liu.Gong.Liu.2022,Albergo.Boffi.VandenEijnden.2023`. In allen drei Formulierungen lernt ein einziges neuronales Netz die unverrauschten Daten vorherzusagen. Gleichwertige Vorhersageziele sind das hinzugefügte Rauschen, der Score $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ oder die Geschwindigkeit entlang des Pfads. Das Netz liefert für jeden Rauschpegel $t \in [0, 1]$ eine Vorhersage. Ein numerischer Sampler integriert diese Vorhersagen vom reinen Rauschen bei $t = 1$ zurück zu den Daten bei $t = 0$.

<!--en:d29c91cc-->
Ersetzen wir die adversariale Minimax-Zielfunktion durch Regression, überwinden wir drei Einschränkungen von GANs. Erstens trainieren wir nur ein Netz, indem wir den quadratischen Fehler minimieren. Zweitens liefert der gewichtete Regressionsverlust eine Variationsschranke für die Log-Likelihood. Über die Wahrscheinlichkeitsfluss-ODE (engl. probability-flow ODE) erhalten wir die exakte Log-Likelihood. Drittens bestraft die Likelihood-Zielfunktion fehlende Modi. In der Standardformulierung dieses Kapitels schätzen wir durch Regression die Geschwindigkeit entlang des linearen Pfads $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\boldsymbol\epsilon$ :cite:`Esser.Kulal.Blattmann.ea.2024`. Daneben betrachten wir die Rauschvorhersage des DDPM auf dem varianzerhaltenden Pfad.

<!--en:bcfb3e03-->
In den ersten drei Abschnitten entwickeln wir die gesamten mathematischen und algorithmischen Grundlagen anhand einer Mischung aus drei zweidimensionalen Gauß-Verteilungen. Hier kennen wir die wahre Dichte, den Score, den A-posteriori-Mittelwert (engl. posterior mean) und das Geschwindigkeitsfeld in geschlossener Form. In :numref:`sec_diffusion-images` gehen wir zu Bildern aus Fashion-MNIST über. Dort führen wir das auf die Zeit konditionierte U-Net, die Auswertung anhand der Fréchet-Distanz und Classifier-Free Guidance ein. Die nächsten vier Abschnitte behandeln den Sampler (:numref:`sec_diffusion-samplers`), die Likelihood-Schranke und den Entwurfsraum (:numref:`sec_diffusion-design-space`), latente Diffusion und Transformer als Basisnetze (:numref:`sec_diffusion-latent`) sowie die Destillation auf wenige Schritte mithilfe von Flussabbildungen (engl. flow maps) (:numref:`sec_diffusion-few-step`). Schließlich erweitern wir in :numref:`sec_diffusion-discrete` die Diffusion auf diskrete Token, indem wir die Token maskieren, statt gaußsches Rauschen hinzuzufügen. Längere mathematische Beweise finden sich in :numref:`chap_mdl-dynamics`. Externe Tutorials, Kurse und Referenzimplementierungen sind in :numref:`sec_diffusion-further-reading` zusammengestellt.

<!--en:ff904380-->
Generative Modelle auf Basis von Diffusion und Flüssen<br>**Entrauschung und Scores · gaußsche Wahrscheinlichkeitspfade · Integration von ODEs und SDEs · U-Net und DiT · schnelle Sampler · Maskierung diskreter Token**

<!--en:76656029-->
::: {.slide title="Warum Diffusion das adversariale Minimax-Spiel ersetzt"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:14851172-->
Ein Generator, der zufälliges Rauschen auf Daten abbildet, hat keinen direkten Zielwert für das Training. Daher müssen GANs einen Diskriminator als Gegenspieler trainieren. Fügen wir schrittweise gaußsches Rauschen hinzu, erhalten wir für jeden Rauschpegel ein explizites Regressionsziel:

<!--en:69e4cf4b-->
- **Regression mit einem einzigen Netz:** Aus dem Trainingsbeispiel $\mathbf{x}_0$ entsteht durch Verrauschen $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$. Ein einziges Netz kann so durch Minimierung des quadratischen Fehlers direkt den unverrauschten Punkt, das hinzugefügte Rauschen, den Score oder die Geschwindigkeit vorhersagen.

<!--en:d614b1a5-->
- **Exakte Likelihoods und Schranken:** Der gewichtete Verlust bei der Entrauschung liefert eine obere Schranke für die negative Log-Likelihood $-\log p_{\boldsymbol\theta}(\mathbf{x}_0)$. Die deterministische Wahrscheinlichkeitsfluss-ODE liefert durch Variablentransformation exakte logarithmische Dichten.

<!--en:9b7f684b-->
- **Garantierte Modusabdeckung:** Minimieren wir die erwartete negative Log-Likelihood $\mathbb{E}_p[-\log p_{\boldsymbol\theta}] = H(p) + D_{\textrm{KL}}(p \,\|\, p_{\boldsymbol\theta})$, fällt jeder ausgelassene Modus der Datenverteilung stark ins Gewicht.

<!--en:62dbfe0d-->
::: {.slide title="Gauß-Pfade und die gemeinsame Zeitskala der Log-SNR"}

<!--en:7dd17f06-->
[Verrauschung im Vorwärtsprozess und Zeitkonventionen]{.kicker}

<!--en:3958f321-->
Jedes kontinuierliche Modell in diesem Kapitel interpoliert zwischen unverrauschten Daten bei $t = 0$ und standardnormalverteiltem Rauschen $\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ bei $t = 1$:

<!--en:4e3aa045-->
- **Linearer Pfad (Standard beim Flow-Matching):** Es gilt $\alpha_t = 1 - t$ und $\sigma_t = t$. Für jedes Trainingspaar bewegt sich der Punkt mit konstanter bedingter Geschwindigkeit $\dot{\mathbf{x}}_t = \boldsymbol\epsilon - \mathbf{x}_0$.

<!--en:e712d695-->
- **Varianzerhaltender Pfad (VP) beim DDPM:** Wegen $\alpha_t^2 + \sigma_t^2 = 1$ bleibt die Varianz über alle $T = 1{,}000$ diskreten Vorwärtsschritte hinweg gleich eins.

<!--en:46f3992f-->
- **Kanonische Zeitskala der Log-SNR:** $\lambda_t = \log(\alpha_t^2 / \sigma_t^2)$ fällt von $+\infty$ bei $t = 0$ auf $-\infty$ bei $t = 1$. Sampler integrieren stets rückwärts von $t = 1$ bis $t = 0$.

<!--en:172b8637-->
::: {.slide title="Ein A-posteriori-Mittelwert verbindet vier Regressionsziele"}

<!--en:75ce520f-->
[Mathematische Grundlagen]{.kicker}

<!--en:552bed7f-->
Bei gegebener Eingabe minimiert der bedingte Erwartungswert den erwarteten quadratischen Fehler. Beim Training mit einem Zielwert pro Beispiel lernen wir daher den A-posteriori-Mittelwert $\hat{\mathbf{x}}_0(\mathbf{x}_t) = \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]$:

<!--en:0c6810e1-->
- **Tweedies Identität:** Der A-posteriori-Mittelwert des unverrauschten Punkts und der A-posteriori-Mittelwert des Rauschens $\hat{\boldsymbol\epsilon} = \mathbb{E}[\boldsymbol\epsilon \mid \mathbf{x}_t]$ bestimmen den Score der Randdichte $p_t$:
  $$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = -\frac{\hat{\boldsymbol\epsilon}(\mathbf{x}_t)}{\sigma_t}$$

<!--en:e7977c67-->
- **Affine Äquivalenz der Regressionsziele:** Es gilt $\mathbf{x}_t = \alpha_t \hat{\mathbf{x}}_0 + \sigma_t \hat{\boldsymbol\epsilon}$. Daher lassen sich die Vorhersage des unverrauschten Punkts $\hat{\mathbf{x}}_0$, das Rauschen $\hat{\boldsymbol\epsilon}$, der Score $\nabla \log p_t$ und die marginale Geschwindigkeit $\mathbf{u}_t = \dot\alpha_t \hat{\mathbf{x}}_0 + \dot\sigma_t \hat{\boldsymbol\epsilon}$ durch affine Transformationen ineinander überführen.

<!--en:e8d2cc35-->
- **Integration rückwärts in der Zeit:** Ein numerischer ODE- oder SDE-Löser integriert $\mathbf{u}_t(\mathbf{x})$ von $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ zurück bis $\mathbf{x}_0$.

<!--en:9f3d2260-->
::: {.slide title="Umrechnung zwischen den Zeitkonventionen der Literatur"}

<!--en:7c962db8-->
[Zeitangaben in Arbeiten zu Diffusion und Flow-Matching]{.kicker}

<!--en:19cde150-->
Arbeiten zu Diffusionsmodellen setzen die unverrauschten Daten bei $t = 0$ an. Viele Arbeiten zum Flow-Matching kehren die Zeitrichtung um und setzen das Rauschen bei $t = 0$ an:

<!--en:9dc1a6ff-->
- **Unsere Konvention (Standard bei Diffusion):** Bei $t = 0$ liegen unverrauschte Daten vor ($\mathbf{x}_0$), bei $t = 1$ reines Rauschen ($\mathbf{x}_1 = \boldsymbol\epsilon$). Zur Erzeugung von Daten integrieren wir von $t = 1$ zurück bis $t = 0$.

<!--en:9df74f98-->
- **Umrechnung beim Lesen von Arbeiten zum Flow-Matching:** Ersetzen Sie die dortige Zeitvariable $t_{\textrm{FM}}$ durch $1 - t$. Ordnen Sie das anfängliche Rauschen $\mathbf{x}_0^{\textrm{FM}}$ unserer Größe $\boldsymbol\epsilon$ und die Zieldaten $\mathbf{x}_1^{\textrm{FM}}$ unserer Größe $\mathbf{x}_0$ zu. Kehren Sie das Vorzeichen jedes Geschwindigkeitsvektors um ($\mathbf{v}_t^{\textrm{FM}} = -\mathbf{u}_{1-t}$).

<!--en:c84ca325-->
- **Plausibilitätsprüfung mit der Log-SNR:** Prüfen Sie stets, ob die Zeit $t = 0.8$ in der jeweiligen Arbeit einem hohen Signal-Rausch-Verhältnis ($\lambda > 0$, überwiegend Daten) oder einem niedrigen ($\lambda < 0$, überwiegend Rauschen) entspricht.

<!--en:7818f8e3-->
::: {.slide title="Durchgängige Benchmarks in Kapitel §{num:chap_diffusion}"}

<!--en:f3f40f68-->
[Empirische Prüfung]{.kicker}

<!--en:12d83417-->
Wir prüfen jede theoretische Aussage und jede Architekturentscheidung anhand kontrollierter Benchmarks. Dafür stehen exakte Referenzwerte oder kalibrierte Metriken im Merkmalsraum zur Verfügung:

<!--en:675e2b20-->
- **Zweidimensionale Gauß-Mischung mit drei Modi (§{sec_diffusion-denoising}–§{num:sec_diffusion-integration}, §{sec_diffusion-design-space}, §{sec_diffusion-few-step}):** Die Gewichte betragen $(0.5, 0.3, 0.2)$, die Breite ist $s = 0.5$. Dichte, Score, A-posteriori-Mittelwert, Geschwindigkeit und Entropie ($\approx 2.5$ nats) liegen in geschlossener Form vor.

<!--en:8cfbe205-->
- **Pixel und latente Darstellungen von Fashion-MNIST (§{sec_diffusion-images}–§{num:sec_diffusion-few-step}):** Bilder der Größe $28 \times 28$ mit Werten in $[-1, 1]$ und latente VAE-Darstellungen der Größe $7 \times 7 \times 4$. Wir messen die Fréchet-Distanz (FD). Als Referenz dient das Grundniveau der Distanz zwischen zwei endlichen Stichproben realer Bilder.

<!--en:086ba720-->
- **Diskrete Tokensequenzen (§{sec_diffusion-discrete}):** Paare quantisierter $32 \times 32$-Gitter und binäre Bilder von Kleidungsstücken mit je 784 Token. Wir maskieren Token mit $[\textrm{M}]$.

<!--en:abec5550-->
::: {.slide title="Kapitel §{num:chap_diffusion} im Überblick: vom Score-Matching zu Token"}

<!--en:123f19cf-->
[Wegweiser durch das Kapitel]{.kicker}

<!--en:61afbee2-->
- **§§{sec_diffusion-denoising}–§{num:sec_diffusion-integration} Theoretische Grundlagen:** Denoising-Score-Matching, Tweedies Formel, Gauß-Pfade ($\lambda_t$) und Rückwärtstransport mit ODEs oder SDEs. Das trainierte MLP trifft die exakten Gewichte des zweidimensionalen Referenzmodells bis auf $0.004$.

<!--en:a2ad194e-->
- **§§{sec_diffusion-images}–§{num:sec_diffusion-design-space} Bilder, schnelle Löser und Entwurfsraum:** U-Net mit CFG ($\approx 90\%$ Klassenübereinstimmung), Heun- und Mehrschrittverfahren mit 20 Funktionsauswertungen, kontinuierliche ELBO ($w(\lambda) \equiv 1$) und EDM-Vorkonditionierung.

<!--en:1d327088-->
- **§§{sec_diffusion-latent}–§{num:sec_diffusion-discrete} Latente Darstellungen, Erzeugung in wenigen Schritten und diskrete Token:** KL-Autoencoder und DiT mit `adaLN-Zero` (Rechenaufwand um einen Faktor $>10\times$ geringer, gemessen in FLOPs), Destillation von Flussabbildungen zwischen zwei Zeitpunkten ($2$ Sprünge entsprechen $15$ Euler-Schritten) und maskierte Diffusion mit absorbierendem Zustand ($0.17$ bpp).
