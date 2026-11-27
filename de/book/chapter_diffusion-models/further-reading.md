<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:243f93d1-->
Dieser Abschnitt versammelt Monografien, Tutorials, Referenzimplementierungen und Originalarbeiten, die die in diesem Kapitel entwickelten Methoden vertiefen.

<!--en:21b2a03a-->
## Monografien, Kurse und Code

<!--en:08face5d-->
- [*The Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890) :cite:`Lai.Song.Kim.ea.2025` leitet Diffusionsmodelle auf drei Wegen her: mit variationellen, scorebasierten und flussbasierten Ansätzen. Das Buch zeigt, wo diese Ansätze zusammenlaufen.

<!--en:7dd3f191-->
- Die [Skripte, Vorlesungen und praktischen Übungen](https://diffusion.csail.mit.edu/) des MIT-Kurses 6.S184 :cite:`Holderrieth.Erives.2025` behandeln Flow-Matching, Diffusion Transformer im latenten Raum und diskrete Diffusion.

<!--en:7224b94e-->
- [*Diffusion Meets Flow Matching*](https://diffusionflow.github.io/) :cite:`Gao.Hoogeboom.Heek.ea.2024` zeigt anhand interaktiver Abbildungen, dass DDIM und der Euler-Sampler für Flow-Matching bei entsprechender Umskalierung des Rauschplans äquivalent sind.

<!--en:94286aad-->
- [*Flow Matching Guide and Code*](https://arxiv.org/abs/2412.06264) :cite:`Lipman.Havasi.Holderrieth.ea.2024` und die zugehörige [Bibliothek](https://github.com/facebookresearch/flow_matching) liefern vollständige Beweise und Referenzcode zu :numref:`sec_diffusion-gaussian-paths`.

<!--en:ecb3c42c-->
- [*Step-by-Step Diffusion*](https://arxiv.org/abs/2406.08929) :cite:`Nakkiran.Bradley.Zhou.ea.2025` leitet deterministische und stochastische Sampler (:numref:`sec_diffusion-integration` und :numref:`sec_diffusion-samplers`) allein mit elementarer Wahrscheinlichkeitsrechnung her und kommt dabei ohne stochastische Analysis aus.

<!--en:292b57c7-->
- In seinem [Blog](https://sander.ai/) untersucht Sander Dieleman die Entrauschung (engl. denoising) im Frequenzraum :cite:`Dieleman.2024`, latente Darstellungen (engl. latent representations) :cite:`Dieleman.2025` und Flussabbildungen (engl. flow maps) :cite:`Dieleman.2026`. Außerdem vergleicht er kontinuierliche und diskrete Diffusion für die Sprachmodellierung :cite:`Dieleman.2026b`.

<!--en:f4863c1d-->
- Referenzimplementierungen finden sich etwa im [offiziellen DDPM-Repository](https://github.com/hojonathanho/diffusion) :cite:`ho2020denoising`, im [Repository zu Score SDE](https://github.com/yang-song/score_sde_pytorch) :cite:`song2021score` und im [MDLM-Repository](https://github.com/kuleshov-group/mdlm) :cite:`Sahoo.Arriola.Schiff.ea.2024`.

<!--en:c50e15d7-->
- Der [Kurs *Deep Generative Models*](https://kuleshov-group.github.io/dgm-website/) :cite:`Kuleshov.2023`, Yang Songs [Tutorial zu scorebasierten Modellen](https://yang-song.net/blog/2021/score/) :cite:`Song.2021`, Lilian Wengs [*What Are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) :cite:`Weng.2021` und die Arbeit von :citet:`Luo.2022` geben einen leicht verständlichen Überblick über Score-Matching und variationelle Diffusion.

<!--en:bad89039-->
Fünf Originalarbeiten sind besonders grundlegend. :citet:`Karras.Aittala.Aila.ea.2022` (EDM) arbeiten die getrennten Rollen von Sampler, Vorkonditionierung und Rauschplan heraus. :citet:`Esser.Kulal.Blattmann.ea.2024` (Stable Diffusion 3) kombinieren Rectified Flow mit multimodalen Diffusion Transformern. :citet:`Lu.Zhou.Bao.ea.2025` (DPM-Solver++) untersuchen Lösungsverfahren höherer Ordnung bei Einsatz von Guidance. :citet:`Lu.Song.2025` (sCM) stabilisieren große Konsistenzmodelle in kontinuierlicher Zeit. :citet:`Sahoo.Arriola.Schiff.ea.2024` (MDLM) geben eine übersichtliche Formulierung der maskierten diskreten Diffusion an. Verwandte Themen behandeln :numref:`sec_gan-further-reading` (adversarielle Destillation, Destillation durch Verteilungsabgleich und Bewertung mit FID) und :numref:`sec_mdl-dynamics-further-reading` (exponentielle Integratoren, Schranken für den Score-Fehler und Schrödinger-Brücken).

<!--en:0bde3a80-->
## Energiebasierte Modelle und Score-Matching

<!--en:358d6d8c-->
**Training energiebasierter Modelle.** :citet:`Song.Kingma.2021` geben einen Überblick über das Maximum-Likelihood-Training für unnormierte Dichten. Bei der kontrastiven Divergenz starten kurze MCMC-Ketten an den Datenpunkten :cite:`Hinton.2002`. Dadurch findet zwischen getrennten Modi keine Durchmischung statt (:numref:`sec_diffusion-denoising`). Bei der persistenten kontrastiven Divergenz bleiben die Zustände der Ketten über Parameterupdates hinweg erhalten :cite:`Tieleman.2008,Du.Mordatch.2019`.

<!--en:2edb1f15-->
**Rauschkontrastive Schätzung.** :citet:`Gutmann.Hyvarinen.2012` behandeln die Normierungskonstante als lernbaren Parameter, statt sie durch Differenzieren zu eliminieren. Sie trainieren das Modell mittels logistischer Regression, die Daten von Samples aus einer bekannten Rauschverteilung unterscheidet.

<!--en:49172c91-->
**Grenzen des Score-Matchings.** Trennt ein Gebiet geringer Dichte zwei Modi, ist die Fisher-Divergenz unempfindlich gegenüber ihren Modusgewichten (:numref:`sec_diffusion-denoising`). Dieses Phänomen untersuchen :citet:`Wenliang.Kanagawa.2020`. :citet:`Koehler.Heckett.Risteski.2022` leiten mithilfe isoperimetrischer Konstanten Schranken für die statistische Effizienz des Score-Matchings her.

<!--en:843ff3c7-->
**Faustregeln für Rauschpegel.** :citet:`Song.Ermon.2020` geben vier praktische Regeln für eine geometrische Stufenfolge von Rauschpegeln (:numref:`sec_diffusion-gaussian-paths`). Der höchste Rauschpegel entspricht dem größten Abstand zwischen zwei Trainingspunkten. Benachbarte Pegel sollen so eng beieinanderliegen, dass sich die Schalen hoher Wahrscheinlichkeit überlappen (Übung :exref:`ex_gaussian-paths-ladder-spacing` in :numref:`sec_diffusion-gaussian-paths`). Hinzu kommen die Abstimmung der Langevin-Schrittweite und die Bildung eines exponentiellen gleitenden Mittelwerts der Gewichte.

<!--en:43c71512-->
**Empirische Bayes-Schätzung.** Tweedies Formel stammt aus der empirischen Bayes-Schätzung :cite:`Robbins.1956`. :citet:`Efron.2011` machte ihre Anwendung zur Korrektur des Selektionsbias bei hochdimensionalen Schätzproblemen bekannt.

<!--en:6b534ab1-->
## Netze für Bilder und Guidance

<!--en:f06c3c9d-->
**Große U-Net-Architekturen.** :citet:`Dhariwal.Nichol.2021` (ADM) ergänzten das DDPM-U-Net um Attention auf mehreren Auflösungsstufen, residuale Neuabtastung und adaptive Gruppennormalisierung. :citet:`Karras.Aittala.Lehtinen.ea.2024` (EDM2) gestalteten das U-Net so um, dass die Größenordnungen der Aktivierungen und Gewichte erhalten bleiben. Außerdem führten sie ein Verfahren ein, das den exponentiellen gleitenden Mittelwert (EMA) nachträglich aus regelmäßig gespeicherten Checkpoints rekonstruiert.

<!--en:6f1d843b-->
**Guidance ohne Sättigung.** Hohe Werte der Guidance-Skala können Pixelintensitäten in die Sättigung treiben (Übung :exref:`ex_image-diffusion-thresholding` in :numref:`sec_diffusion-images`). :citet:`Sadat.Hilliges.Weber.2025` zerlegen das Guidance-Update in zwei Komponenten: eine parallel zur bedingten Vorhersage und eine orthogonal dazu. Sie dämpfen die parallele Komponente, um Sättigung zu verhindern.

<!--en:8ecba538-->
**Räumliche Konditionierung.** ControlNet :cite:`Zhang.Rao.Agrawala.2023` konditioniert ein vortrainiertes Text-zu-Bild-Modell auf räumliche Karten für Kanten, Tiefe oder Körperhaltung. Dazu dient eine trainierbare Kopie des Encoders, die über mit null initialisierte Schichten an das Modell angebunden ist.

<!--en:d3bea0c1-->
## Evaluation, Auswendiglernen und Generalisierung

<!--en:a8fd78b6-->
**Übereinstimmung mit dem Prompt.** Der Fréchet-Abstand erfasst die Ähnlichkeit von Verteilungen und gibt daher keinen Aufschluss darüber, wie gut ein Bild zum Prompt passt. CLIPScore :cite:`Hessel.Holtzman.Forbes.ea.2021` misst dazu die Ähnlichkeit zwischen Bild und Bildbeschreibung in einem gemeinsamen Einbettungsraum. GenEval :cite:`Ghosh.Hajishirzi.Schmidt.2023` prüft mit einem Objektdetektor Anzahl, Positionen und Eigenschaften der Objekte.

<!--en:26873971-->
**Auswendiglernen.** :citet:`Carlini.Hayes.Nasr.ea.2023` extrahieren aus großen Text-zu-Bild-Modellen Bilder, die mit Trainingsbildern nahezu identisch sind. :citet:`Biroli.Bonnaire.deBortoli.ea.2024` unterscheiden im Rückwärtsprozess zwei dynamische Regime: die Festlegung auf eine Klasse (Artbildung) und das Auswendiglernen (engl. memorization) einzelner Trainingspunkte (Kollaps).

<!--en:f95f5085-->
**Generalisierung von Entrauschern.** Der exakte empirische Minimierer des Entrauschungsziels lernt einen endlichen Trainingsdatensatz auswendig. Dennoch zeigen :citet:`Bertrand.Gagneux.Massias.ea.2025`, dass das Rauschen der Zielgrößen allein die Generalisierung nicht erklärt. Auf disjunkten Teilmengen trainierte Entrauscher lernen nahezu identische Score-Felder :cite:`Kadkhodaie.Guth.Simoncelli.ea.2024`. Diese sind weitgehend durch die Lokalität und Äquivarianz der Faltung bestimmt :cite:`Kamb.Ganguli.2025`. Einen Überblick geben :citet:`Massias.Bertrand.2026`.

<!--en:d29b7e35-->
## Sampler, Bildbearbeitung und inverse Probleme

<!--en:f945cc79-->
**Lösungsverfahren für gesteuertes Sampling.** Hohe Werte der Guidance-Skala destabilisieren Verfahren höherer Ordnung, die auf Rauschvorhersage beruhen (Übung :exref:`ex_samplers-dpm-solver` in :numref:`sec_diffusion-samplers`). DPM-Solver++ :cite:`Lu.Zhou.Bao.ea.2025` parametrisiert die Updates des Mehrschrittverfahrens stattdessen über die Schätzung des rauschfreien Bildes. UniPC :cite:`Zhao.Bai.Rao.ea.2023` ergänzt einen Korrektorschritt ohne zusätzlichen Rechenaufwand.

<!--en:278a52df-->
**Stochastische Elemente in ODE-Lösern.** :citet:`Karras.Aittala.Aila.ea.2022` fügen vor jedem Heun-Schritt eine kleine Menge Rauschen hinzu. Beim Restart-Sampling :cite:`Xu.Deng.Cheng.ea.2023` wechseln sich deterministische ODE-Integration und größere Sprünge in Vorwärtsrichtung durch Zugabe von Rauschen ab. So verringert sich der angesammelte Diskretisierungsfehler.

<!--en:c02f2f27-->
**Exakte Inversion.** Bei der DDIM-Inversion (:numref:`sec_diffusion-samplers`) summieren sich unter Classifier-Free Guidance die Diskretisierungsfehler. Daher optimiert die Nulltext-Inversion :cite:`Mokady.Hertz.Aberman.ea.2023` die unbedingte Einbettung in der Umgebung einer Trajektorie ohne Guidance. EDICT :cite:`Wallace.Gokul.Naik.2023` erreicht exakte Umkehrbarkeit durch gekoppelte affine Transformationen.

<!--en:d5ace9b1-->
**Inverse Probleme ohne erneutes Training.** Bei linearen inversen Problemen wie der Beseitigung von Bildunschärfe oder der Superauflösung approximiert Diffusion Posterior Sampling (DPS) :cite:`Chung.Kim.Mccann.ea.2023` den Gradienten der Mess-Likelihood am A-posteriori-Mittelwert $\hat{\mathbf{x}}_0(\mathbf{x}_t)$. DDRM :cite:`Kawar.Elad.Ermon.ea.2022` arbeitet im Spektralraum des Messoperators.

<!--en:01e8f000-->
## Rauschpläne, Gewichtung und Likelihood

<!--en:d3bf9c07-->
**Gelernte Rückwärtsvarianzen.** Neben dem Kosinusplan (:numref:`sec_diffusion-gaussian-paths`) führen :citet:`Nichol.Dhariwal.2021` eine Parametrisierung der Rückwärtsvarianz ein. Sie interpolieren zwischen $\tilde\beta_k$ und $\beta_k$ und lernen die Interpolation anhand der Variationsschranke.

<!--en:ad9f0f6f-->
**Min-SNR-Gewichtung.** :citet:`Hang.Gu.Li.ea.2023` gewichten den quadratischen Fehler bei der Vorhersage des rauschfreien Bildes mit $\min(\mathrm{SNR}_t, \gamma)$. Gleichwertig dazu ist die Gewichtung des Fehlers der Rauschvorhersage mit $\min(1, \gamma e^{-\lambda})$. So verhindern sie, dass Schritte mit geringem Rauschpegel die Gradientenupdates dominieren.

<!--en:189c8f12-->
**Likelihood-Optimierung.** Variationelle Diffusionsmodelle (VDM) :cite:`Kingma.Salimans.Poole.ea.2021` optimieren den Rauschplan, um die Varianz des ELBO-Schätzers zu minimieren. :citet:`Song.Durkan.Murray.ea.2021` leiten den Plan für die Likelihood-Gewichtung beim Score-Matching in kontinuierlicher Zeit her. :citet:`Zheng.Lu.Chen.ea.2023` verbessern das Training und die Evaluation mit exakter Likelihood für die Wahrscheinlichkeitsfluss-ODE.

<!--en:6e0a285c-->
## Latente Darstellungen, Text und Video

<!--en:2b27eb53-->
**Konditionierung auf Text.** Große Text-zu-Bild-Modelle nutzen die Ausgabe eingefrorener Textencoder von CLIP :cite:`radford2021learning` oder T5 :cite:`raffel2020exploring` zur Konditionierung. Dazu verwenden sie Cross-Attention (:numref:`sec_diffusion-latent`) oder gemeinsame Self-Attention über Bild und Text :cite:`Esser.Kulal.Blattmann.ea.2024`. :citet:`saharia2022photorealistic` zeigten, dass eine Vergrößerung des Textencoders die Übereinstimmung mit dem Prompt stärker verbessert als eine Vergrößerung des U-Nets.

<!--en:e46427f5-->
**Abgleich von Darstellungen und Autoencoder.** REPA :cite:`Yu.Kwak.Jang.ea.2025` beschleunigt das Training eines DiT im latenten Raum, indem es versteckte Zustände in Zwischenschichten an die Darstellungen eines vortrainierten Bildencoders angleicht. Deep Compression Autoencoders (DC-AE) :cite:`Chen.Cai.Chen.ea.2025` steigern den Faktor für das räumliche Downsampling auf über $8\times$. Repräsentationsautoencoder :cite:`Zheng.Ma.Tong.ea.2026` kombinieren einen eingefrorenen semantischen Encoder wie DINO mit einem gelernten Decoder.

<!--en:f2481849-->
**Videogenerierung.** Videodiffusionsmodelle verwenden räumlich-zeitliche 3D-Autoencoder. Die Attention über Raum und Zeit berechnen sie faktorisiert oder gemeinsam :cite:`Ho.Salimans.Gritsenko.ea.2022,Blattmann.Dockhorn.Kulal.ea.2023,Polyak.Zohar.Brown.ea.2024`.

<!--en:0456db4f-->
## Generierung in wenigen Schritten

<!--en:90de5be8-->
**Progressive Destillation und Destillation von Guidance.** Die progressive Destillation :cite:`Salimans.Ho.2022` halbiert wiederholt die Schrittzahl des Lehrers. Mit ihr wurde die $\mathbf{v}$-Vorhersage eingeführt, um bei niedrigem Signal-Rausch-Verhältnis numerische Stabilität zu erreichen. :citet:`Meng.Rombach.Gao.ea.2023` destillieren die Classifier-Free Guidance in einen einzigen Durchlauf, der auf die Guidance-Skala $\gamma$ konditioniert ist.

<!--en:510cb2e1-->
**Konsistenz und Flussabbildungen für große Modelle.** Das verbesserte Konsistenztraining (iCT) :cite:`Song.Dhariwal.2024` kommt in diskreter Zeit ohne Lehrer aus. Für kontinuierliche Zeit stabilisiert sCM :cite:`Lu.Song.2025` das Konsistenztraining im ImageNet-Maßstab mithilfe trigonometrischer Pfade und einer Normierung der Tangenten. Improved MeanFlow :cite:`Geng.Lu.Wu.ea.2026` verwendet die Momentangeschwindigkeiten direkt als Regressionsziele. :citet:`Boffi.Albergo.VandenEijnden.2025` untersuchen die Selbstdestillation anhand eulerscher und lagrangescher Identitäten sowie der Verkettungsidentität für Flussabbildungen.

<!--en:cc839eee-->
## Diskrete Diffusion

<!--en:ccc7c374-->
**Alternative Übergangskerne und erneute Maskierung.** Diskretes Flow-Matching :cite:`Gat.Remez.Shaul.ea.2024` beschreibt die diskrete Generierung durch Pfade von Wahrscheinlichkeitsverteilungen auf dem Simplex. Übergangskerne mit gleichverteiltem Rauschen erlauben es, Token in jedem Schritt zu korrigieren. :citet:`Sahoo.Deschenaux.Gokaslan.ea.2025` stellen einen Zusammenhang zwischen diskreter Diffusion mit gleichverteiltem Rauschen und gaußscher Diffusion mit Schwellenwertbildung her. ReMDM :cite:`Wang.Schiff.Sahoo.ea.2025` führt die erneute Maskierung in die absorbierende Diffusion ein. :citet:`Kim.Shah.Kontonis.ea.2025` zeigen, wie stark es sich bei Denkaufgaben wie Sudoku auswirkt, wenn man die Token in der Reihenfolge ihrer Konfidenz demaskiert.

<!--en:badadf1b-->
**Diffusionssprachmodelle.** LLaDA :cite:`Nie.Zhu.You.ea.2025` trainiert ein maskiertes Diffusionssprachmodell mit acht Milliarden Parametern von Grund auf. Dream :cite:`Ye.Xie.Zheng.ea.2025` geht von einem autoregressiven Checkpoint aus. Blockdiffusion :cite:`Arriola.Gokaslan.Chiu.ea.2025` kombiniert autoregressive Generierung von Block zu Block mit paralleler Diffusion innerhalb der Blöcke. Fast-dLLM :cite:`Wu.Zhang.Xue.ea.2026` nutzt zusätzlich den KV-Cache mehrfach (:numref:`sec_kv-cache`). :citet:`Zheng.Chen.Mao.ea.2025` weisen darauf hin, dass Gumbel-Max-Sampling mit 32 Bit die Sampling-Temperatur künstlich senkt. Dadurch fällt die gemessene generative Perplexität zu günstig aus, während die Entropie der Samples sinkt.

<!--en:224e09cf-->
## Post-Training und Anwendungen jenseits von Bild und Text

<!--en:6f051c20-->
**Reinforcement Learning und Präferenzoptimierung.** DDPO :cite:`Black.Janner.Du.ea.2024` und Flow-GRPO :cite:`Liu.Liu.Liang.ea.2025` verwenden Strategiegradienten (:numref:`sec_rl_sequences`), um die stochastischen Rückwärtsübergänge von Diffusions- und Flussmodellen zu aktualisieren. Diffusion-DPO :cite:`Wallace.Dang.Rafailov.ea.2024` passt die direkte Präferenzoptimierung an: Die ELBO des Diffusionsmodells ersetzt die Log-Likelihood des Modells, die sich nicht praktikabel berechnen lässt.

<!--en:aafea46a-->
**Audio, Robotik und Proteine.** DiffWave :cite:`Kong.Ping.Huang.ea.2021` wendet Diffusion direkt auf die Wellenform von Audiosignalen an. Diffusionsstrategien :cite:`Chi.Feng.Du.ea.2023` modellieren multimodale Aktionstrajektorien von Robotern, bedingt auf visuelle Beobachtungen. So vermeiden sie das Problem der direkten Strategieregression mit dem mittleren quadratischen Fehler (MSE) als Verlustfunktion: Sie würde Aktionen aus verschiedenen Modi mitteln. Multimodale Proteinmodelle :cite:`Campbell.Yim.Barzilay.ea.2024` wenden Diffusion gemeinsam auf kontinuierliche 3D-Atomkoordinaten und diskrete Aminosäuresequenzen an.

<!--en:f75d086a-->
[Diskussion](https://d2l.discourse.group/t/36894)
