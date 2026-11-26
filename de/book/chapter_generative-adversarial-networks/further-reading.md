<!--en:7902ddfd-->
# Weiterführende Literatur

<!--en:34c6c7c0-->
Dieser Abschnitt verbindet die zentralen Konzepte des Kapitels mit weiterführender Literatur und Open-Source-Implementierungen. Dazu gehören die Schätzung von Dichteverhältnissen, die Formulierung von Zielfunktionen als Divergenzen, relativistische Zielfunktionen, Gradientenstrafen und adversariales Post-Training.

<!--en:2683e95e-->
## Empfohlene Einstiegslektüre und Quellcode

<!--en:e2cac79e-->
1. **Lokale Konvergenz und Gradientenstrafen.** :citet:`Mescheder.Geiger.Nowozin.2018` beweisen den in :numref:`sec_gan_convergence` zitierten Satz zur lokalen Konvergenz. Sie zeigen, dass um null zentrierte Gradientenstrafterme ($R_1$ und $R_2$) alternierende Gradientenupdates in der Nähe des Gleichgewichts stabilisieren. Dort versagen sowohl die Begrenzung der Gewichte als auch um eins zentrierte Strafterme (WGAN-GP). Ihr [Repository zur GAN-Stabilität](https://github.com/LMescheder/GAN_stability) enthält Implementierungen des Dirac-GAN und der Regularisierungsterme $R_1$ und $R_2$.

<!--en:16e9e296-->
2. **Modernisierte GAN-Baseline (R3GAN).** :citet:`Huang.Gokaslan.Kuleshov.ea.2024` kombinieren den Verlust für die relativistische Paarbildung mit dem Strafterm $R_1 + R_2$ und modernen Faltungsblöcken. Sie zeigen, dass eine gut regularisierte Zielfunktion viele der Ad-hoc-Maßnahmen überflüssig macht, mit denen frühere GAN-Architekturen das Training stabilisierten ([offizielles R3GAN-Repository](https://github.com/brownvc/R3GAN)).

<!--en:971361fe-->
3. **Bewertungsmetriken (Präzision, Trefferquote, FID, KID).** :citet:`Kynkaanniemi.Karras.Laine.ea.2019` führen die $k$-NN-Metriken für Präzision und Trefferquote aus :numref:`subsec_gan_precision_recall` ein. Damit lassen sich Bildtreue und Modusabdeckung getrennt messen. Das [StyleGAN3-Repository](https://github.com/NVlabs/stylegan3) :cite:`Karras.Aittala.Laine.ea.2021` enthält Referenzimplementierungen für die Auswertung von FID, KID, Präzision und Trefferquote im großen Maßstab.

<!--en:fda0b759-->
4. **Adversariale Diffusionsdestillation.** DMD2 :cite:`Yin.Gharbi.Park.ea.2024` kombiniert Score-Destillation zum Abgleich der Verteilungen mit einem Diskriminator, der anhand echter Bilder trainiert wird. So kann der Generator als Schüler in einem einzigen Schritt bessere Ergebnisse erzielen als sein Lehrer, ein Diffusionsmodell mit mehreren Schritten.

<!--en:e9b7586f-->
5. **Interaktive Visualisierung und historische Übersichten.** Das [Tutorial der NIPS 2016](https://arxiv.org/abs/1701.00160) :cite:`Goodfellow.2016` und die [ganhacks-Checkliste](https://github.com/soumith/ganhacks) beschreiben frühe Heuristiken für das GAN-Training. [GAN Lab](https://poloclub.github.io/ganlab/) :cite:`Kahng.Thorat.Chau.ea.2018` veranschaulicht die Dynamik von Generator und Diskriminator in zwei Dimensionen direkt im Browser. In einem groß angelegten Benchmark zeigen :citet:`Lucic.Kurach.Michalski.ea.2018`, dass viele adversariale Zielfunktionen (:numref:`sec_gan_objectives`) bei gleichem Budget für die Hyperparameteroptimierung vergleichbare FID-Werte erreichen.

<!--en:9a32726f-->
## Zielfunktionen und Minimax-Dynamik

<!--en:0ad069d1-->
- **Kodierung der Ziellabel im Least-Squares GAN (LSGAN).** Wie :numref:`sec_gan_objectives` zeigt, berechnet der Kleinste-Quadrate-Diskriminator mit 0–1-Kodierung den Wert $\tfrac14 \chi^2(p \,\|\, m)$. :citet:`Mao.Li.Xie.ea.2017` untersuchen allgemeine Ziellabel $b$ (echt), $a$ (generiert) und $c$ (Ziel des Generators). Sie zeigen, dass sich die Zielfunktion des Generators bei optimalem Kritiker auf $\tfrac14 (a + b - 2c)^2 + \tfrac14 (b - a)^2 \chi^2(p \,\|\, m)$ vereinfacht. Das ergibt bis auf eine additive Konstante ein positives Vielfaches derselben $\chi^2$-Divergenz für alle $b \neq a$.

<!--en:66c091ce-->
- **Ausgabeaktivierungen für $f$-GANs.** Die Ausgabe $t$ des Kritikers muss in dem Bereich liegen, in dem die konvex Konjugierte $f^*(t)$ endlich ist (:numref:`sec_gan_objectives`). Dazu wenden :citet:`Nowozin.Cseke.Tomioka.2016` je nach Definitionsbereich eine andere Ausgabeaktivierung $t(v)$ auf die Rohausgabe $v$ des Netzes an. Für die vorwärts gerichtete KL-Divergenz verwenden sie die Identität, für die umgekehrte KL-Divergenz $-e^{-v}$ ($t < 0$) und für die Jensen-Shannon-Divergenz $\log 2 - \mathrm{softplus}(-v)$ ($t < \log 2$).

<!--en:5d940e57-->
- **Allgemeine relativistische Divergenzen.** :citet:`Jolicoeur-Martineau.2020` beweist, dass der kalibrierte Optimalwert der Zielfunktion für die relativistische Paarbildung (:numref:`sec_gan_relativistic`) eine gültige Divergenz definiert. Das gilt für jede konkave Auszahlungsfunktion $\ell$ mit $\ell(0) = 0$, $\ell'(0) \neq 0$ und $\sup_{t > 0} \ell(t) > 0$.

<!--en:83ff3113-->
- **Optimierungsalgorithmen für bilineare Minimax-Probleme.** Beim bilinearen Problem $\min_\theta \max_\psi \theta\psi$ vergrößert der simultane Gradientenabstieg mit Schrittweite $h$ den Abstand zum Ursprung in jedem Schritt um den Faktor $\sqrt{1 + h^2}$. Das *Extragradientenverfahren* :cite:`Korpelevich.1976` wertet den Gradienten an einem vorläufigen Punkt aus, der einen Schritt vorausliegt. Dadurch schrumpft der Abstand um den Faktor $\sqrt{1 - h^2 + h^4} < 1$ für $0 < h < 1$. Der *optimistische Gradientenabstieg* :cite:`Daskalakis.Ilyas.Syrgkanis.ea.2018` nähert diesen Vorgriff mit dem Gradienten des vorigen Schritts an. Negatives Momentum :cite:`Gidel.Askari-Hemmat.Pezeshki.ea.2019` stabilisiert alternierende Updates. Diese Verfahren ändern den Optimierer. Große GANs zur Bilderzeugung ändern dagegen die Zielfunktion durch die Gradientenstrafterme $R_1$ und $R_2$ (:numref:`sec_gan_convergence`).

<!--en:6dbea5a1-->
## Feste und gelernte Diskrepanzen

<!--en:cef17807-->
Lässt sich ein Generator ohne Kritikernetz trainieren, indem er einen festen Abstand zwischen zwei Stichproben direkt minimiert? Generative Moment Matching Networks :cite:`Li.Swersky.Zemel.2015` minimieren die MMD mit festem Kernel (:numref:`sec_gan_objectives`). Die implizite Maximum-Likelihood-Schätzung (IMLE) :cite:`Li.Malik.2018` ordnet jedem echten Beispiel das nächstgelegene generierte Sample zu, um Modusausfall zu verhindern. Sliced-Wasserstein-Generatoren :cite:`Deshpande.Zhang.Schwing.2018` mitteln die eindimensionalen Abstände des optimalen Transports über zufällige Projektionen. Bei natürlichen Bildern erfassen feste Kernel im Pixelraum und zufällige Projektionen wahrnehmbare Unterschiede jedoch nur unzureichend. Deshalb lernt MMD-GAN den Kernel adversarial :cite:`Li.Chang.Cheng.ea.2017,Binkowski.Sutherland.Arbel.ea.2018`.

<!--en:acd8039a-->
## Architekturen, Merkmalsräume und Dateneffizienz

<!--en:cff84a2d-->
- **Entwicklung der Architekturen (2015–2021).** Zwischen DCGAN :cite:`Radford.Metz.Chintala.2015` und R3GAN :cite:`Huang.Gokaslan.Kuleshov.ea.2024` zielten Neuerungen in der Architektur auf stabileres Training und die Erzeugung hochauflösender Bilder. Dazu zählen das schrittweise Vergrößern der Netze :cite:`Karras.Aila.Laine.ea.2017` sowie die spektrale Normalisierung und die Self-Attention in SAGAN und BigGAN :cite:`Miyato.Kataoka.Koyama.ea.2018,Zhang.Goodfellow.Metaxas.ea.2019,Brock.Donahue.Simonyan.2019`. Bei StyleGAN kamen ein Netz zur Abbildung im latenten Raum, die Modulation und Demodulation durch den Stil sowie die aliasfreie Neuabtastung hinzu :cite:`Karras.Laine.Aila.2019,Karras.Laine.Aittala.ea.2020,Karras.Aittala.Laine.ea.2021`.

<!--en:65c9d879-->
- **Merkmalsräume und Fallstricke bei Metriken.** :citet:`Kynkaanniemi.Karras.Aittala.ea.2023` zeigen, dass die Klassengrenzen von ImageNet die Merkmale von Inception-v3 stark prägen. Die selbstüberwacht gelernten Merkmale von DINOv2 stimmen besser damit überein, wie Menschen Bilder beurteilen :cite:`Stein.Cresswell.Hosseinzadeh.ea.2023,Karras.Aittala.Kynkaanniemi.ea.2024`. Aliasing bei der Änderung der Bildgröße kann die FID verfälschen :cite:`Parmar.Zhang.Zhu.2022`. Bei endlicher Stichprobengröße skaliert die Verzerrung der FID mit $O(1/n)$, wobei die Steigung vom Modell abhängt :cite:`Chong.Forsyth.2020`.

<!--en:4d1bda5c-->
- **Vortrainierte Merkmale im Kritiker.** Projected GAN :cite:`Sauer.Chitta.Muller.ea.2021` und StyleGAN-T :cite:`Sauer.Karras.Laine.ea.2023` setzen kleine Diskriminatorköpfe auf vortrainierte Basisnetze für die Bildverarbeitung. Die Gewichte der Basisnetze bleiben fest. Dieser Aufbau beschleunigt die Konvergenz. Zur Bewertung sollte man dann stets einen unabhängigen Merkmalsextraktor verwenden.

<!--en:caecb887-->
- **Verteilung des Rechenaufwands für Gradientenstrafen.** Da $R_1$ und $R_2$ eine Backpropagation zweiter Ordnung durch den Kritiker erfordern, berechnet StyleGAN2 den Strafterm nur einmal alle 16 Minibatches (*intervallweise Regularisierung* (engl. lazy regularization), :citet:`Karras.Laine.Aittala.ea.2020`). Beim Post-Training großer Diffusionsmodelle für Videos lässt sich $R_1$ durch endliche Differenzen bei kleinen gaußschen Störungen der Eingabe annähern :cite:`Lin.Xia.Ren.ea.2025`.

<!--en:42142a37-->
- **Datenaugmentierung ohne Übertragung auf die generierten Bilder.** Bei kleinen Datensätzen lernen Kritiker die Trainingsbilder schnell auswendig (:numref:`sec_dcgan`). Adaptive Discriminator Augmentation (ADA) :cite:`Karras.Aittala.Hellsten.ea.2020` und DiffAugment :cite:`Zhao.Liu.Lin.ea.2020` wenden innerhalb des Gradientenpfads sowohl auf echte als auch auf generierte Bilder invertierbare Transformationen an, *deren Effekte der Generator nicht übernimmt*. So verhindern sie eine Überanpassung des Kritikers, ohne die Zielverteilung zu verfälschen.

<!--en:84f41cf8-->
## Konditionierung, Bildübersetzung und moderne Hybridsysteme

<!--en:2a5e6f9e-->
- **Schichten zur Merkmalsmodulation.** Die kanalweise affine Modulation $a(c) \odot h + b(c)$ aus :numref:`sec_gan_conditional` findet sich in Conditional Batch Normalization :cite:`De-Vries.Strub.Mary.ea.2017`, in FiLM :cite:`Perez.Strub.De-Vries.ea.2018`, in Adaptive Instance Normalization (AdaIN) für den Stiltransfer :cite:`Huang.Belongie.2017` und in der räumlich variablen Modulation von SPADE für die semantische Bilderzeugung :cite:`Park.Liu.Wang.ea.2019`.

<!--en:d13d9eec-->
- **Bild-zu-Bild-Übersetzung mit und ohne Bildpaare.** Pix2pix :cite:`Isola.Zhu.Zhou.ea.2017` kombiniert einen U-Net-Generator mit einem Diskriminator, der einzelne Bildausschnitte bewertet (PatchGAN). Dadurch werden $L_1$-Rekonstruktionen schärfer. CycleGAN :cite:`Zhu.Park.Isola.ea.2017` ermöglicht die Übersetzung zwischen Domänen ohne Bildpaare, indem es Zykluskonsistenz erzwingt ($F(G(x)) \approx x$). Unterscheiden sich die Domänen jedoch in ihrer Entropie, können zykluskonsistente Abbildungen Informationen über das Ausgangsbild steganografisch in hochfrequenten Störungen verbergen :cite:`Chu.Zhmoginov.Sandler.2017`.

<!--en:900beb96-->
- **Tokenizer, Destillation auf wenige Schritte und Vocoder.** Bei visuellen Tokenizern (VQGAN, ViTok) verbessern adversariale Decoderverluste die wahrgenommene Rekonstruktionsqualität auf Kosten des pixelweisen PSNR :cite:`Esser.Rombach.Ommer.2021,Hansen-Estruch.Yan.Chung.ea.2025,Hansen-Estruch.Chen.Ramanujan.ea.2026`. Bei der Diffusionsdestillation setzen LADD :cite:`Sauer.Boesel.Dockhorn.ea.2024` (verwendet in FLUX.1-schnell, :citet:`BlackForestLabs.2024`) und SDXL-Lightning :cite:`Lin.Wang.Yang.2024` Diskriminatoren direkt im latenten Raum oder im Raum der Bildausschnitte ein. MeanFlow :cite:`Geng.Deng.Bai.ea.2025` und Konsistenzmodelle in kontinuierlicher Zeit :cite:`Lu.Song.2025` untersuchen Regressionszielfunktionen für wenige Schritte ohne Kritiker. In der Sprachsynthese erweitert RAF :cite:`Lee.Choi.2026` das Modell BigVGAN :cite:`Lee.Ping.Ginsburg.ea.2023` um die Zielfunktion für die relativistische Paarbildung aus :numref:`sec_gan_relativistic`.

<!--en:d598dcf4-->
- **Likelihood-Modelle als implizite Kritiker (DDO).** Bei der Direct Discriminative Optimization :cite:`Zheng.Chen.Chen.ea.2025` wird ein Likelihood- oder Diffusionsmodell $q_\theta$ im Vergleich zu einer eingefrorenen Referenzkopie $q_{\textrm{ref}}$ feinabgestimmt. Das Verfahren parametrisiert den Kritiker selbst als logarithmiertes Likelihood-Verhältnis $D_\theta(x) = \beta \log(q_\theta(x) / q_{\textrm{ref}}(x))$. Dadurch entfällt das separate Diskriminatornetz (Aufgabe :exref:`ex_adversarial-losses-ddo-plays-log-loss`).

<!--en:78ea4de4-->
[Diskussion](https://d2l.discourse.group/t/36874)
