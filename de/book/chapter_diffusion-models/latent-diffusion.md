<!--en:15ae8a37-->
# Latente Diffusion und Transformer

<!--en:d62ada5c-->
Arbeitet ein Diffusionsmodell direkt im Pixelraum, wertet es das Netz zur Entrauschung bei jedem Sampling-Schritt an jedem Pixel aus. Ein Graustufenbild der Größe $28 \times 28$ enthält nur $784$ Werte, ein RGB-Foto der Größe $512 \times 512$ dagegen $786{,}432$. Wie :numref:`subsec_diffusion-design-space-scales` gezeigt hat, beschreiben die meisten dieser hochfrequenten Dimensionen feine Texturen, die wir nicht wahrnehmen. *Latente Diffusion* :cite:`Rombach.Blattmann.Lorenz.ea.2022` trennt die perzeptuelle Kompression (engl. perceptual compression) von der Erzeugung semantischer Inhalte. Zunächst komprimiert ein Variationsautoencoder jedes Bild zu einem niedrigerdimensionalen räumlichen Gitter aus *latenten Darstellungen* (engl. latents). In diesem latenten Raum benötigt das Diffusionsmodell für Training und Sampling nur einen Bruchteil der FLOPs pro Schritt. Der Decoder läuft für jedes erzeugte Sample nur einmal.

<!--en:7c154fd2-->
Eine zweite Änderung der Architektur ersetzt das konvolutionäre U-Net durch einen *Diffusion Transformer* (DiT) :cite:`Peebles.Xie.2023`. Er verarbeitet das Bild oder das latente Gitter als Sequenz von Patch-Token, die er durch adaptive Schichtnormalisierung (engl. adaptive layer normalization, `adaLN-Zero`) moduliert. In diesem Abschnitt trainieren wir einen KL-regularisierten konvolutionären Autoencoder auf Fashion-MNIST. Auf seinen latenten Darstellungen der Größe $7 \times 7 \times 4$ lernen wir das Geschwindigkeitsfeld mit Flow-Matching. Wir untersuchen den Zielkonflikt zwischen der Verzerrung durch den Decoder und der Modellierbarkeit der latenten Darstellungen :cite:`Dieleman.2025`. Schließlich vergleichen wir U-Net und DiT als Grundarchitekturen bei unterschiedlichem Rechenaufwand.

<!--en:6a9d3aa9-->
## Perzeptuelle Kompression

<!--en:f46e8f27-->
### Die Kosten der Verarbeitung im Pixelraum

<!--en:90d14b59-->
Bei jedem Sampling-Schritt werten wir das Netz auf dem gesamten Bild aus. Eine Faltung wendet ihren Faltungskern an jedem Pixel an. Daher wachsen die Kosten eines U-Net mit der Anzahl der Pixel. Attention vergleicht jede Position mit jeder anderen, ihre Kosten wachsen also quadratisch mit dieser Anzahl. Ein Farbbild der Größe $512 \times 512$ enthält $512 \cdot 512 \cdot 3 = 786{,}432$ Zahlen. Ein Sampler mit fünfzig Schritten greift auf jede davon fünfzigmal zu. Die meisten dieser Zahlen beschreiben Details. Stellen wir ein Bild in einer Fourier-Basis dar, so wächst die Anzahl der Frequenzen bis zu $k$ Perioden pro Bild wie $k^2$. Daher entfallen die meisten Koordinaten auf hohe Frequenzen. In :numref:`subsec_diffusion-design-space-scales` sank die Leistung der Fashion-MNIST-Bilder zwischen einer und vierzehn Perioden pro Bild um drei Größenordnungen. Die vielen Koordinaten für feine Details tragen also wenig zur Leistung des Bildes bei.

<!--en:eab3eae3-->
Mit der Rate-Verzerrungs-Theorie können wir beschreiben, was sich verwerfen lässt. Die Rate-Verzerrungs-Funktion gibt für jede Verzerrung die Mindestzahl an Bits an, die ein Code pro Bild benötigt (:numref:`subsec_mdl-rate-distortion`). Welche fehlenden Bits beim Betrachten auffallen, hängt von der menschlichen Wahrnehmung ab. Der quadratische Fehler beantwortet diese Frage nicht. :citet:`Rombach.Blattmann.Lorenz.ea.2022` stützen ihre Antwort auf eine Beobachtung. Ein Likelihood-Modell für natürliche Bilder verbraucht die meisten Bits am Ende der Kurve mit geringer Verzerrung, und zwar für Details, die niemand wahrnimmt. Diese *perzeptuelle Kompression* überlassen sie einem Autoencoder. Die *semantische Kompression*, also die Modellierung dessen, was das Bild zeigt, übernimmt das Diffusionsmodell. Der Autoencoder von Stable Diffusion bildet ein Bild der Größe $512 \times 512 \times 3$ auf eine latente Darstellung der Größe $64 \times 64 \times 4$ ab. Die Anzahl der Zahlen sinkt um den Faktor 48. Jeder Diffusionsschritt arbeitet auf der latenten Darstellung. Der Decoder läuft einmal pro Sample.

<!--en:147e8a54-->
Was macht eine gute latente Darstellung aus? Sie muss genug Information bewahren, damit der Decoder das Bild wiederherstellen kann. Zugleich muss das Diffusionsmodell sie leicht lernen können. Diese Ziele stehen in Konflikt. Speichert die latente Darstellung jedes Pixel, gelingt die Rekonstruktion perfekt, doch die Darstellung ist genauso schwer zu modellieren wie die Pixel selbst. :citet:`Dieleman.2025` benennt die drei Größen dieses Zielkonflikts. Die *Rate* gibt an, wie viele Zahlen die latente Darstellung enthält, die *Verzerrung* (engl. distortion), wie viel der Decoder verliert, und die *Modellierbarkeit* (engl. modelability), wie gut ein generatives Modell die latenten Darstellungen erfasst. Unser Autoencoder bildet 784 Zahlen auf ein $7 \times 7$-Gitter mit vier Kanälen ab. Das entspricht einer Rate von 196 Zahlen (:numref:`fig_diffusion-latent-pipeline`).

<!--en:77f4f66a-->
![Latente Diffusion. Der Encoder bildet ein Bild auf eine kleinere latente Darstellung ab. Das Diffusionsmodell arbeitet beim Training und Sampling auf latenten Darstellungen. Der Decoder bildet eine gezogene latente Darstellung wieder auf ein Bild ab. Oben der Autoencoder für Fashion-MNIST, von $28 \times 28 \times 1$ (784 Zahlen) auf $7 \times 7 \times 4$ (196). Unten der Autoencoder von Stable Diffusion, von $512 \times 512 \times 3$ (786 432) auf $64 \times 64 \times 4$ (16 384).](../img/mdl-diffusion-latent-pipeline.svg)

<!--en:add4c25c-->
### Der Autoencoder

<!--en:e8fc9287-->
Von den drei Größen ist die Modellierbarkeit am wenigsten anschaulich. Betrachten wir deshalb, welche Anforderungen ein Diffusionsmodell an die latente Darstellung stellt. Erstens braucht es eine bekannte Skala, auf die das Rauschen des Gauß-Pfads abgestimmt ist. Zweitens sollte es keine vereinzelten Ausreißer geben. Drittens sollten benachbarte latente Darstellungen zu ähnlichen Bildern dekodieren, damit auch eine leicht fehlerhafte Darstellung ein plausibles Bild ergibt. Trainieren wir einen Autoencoder allein auf Rekonstruktion, ist keine dieser Eigenschaften garantiert. Er könnte seine Codes mit einer Skala von tausend speichern oder zwei benachbarte Codes als Schuh und als Hemd dekodieren.

<!--en:d40c20e7-->
Der Variationsautoencoder :cite:`Kingma.Welling.2014` bietet für alle drei Anforderungen einen Ansatz. Der Encoder gibt für jede der 196 latenten Zahlen einen Mittelwert $\boldsymbol\mu$ und eine Standardabweichung $\mathbf{s}$ aus. Wir ziehen die latente Darstellung als $\mathbf{z} = \boldsymbol\mu + \mathbf{s} \odot \boldsymbol\xi$ mit standardnormalverteiltem $\boldsymbol\xi$. Der Decoder bildet $\mathbf{z}$ auf eine Rekonstruktion $\hat{\mathbf{x}}$ ab. Zum Rekonstruktionsfehler kommt im Verlust ein Term hinzu, der die Verteilung jeder latenten Zahl an eine Standardnormalverteilung annähert:

<!--en:b35f2e80-->
Die Summe ist die Kullback-Leibler-Divergenz $D_{\mathrm{KL}}\big(\mathcal{N}(\boldsymbol\mu, \mathrm{diag}(\mathbf{s}^2))
\,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\big)$, der entsprechende Term der unteren Evidenzschranke (:numref:`sec_mdl-latent-em-elbo`). Ihr Gewicht ist $\beta$ (nicht zu verwechseln mit $\beta_k$ aus dem Rauschplan). Im Hinblick auf die drei Anforderungen bewirkt der Verlust zweierlei. Er zieht jedes $\mu_j$ gegen null und jedes $s_j$ gegen eins. Das legt die Skala fest und bestraft Ausreißer. Ziehen wir $\mathbf{z}$ beim Training mit Rauschen der Stärke $\mathbf{s}$, sieht der Decoder Codes, die um etwa $\mathbf{s}$ schwanken. Er lernt, sie ähnlich zu dekodieren, wie es die dritte Anforderung verlangt.

<!--en:e4dd6f0c-->
Wie stark diese Effekte ausfallen, hängt von $\beta$ ab. Mit unserer Wahl $\beta = 10^{-4}$ dominiert weiterhin die Rekonstruktion. Bei diesem Gewicht legt der KL-Term die Skala fest und verhindert, dass das Rauschen des Encoders auf null fällt. Er ist jedoch viel zu schwach, um eine ähnliche Dekodierung benachbarter latenter Darstellungen zu erzwingen. Ob das ins Gewicht fällt, ist zunächst offen. Wir messen es weiter unten.

<!--en:64af9974-->
Die Autoencoder großer Systeme nutzen weitere Trainingssignale. Meist trainiert man sie mit einem Kritiker, der Bild-Patches bewertet, und mit einem perzeptuellen Verlust. Denn der quadratische Fehler mittelt plausible Texturen zu einem unscharfen Bild (:numref:`sec_gan_beyond`). Beim neueren großen Autoencoder ViTok-v2 ersetzte ein fester perzeptueller Verlust den Kritiker, als das Modell vergrößert wurde :cite:`Hansen-Estruch.Chen.Ramanujan.ea.2026`. Unser Autoencoder braucht beides nicht. Er behält ein Viertel der Zahlen eines kleinen Graustufenbildes mit schwarzem Hintergrund bei, der Autoencoder von Stable Diffusion dagegen nur ein Achtundvierzigstel der Zahlen eines Fotos. Andere Autoencoder quantisieren die latente Darstellung zu diskreten Token :cite:`Esser.Rombach.Ommer.2021,Chang.Zhang.Jiang.ea.2022`. So entsteht ein beschränkter Code, dessen Skala keiner Kalibrierung bedarf und der sich als Eingabe für die maskierte Diffusion (:numref:`sec_diffusion-discrete`) eignet.

<!--en:e00a796f-->
Unser Autoencoder verwendet aufeinanderfolgende Residualblöcke des U-Net mit den Breiten 64 und 128, ohne Einbettung. Der Encoder halbiert die Auflösung zweimal. So entsteht ein $7 \times 7$-Gitter mit einem Mittelwert und einer logarithmischen Varianz pro latentem Kanal. Der Decoder ist spiegelbildlich aufgebaut.

<!--en:2e80b005-->
Wir trainieren den Autoencoder über 20 000 Updates mit `d2l.train_regression` aus :numref:`sec_diffusion-denoising`. Die Funktion minimiert einen Verlust mit Adam und gibt das trainierte Modell, einen gleitenden Mittelwert seiner Gewichte und die Verlustwerte zurück. Die Lernrate sinkt entlang einer Kosinuskurve auf null. Daher brauchen wir die Gewichte am Ende nicht zu mitteln. Vorab ein Hinweis zu den Kosten. In diesem Abschnitt trainieren wir zwei Autoencoder und vierzehn Diffusionsmodelle mit jeweils 20 000 Updates. Auf einer einzelnen RTX 4090 läuft das gesamte Notebook etwa anderthalb Stunden, auf einer CPU erheblich länger.

<!--en:1cc1ca62-->
Die erste Zeile zeigt acht Testbilder, die zweite die Ausgabe des Decoders für die zugehörigen latenten Mittelwerte.

<!--en:97f0e3e5-->
Die Rekonstruktionen lassen sich kaum von den Originalen unterscheiden. Wir beurteilen sie mit dem Fréchet-Abstand (engl. Fréchet distance, FD) aus :numref:`sec_diffusion-images`. Ein kleiner Klassifikator, `d2l.FeatureCNN`, extrahiert Merkmale aus zwei Bildmengen. An jede Merkmalsmenge passen wir eine Gauß-Verteilung an und messen den Abstand der beiden Verteilungen. Auch zwei Mengen echter Bilder unterscheiden sich durch Zufall. Daher geben wir als *Referenzwert* den FD zwischen zwei disjunkten Mengen von je 5000 Testbildern aus. Alle FD-Werte in diesem Abschnitt beruhen auf 5000 Bildern. Bei 1000 Bildern liegt der Referenzwert etwa fünfmal so hoch (:numref:`subsec_diffusion-fd-reference`).

<!--en:1e0ba686-->
Mit `fd` kodieren und dekodieren wir nun die übrigen 5000 Testbilder `X_test[5000:]` in Batches. Dabei messen wir sowohl den quadratischen Rekonstruktionsfehler pro Pixel als auch den Fréchet-Abstand der Rekonstruktionen `fd_rec`:

<!--en:03882441-->
Die Rekonstruktionen erreichen einen FD von 0.6 bis 0.8 bei einem Referenzwert von 0.17. Der quadratische Fehler beträgt etwa 0.004 pro Pixel. Dieser Abstand misst die Verzerrung durch den Autoencoder. Zugleich entspricht er dem Wert eines perfekten latenten Modells. Zöge das Diffusionsmodell die latenten Darstellungen genau so, wie der Encoder sie erzeugt, würde der Decoder sie bis auf das geringe Rauschen des Encoders in Rekonstruktionen überführen. An diesem Wert messen wir im Folgenden jedes latente Modell. Weshalb er als Referenz dient und keine untere Schranke ist, zeigt der spätere Versuch mit gestörten Codes.

<!--en:960b0453-->
Wir kodieren den Trainingsdatensatz einmal. Dann geben wir die Standardabweichung der latenten Darstellungen, den Mittelwert von $\mathbf{s}$ aus dem Encoder und den Exzess (engl. excess kurtosis) aller latenten Werte aus. Bei einer Normalverteilung ist der Exzess null. Er ist groß, wenn einige wenige Werte weit außen liegen.

<!--en:d619b862-->
Die Standardabweichung der latenten Darstellungen liegt knapp unter eins. Wir multiplizieren sie daher mit einem Faktor $c$ von etwa 1.05. Damit haben sie wie das Rauschen des Pfads Einheitsvarianz. Der Exzess liegt nahe null oder ist negativ. Es gibt also keine schweren Verteilungsränder. Zwei der drei Anforderungen sind erfüllt. Die Standardabweichung des Encoders beträgt 0.02 bis 0.03. Beim Training sah der Decoder also jeden Code mit Rauschen von etwa 3 Prozent der Streuung der latenten Darstellungen, fast nie mit mehr als einem Zehntel. Die dritte Anforderung erfüllt der schwache KL-Term nicht. Er stellt nicht sicher, dass leicht fehlerhafte latente Darstellungen plausible Bilder ergeben.

<!--en:8aa0cf44-->
## Diffusion im latenten Raum

<!--en:5c74e200-->
### Dieselbe Regression auf latenten Darstellungen

<!--en:a0951d82-->
Sobald der Autoencoder feststeht, ändern sich nur die Daten. Das macht den Ansatz attraktiv. Alles, was wir in den vorigen Abschnitten erarbeitet haben, lässt sich übertragen. Für jeden Batch ziehen wir die latenten Darstellungen von 128 Trainingsbildern, $\mathbf{z}_0 =
c\,(\boldsymbol\mu + \mathbf{s} \odot \boldsymbol\xi)$, jeweils neu aus der Verteilung des Encoders, wie bei :citet:`Rombach.Blattmann.Lorenz.ea.2022`. Wir trainieren auf dem linearen Pfad aus :numref:`sec_diffusion-gaussian-paths`. Dazu bilden wir $\mathbf{z}_t = (1 - t)\,\mathbf{z}_0 + t\,\boldsymbol\epsilon$ und verwenden die Geschwindigkeit $\boldsymbol\epsilon - \mathbf{z}_0$ als Zielgröße für die Regression der Netzausgabe.

<!--en:b9148a39-->
Der Skalenfaktor ist wichtig, weil der Pfad die Daten mit Rauschen der Varianz eins mischt. Latente Darstellungen mit einer Standardabweichung von fünf würden die Log-SNR zu jedem Zeitpunkt um $\log 25$ verschieben (Übung :exref:`ex_latent-scale-factor`). Deshalb multipliziert Stable Diffusion seine latenten Darstellungen mit 0.18215. Als Netz dient das U-Net aus :numref:`sec_diffusion-images` mit je vier Eingabe- und Ausgabekanälen. Für Bilder und latente Darstellungen genügt dieselbe Funktion `loss_fn`.

<!--en:72605a81-->
Als Referenz trainieren wir dasselbe U-Net auf Pixeln. Beide Trainingsläufe umfassen 20 000 Updates, doppelt so viele wie in :numref:`sec_diffusion-samplers`. Denn die Transformer weiter unten benötigen längere Trainingsläufe, und das Pixelmodell dient auch ihnen als Referenz.

<!--en:cda5c60c-->
`sample` erzeugt aus fest vorgegebenem Rauschen 5000 Samples mit 50 Euler-Schritten von `d2l.ode_sample` (:numref:`sec_diffusion-integration`). Die Funktion `decode` macht die Skalierung rückgängig und dekodiert die latenten Darstellungen. Die erste Zeile unten zeigt Samples des Pixelmodells, die zweite dekodierte Samples des latenten Modells.

<!--en:ad554a23-->
In beiden Zeilen sind Kleidungsstücke zu erkennen. Das latente Modell erreicht einen FD von etwa 2 (2.0 bis 2.2), das Pixelmodell etwa 1.5. Wie alle Modelle in diesem Abschnitt arbeiten beide ohne Konditionierung. Ihre Werte lassen sich daher nicht mit dem klassenbedingten Wert von 0.8 aus :numref:`sec_diffusion-images` vergleichen.

<!--en:f3f6c787-->
Wie viel Rechenarbeit spart das latente Modell? `train_step_gflops` zählt die Gleitkommaoperationen eines Trainingsschritts in Milliarden (GFLOP). Ein Schritt umfasst den Vorwärts- und den Rückwärtsdurchlauf für 128 Eingaben.

<!--en:139c4f5e-->
Wir werten `train_step_gflops` für das latente U-Net mit vier Kanälen (`latent_shape`) und für das Pixel-U-Net mit einem Kanal (`image_shape`) aus. So vergleichen wir die Rechenkosten pro Schritt direkt:

<!--en:5aa239c7-->
Das latente U-Net benötigt etwa ein Dutzend GFLOP pro Schritt, das Pixel-U-Net etwa

<!--en:b90c7543-->
200. Jede Auswertung des latenten Netzes kostet weniger als ein Zehntel der Rechenoperationen. Das gilt sowohl im Training als auch bei jedem Sampling-Schritt.

<!--en:b4719f82-->
### Fehlertoleranz des Decoders und Modellfehler

<!--en:b66ef335-->
Das latente Modell rechnet also erheblich günstiger, schneidet aber etwas schlechter ab. Sein FD ist etwa 1.4-mal so groß wie der des Pixelmodells und etwa dreimal so groß wie der der Rekonstruktionen. In Dielemans Begriffen geht eine Rate von 196 Zahlen mit einer Verzerrung von 0.6 bis 0.8 einher. Der Abstand von dort bis zum FD der Samples misst die Modellierbarkeit. Woher kommt diese Lücke? Zwei Ursachen liegen nahe. Möglicherweise zieht das latente Modell andere Darstellungen, als der Encoder erzeugt. Dann liegt der Fehler beim Modell selbst. Vielleicht weichen die gezogenen Darstellungen aber nur geringfügig ab. Der Decoder könnte daraus schlechte Bilder erzeugen, weil er beim Training nur Codes mit Abweichungen von etwa 3 Prozent der Streuung kennengelernt hat. Der schwache KL-Term macht diese zweite Erklärung plausibel.

<!--en:aa30e078-->
Um die Ursachen zu unterscheiden, bauen wir einen Decoder, der fehlerhafte Codes toleriert. Wir prüfen seine Toleranz und trainieren anschließend latente Modelle auf den zugehörigen latenten Darstellungen. Liegt die Lücke an mangelnder Toleranz, sollten die Samples im FD näher an die Rekonstruktionen rücken. Dafür trainieren wir einen zweiten Autoencoder mit zehnfachem KL-Gewicht. Zum Test dekodieren wir absichtlich gestörte Darstellungen. Zu den latenten Mittelwerten der anderen 5000 Testbilder addieren wir gaußsches Rauschen, dessen Stärke dem $\sigma$-Fachen der Standardabweichung der latenten Darstellungen entspricht. Überlegen wir vorab, was zu erwarten ist. Der zweite Encoder sollte seine Codes stärker verrauschen. Sein Decoder sollte daher größere Fehler tolerieren und schlechter rekonstruieren.

<!--en:b2f1ae2c-->
`perturbed_decodes` gibt die Standardabweichung des Encoders in Einheiten der Standardabweichung der latenten Darstellungen aus, außerdem den Rekonstruktionsfehler und den FD der Rekonstruktionen. Für jedes $\sigma$ gibt die Funktion die mittlere quadratische Änderung des dekodierten Bildes und den zugehörigen FD aus.

<!--en:41722e85-->
Beide Vorhersagen treffen zu. Mit dem zehnfachen Gewicht verrauscht der Encoder jeden Code um etwa ein Fünftel der Streuung der latenten Darstellungen, also siebenmal so stark wie zuvor. Der FD seiner Rekonstruktionen verschlechtert sich um etwa 0.3. Bei einer Rauschstärke von der halben Streuung steigt der FD des schwach regularisierten Decoders sprunghaft über 3. Beim toleranten Decoder bleibt er nahe 1.3. Dabei ändern sich die dekodierten Bilder in beiden Fällen ähnlich stark, um 0.053 bis 0.055 pro Pixel gegenüber 0.061 bis 0.062. Der stärkere KL-Term hat dem Decoder nicht beigebracht, einen fehlerhaften Code zu ignorieren. Er hat gelernt, ihn als anderes, aber plausibles Kleidungsstück zu dekodieren. Genau das honoriert der Fréchet-Abstand.

<!--en:9578d355-->
Die Ausgabe enthält auch eine kleine Überraschung. Bei beiden Autoencodern senkt etwas Rauschen den FD unter den Wert der Rekonstruktionen, obwohl sich die dekodierten Bilder von den Originalen entfernen. Das gilt bei $\sigma = 0.1$ für beide Modelle, und bei $\sigma = 0.3$ sinkt der Wert für den toleranten Decoder weiter. Vermutlich ergibt das Dekodieren des Mittelwerts jedes Codes ein etwas zu glattes Bild, eine Mittelung über die Codes aus dem Training des Decoders. Etwas Rauschen könnte einen Teil der Vielfalt echter Bilder wiederherstellen. Keine der Codezellen hier prüft diese Erklärung. In jedem Fall klären die Zahlen den entscheidenden Punkt. Der Rekonstruktions-FD dient als Referenz. Ein latentes Modell kann diesen Wert unterschreiten.

<!--en:bae8cb07-->
Liefert der tolerantere Decoder bessere Samples? `latent_unet_fd` wendet das obige Verfahren auf die latenten Darstellungen eines gegebenen Autoencoders an. Die Funktion trainiert ein latentes U-Net über 20 000 Updates, erzeugt Samples mit 50 Euler-Schritten und gibt den FD der dekodierten Samples zurück. Wir trainieren pro Autoencoder zwei U-Nets mit zwei verschiedenen Zufallsstartwerten.

<!--en:54d1facf-->
Nein. Jedes latente U-Net mit zehnfachem KL-Gewicht schneidet schlechter ab als jedes mit geringem Gewicht. Der Unterschied beträgt etwa einen halben Punkt (0.4 bis 0.8). Er ist damit mindestens zweieinhalbmal so groß wie die Streuung zwischen den beiden Zufallsstartwerten bei jedem der beiden Gewichte. Beim Autoencoder mit zehnfachem Gewicht liegt der FD der Samples auch weiter über dem seiner eigenen Rekonstruktionen. Läge die Ursache in mangelnder Toleranz, hätte der tolerante Decoder die Lücke verkleinert. Das geschah in keinem Lauf. Der Versuch mit gestörten Codes liefert einen zweiten Hinweis. Bei einer Rauschstärke von der halben Streuung der latenten Darstellungen erreicht der tolerante Decoder einen FD von nur etwa 1.3. Die auf seinen Darstellungen trainierten latenten Modelle kommen dagegen auf 2.6 bis 2.8. Ihre Fehler gehen also über kleine zufällige Störungen der richtigen latenten Darstellungen hinaus. Entweder sind sie größer oder sie sind anderer Art. Bei unserer Modellgröße geht die Lücke auf den Fehler des latenten Modells selbst zurück.

<!--en:91ed84dd-->
Dennoch spielt der Autoencoder eine Rolle. Mit demselben Trainingsverfahren schnitt dasselbe U-Net auf den latenten Darstellungen des einen Encoders deutlich schlechter ab als auf denen des anderen. Der Encoder bestimmt also, wie schwer die Aufgabe des latenten Modells ist. Genau das meint Dieleman mit Modellierbarkeit. Sie ist eine Eigenschaft des Autoencoders, die nur ein trainiertes Diffusionsmodell messen kann. Die Wahl des KL-Gewichts hat also Folgen. Hier ist das kleine Gewicht die richtige Wahl, denn der stärkere KL-Term verschlechterte sowohl die Rekonstruktionen als auch die Samples. In Übung :exref:`ex_latent-bottleneck` untersuchen Sie noch kleinere und noch größere Gewichte.

<!--en:cb29a3e1-->
Wann lässt sich eine latente Darstellung also leicht modellieren? Arbeiten seit 2024 deuten darauf hin, dass der Inhalt der Darstellung mehr zählt als das KL-Gewicht. REPA :cite:`Yu.Kwak.Jang.ea.2025` ergänzt den Verlust um einen Term, der die versteckten Zustände des Diffusionsmodells an den Merkmalen eines vortrainierten Bild-Encoders wie DINOv2 ausrichtet. So erreicht das Modell eine gegebene Sample-Qualität in einem Bruchteil der Trainingszeit. Repräsentationsautoencoder :cite:`Zheng.Ma.Tong.ea.2026` gehen weiter und verzichten ganz auf den KL-regularisierten Encoder. Ein eingefrorener vortrainierter Encoder liefert die latente Darstellung. Nur der Decoder lernt, diese Abbildung umzukehren. Eine solche Darstellung hat weit mehr Kanäle als eine auf Kompression ausgelegte. Nach unserer bisherigen Überlegung wäre das ein Nachteil. Dennoch lernt ein Diffusionsmodell sie leichter. Benachbarte Punkte in diesem latenten Raum stehen für ähnliche Objekte statt für ähnliche Pixel. Offenbar zählt die semantische Ordnung mehr als eine bloß kleine und glatte Darstellung.

<!--en:04c835a6-->
Bei $28 \times 28$ lohnt sich dieser Tausch kaum. Das Pixelmodell ist ohnehin kostengünstig. Der Autoencoder benötigt einen eigenen Trainingslauf, und die Samples des latenten Modells schneiden schlechter ab.

<!--en:cde99bc9-->
## Der Transformer als Grundarchitektur

<!--en:db77d44f-->
Das U-Net übernimmt die Annahmen eines konvolutionären Netzes. Benachbarte Pixel sind am wichtigsten, und dieselben Merkmale können überall auftreten. Der Vision Transformer aus :numref:`sec_vision-transformer` kommt mit weniger solchen Annahmen aus. Bei der Bildklassifikation bleibt er hinter konvolutionären Netzen zurück, wenn er mit wenig Daten trainiert wird. Nach einem Pre-Training mit sehr vielen Daten übertrifft er sie. :citet:`Peebles.Xie.2023` untersuchten, ob das auch für Diffusion gilt.

<!--en:6a7dc2cc-->
### Patches und Token

<!--en:932a4099-->
Der DiT liest seine Eingabe wie der Vision Transformer. Er zerlegt das Bild in Patches der Größe $p \times p$ und bildet jeden Patch linear auf einen Vektor der Breite $d$ ab. Dies leistet eine Faltung mit Kerngröße und Stride $p$. Anschließend addiert er eine gelernte Positionseinbettung und verarbeitet die Token mit Transformer-Blöcken mit vorangestellter Normalisierung (:numref:`sec_transformer-block`). Bei $p = 4$ entstehen aus einem Bild der Größe $28 \times 28$ insgesamt 49 Token mit je 16 Pixeln. Für unsere latenten Darstellungen mit $p = 1$ entspricht jede der 49 Gitterpositionen einem Token aus vier Zahlen.

<!--en:7e320bf6-->
Die Ausgabe unterscheidet sich von der eines Klassifikators. Ein Entrauscher gibt eine Vorhersage in der Form seiner Eingabe zurück. Daher bildet der DiT jedes Token mit einer abschließenden linearen Schicht auf die Werte seines Patches ab und setzt die Patches wieder zusammen. Jedes Token sagt die Geschwindigkeit seines eigenen Patches voraus. Über Attention fließen Informationen aus allen anderen Patches ein. Bei einem Token pro Pixel hätte das Bild 16-mal so viele Token, und die Attention würde 256-mal so viel Rechenarbeit kosten.

<!--en:e09b131b-->
### Adaptive Schichtnormalisierung

<!--en:750ea5db-->
Der Transformer benötigt auch die Zeit $t$ und gegebenenfalls ein Label. Das U-Net aus :numref:`sec_diffusion-images` löst diese Aufgabe mit einer gelernten Projektion der eingebetteten Zeit und des eingebetteten Labels. Es addiert die Projektion zu den Merkmalskarten jedes Blocks, eine Verschiebung pro Kanal. Die adaptive Schichtnormalisierung behält die Verschiebung bei und ergänzt einen Skalenfaktor und ein Gate.

<!--en:fa367693-->
Die Schichtnormalisierung standardisiert jedes Token und wendet eine gelernte Skalierung und Verschiebung an. Bei der *adaptiven Schichtnormalisierung* ergeben sich die Skalierung $\mathbf{a}$ und die Verschiebung $\mathbf{b}$ aus der Bedingung $\mathbf{c}$, der Summe aus Zeit- und Labeleinbettung. Die Bedingung bestimmt außerdem ein *Gate* $\mathbf{g}$ für die Ausgabe jeder Teilschicht. Ein Block berechnet

<!--en:44a699ba-->
Dabei hat $\mathrm{LN}$ weder eine eigene Skalierung noch eine eigene Verschiebung. Eine einzige lineare Schicht bildet $\mathrm{SiLU}(\mathbf{c})$ auf die sechs Vektoren ab, die für alle Token eines Bildes gelten. So genügt pro Bild ein kleines Matrixprodukt, um jedes Merkmal jeder Teilschicht je nach Rauschpegel unterschiedlich stark zu gewichten.

<!--en:74ae89a3-->
Die lineare Schicht ist mit null initialisiert, daher der Name *AdaLN-Zero*. Anfangs sind damit alle Gates null, jeder Block berechnet die Identität, und auch die mit null initialisierte Ausgabeschicht gibt null aus. Zu Beginn des Trainings sagt das Netz also überall die Geschwindigkeit null voraus. Jeder Block wird wirksam, sobald seine Gates wachsen (:numref:`fig_diffusion-dit-block`). Das wichtigste Detail des Blocks ist diese Initialisierung der Gates mit null. :citet:`Peebles.Xie.2023` verglichen den Entwurf mit drei Alternativen. Sie hängten die Bedingung als zusätzliche Token an, banden sie über Cross-Attention ein oder nutzten adaptive Schichtnormalisierung ohne Gates. Von den vier Varianten lieferte AdaLN-Zero die besten Samples. Übung :exref:`ex_latent-zero-gate` zeigt, wie sich der Start bei null auf die ersten zweitausend Updates auswirkt.

<!--en:8fce95cf-->
Eine Zeit oder ein Label liefert einen Vektor pro Bild. Dazu passt die adaptive Schichtnormalisierung. Eine Bildbeschreibung hingegen liefert eine Sequenz von Vektoren, einen pro Token aus einem eingefrorenen Text-Encoder. Dafür bietet sich Cross-Attention an. Bei der *Cross-Attention* liefern die $m$ Bildtoken die Abfragen und die $n$ Token der Bildbeschreibung die Schlüssel und Werte. Die Ausgabe hat daher eine Zeile pro Bildtoken und behält die Form des Bildes bei :cite:`Rombach.Blattmann.Lorenz.ea.2022`. Der multimodale DiT von :citet:`Esser.Kulal.Blattmann.ea.2024` fasst Bildtoken und Token der Beschreibung stattdessen zu einer Sequenz zusammen. Lassen wir die Bildbeschreibung beim Training zufällig weg, dient die leere Beschreibung als Null-Label für die Classifier-Free Guidance (:numref:`subsec_diffusion-images-guided`).

<!--en:b5a2fb8a-->
![Der Diffusion Transformer. Links werden Patches der Größe $4 \times 4$ zu 49 Token mit gelernten Positionen. In der Mitte ein DiT-Block. Die Bedingung $\mathbf{c}$, die Summe aus Zeit- und Labeleinbettung, bestimmt für jede Teilschicht Verschiebung, Skalierung und Gate. Die Gates starten bei null. Rechts bilden eine abschließende adaptive Schichtnormalisierung und eine mit null initialisierte lineare Schicht jedes Token auf seinen Patch ab.](../img/mdl-diffusion-dit-block.svg)

<!--en:1a2041f2-->
`DiTBlock` baut auf dem `ViTBlock` aus :numref:`sec_vision-transformer` auf und verwendet `ada` als lineare Abbildung.

<!--en:acf5df53-->
`DiT` ergänzt die Patch-Einbettung, die gelernten Positionen, die Zeit- und Labeleinbettungen sowie die abschließende adaptive Schichtnormalisierung und die lineare Ausgabeschicht.

<!--en:d3e06193-->
Wir bauen einen DiT der Breite 256 und Tiefe 6 mit vier Köpfen und geben ihm acht Bilder mit den zugehörigen Zeiten.

<!--en:f84afe55-->
Wie erwartet ist die Ausgabe des untrainierten Netzes exakt null. Der DiT hat 7.39 Millionen Parameter, sechsmal so viele wie das U-Net. Dennoch zeigt der folgende Skalierungsversuch, dass beide pro Schritt etwa gleich viele Rechenoperationen benötigen. Der Unterschied liegt darin, wie sie ihre Gewichte verwenden. Eine Faltung wendet wenige Gewichte an vielen Positionen an, der DiT dagegen viele Gewichte auf nur 49 Token.

<!--en:aedce44e-->
### Skalierung mit dem Rechenaufwand

<!--en:d952acb2-->
Ist der Transformer besser als das U-Net? Für eine einzelne Modellgröße lässt sich das nicht allgemein beantworten, denn bei beiden Architekturen schlägt ein größeres Netz ein kleineres. Entscheidend ist, wie stark sich die jeweilige Architektur mit zunehmender Größe verbessert. :citet:`Peebles.Xie.2023` trainierten DiTs in vier Größen mit je drei Patch-Größen. Bei fester Anzahl an Trainingsschritten sank der FID stetig, wenn die Zahl der Rechenoperationen pro Vorwärtsdurchlauf stieg. Dabei spielte es keine Rolle, ob das Netz größer wurde oder die Patches kleiner. Ihr größtes Modell erreichte nach sieben Millionen Schritten einen neuen FID-Bestwert für die klassenbedingte Bilderzeugung auf ImageNet bei $256 \times 256$.

<!--en:d503c5f7-->
Wir wiederholen die Messung in unserem Größenbereich für beide Grundarchitekturen. Bei den beiden DiTs erhöhen wir Breite und Tiefe gemeinsam. Wie bei den meisten Modellen der Studie entfallen 64 Dimensionen auf jeden Kopf. DiT-S hat Breite 128, Tiefe 4 und zwei Köpfe. DiT-M ist das Modell von oben. Bei den beiden U-Nets verdoppeln wir die Kanalzahl von `ch=32` auf `ch=64` wie beim obigen Pixelmodell. Wir trainieren jedes Modell auf Pixeln mit demselben Verlust und demselben Verfahren über 20 000 Updates, jeweils zweimal mit unterschiedlichen Zufallsstartwerten. Als Maß für die Größe verwenden wir die FLOPs eines Trainingsschritts.

<!--en:c191b609-->
Für jedes Modell tragen wir den mittleren FD der beiden Läufe gegen die FLOPs pro Schritt auf. Beide Achsen sind logarithmisch.

<!--en:d99ede63-->
Jeder der folgenden Unterschiede ist mindestens doppelt so groß wie die Streuung zwischen den beiden Läufen jedes beteiligten Modells. Drei Aussagen lassen sich daher sicher treffen. Die erste überrascht. DiT-S hat mit 1.26 gegenüber 1.20 Millionen etwa gleich viele Parameter wie das U-Net mit `ch=64`, aber einen dreimal so hohen FD. Selbst DiT-M bleibt mit sechsmal so vielen Parametern hinter diesem U-Net zurück. Bei Kleidungsbildern der Größe $28 \times 28$ bringen die in der Faltung verankerten Annahmen also mehr als eine Versechsfachung der Parameterzahl.

<!--en:c6692f6d-->
Auch gemessen am Rechenaufwand liegt das U-Net bei beiden Größen vorn, wenn auch weniger deutlich. Bei den kleinen Modellen erreicht das U-Net mit `ch=32` bei etwa 50 GFLOP pro Schritt einen FD von etwa 2.5. DiT-S benötigt ein Drittel weniger Rechenoperationen und erreicht 4.5 bis 4.9. Bei knapp 200 GFLOP erreicht das U-Net etwa 1.5 und DiT-M 1.7 bis 1.9. Der Transformer gewinnt jedoch stärker hinzu, wenn er wächst. Beim DiT sinkt der FD mit dem 5.8-Fachen des Rechenaufwands um den Faktor 2.5 bis 2.8. Beim U-Net sinkt er mit dem Vierfachen des Rechenaufwands um etwa den Faktor 1.65. Die Gerade des Transformers fällt im Diagramm also steiler ab.

<!--en:3ceb1ea8-->
Die folgenden Zeilen zeigen die ersten acht Samples von DiT-S, DiT-M und den beiden U-Nets. Alle Modelle starten mit demselben Rauschen.

<!--en:4bc12576-->
In den meisten Spalten erzeugen alle vier Modelle dasselbe Kleidungsstück. Jedes Modell integriert eine gewöhnliche Differentialgleichung (ODE), die einem Rauschbild genau ein Bild zuordnet. In ein oder zwei Spalten ändert sich das Kleidungsstück selbst. An diesen acht Spalten lässt sich kaum ablesen, dass die oberste Zeile einen dreimal so hohen FD erreicht wie die unterste. Der Fréchet-Abstand vergleicht Tausende von Samples anhand der Merkmale eines Klassifikators. Wir sehen nur acht.

<!--en:ab886351-->
Unsere vier Messpunkte reichen nicht aus, um den Befund zu belegen, der den Transformer zur Grundarchitektur für große Modelle gemacht hat. Er verbessert sich über mehrere Größenordnungen des Rechenaufwands hinweg, und dazu genügt es, Breite und Tiefe zu erhöhen. Diesen Trend haben Peebles und Xie gezeigt. Unser Diagramm zeigt den steileren Abfall, aber keinen Schnittpunkt. Für den Transformer spricht außerdem, dass seine Berechnungen als große Matrixprodukte ablaufen. Dadurch lassen sich die Verfahren zum Training großer Sprachmodelle übertragen.

<!--en:84e1dd3c-->
## Wahl der Darstellung und der Grundarchitektur

<!--en:b4a2cb96-->
Ein letzter Versuch verbindet die beiden Teile dieses Abschnitts. Bringt der Transformer im latenten Raum einen Vorteil? Wir trainieren DiT-M auf den latenten Darstellungen mit Patch-Größe 1. So verarbeitet er wie im Pixelraum 49 Token und benötigt pro Schritt gleich viele Rechenoperationen wie dort, zwölf- bis siebzehnmal so viele wie das latente U-Net.

<!--en:9f37d6f5-->
Der latente DiT-M erreicht einen FD von etwa 2 (1.8 bis 2.2). Das entspricht ungefähr den latenten U-Nets (1.9 bis 2.2) und liegt wie bei diesen etwa beim Dreifachen des Rekonstruktions-FD. Umgekehrt hilft die latente Darstellung hier auch dem Transformer nicht. Bei gleichem Rechenaufwand ist der latente DiT-M nicht besser als der Pixel-DiT-M. Das beste Pixelmodell übertrifft jedes latente Modell dieses Abschnitts.

<!--en:ce7fc6ea-->
Wann rechtfertigt die latente Darstellung also den Autoencoder? Die Einsparung wächst mit der Pixelzahl. Die Nachteile, nämlich die Verzerrung durch den Autoencoder und die schwierigere Modellierung, wachsen nicht mit. Bei $512 \times 512$ verarbeitet ein U-Net auf der latenten Darstellung von Stable Diffusion um den Faktor 64 weniger Positionen. Der Rechenaufwand sinkt etwa um denselben Faktor. Unser Pixel-U-Net benötigte bei sechzehnmal so vielen Positionen 13- bis 18-mal so viele Rechenoperationen wie das latente U-Net. Die Einsparung fällt bei jedem Trainingsupdate und bei jedem der fünfzig Sampling-Schritte an. Bei dieser Bildgröße wiegt sie die Kosten des Autoencoders auf.

<!--en:8626af46-->
Für einen Transformer sieht die Rechnung anders aus. Mit Patch-Größe 2 verarbeitet ein DiT auf der latenten Darstellung der Größe $64 \times 64 \times 4$ genau 1024 Token. Mit Patch-Größe 16 auf Pixeln sind es ebenfalls 1024 Token bei fast gleichen Kosten (Übung :exref:`ex_latent-compression-arithmetic`). Der Autoencoder spart dem Transformer keine Token. Er ersetzt eine lineare Einbettung von $16 \cdot 16 \cdot 3 =
768$ rohen Pixelwerten durch eine gelernte nichtlineare Einbettung. Eine lineare Einbettung so vieler Werte kommt mit Rauschen schlecht zurecht. Das Rauschen füllt alle 768 Dimensionen, während das Bild nahe einer Menge wesentlich niedrigerer Dimension liegt. Zwei neuere Entwürfe für Pixel-Transformer greifen dieses Problem auf. :citet:`Li.He.2026` behalten die großen Patches bei und sagen das rauschfreie Bild anstelle des Rauschens oder der Geschwindigkeit voraus (Übung :exref:`ex_latent-large-patches`). Sanduhr-Transformer :cite:`Crowson.Baumann.Birch.ea.2024` übernehmen die Struktur des U-Net mit mehreren Auflösungen. Auf den feinen Stufen verwenden sie nur lokale Attention, sodass ihre Kosten lediglich linear mit der Pixelzahl wachsen.

<!--en:00e49096-->
Für niedrig aufgelöste Bilder empfiehlt sich also Diffusion mit einem U-Net direkt auf den Pixeln, sofern das Rechenbudget dafür ausreicht. Für hochaufgelöste Bilder empfiehlt sich Diffusion auf den latenten Darstellungen eines schwach KL-regularisierten Autoencoders. Er ist ein zentraler Teil der generativen Pipeline, denn er bestimmt sowohl die Verzerrung der Rekonstruktionen als auch die Modellierbarkeit der latenten Darstellungen. Für die Skalierung auf große Rechenbudgets bietet sich ein DiT als Grundarchitektur an.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:df3b398f-->
Bei der latenten Diffusion komprimiert ein KL-regularisierter Autoencoder die Bilder zu einem räumlichen Gitter niedrigerer Dimension. Wir skalieren die latenten Darstellungen auf Standardabweichung eins und verwenden im latenten Raum die übliche Regression für Diffusion oder Flow-Matching. Beim Autoencoder stehen Rate (Dimension der latenten Darstellung), Verzerrung (Rekonstruktions-FD) und Modellierbarkeit (wie leicht das Diffusionsmodell die Verteilung der latenten Darstellungen lernt) in einem Zielkonflikt. Ein höheres KL-Gewicht $\beta$ macht den Decoder toleranter gegenüber zufälligen Störungen im latenten Raum. Zugleich verschlechtert es die Rekonstruktionsqualität und den FD der erzeugten Samples. Als Grundarchitektur zur Entrauschung zerlegt ein Diffusion Transformer (DiT) das Bild oder das latente Gitter in Patch-Token. Die Konditionierung übernimmt `adaLN-Zero`, eine adaptive Schichtnormalisierung mit auf null initialisierten Gates in den Residualzweigen. Bei kleinen Rechenbudgets auf Bildern der Größe $28 \times 28$ übertrifft das konvolutionäre U-Net den DiT bei gleichen FLOPs. Mit zunehmendem Rechenaufwand pro Schritt verbessert sich der DiT jedoch stärker.

<!--en:2294243a-->
## Übungen

<!--en:59096950-->
1. **Die Skala der latenten Darstellungen.** Der lineare Pfad mischt die Daten mit Rauschen der Varianz eins.
    1. Jeder latente Wert habe die Standardabweichung $\varsigma$ statt eins. Zeigen Sie, dass sein Signal-Rausch-Verhältnis zur Zeit $t$ gleich $\varsigma^2 (1 - t)^2 / t^2$ ist. Damit verschiebt sich die Log-SNR zu jedem Zeitpunkt um $\log \varsigma^2$.
    1. Zu welcher Zeit hat eine latente Darstellung mit $\varsigma = 5$ dasselbe Signal-Rausch-Verhältnis wie eine Darstellung mit $\varsigma = 1$ zur Zeit $t = \tfrac12$? Welcher Anteil gleichverteilt gezogener Trainingszeiten fällt dann in den Bereich, in dem das Signal das Rauschen noch überwiegt?
    1. [code] Trainieren Sie das latente U-Net auf latenten Darstellungen, die Sie mit $5c$ statt mit $c$ multiplizieren, und vergleichen Sie den FD nach dem Dekodieren. Setzen Sie das Ergebnis in Beziehung zur Verschiebung des Rauschplans mit der Auflösung (:numref:`subsec_diffusion-design-space-scales`).
   :exlabel:`ex_latent-scale-factor`

<!--en:cf232065-->
1. **Token bei hoher Auflösung.** Der Autoencoder von Stable Diffusion bildet ein Bild der Größe $512 \times 512 \times 3$ auf $64 \times 64 \times 4$ ab.
    1. Wie viele Token verarbeitet ein DiT mit Patch-Größe 2 auf der latenten Darstellung und mit Patch-Größe 16 auf den Pixeln? Wie viele Zahlen enthält jedes Token in den beiden Fällen?
    1. Ein DiT-Block der Breite $d$ hat etwa $12 d^2$ Gewichte. Ein Trainingsschritt kostet etwa sechs FLOPs pro Gewicht und Token zuzüglich der Attention. Schätzen Sie die FLOPs eines Trainingsschritts mit einem Batch von 256 für einen DiT der Breite 1152 und Tiefe 28 auf den latenten Darstellungen mit Patch-Größe 2. Wie groß ist der Anteil der Attention?
    1. Wiederholen Sie die Schätzung für Patch-Größe 16 auf den Pixeln. Welchen Anteil der Kosten pro Schritt spart der Autoencoder dem Transformer? Wie viel spart er einem U-Net, dessen Kosten mit der Anzahl der Positionen wachsen?
   :exlabel:`ex_latent-compression-arithmetic`

<!--en:2883ca35-->
1. [code] **Breite und Gewicht des Autoencoders.** Rate und KL-Gewicht sind die beiden Einstellungen des Autoencoders.
    1. Trainieren Sie Autoencoder mit 2, 4 und 8 latenten Kanälen. Geben Sie für jeden den Rekonstruktions-FD an. Trainieren Sie auf jeder Variante ein latentes U-Net und geben Sie den FD seiner dekodierten Samples an. Welche der beiden Zahlen ändert sich stärker? Schließt sich die Lücke zwischen ihnen, wenn die latente Darstellung größer wird? Was legt das über die vier Kanäle im Autoencoder von Stable Diffusion und die 16 oder mehr Kanäle späterer Autoencoder nahe?
    1. Erweitern Sie den Versuch mit gestörten latenten Darstellungen aus diesem Abschnitt auf $\beta \in \{0, 10^{-2}\}$. Geben Sie auch die Standardabweichungen der latenten Darstellungen und des Encoders sowie den Exzess aus. Bei $\beta = 0$ bindet nichts die Skala der latenten Darstellungen an eins. Weshalb macht der Skalenfaktor $c$ sie dennoch nutzbar?
    1. Trainieren Sie latente U-Nets auf beiden neuen Autoencodern. Gilt die im Text beobachtete Rangfolge, nach der ein kleineres KL-Gewicht bessere Samples ergibt, bis hin zu $\beta = 0$?
   :exlabel:`ex_latent-bottleneck`

<!--en:4a9dd609-->
1. [code] **Warum bei null beginnen?** Ersetzen Sie die Nullinitialisierung der `ada`-Schichten in `d2l.DiTBlock` und `d2l.DiT` durch die Standardinitialisierung des Frameworks.
    1. Geben Sie für beide Initialisierungen an, wie groß die Ausgabewerte des untrainierten Netzes sind.
    1. Trainieren Sie beide DiT-M-Modelle über 2000 Updates und plotten Sie ihre Verluste. Erklären Sie den Unterschied anhand der Identitätsabbildung bei der Initialisierung.
   :exlabel:`ex_latent-zero-gate`

<!--en:8b81a146-->
1. **Cross-Attention als Abfrage.** Die Bildtoken $\mathbf{Y} \in \mathbb{R}^{m \times d}$ fragen die Token der Bildbeschreibung $\mathbf{X} \in \mathbb{R}^{n \times d}$ ab. Dabei sind $\mathbf{Q} = \mathbf{Y}\mathbf{W}_q$ die Abfragen, $\mathbf{K} = \mathbf{X}\mathbf{W}_k$ die Schlüssel und $\mathbf{V} = \mathbf{X}\mathbf{W}_v$ die Werte. Die Schicht gibt $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d})\,\mathbf{V}$ aus.
    1. Zeigen Sie, dass jede Bildposition eine Konvexkombination der Wertvektoren der Bildbeschreibung erhält. Zeigen Sie für eine leere Beschreibung aus einem Token, dass jede Position denselben Vektor erhält. Für die unbedingte Vorhersage bei der Guidance wirkt Cross-Attention damit wie eine gelernte konstante Verschiebung.
    1. Nehmen Sie an, die Werte ergäben sich stattdessen aus dem Bild, also $\mathbf{V} = \mathbf{Y}\mathbf{W}_v$. Welche Formen passen bei $m \neq n$ nicht zusammen? Was würde eine solche Schicht bei $m = n$ berechnen?
    1. Bei Cross-Attention hängen Schlüssel und Werte der Beschreibung nicht von $\mathbf{x}_t$ ab. In einem multimodalen DiT aktualisieren die Bildtoken in jedem Block die Token der Beschreibung. Betrachten Sie eine Beschreibung mit 77 Token, 1024 Bildtoken und 50 Schritten mit Guidance. Welche Berechnungen für die Beschreibung kann ein Modell mit Cross-Attention einmal pro Sample erledigen? Welcher Anteil der Berechnungen über Token entfällt bei einem multimodalen DiT auf die Beschreibung?

    *Teil (b) ist angelehnt an CMU 10-423/623, [Generative KI](https://www.cs.cmu.edu/~mgormley/courses/10423/), Aufgabenblatt 4, Frage 2.3 (Frühjahr 2026).*
   :exlabel:`ex_latent-cross-attention`

<!--en:52bc1f44-->
1. [code] **Große Patches und das rauschfreie Bild als Zielgröße.** ● Trainieren Sie `DiT` mit `patch_size=7` auf Pixeln, sodass jedes der 16 Token 49 Pixelwerte enthält. Trainieren Sie einmal mit Breite 32 und einem Kopf und einmal mit Breite 256, jeweils über 10 000 Updates. Verwenden Sie für jede Breite zwei Parametrisierungen der Ausgabe bei gleichem Geschwindigkeitsverlust. Entweder gibt das Netz die Geschwindigkeit aus oder es gibt $\hat{\mathbf{x}}_0$ aus und Sie berechnen die Geschwindigkeit als $(\mathbf{x}_t - \hat{\mathbf{x}}_0)/t$, wobei Sie $t$ nach unten auf $0.05$ begrenzen.
    1. Geben Sie die FD-Werte der vier Modelle in einer $2 \times 2$-Tabelle an.
    1. Hilft die Vorhersage des rauschfreien Bildes bei Breite 32, wenn ein Token mehr Werte enthält, als das Netz Kanäle hat? Schließt sich die Lücke bei Breite 256?
    1. Erklären Sie das Ergebnis mit der Überlegung aus :numref:`sec_diffusion-design-space`. Ein Bild liegt nahe einer Menge, deren Dimension wesentlich kleiner ist als die Pixelzahl, während das Rauschen alle Dimensionen ausfüllt. Vergleichen Sie dies mit dem Befund von :citet:`Li.He.2026`.
   :exlabel:`ex_latent-large-patches`

<!--en:babfd9db-->
[Diskussion](https://d2l.discourse.group/t/36892)

<!--en:74e0d6a5-->
Latente Diffusion und Diffusion Transformer (DiT)<br>**Perzeptuelle Kompression (VAE) · Rate, Verzerrung und Modellierbarkeit · Skalierung latenter Darstellungen · Patch-Token und AdaLN-Zero · Skalierung von U-Net und DiT**

<!--en:06d6c5fe-->
::: {.slide title="Warum Diffusion im Pixelraum so viel Rechenleistung braucht"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:71e64ad8-->
- **Quadratische und quartische Skalierung mit der Seitenlänge:** Ein Bild der Größe $512 \times 512 \times 3$ enthält $786{,}432$ Werte ($1{,}000\times$ so viele wie bei $28 \times 28$). Self-Attention skaliert wie $(HW)^2$, und zwar in allen $50$ Sampling-Schritten.

<!--en:69aae63c-->
- **Kaum wahrnehmbare hochfrequente Bits:** Die meisten Pixeldimensionen beschreiben feine Körnung, die dem Auge kaum auffällt.

<!--en:b407520e-->
- **Zwei Durchbrüche:** Ein perzeptueller Autoencoder komprimiert Bilder zu latenten Darstellungen (Rombach et al., 2022). Diffusion Transformer mit Patch-Token ersetzen U-Nets (Peebles & Xie, 2023).

<!--en:ed6a8906-->
::: {.slide title="Perzeptuelle und semantische Kompression"}

<!--en:fc943e9d-->
[Die Pipeline der latenten Diffusion]{.kicker}

<!--en:1d75dc3c-->
Bei der latenten Diffusion übernehmen zwei Komponenten die Bilderzeugung: ein eingefrorener **perzeptueller Autoencoder** $(\mathcal{E}, \mathcal{D})$ und ein **latentes Diffusionsmodell**, trainiert auf $\mathbf{z}_0 = \mathcal{E}(\mathbf{x}_0)$:

<!--en:69a89b46-->
- **Räumliche Kompressionsverhältnisse:**
  - *Fashion-MNIST-Benchmark:* $28 \times 28 \times 1$ ($784$ Pixel) $\to$ $7 \times 7 \times 4$ ($196$ latente Werte, Kompressionsfaktor $4\times$, um den Faktor $16\times$ weniger räumliche Positionen).
  - *Stable Diffusion:* $512 \times 512 \times 3$ ($786{,}432$ Werte) $\to$ $64 \times 64 \times 4$ ($16{,}384$ latente Werte, **Kompressionsfaktor $48\times$**, um den Faktor $64\times$ weniger räumliche Positionen).

<!--en:1291462e-->
- **Zielkonflikt zwischen drei Größen (Dieleman, 2025):**
  - **Rate:** Anzahl latenter Dimensionen ($196$).
  - **Verzerrung:** Rekonstruktionsverlust des Decoders beim Vergleich von $\mathcal{D}(\mathcal{E}(\mathbf{x}_0))$ mit $\mathbf{x}_0$.
  - **Modellierbarkeit:** Wie leicht das Diffusionsmodell die Verteilung der latenten Darstellungen lernt.

<!--en:d14041c3-->
::: {.slide title="Regularisierung des latenten Raums im Autoencoder"}

<!--en:52b08bb2-->
[Warum herkömmliche Autoencoder für Diffusion nicht ausreichen]{.kicker}

<!--en:9621ce66-->
- **Drei Anforderungen an latente Darstellungen für die Diffusion:** (1) Einheitsskala passend zu $\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, (2) keine Ausreißer in schweren Verteilungsrändern und (3) eine glatte Dekodierungsmannigfaltigkeit durch $\mathbf{z} = \boldsymbol\mu + \mathbf{s} \odot \boldsymbol\xi$.

<!--en:30d38158-->
- **Geringes KL-Gewicht ($\beta = 10^{-4}$):** Hält die Skala nahe $1$ und verhindert, dass das Rauschen des Encoders verschwindet ($s_j \to 0$). Die Rekonstruktionen bleiben scharf.

<!--en:9638303a-->
::: {.slide title="Implementierung des KL-regularisierten Autoencoders" only="pytorch"}

<!--en:b6fd9d82-->
[Konvolutionärer Encoder und Decoder mit Residualblöcken]{.kicker}

<!--en:80aa98bc-->
Der `Autoencoder` verringert die Auflösung gemäß $28 \times 28 \times 1 \to 14 \times 14 \to 7 \times 7 \times 8$. Die Ausgabe zerfällt in $4$ Kanäle für Mittelwerte und $4$ für begrenzte logarithmische Varianzen. Der Decoder durchläuft die Hierarchie in umgekehrter Richtung:

<!--en:35f34d55-->
::: {.slide title="Implementierung des KL-regularisierten Autoencoders" only="jax"}

<!--en:b6fd9d82-2-->
[Konvolutionärer Encoder und Decoder mit Residualblöcken]{.kicker}

<!--en:80aa98bc-2-->
Der `Autoencoder` verringert die Auflösung gemäß $28 \times 28 \times 1 \to 14 \times 14 \to 7 \times 7 \times 8$. Die Ausgabe zerfällt in $4$ Kanäle für Mittelwerte und $4$ für begrenzte logarithmische Varianzen. Der Decoder durchläuft die Hierarchie in umgekehrter Richtung:

<!--en:83ebf44d-->
::: {.slide title="Rekonstruktionen des Autoencoders und Referenzwert für die Verzerrung"}

<!--en:b8fc1ea1-->
[Rekonstruktionsqualität auf Fashion-MNIST messen]{.kicker}

<!--en:0147e25e-->
Der konvolutionäre `Autoencoder` mit Residualblöcken ($2.4\text{M}$ Parameter) durchläuft $20{,}000$ Adam-Updates mit kosinusförmig sinkender Lernrate bei $\beta = 10^{-4}$. Zum Vergleich für $5{,}000$ rekonstruierte Testbilder dient der FD zwischen zwei Mengen von je $5{,}000$ echten Bildern ($0.17$):

<!--en:0a8301ac-->
**Referenzwert der Rekonstruktionen ($\mathrm{FD} \approx 0.65$–$0.80$):** Erzeugte ein latentes Diffusionsmodell exakt die Verteilung der latenten Darstellungen des Encoders, ergäben seine dekodierten Samples diesen Rekonstruktions-FD.

<!--en:21be4406-->
::: {.slide title="Skala und Kurtosis der latenten Werte bestimmen"}

<!--en:b0a9ba84-->
[Anpassung an den Rauschplan mit Einheitsvarianz]{.kicker}

<!--en:cabf546c-->
- **Erforderlicher Skalenfaktor $c = 1/\mathrm{std}(\mathbf{z})$:** Unskalierte latente Darstellungen mit Standardabweichung $\varsigma$ verschieben die Log-SNR um $+\log \varsigma^2$ (Stable Diffusion hat $\varsigma \approx 5.5$ und verwendet $c = 0.18215$, um eine Verschiebung um $+3.4\text{ nat}$ zu vermeiden).

<!--en:44b2f62c-->
- **Subgaußsche Verteilungsränder:** Ein Exzess $\le 0$ bestätigt, dass die KL-Strafe Ausreißer in schweren Verteilungsrändern beseitigt hat.

<!--en:8554a7bc-->
::: {.slide title="Pixel-U-Net und latentes U-Net im Vergleich: Qualität und FLOPs"}

<!--en:2c554576-->
[Dasselbe Flow-Matching im latenten Raum]{.kicker}

<!--en:d005019c-->
Mit skalierten latenten Darstellungen $\mathbf{z}_0 = c\,(\boldsymbol\mu + \mathbf{s} \odot \boldsymbol\xi) \in \mathbb{R}^{4 \times 7 \times 7}$ bleiben der Regressionsverlust für die Geschwindigkeit $\mathbb{E}\|\mathbf{u}_{\boldsymbol\theta}(\mathbf{z}_t, t) - (\boldsymbol\epsilon - \mathbf{z}_0)\|^2$ und der ODE-Sampler mit 50 Euler-Schritten unverändert:

<!--en:ba370568-->
- **Rechenaufwand pro Schritt um den Faktor $>13\times$ geringer:** Das latente U-Net (Eingabe $7 \times 7$) benötigt nur **$\approx 12$–$15\text{ GFLOP}$** pro Trainingsschritt, das Pixel-U-Net dagegen **$\approx 200\text{ GFLOP}$** (Eingabe $28 \times 28$).

<!--en:8df8678b-->
- **Lücke durch begrenzte Modellierbarkeit bei $28 \times 28$:** Dekodierte latente Samples erreichen $\mathrm{FD} \approx 2.0$–$2.2$, das Pixel-U-Net $\approx 1.5$ und die Rekonstruktionen des Autoencoders $\approx 0.7$.

<!--en:bbed38e7-->
::: {.slide title="Woher kommt die Lücke: Decoder oder Diffusionsmodell?"}

<!--en:61ec7853-->
[Test mit 10× so hohem KL-Gewicht (β = 10⁻³)]{.kicker}

<!--en:e1b3f5d6-->
Warum erreichen dekodierte Samples aus der latenten Diffusion $\mathrm{FD} \approx 2.0$, Rekonstruktionen dagegen $\approx 0.7$? Das Training eines zweiten Autoencoders `ae_10` mit $10\times$ so hohem KL-Gewicht ($\beta = 10^{-3}$) prüft zwei konkurrierende Hypothesen:

<!--en:654a0d37-->
- **Hypothese A – mangelnde Fehlertoleranz des Decoders:** Bei $\beta = 10^{-4}$ verrauscht der Encoder die Codes nur um $\mathbf{s}/\mathrm{std}(\boldsymbol\mu) \approx 3\%$. Schon knapp außerhalb der Mannigfaltigkeit könnte der Decoder aus den latenten Darstellungen des Diffusionsmodells unbrauchbare Bilder erzeugen.

<!--en:376d8efb-->
- **Hypothese B – Modellfehler der latenten Diffusion:** Das Diffusions-U-Net macht selbst große Fehler bei der Modellierung der Verteilung des komprimierten latenten Codes der Größe $7 \times 7 \times 4$.

<!--en:f248b576-->
::: {.slide title="Störungstoleranz und Sample-Qualität des Gesamtsystems"}

<!--en:822f9bb4-->
[Dielemans Zielkonflikt um die Modellierbarkeit im Experiment]{.kicker}

<!--en:7bb97652-->
Das Training zweier latenter U-Nets (Zufallsstartwerte 10 und 11) auf den latenten Darstellungen jedes Autoencoders zeigt, welche Hypothese die Lücke erklärt:

<!--en:2bf3e612-->
- **$\beta = 10^{-3}$ macht den Decoder tolerant gegenüber zufälligem Rauschen:** Gaußsches Rauschen mit $\sigma = 0.5$ auf den latenten Darstellungen echter Testbilder verschlechtert `ae_10` nur auf $\mathrm{FD} \approx 1.3$ (beim Decoder mit $\beta = 10^{-4}$ steigt der Abstand auf $\mathrm{FD} > 3.0$).

<!--en:b866529e-->
- **Dennoch liefert $\beta = 10^{-3}$ im Gesamtsystem *schlechtere* Diffusionssamples ($\mathrm{FD} \approx 2.7$ gegenüber $2.1$):** Eine zufällige Störung der latenten Darstellung um $50\%$ ergibt bei `ae_10` nur einen FD von $1.3$. Trainierte latente U-Nets erreichen etwa $2.7$. Die Lücke entsteht also durch **Modellfehler der latenten Diffusion**, nicht durch die Störempfindlichkeit des Decoders.

<!--en:aac03969-->
- **Fazit:** Das KL-Gewicht des Autoencoders klein halten ($\beta = 10^{-4}$), damit die Rekonstruktionen scharf bleiben und sich die latenten Kanäle leicht modellieren lassen.

<!--en:57c617a3-->
::: {.slide title="Diffusion Transformer (DiT): Bilder in Patches zerlegen"}

<!--en:8a135cce-->
[Attention über eine Sequenz ersetzt Faltungen]{.kicker}

<!--en:7d317e6c-->
- **Tokenisierung in Patches (`PatchEmbedding`):** Eine Faltung mit Stride $p$ und Kerngröße $p \times p$ zerlegt $H \times W \times C$ in $L = (H/p)(W/p)$ Token der Dimension $d$ ($L = 49$ für $28 \times 28$ Pixel bei $p = 4$ oder für latente Darstellungen der Größe $7 \times 7$ bei $p = 1$).

<!--en:5729466a-->
- **$p$ bestimmt die quadratischen Attention-Kosten:** Self-Attention skaliert wie $L^2 = (HW/p^2)^2$ (bei $p = 1$ auf $28 \times 28$ Pixeln benötigt sie $256\times$ so viele FLOPs wie bei $p = 4$).

<!--en:41441b5d-->
- **Lineare Ausgabeschicht zur Bildrekonstruktion:** Projiziert jedes Token gemäß $\mathbb{R}^d \to \mathbb{R}^{p^2 C}$ und stellt die Form $(H, W, C)$ wieder her.

<!--en:0fbf0bc5-->
::: {.slide title="Konditionierung von Transformern mit AdaLN-Zero"}

<!--en:e48edd75-->
[Adaptive Schichtnormalisierung mit auf null initialisierten Gates]{.kicker}

<!--en:3532f00a-->
Wie sollen die Zeit $t$ und das Klassenlabel $y$ jeden Transformer-Block steuern? Bei Peebles und Xie (2023) schnitt die **adaptive Schichtnormalisierung mit auf null initialisierten Gates (`adaLN-Zero`)** besser ab als Cross-Attention und die Verkettung von Token:

<!--en:423d4fee-->
- **Sechs Modulationsvektoren pro Block:** Eine lineare Schicht bildet $\mathrm{SiLU}(\mathbf{c})$ auf $(\mathbf{b}_1, \mathbf{a}_1, \mathbf{g}_1, \mathbf{b}_2, \mathbf{a}_2, \mathbf{g}_2) \in \mathbb{R}^{6d}$ ab. Für alle $L$ Token gelten dieselben Vektoren.

<!--en:c1659b5b-->
- **Initialisierung als Identität (`Zero`):** Die linearen `ada`-Schichten und die letzte Ausgabeprojektion mit **null** zu initialisieren ergibt $\mathbf{g}_1 = \mathbf{g}_2 = \mathbf{0}$. Jeder Block beginnt daher als exakte Identitätsabbildung, und das untrainierte Netz gibt überall $\mathbf{0}$ aus.

<!--en:8d50322e-->
::: {.slide title="Implementierung von DiTBlock mit AdaLN-Zero"}

<!--en:4ec82074-->
[ViTBlock um eine mit null initialisierte Modulation erweitern]{.kicker}

<!--en:92d2889c-->
`DiTBlock` deaktiviert die eingebauten affinen Parameter von `LayerNorm` und sagt sechs Modulationsvektoren `(shift1, scale1, gate1, shift2, scale2, gate2)` aus dem Bedingungsvektor `c` voraus:

<!--en:79db20af-->
::: {.slide title="Textkonditionierung mit Cross-Attention oder multimodalem DiT"}

<!--en:398b9960-->
[Konditionierung auf Tokensequenzen variabler Länge]{.kicker}

<!--en:396204d2-->
- **Abfrage per Cross-Attention (Rombach et al., 2022):** Bildtoken liefern Abfragen $\mathbf{Q}$, eingefrorene Texteinbettungen liefern Schlüssel und Werte $\mathbf{K}, \mathbf{V}$ (einmal pro Prompt vor der Sampling-Schleife mit 50 Schritten berechnet).

<!--en:559a431e-->
- **Gemeinsamer multimodaler DiT (MM-DiT, Esser et al., 2024):** Verkettet $m$ Bildtoken und $n$ Texttoken zu einer Sequenz der Länge $m + n$ für bidirektionale Self-Attention in jedem Block.

<!--en:a8b2540f-->
::: {.slide title="Formen, Parameter und Nullinitialisierung des DiT prüfen"}

<!--en:53a01b93-->
[Wiederverwendung von Gewichten in Faltungen und Transformern]{.kicker}

<!--en:9c1ed749-->
Ein Batch mit $8$ Bildern durchläuft den `DiT` mit 6 Blöcken und Breite 256 (`patch_size = 4`, $4$ Köpfe). Damit lassen sich Tokenformen und die exakte Nullinitialisierung prüfen:

<!--en:b3a95bd7-->
- **Warum DiT-M bei ähnlichen FLOPs $6\times$ so viele Parameter ($7.39\text{M}$) wie das U-Net ($1.20\text{M}$) hat:** Bei voller Auflösung verwenden die Faltungen des U-Net einen kleinen $3 \times 3$-Faltungskern an $28 \times 28 = 784$ Positionen. Der DiT wendet große, dicht besetzte $d \times 4d$-Matrizen auf nur $L = 49$ Patch-Token an.

<!--en:770691a6-->
::: {.slide title="Skalierung von U-Net und DiT bei Diffusion im Pixelraum"}

<!--en:9edbf7d9-->
[Parameter und GFLOP pro Schritt auf Fashion-MNIST]{.kicker}

<!--en:670991ae-->
Zwei Transformer-Größen (`DiT-S` mit Breite 128 und Tiefe 4, `DiT-M` mit Breite 256 und Tiefe 6) und zwei U-Net-Breiten (`ch = 32` und `ch = 64`) durchlaufen auf Pixeln je $20{,}000$ Updates mit jeweils zwei Zufallsstartwerten:

<!--en:604d1fe0-->
- **Bei kleinen Modellen gewinnt der induktive Bias der Faltung:** `DiT-S` ($1.26\text{M}$ Parameter) hat etwa so viele Parameter wie `U-Net 64` ($1.20\text{M}$ Parameter), aber einen **$3\times$ so hohen FD** ($\approx 4.7$ gegenüber $1.5$).

<!--en:245fa1af-->
- **Steilere Skalierungskurve für DiT:** Steigt der Rechenaufwand des DiT um den Faktor $5.8\times$, sinkt der FD um den **Faktor $2.7\times$** ($4.7 \to 1.8$). Beim U-Net sinkt der FD bei einem Anstieg um den Faktor $4\times$ um den **Faktor $1.65\times$** ($2.5 \to 1.5$).

<!--en:9c1b06dd-->
::: {.slide title="Samples verschiedener Architekturen und Größen"}

<!--en:9f804295-->
[DiT-S, DiT-M, U-Net 32 und U-Net 64 im Vergleich]{.kicker}

<!--en:60d6d44f-->
Acht Samples aus demselben Anfangsrauschen zeigen, wie ähnlich die ODE-Trajektorien der vier Pixelmodelle verlaufen:

<!--en:728a8c96-->
- **Gemeinsame ODE-Abbildung:** In fast jeder Spalte bilden beide DiTs und beide U-Nets dasselbe Anfangsrauschen auf dasselbe Kleidungsstück in derselben Pose ab. Alle vier approximieren dieselbe Wahrscheinlichkeitsfluss-ODE.

<!--en:440f6e4f-->
- **Was der FD zusätzlich zum visuellen Vergleich erfasst:** Die 8 Vorschaubilder in Zeile 1 (`DiT-S`, $\mathrm{FD} \approx 4.7$) und Zeile 4 (`U-Net 64`, $\mathrm{FD} \approx 1.5$) sehen ähnlich aus. Über $5{,}000$ Samples erfasst der FD feine Verzerrungen der Konturen und Verschiebungen der Kovarianz.

<!--en:6845be1f-->
::: {.slide title="Latente Darstellungen und Transformer: Wann lohnt sich welche Architektur?" layout="tight"}

<!--en:f15aef4a-->
[Entscheidungshilfe für die Architekturwahl]{.kicker}

<!--en:b69861f9-->
- **Niedrige Auflösung ($\le 64^2$):** **Pixel-U-Net** verwenden. Die Verarbeitung der Pixel kostet wenig und vermeidet die Lücke durch die begrenzte Modellierbarkeit der Autoencoder-Darstellungen ($\mathrm{FD} \approx 1.5$ gegenüber $2.0$).

<!--en:8e1e9424-->
- **Hohe Auflösung ($512^2+$) bei großem Rechenbudget:** **Latenten DiT ($p=2$)** verwenden. Die Zahl räumlicher Positionen sinkt um den Faktor $64\times$, und die Verarbeitung der Token im Transformer skaliert gut mit den verfügbaren GFLOP.

<!--en:70b94762-->
- **Pixel-DiTs mit großen Patches (Li und He, 2026):** Das rauschfreie $\hat{\mathbf{x}}_0$ vorhersagen, um hochdimensionales orthogonales Rauschen durch Projektion zu entfernen.

<!--en:8c44d4c2-->
::: {.slide title="Latente Darstellungen und Transformer: Wann lohnt sich welche Architektur? (Code)" layout="tight"}

<!--en:f15aef4a-2-->
[Entscheidungshilfe für die Architekturwahl]{.kicker}

<!--en:0d017295-->
::: {.slide title="Verständnisfragen: Skalierung latenter Darstellungen, AdaLN-Zero und Token"}

<!--en:929b7671-->
[Wissen überprüfen]{.kicker}

<!--en:560552b6-->
Fragen zu Autoencodern für latente Diffusion und zu Diffusion Transformern:

<!--en:14ca7d79-->
1. **Der Skalenfaktor $0.18215$ von Stable Diffusion:** Stable Diffusion multipliziert die latenten Darstellungen des Autoencoders mit $c = 0.18215$ und addiert dann gaußsches Rauschen mit Einheitsvarianz. Wie groß war die ursprüngliche Standardabweichung $\varsigma$? Wie würde sich das Weglassen von $c$ auswirken? Geben Sie die Verschiebung der Log-SNR $\lambda_t$ in Nat an.

<!--en:3ef0e43b-->
2. **Partielle Nullinitialisierung im DiT:** In jedem `DiTBlock` startet das `ada`-Gate bei null. Auch die Modulation der abschließenden Normalisierung startet bei null. *Ausgenommen* ist die letzte lineare Ausgabeschicht `self.out`, die eine übliche Zufallsinitialisierung erhält. Was gibt der untrainierte DiT zur Zeit $t$ aus?

<!--en:557b98b6-->
**Lösungen:**

<!--en:80cf7ccb-->
1. Die ursprüngliche Standardabweichung war $\varsigma = 1 / 0.18215 \approx \mathbf{5.49}$ (Varianz $\varsigma^2 \approx 30.1$). Ohne $c$ wächst die Signalvarianz um den Faktor $30\times$ bei jedem $t$. Damit steigt $\lambda_t$ um $+\log(30.1) \approx \mathbf{+3.4\text{ nats}}$.

<!--en:d5a47a12-->
2. Jeder `DiTBlock` berechnet exakt die Identität ($\mathbf{g}_1 = \mathbf{g}_2 = \mathbf{0}$). Das Netz gibt daher `self.out(self.ln(patch(x) + pos))` aus, eine von null **verschiedene** lineare Projektion der normalisierten Eingabe-Patches, die **von der Zeit $t$ völlig unabhängig** ist.

<!--en:a993a288-->
::: {.slide title="Verständnisfragen: Skalierung latenter Darstellungen, AdaLN-Zero und Token (Fortsetzung)"}

<!--en:929b7671-2-->
[Wissen überprüfen]{.kicker}

<!--en:e5b470e9-->
3. **Tokenzahl und Skalierung der Attention bei $1024 \times 1024$:** Ein Autoencoder reduziert die Auflösung um den Faktor $8\times$. Die latenten Patches haben die Größe $2 \times 2$ ($p = 2$). Wie viele Token $L$ verarbeitet ein DiT für ein Bild der Größe $1024 \times 1024$ im Vergleich zu $512 \times 512$? Um welchen Faktor steigen die Kosten der Self-Attention?

<!--en:23db34b8-->
**Lösung:**

<!--en:cf736bda-->
3. Das latente Gitter hat die Größe $128 \times 128 \times 4$. Mit $p = 2$ ergeben sich also $L = 64^2 = \mathbf{4{,}096\text{ tokens}}$ ($4\times$ so viele wie die $1{,}024$ Token bei $512^2$). Die quadratischen Kosten der Self-Attention $O(L^2)$ steigen um den **Faktor $16\times$**.

<!--en:aa6b5dac-->
::: {.slide title="Zusammenfassung: Latente Diffusion und Transformer als Grundarchitektur"}

<!--en:c5a222f8-->
[Die wichtigsten Ergebnisse]{.kicker}

<!--en:32bcc062-->
| Komponente oder Konzept | Mathematische Form oder Architektur | Empirischer Befund in §{sec_diffusion-latent} | Empfehlung für den Entwurf |

<!--en:5d245b9b-->
| **Perzeptueller Autoencoder** | $\frac{1}{d}\|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \frac{\beta}{d_z}D_{\mathrm{KL}}(\mathcal{N}(\boldsymbol\mu, \mathbf{s}^2) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I}))$ | $\beta = 10^{-4}$ ergibt $\mathrm{FD}_{\textrm{rec}} \approx 0.7$, Exzess $\le 0$ | KL-Gewicht klein halten ($\beta \approx 10^{-4}$), um die Modellierbarkeit zu erhalten |

<!--en:1df2d269-->
| **Skalierung latenter Darstellungen** | $\mathbf{z}_0 = c\,(\boldsymbol\mu + \mathbf{s} \odot \boldsymbol\xi), \quad c = 1 / \mathrm{std}(\mathbf{z})$ | Hier $c \approx 1.05$ (bei Stable Diffusion $c = 0.18215$) | Erforderlich, damit die latente Signalvarianz der Einheitsvarianz des Rauschens $\boldsymbol\epsilon$ entspricht |

<!--en:8ff1bbb8-->
| **Latentes U-Net ($7 \times 7 \times 4$)** | Übliches Flow-Matching für die Geschwindigkeit auf $\mathbf{z}_t$ | Senkt den Rechenaufwand pro Schritt um den **Faktor $13\text{--}17\times$** ($200 \to 12\text{ GFLOP}$) | Wesentlich für konvolutionäre Modelle bei hoher Auflösung ($512^2+$) |

<!--en:760381f0-->
| **Patch-Zerlegung im DiT** | Faltung mit Stride $p$ $\to L = (H/p)(W/p)$ Token | $p = 4$ bildet $28 \times 28$ auf $L = 49$ Token ab (je $16$ Pixel) | $p$ nach den quadratischen Attention-Kosten $O(L^2)$ und der Patch-Dimension wählen |

<!--en:c0a32bca-->
| **Modulation mit `adaLN-Zero`** | $\mathbf{X} + \mathbf{g} \odot f\big(\mathrm{LN}(\mathbf{X}) \odot (\mathbf{1} + \mathbf{a}) + \mathbf{b}\big)$ | Untrainierte Ausgabe ist identisch $\mathbf{0}$, stabilisiert tiefe DiTs | Alle Gates zur Konditionierung und die letzte Ausgabeschicht mit null initialisieren |

<!--en:0fbe1911-->
| **Skalierung von U-Net und DiT** | Konvolutionäre Hierarchie gegenüber isotropen Token | U-Net gewinnt bei $\le 200\text{ GFLOP}$, DiT hat die steilere Verbesserungskurve | U-Net bei kleinem Budget, DiT bei wachsendem Rechenaufwand |
