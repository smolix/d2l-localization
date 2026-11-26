# Decisions log: Deutsch (de)

## chapter_installation/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „lock file“ → „Lockdatei“, „notebook bundle“ → „Notebook-Paket“.

## chapter_notation/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „nuclear norm (trace norm)“ → „Nuklearnorm (Spurnorm)“.
- „calculus“ → „Analysis“, „indicator function“ → „Indikatorfunktion“, „cardinality“ → „Mächtigkeit“.

## chapter_notation/index.md#u02 (Codex Astra (medium), 2026-11-26)

- „state-value function“ → „Zustandswertfunktion“, „advantage function“ → „Vorteilsfunktion“.
- „state space model“ → „Zustandsraummodell“, „feedthrough term“ → „Durchgriffsterm“.

## index.md#u01 (Codex Astra (medium), 2026-11-26)

- “Image Models” → „Bildmodelle“, “Attic” → „Archiv“ (first-edition chapters retained for reference).

## chapter_introduction/index.md#u01 (Codex Astra (medium), 2026-11-26)

- “wake word” → „das Aktivierungswort“.
- “model family” → „die Modellfamilie“.
- “surrogate loss” → „der Ersatzverlust“.

## chapter_introduction/index.md#u02 (Codex Astra (medium), 2026-11-26)

- “feedback loop” → „die Rückkopplungsschleife“.
- “tagging” → „die Verschlagwortung“ for assigning labels.
- “credit assignment” → „die Zuordnung des Erfolgs“ in reinforcement learning.

## chapter_preface/index.md#u01 (Codex Astra (medium), 2026-11-26)

- Book part titles: “Basics” → „Grundlagen“, “Advanced” → „Fortgeschrittene Themen“, “Mathematics for Deep Learning” → „Mathematik für Deep Learning“, “Language Models” → „Sprachmodelle“.

## chapter_preface/index.md#u02 (Codex Astra (medium), 2026-11-26)

- “Tools for Deep Learning” → „Werkzeuge für Deep Learning“.

## chapter_preliminaries/index.md#u01 (Codex Astra (medium), 2026-11-26)

- “Preliminaries” → „Handwerkszeug“ for the opening chapter of the Basics part.

## chapter_preliminaries/pandas.md#u01 (Codex Astra (medium), 2026-11-26)

- „data leakage“ → „Datenleckage“, „missingness indicator“ → „Indikator für fehlende Werte“.

## chapter_preliminaries/autograd.md#u01 (Codex Astra (medium), 2026-11-26)

- “detach” → „abtrennen“ for disconnecting a tensor from its computation history.
- “gradient tracking” → „Gradientenverfolgung“, “micro-batch” → „Mikrobatch“.

## chapter_preliminaries/ndarray.md#u01 (Codex Astra (medium), 2026-11-26)

- “view” → „Sicht“, “slice” → „Slice“, “order” (number of tensor axes) → „Ordnung“.

## chapter_preliminaries/probability.md#u01 (Codex Astra (medium), 2026-11-26)

- „sample space“ → „Ergebnisraum“, „estimator“ → „Schätzer“, „unbiased“ → „erwartungstreu“.
- „biased coin“ → „unfaire Münze“. Die Verzerrung eines Schätzers ist ein anderer Begriff.

## chapter_preliminaries/probability.md#u02 (Codex Astra (medium), 2026-11-26)

- „explaining away“ → „Wegerklären“, „false-positive rate“ → „Falsch-Positiv-Rate“.

## chapter_linear-regression/linear-regression.md#u01 (Codex Astra (medium), 2026-11-26)

- “design matrix” → „Entwurfsmatrix“.

## chapter_linear-regression/linear-regression.md#u02 (Codex Astra (medium), 2026-11-26)

- “noise ball” → „Kugel um das Minimum“, glossed on first use.

## chapter_linear-regression/linear-regression.md#u03 (Codex Astra (medium), 2026-11-26)

- “pinball loss” → „Pinball-Verlust“.

## chapter_linear-regression/linear-regression-concise.md#u01 (Codex Astra (medium), 2026-11-26)

- „noise floor“ → „Rauschgrenze“ für den durch das Rauschen bedingten Grenzwert des Verlusts.
- „from-scratch implementation“ → „eigene Implementierung“, „concise implementation“ → „kompakte Implementierung“.

## chapter_linear-regression/linear-regression-scratch.md#u01 (Codex Astra (medium), 2026-11-26)

- „excess loss“ → „Überschussverlust“ für den Verlustanteil oberhalb der rauschbedingten Untergrenze.
- „noise floor“ → „rauschbedingte Untergrenze“ für den durch das Rauschen gesetzten Mindestwert des Verlusts.

## chapter_linear-regression/oo-design.md#u01 (Codex Astra (medium), 2026-11-26)

- „training harness“ → „Trainingsgerüst“, „epoch mean“ → „Epochenmittelwert“, „progress board“ → „Fortschrittsanzeige“.

## chapter_linear-classification/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „importance weighting“ → „Wichtigkeitsgewichtung“.
- „learned template“ → „erlernte Schablone“ für das räumliche Gewichtsmuster einer Klasse.

## chapter_linear-regression/generalization.md#u01 (Codex Astra (medium), 2026-11-26)

- „true risk“ → „wahres Risiko“, „squared bias“ → „quadrierte Verzerrung“, „estimation variance“ → „Schätzvarianz“.
- „polynomial features“ → „Polynommerkmale“, „sample complexity“ → „Stichprobenkomplexität“.
- „fold“ → „Gruppe“ bei der Kreuzvalidierung, „leave-one-out cross-validation“ → „Leave-one-out-Kreuzvalidierung“.

## chapter_linear-regression/weight-decay.md#u01 (Codex Astra (medium), 2026-11-26)

- „norm penalty“ → „Strafterm auf Basis einer Norm“, „shrinkage factor“ → „Schrumpfungsfaktor“.

## chapter_linear-regression/weight-decay.md#u02 (Codex Astra (medium), 2026-11-26)

- „least-squares estimate“ → „Kleinste-Quadrate-Schätzung“, „effective number of parameters“ → „effektive Parameterzahl“.

## chapter_linear-classification/image-classification-dataset.md#u01 (Codex Astra (medium), 2026-11-26)

- „channel-first“ und „channel-last“ → „Channel-First“ und „Channel-Last“ für die Achsenreihenfolge.
- „dataset split“ → „Teildatensatz“ für einen Teil der Daten, „Aufteilung“ für die Unterteilung selbst.
- „worker“ → „Arbeitsprozess“ beim parallelen Laden von Daten.

## chapter_linear-classification/softmax-regression.md#u01 (Codex Astra (medium), 2026-11-26)

- „shift invariance“ → „Verschiebungsinvarianz“, „decision region“ → „Entscheidungsregion“.
- „hard assignment“ → „harte Zuordnung“, „soft assignment“ → „weiche Zuordnung“.

## chapter_linear-classification/softmax-regression.md#u02 (Codex Astra (medium), 2026-11-26)

- „zero-one error“ → „Null-Eins-Fehler“, „confidence“ → „Konfidenz“.

## chapter_linear-classification/softmax-regression.md#u03 (Codex Astra (medium), 2026-11-26)

- „temperature scaling“ → „Temperaturskalierung“.

## chapter_linear-classification/softmax-regression-scratch.md#u01 (Codex Astra (medium), 2026-11-26)

- „probability clamping“ → „Begrenzung der Wahrscheinlichkeit nach unten“.
- „fused loss“ → „kombinierte Verlustfunktion“, die Softmax-Funktion und Logarithmus in einer Operation zusammenfasst.

## chapter_linear-classification/classification.md#u01 (Codex Astra (medium), 2026-11-26)

- „reliability diagram“ → „Zuverlässigkeitsdiagramm“, „temperature scaling“ → „Temperaturskalierung“.
- „confidence“ → „Konfidenz“, „confidence bin“ → „Konfidenzbereich“ (Intervall, in das die vorhergesagte Konfidenz fällt).
- „true-positive rate“ → „Richtig-Positiv-Rate“, „positive predictive value“ → „positiver Vorhersagewert“.
- „precision-recall curve“ → „Präzisions-Trefferquoten-Kurve“, „abstention“ → „Enthaltung“.

## chapter_linear-classification/generalization-classification.md#u01 (Codex Astra (medium), 2026-11-26)

- „population error“ → „Populationsfehler“, „coverage“ → „Überdeckung“, „confidence radius“ → „Konfidenzradius“.
- „shatter“ → „zertrümmern“, „labeling“ → „Labelzuweisung“, „memorizer“ → „auswendig lernendes Modell“.
- „feature map“ → „Merkmalsabbildung“ für die Abbildung in einen Merkmalsraum.

## chapter_multilayer-perceptrons/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „linear collapse“ → „Reduktion auf eine lineare Abbildung“.
- „parameter memory“ → „Parameterspeicher“, „machine precision“ → „Maschinengenauigkeit“.

## chapter_linear-classification/environment-and-distribution-shift.md#u01 (Codex Astra (medium), 2026-11-26)

- „concept shift“ → „Konzeptverschiebung“, „domain classifier“ → „Domänenklassifikator“.
- „source distribution“ → „Quellverteilung“, „target distribution“ → „Zielverteilung“, „target risk“ → „Zielrisiko“.
- „support overlap“ → „Überdeckung des Zielträgers“, wenn der Träger der Quellverteilung den gesamten Zielträger enthalten muss.
- „classifier two-sample test“ → „klassifikatorbasierter Zweistichprobentest“.

## chapter_linear-classification/environment-and-distribution-shift.md#u02 (Codex Astra (medium), 2026-11-26)

- „performative prediction“ → „performative Vorhersage“.
- „Bayes logit adjustment“ → „Logit-Anpassung nach Bayes“.

## chapter_multilayer-perceptrons/mlp.md#u01 (Codex Astra (medium), 2026-11-26)

- „pre-activation“ → „Präaktivierung“, „hidden space“ → „Raum der versteckten Aktivierungen“.
- „expressive power“ → „Ausdrucksstärke“, „hinge“ → „Knickfunktion“, „knot“ → „Knoten“, „tent function“ → „Zeltfunktion“.
- „space folding“ → „Raumfaltung“, „linear piece“ → „linearer Abschnitt“.

## chapter_multilayer-perceptrons/mlp.md#u02 (Codex Astra (medium), 2026-11-26)

- „ridge function“ → „Ridge-Funktion“.

## chapter_multilayer-perceptrons/mlp-implementation.md#u01 (Codex Astra (medium), 2026-11-26)

- „multiply-add“ → „Multiplikationsaddition“, „pre-activation“ → „Voraktivierung“, „wall-clock time“ → „Laufzeit“ für die tatsächlich verstrichene Zeit.

## chapter_multilayer-perceptrons/backprop.md#u01 (Codex Astra (medium), 2026-11-26)

- „upstream gradient“ → „eingehender Gradient“, „pre-activation“ → „Präaktivierung“, „backward recurrence“ → „Rückwärtsrekursion“.
- „broadcasted bias addition“ → „Bias-Addition durch Broadcasting“.

## chapter_multilayer-perceptrons/numerical-stability-and-init.md#u01 (Codex Astra (medium), 2026-11-26)

- „mean square“ → „mittleres Quadrat“, bei Gradienten komponentenweise.
- „fan-in“ und „fan-out“ → „Fan-in“ und „Fan-out“ (Anzahl der Eingänge und Ausgänge).

## chapter_multilayer-perceptrons/dropout.md#u01 (Codex Astra (medium), 2026-11-26)

- „co-adaptation“ → „Koadaptation“, „label noise“ → „Labelrauschen“, „thinned subnetwork“ → „ausgedünntes Teilnetz“, „adaptive ridge penalty“ → „adaptiver Ridge-Strafterm“.

## chapter_multilayer-perceptrons/generalization-deep.md#u01 (Codex Astra (medium), 2026-11-26)

- „minimum-norm interpolator“ → „Interpolant kleinster Norm“, „maximum-margin separator“ → „Maximum-Margin-Trennebene“.
- „memorization“ → „Auswendiglernen“, „random features“ → „zufällige Merkmale“.

## chapter_multilayer-perceptrons/generalization-deep.md#u02 (Codex Astra (medium), 2026-11-26)

- „patience“ → „Wartezeit“ für die Zahl der Epochen, die die Abbruchregel ohne hinreichende Verbesserung abwartet.
- „grokking“ → „Grokking“.

## chapter_builders-guide/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „module tree“ → „Modulbaum“, „state tree“ → „Zustandsbaum“, „buffer“ → „Puffer“.
- „lazy shape inference“ → „verzögerte Bestimmung von Tensorformen“.
- „master weights“ → „Hauptgewichte“, „straight-through estimator“ → „Straight-Through-Schätzer“.
- „hook“ → „der Hook“, „backbone“ → „Basisnetz“, „foundation model“ → „Basismodell“.

## chapter_multilayer-perceptrons/kaggle-house-price.md#u01 (Codex Astra (medium), 2026-11-26)

- „linear baseline“ → „lineares Referenzmodell“, „mean imputation“ → „Mittelwertimputation“, „log-RMSE“ → „Log-RMSE“.

## chapter_multilayer-perceptrons/kaggle-house-price.md#u02 (Codex Astra (medium), 2026-11-26)

- „target encoding“ → „Zielwertkodierung“, „median imputation“ → „Medianimputation“.

## chapter_builders-guide/model-construction.md#u01 (Codex Astra (medium), 2026-11-26)

- „child module“ → „Kindmodul“, „module tree“ → „Modulbaum“, „registry“ → „Register“, „sequential container“ → „sequenzieller Container“.
- „lazy initialization“ → „verzögerte Initialisierung“.

## chapter_builders-guide/model-construction.md#u02 (Codex Astra (medium), 2026-11-26)

- „body“ → „Hauptzweig“ beim Residualblock, „builder“ → „Aufbaufunktion“, „dry run“ → „Probelauf“.

## chapter_builders-guide/parameters-state-memory.md#u01 (Codex Astra (medium), 2026-11-26)

- „freezing“ → „Einfrieren“, „residual MLP“ → „Residual-MLP“, „device move“ → „Gerätewechsel“.

## chapter_builders-guide/parameters-state-memory.md#u02 (Codex Astra (medium), 2026-11-26)

- „output head“ → „Ausgabekopf“, „meta device“ → „Meta-Gerät“, „evaluation mode“ → „Auswertungsmodus“.

## chapter_builders-guide/init.md#u01 (Codex Astra (medium), 2026-11-26)

- „gain“ → „Verstärkungsfaktor“, „initializer“ → „Initialisierer“, „truncated normal“ → „gestutzte Normalverteilung“.
- „body“ und „residual branch“ → „Residualzweig“, „expansion layer“ → „Erweiterungsschicht“.
- „kernel“ → „Gewichtsmatrix“ bei linearen Schichten in Keras und NNX.

## chapter_builders-guide/custom-layers.md#u01 (Codex Astra (medium), 2026-11-26)

- „custom layer“ → „eigene Schicht“, „gain“ → „Verstärkungsfaktor“ je Komponente, „gain vector“ → „Verstärkungsvektor“.

## chapter_builders-guide/custom-layers.md#u02 (Codex Astra (medium), 2026-11-26)

- „surrogate gradient“ → „Ersatzgradient“, „surrogate derivative“ → „Ersatzableitung“, „gradient check“ → „Gradientenprüfung“, „linear head“ → „lineare Ausgabeschicht“.

## chapter_builders-guide/numerics.md#u01 (Codex Astra (medium), 2026-11-26)

- „subnormal number“ → „subnormale Zahl“, „grid spacing“ → „Rasterabstand“.
- „dtype promotion“ → „automatische Typumwandlung“, „cast“ → „umwandeln“.

## chapter_builders-guide/numerics.md#u03 (Codex Astra (medium), 2026-11-26)

- „accumulator“ → „Akkumulator“, „shifted log-sum-exp“ → „Log-Sum-Exp mit Verschiebung“.

## chapter_builders-guide/saving-loading.md#u01 (Codex Astra (medium), 2026-11-26)

- „state dictionary“ → „Zustandswörterbuch“, „header“ → „Dateikopf“ bei serialisierten Dateien, „memory mapping“ → „Speicherabbildung“.

## chapter_builders-guide/saving-loading.md#u02 (Codex Astra (medium), 2026-11-26)

- „manifest“ → „Manifestdatei“ für das Verzeichnis der Dateien eines Checkpoints.

## chapter_builders-guide/gpus-devices-memory.md#u01 (Codex Astra (medium), 2026-11-26)

- „device placement“ → „Gerätezuordnung“, „command queue“ → „Befehlswarteschlange“, „pinned memory“ → „auslagerungsgesperrter Speicher“.

## chapter_builders-guide/gpus-devices-memory.md#u02 (Codex Astra (medium), 2026-11-26)

- „caching allocator“ → „Speicherallokator mit Cache“, „step memory“ → „Speicherbedarf pro Schritt“.

## chapter_convolutional-neural-networks/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „locality“ → „Lokalität“.
- „activation map“ → „Aktivierungskarte“.

## chapter_builders-guide/reproducibility-inspection.md#u01 (Codex Astra (medium), 2026-11-26)

- „random stream“ → „Zufallsfolge“, „generator object“ → „Generatorobjekt“, „forward hook“ und „backward hook“ → „Vorwärts-Hook“ und „Rückwärts-Hook“.

## chapter_builders-guide/reproducibility-inspection.md#u02 (Codex Astra (medium), 2026-11-26)

- „body“ eines Residualblocks → „Residualzweig“, „bitwise determinism“ → „bitgenauer Determinismus“.

## chapter_convolutional-neural-networks/conv-layer.md#u01 (Codex Astra (medium), 2026-11-26)

- „edge map“ → „Kantenkarte“, „edge detector“ → „Kantendetektor“.

## chapter_convolutional-neural-networks/conv-layer.md#u02 (Codex Astra (medium), 2026-11-26)

- „patch matrix“ → „Ausschnittsmatrix“, mit einem als Zeile geschriebenen Eingabeausschnitt pro Fensterposition.

## chapter_convolutional-neural-networks/why-conv.md#u01 (Codex Astra (medium), 2026-11-26)

- „sample“ bei einem diskreten Signal → „Abtastwert“, „unit impulse“ → „Einheitsimpuls“.
- „response map“ → „Antwortkarte“, „score map“ → „Score-Karte“, „offset“ bei Gitterkoordinaten → „Versatz“.
- „sample efficiency“ → „Dateneffizienz“, „locality“ → „Lokalität“.
- „Where's Waldo“ → „Wo ist Walter?“ (deutscher Titel), entsprechend „Walter“ im Text.

## chapter_convolutional-neural-networks/why-conv.md#u02 (Codex Astra (medium), 2026-11-26)

- „rotation equivariance“ → „Rotationsäquivarianz“.

## chapter_convolutional-neural-networks/padding-and-strides.md#u01 (Codex Astra (medium), 2026-11-26)

- „reflect padding“ und „mirror padding“ → „spiegelndes Padding“, „replicate padding“ → „Padding durch Wiederholung“.
- „kernel tap“ → „Abtastpunkt des Faltungskerns“, „shape-preserving padding“ → „formerhaltendes Padding“.

## chapter_convolutional-neural-networks/padding-and-strides.md#u02 (Codex Astra (medium), 2026-11-26)

- „patchify stem“ → „Patchify-Eingangsschicht“, „gridding artifacts“ → „Gitterartefakte“.

## chapter_convolutional-neural-networks/channels.md#u01 (Codex Astra (medium), 2026-11-26)

- „depthwise-separable convolution“ → „tiefenweise separierbare Faltung“.

## chapter_convolutional-neural-networks/pooling.md#u01 (Codex Astra (medium), 2026-11-26)

- „shift tolerance“ → „Toleranz gegenüber Verschiebungen“, auf Folien auch „Verschiebungstoleranz“.
- „peak activation“ → „Spitzenaktivierung“.

## chapter_convolutional-neural-networks/pooling.md#u02 (Codex Astra (medium), 2026-11-26)

- „softmax pooling“ → „Softmax-Pooling“, „minimum pooling“ → „Minimum-Pooling“.
- „blur-pool“ → „BlurPool“ (Schreibweise wie auf den Folien).

## chapter_convolutional-modern/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „stem, body, head“ → „Eingangsteil, Hauptteil, Kopf“ für die Gliederung eines ganzen Netzes.
- „dense head“ → „vollvernetzter Kopf“, „shortcut“ → „Sprungverbindung“, „compound scaling“ → „kombinierte Skalierung“.

## chapter_convolutional-neural-networks/lenet.md#u01 (Codex Astra (medium), 2026-11-26)

- „convolutional encoder“ → „Faltungsencoder“, „dense classification head“ → „vollständig verbundener Klassifikationskopf“, kurz „vollständig verbundener Kopf“.

## chapter_convolutional-modern/alexnet.md#u01 (Codex Astra (medium), 2026-11-26)

- „classifier head“ und „classification head“ → „Klassifikationskopf“, „top-5 error“ → „Top-5-Fehler“.
- „color-opponent blobs“ → „Flecken in Gegenfarben“.

## chapter_convolutional-modern/batch-norm.md#u01 (Codex Astra (medium), 2026-11-26)

- „group normalization“ → „Gruppennormalisierung“, „scale invariance“ → „Skaleninvarianz“, „peak learning rate“ → „maximale Lernrate“.
- „learned scale and shift“ → „gelernter Skalierungsfaktor und gelernte Verschiebung“, „decay“ bei gleitenden Mittelwerten → „Abklingfaktor“.

## chapter_convolutional-modern/batch-norm.md#u02 (Codex Astra (medium), 2026-11-26)

- „instance normalization“ → „Instanznormalisierung“, „running estimates“ → „gleitende Schätzwerte“.

## chapter_convolutional-modern/batch-norm.md#u03 (Codex Astra (medium), 2026-11-26)

- „normalizer-free networks“ → „Netze ohne Normalisierungsschichten“.

## chapter_convolutional-modern/blocks.md#u01 (Codex Astra (medium), 2026-11-26)

- „quarter-width VGG“ → „VGG mit einem Viertel der Breite“, „convolutional trunk“ → „Faltungsteil“.
- „cosine schedule with warmup“ → „Kosinusplan mit Aufwärmphase“, „peak learning rate“ → „maximale Lernrate“.

## chapter_convolutional-modern/blocks.md#u02 (Codex Astra (medium), 2026-11-26)

- „pooling head“ → „Pooling-Kopf“, „backbone“ → „Grundnetz“.

## chapter_convolutional-modern/resnet.md#u01 (Codex Astra (medium), 2026-11-26)

- „plain network/block“ → „unverzweigtes Netz/unverzweigter Block“ im Vergleich mit ResNet, „degradation problem“ → „Degradationsproblem“, „nested function classes“ → „geschachtelte Funktionenklassen“.

## chapter_convolutional-modern/resnet.md#u03 (Codex Astra (medium), 2026-11-26)

- „dense block“ → „dichter Block“, „growth rate“ → „Wachstumsrate“, „transition layer“ → „Übergangsschicht“.

## chapter_convolutional-modern/training-recipes.md#u01 (Codex Astra (medium), 2026-11-26)

- „residual branch“ → „Residualzweig“, „shadow copy“ bei der Gewichtsmittelung → „Schattenkopie“.
- „stochastic weight averaging“ → „stochastische Gewichtsmittelung“, „effective averaging horizon“ → „effektives Zeitfenster“.

## chapter_convolutional-modern/efficient-convnets.md#u01 (Codex Astra (medium), 2026-11-26)

- „structural re-parameterization“ → „strukturelle Reparametrisierung“, „inverted bottleneck“ → „invertierter Flaschenhals“.

## chapter_convolutional-modern/efficient-convnets.md#u02 (Codex Astra (medium), 2026-11-26)

- „BN folding“ → „Einrechnen der Batchnormalisierung in die Faltung“, „branch fusion“ → „Fusion der Zweige“.

## chapter_convolutional-modern/convnext.md#u01 (Codex Astra (medium), 2026-11-26)

- „patchify stem“ → „Patch-Eingangsteil“, „layer scale“ → „Schichtskalierung“, „inverted bottleneck“ → „invertierter Flaschenhals“.

## chapter_convolutional-modern/convnext.md#u03 (Codex Astra (medium), 2026-11-26)

- „deformable convolution“ → „verformbare Faltung“.

## chapter_recurrent-neural-networks/index.md#u01 (Codex Astra (medium), 2026-11-26)

- “bits per byte” → „Bit pro Byte“ (bpb).
- “rollout” → „wiederholte Vorhersage“ im Kontext autoregressiver Sequenzmodelle.
- “count-based language model” → „häufigkeitsbasiertes Sprachmodell“.
- “truncated sampling” → „Sampling aus abgeschnittenen Verteilungen“.

## chapter_convolutional-modern/cnn-design.md#u01 (Codex Astra (medium), 2026-11-26)

- „design space“ → „Entwurfsraum“, „neural architecture search“ → „neuronale Architektursuche“.
- „stage depth/width“ → „Stufentiefe/Stufenbreite“, „group width/count“ → „Gruppenbreite/Gruppenzahl“, „bottleneck ratio“ → „Flaschenhalsverhältnis“.

## chapter_recurrent-neural-networks/sequence.md#u01 (Codex Astra (medium), 2026-11-26)

- „fixed window“ → „festes Fenster“, „window length“ → „Fensterlänge“.
- „latent summary“ → „latente Zusammenfassung“.
- „Markov condition“ → „Markov-Bedingung“, „Markov model“ → „Markov-Modell“.
- „rollout“ → „rekursive Vorhersage“, bei Bedarf „rekursive Mehrschrittvorhersage“.

## chapter_recurrent-neural-networks/sequence.md#u02 (Codex Astra (medium), 2026-11-26)

- „one-step prediction“ → „Einschrittvorhersage“, „k-step-ahead prediction“ → „k-Schritt-Vorhersage“.
- „lag features“ → „zeitversetzte Merkmale“, „moving average“ → „gleitender Mittelwert“, „phase lag“ → „Phasenverzögerung“.

## chapter_recurrent-neural-networks/sequence.md#u03 (Codex Astra (medium), 2026-11-26)

- „regime switching“ → „Regimewechsel“.

## chapter_recurrent-neural-networks/text-sequence.md#u01 (Codex Astra (medium), 2026-11-26)

- “merge” → „Verschmelzung“, als Verb „verschmelzen“. “merge rank” → „Verschmelzungsrang“, “merge tree” → „Verschmelzungsbaum“.
- “pre-tokenization” → „Vortokenisierung“, “special token” → „Sondertoken“.

## chapter_recurrent-neural-networks/text-sequence.md#u03 (Codex Astra (medium), 2026-11-26)

- “fertility” → „Fertilität“ (mittlere Anzahl der Token pro Wort in einer Sprache).
- “glitch token” → „Glitch-Token“.

## chapter_recurrent-neural-networks/language-model.md#u01 (Codex Astra (medium), 2026-11-26)

- „count table“ → „Häufigkeitstabelle“, „unigram / bigram / trigram“ → „Unigramm / Bigramm / Trigramm“.
- „Zipf's law“ → „Zipf-Gesetz“, „heavy tail“ → „schwerer Rand“, „effective branching factor“ → „effektiver Verzweigungsfaktor“.

## chapter_recurrent-neural-networks/language-model.md#u02 (Codex Astra (medium), 2026-11-26)

- „absolute discounting“ → „absolute Diskontierung“, „continuation count“ → „Fortsetzungshäufigkeit“, „pseudocount“ → „Pseudohäufigkeit“.
- „Laplace smoothing“ → „Laplace-Glättung“, „interpolated Kneser–Ney smoothing“ → „interpolierte Kneser-Ney-Glättung“.

## chapter_recurrent-neural-networks/bptt.md#u01 (Codex Astra (medium), 2026-11-26)

- „state Jacobian“ → „Jacobi-Matrix des Zustandsübergangs“, „lag“ → „zeitlicher Abstand“, kurz „Abstand“.
- „detach“ → „vom Berechnungsgraphen abtrennen“.

## chapter_recurrent-neural-networks/decoding.md#u01 (Codex Astra (medium), 2026-11-26)

- „neural text degeneration“ → „Degeneration neuronal erzeugter Texte“.
- „open-ended generation“ → „offene Textgenerierung“, „temperature scaling“ → „Temperaturskalierung“.

## chapter_recurrent-neural-networks/decoding.md#u02 (Codex Astra (medium), 2026-11-26)

- „unreliable tail“ → „unzuverlässiger Rand“, „support set“ → „Trägermenge“.

## chapter_recurrent-neural-networks/rnn.md#u01 (Codex Astra (medium), 2026-11-26)

- „input projection“ → „Eingabeprojektion“, „recurrent cell“ → „rekurrente Zelle“, „embedding lookup“ → „Einbettungen nachschlagen“.

## chapter_recurrent-neural-networks/rnn.md#u02 (Codex Astra (medium), 2026-11-26)

- „effective context horizon“ → „effektive Kontextlänge“, „validation perplexity“ → „Validierungsperplexität“.

## chapter_optimization/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „step-size schedule“ → „Schrittweitenplan“, „step-size budget“ → „Schrittweitenbudget“.
- „edge of stability“ → „Stabilitätsgrenze“, „excess loss“ → „Überschussverlust“.

## chapter_optimization/gd.md#u01 (Codex Astra (medium), 2026-11-26)

- „edge of stability“ → „Rand der Stabilität“ für das Trainingsregime, „Stabilitätsgrenze“ für die numerische Schranke.
- „progressive sharpening“ → „fortschreitende Zunahme der Schärfe“.
- „strict saddle point“ → „strikter Sattelpunkt“.

## chapter_optimization/gd.md#u02 (Codex Astra (medium), 2026-11-26)

- „full-batch gradient descent“ → „Gradientenabstieg auf dem gesamten Trainingssatz“.
- „preconditioned sharpness“ → „vorkonditionierte Schärfe“.

## chapter_advanced/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „temporal-difference learning“ → „Lernen aus zeitlichen Differenzen“, „Model Predictive Control“ → „modellprädiktive Regelung“.
- „gradient penalty“ → „Gradientenstrafe“, „velocity field“ → „Geschwindigkeitsfeld“, „consistency distillation“ → „Konsistenzdestillation“.

## chapter_optimization/test-models.md#u01 (Codex Astra (medium), 2026-11-26)

- „learning-rate sweep“ → „Lernratensuche“, „usable learning-rate range“ → „nutzbarer Lernratenbereich“.
- „loss landmarks“ → „Referenzwerte für den Verlust“, „nuisance hyperparameter“ → „Störhyperparameter“.
- „shard“ → „Teildatei“ bei einem auf Dateien verteilten Korpus.

## chapter_optimization/test-models.md#u02 (Codex Astra (medium), 2026-11-26)

- „add-one smoothing“ → „Add-eins-Glättung“.

## chapter_optimization/momentum.md#u01 (Codex Astra (medium), 2026-11-26)

- „velocity“ → „Geschwindigkeit“, „memory window“ → „Gedächtnisfenster“, „critical damping“ → „kritische Dämpfung“.
- „curvature-limited“ → „krümmungsbegrenzt“, „noise-limited“ → „rauschbegrenzt“.

## chapter_optimization/momentum.md#u02 (Codex Astra (medium), 2026-11-26)

- „look-ahead point“ → „vorausberechneter Punkt“, „anticipatory damping“ → „vorausschauende Dämpfung“.

## chapter_optimization/sgd.md#u01 (Codex Astra (medium), 2026-11-26)

- „interpolating regime“ → „interpolierendes Regime“, „noise-limited regime“ → „rauschbegrenztes Regime“.
- „sampling without replacement“ → „Ziehen ohne Zurücklegen“, „random reshuffling“ → „zufälliges Mischen“ zu Beginn jeder Epoche.
- „finite-population correction factor“ → „Korrekturfaktor für die endliche Grundgesamtheit“.

## chapter_optimization/lr-scheduler.md#u01 (Codex Astra (medium), 2026-11-26)

- „cumulative learning-rate budget“ → „kumulatives Lernratenbudget“, „terminal learning rate“ → „Endlernrate“, „remaining budget“ → „Restbudget“.
- „cooldown“ → „Abkühlphase“, „stable–then–decay“ → „Plateau mit anschließendem Abfall“. Den Namen „Warmup–Stable–Decay“ mit dem Kürzel WSD beibehalten und die drei Phasen bei der Einführung benennen.

## chapter_optimization/muon.md#u01 (Codex Astra (medium), 2026-11-26)

- „steepest descent“ → „steilster Abstieg“, „sign descent“ → „Vorzeichenabstieg“.
- „norm ball“ → „Normkugel“, „coordinate box“ → „achsenparalleler Würfel“ bei einheitlicher Schranke für alle Koordinaten.
- „orthogonalized gradient“ → „orthogonalisierter Gradient“, „Newton–Schulz iteration“ → „Newton-Schulz-Iteration“.

## chapter_optimization/muon.md#u02 (Codex Astra (medium), 2026-11-26)

- „entrywise RMS“ → „quadratischer Mittelwert der Einträge“, nach Einführung auch „RMS-Wert der Einträge“.

## chapter_optimization/adam.md#u01 (Codex Astra (medium), 2026-11-26)

- „sign descent“ → „Vorzeichenabstieg“, „decay timescale“ → „Zeitskala des Gewichtszerfalls“, „equilibrium norm“ → „Gleichgewichtsnorm“.

## chapter_optimization/adam.md#u02 (Codex Astra (medium), 2026-11-26)

- „rotational equilibrium“ → „Rotationsgleichgewicht“, „relative angular step“ → „relative Winkelschrittweite“.

## chapter_optimization/batch-size.md#u01 (Codex Astra (medium), 2026-11-26)

- „gradient-noise scale“ → „Skala des Gradientenrauschens“, kurz „Rauschskala“.
- „two-batch estimator“ → „Schätzer mit zwei Batchgrößen“.
- „square-root scaling“ → „Wurzelskalierung“, „square-root rule“ → „Wurzelregel“.
- „critical batch size“ → „kritische Batchgröße“, „Pareto frontier“ → „Pareto-Front“.

## chapter_optimization/scaling.md#u01 (Codex Astra (medium), 2026-11-26)

- „proxy model“ → „Stellvertretermodell“, „coordinate check“ → „Koordinatentest“.
- „standard parametrization“ → „Standardparametrisierung“, „Maximal Update Parametrization“ bleibt als Verfahrensname im Original.
- „base width“ → „Basisbreite“, „width multiplier“ → „Breitenfaktor“, „output readout matrix“ → „Ausgabematrix“.
- „zero-shot learning-rate transfer“ → „Übertragung der Lernrate ohne erneute Abstimmung“.

## chapter_optimization/scaling.md#u02 (Codex Astra (medium), 2026-11-26)

- „decay timescale“ → „Zeitskala des Gewichtszerfalls“.
- „spectral scaling condition“ → „spektrale Skalierungsbedingung“, kurz „Spektralbedingung“.

## chapter_optimization/further-reading.md#u01 (Codex Astra (medium), 2026-11-26)

- „mode connectivity“ → „Verbindung zwischen Minima“, bei der Einführung mit englischer Glosse.
- „Hessian-free optimization“ → „Hesse-freie Optimierung“, „Hessian–vector product“ → „Hesse-Vektor-Produkt“.
- „iterate averaging“ → „Mittelung der Iterierten“, „warm restart“ → „Warmstart“.

## chapter_attention/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „soft lookup“ → „weicher Zugriff“, „attention head“ → „Attention-Kopf“, „sliding window“ → „gleitendes Fenster“.

## chapter_optimization/practice.md#u01 (Codex Astra (medium), 2026-11-26)

- „global-norm gradient clipping“ → „Gradienten-Clipping anhand der globalen Norm“.
- „guard“ → „Schutz“, „constant brake“ → „Dauerbremse“.
- „live weights“ → „aktuelle Gewichte“, „weight EMA“ → „EMA der Gewichte“.
- „proxy run“ → „Probelauf“, „clipping fraction“ → „Anteil der Schritte mit Clipping“.

## chapter_attention/queries-keys-values.md#u01 (Codex Astra (medium), 2026-11-26)

- „soft lookup“, „hard lookup“ → „weiches Nachschlagen“, „hartes Nachschlagen“.
- „Nadaraya–Watson kernel regression“, „similarity kernel“, „bandwidth“ → „Nadaraya-Watson-Kernel-Regression“, „Ähnlichkeitskernel“, „Bandbreite“.
- „compatibility score“ → „Kompatibilitäts-Score“.

## chapter_attention/queries-keys-values.md#u02 (Codex Astra (medium), 2026-11-26)

- „attention heatmap“ → „Attention-Heatmap“.

## chapter_attention/attention-scoring.md#u01 (Codex Astra (medium), 2026-11-26)

- „dot-product score“ → „Skalarprodukt-Score“, „head width“ → „Kopfbreite“, „score gap“ → „Score-Abstand“.

## chapter_attention/attention-scoring.md#u02 (Codex Astra (medium), 2026-11-26)

- „packed sequences“ → „gepackte Sequenzen“, „position-aware keys“ → „Schlüssel mit Positionsinformation“.

## chapter_attention/multihead-attention.md#u01 (Codex Astra (medium), 2026-11-26)

- „copy-both task“ → „Aufgabe, beide Werte zu kopieren“, im laufenden Text auch „beim Kopieren beider Werte“.
- „linear readout“ → „lineares Auslesen“, „mixing matrix“ → „Mischmatrix“, „head width“ → „Kopfbreite“.

## chapter_attention/multihead-attention.md#u02 (Codex Astra (medium), 2026-11-26)

- „key–value head“ → „Schlüssel-Wert-Kopf“, in Kurzform „KV-Kopf“.

## chapter_attention/positional-information.md#u01 (Codex Astra (medium), 2026-11-26)

- „permutation equivariance“ → „Permutationsäquivarianz“, „position embedding“ → „Positionseinbettung“, „distance penalty“ → „Abstandsstrafe“.
- „NTK-aware scaling“ → „NTK-bewusste Skalierung“, „content–position cross terms“ → „gemischte Terme aus Inhalt und Position“.

## chapter_attention/attention-at-scale.md#u01 (Codex Astra (medium), 2026-11-26)

- „online softmax“ → „Online-Softmax-Algorithmus“, für die Berechnung „Online-Softmax-Berechnung“.
- „score stripe“ → „Streifen der Score-Matrix“, „tile“ → „Kachel“, „fused kernel“ → „fusionierter Kernel“, „memory traffic“ → „Speicherverkehr“.

## chapter_attention/windowed-linear-attention.md#u01 (Codex Astra (medium), 2026-11-26)

- „sliding-window attention“ → „Attention mit gleitendem Fenster“, „windowed layer“ → „Fensterschicht“.
- „fused kernel“ → „fusionierter Kernel“, „tile“ → „Kachel“, „chunked computation“ → „blockweise Berechnung“.
- „cross-talk“ → „Störbeiträge“ anderer gespeicherter Werte, „decay gate“ → „Vergessensgate“ (das).
- „fast weight programmer“ → „Fast Weight Programmer“ (der).

## chapter_attention/windowed-linear-attention.md#u02 (Codex Astra (medium), 2026-11-26)

- „scalar decay“ → „skalarer Abklingfaktor“.

## chapter_transformers/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „open-weights LLM“ → „großes Sprachmodell mit offenen Gewichten“.
- „conditional computation“ → „bedingte Berechnung“, „speculative decoding“ → „spekulative Dekodierung“.
- „token collapse“ → „Kollaps der Tokenrepräsentationen“.

## chapter_attention/further-reading.md#u01 (Codex Astra (medium), 2026-11-26)

- „circuit“ im mechanistischen Sinn → „Schaltkreis“, „open-weights model“ → „Modell mit offenen Gewichten“.
- „activation patching“, „attribution patching“ → „das Activation-Patching“, „das Attribution-Patching“.
- „associative recall“ → „assoziativer Abruf“, „distance bucket“ → „Abstandsklasse“, „learnable sink logits“ → „lernbare Sink-Logits“.

## chapter_attention/what-attention-computes.md#u01 (Codex Astra (medium), 2026-11-26)

- „previous-token head“ → „Vorgängerkopf“, „match-and-copy head“ → „Kopf zum Abgleichen und Kopieren“.
- „skip-trigram“ → „Skip-Trigramm“, „fixed-offset shortcut“ → „Abkürzung über einen festen Versatz“.

## chapter_attention/what-attention-computes.md#u02 (Codex Astra (medium), 2026-11-26)

- „copy matrix“ → „Kopiermatrix“, „zero ablation“ → „Nullablation“, „mean ablation“ → „Mittelwertablation“.

## chapter_transformers/gpt.md#u01 (Codex Astra (medium), 2026-11-26)

- „decoder-only language model“ → „Sprachmodell mit reinem Decoder“, „byte-level GPT“ → „bytebasiertes GPT“.
- „pre-norm / post-norm“ → „vorgeschaltete / nachgeschaltete Normalisierung“, kurz „Pre-Norm / Post-Norm“.
- „language-modeling head“ → „Ausgabekopf des Sprachmodells“.

## chapter_transformers/gpt.md#u02 (Codex Astra (medium), 2026-11-26)

- „subword token“ → „Subworttoken“, „nucleus sampling“ → „Nucleus-Sampling“ (das).

## chapter_transformers/transformer-block.md#u01 (Codex Astra (medium), 2026-11-26)

- „token spread“ → „Tokenstreuung“, „token mixing“ und „channel mixing“ → „Mischen zwischen Token“ und „Mischen zwischen Kanälen“.
- „identity path“ → „Identitätspfad“, „gated feed-forward network“ → „Feedforward-Netz mit Gate“.

## chapter_transformers/kv-cache.md#u01 (Codex Astra (medium), 2026-11-26)

- „ridge point“ → „Knickpunkt“ der Roofline.
- „single-stream decode“ → „Dekodierung einer einzelnen Sequenz“, „batched decoding“ → „Dekodierung im Batch“.
- „low-rank key–value projections“ → „Schlüssel-Wert-Projektionen mit niedrigem Rang“.
- „uptraining“ → „Weitertraining“ nach dem Zusammenfassen der Köpfe.

## chapter_transformers/kv-cache.md#u02 (Codex Astra (medium), 2026-11-26)

- „mean-pooling“ → „Mittelwert-Pooling“, hier über die Projektionsköpfe einer Gruppe.
- „sink token“ → „Sink-Token“ (das).

## chapter_transformers/vision-transformer.md#u01 (Codex Astra (medium), 2026-11-26)

- „patch token“ → „Patch-Token“, „patch size“ → „Patch-Größe“.
- „grid distance“ → „Gitterabstand“, „position table“ → „Positionstabelle“.

## chapter_transformers/vision-transformer.md#u02 (Codex Astra (medium), 2026-11-26)

- „mean attention distance“ → „mittlerer Attention-Abstand“.

## chapter_transformers/encoders-decoders.md#u01 (Codex Astra (medium), 2026-11-26)

- „encoder-only / decoder-only“ → „reiner Encoder / reiner Decoder“, „wiring“ → „Verschaltung“, „latent bottleneck“ → „latenter Flaschenhals“, „latents“ → „latente Vektoren“.
- „greedy decoding“ → „Greedy-Dekodierung“.

## chapter_transformers/further-reading.md#u01 (Codex Astra (medium), 2026-11-26)

- „soft-capping“ → „glatte Begrenzung“, „draft model“ → „Entwurfsmodell“.
- „multi-head latent attention“ → „Multi-Head Latent Attention“ (die), „sparse upcycling“ → „Sparse Upcycling“ (das).

## chapter_transformers/scaling-laws.md#u01 (Codex Astra (medium), 2026-11-26)

- „non-embedding parameters“ → „Parameter ohne Einbettungen“.
- „scaling grid“ → „Versuchsraster“, „constant-loss contour“ → „Niveaulinie konstanten Verlusts“.
- „over-training for inference“ → „für die Inferenz über das Chinchilla-Optimum hinaus trainieren“.

## chapter_recurrent-modern/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „sequence mixer“ → „Sequenzmischer“, „matrix memory“ → „Matrixspeicher“, „delta rule“ → „Delta-Regel“.
- „readout“ → „Auslesefunktion“ für die Funktion, die aus dem Zustand die Ausgabe berechnet.

## chapter_transformers/moe.md#u01 (Codex Astra (medium), 2026-11-26)

- „routing collapse“ → „Routing-Kollaps“, „auxiliary loss“ → „Hilfsverlust“, „loss-free bias controller“ → „Bias-Regler ohne Hilfsverlust“.
- „routed expert“ → „vom Router auswählbarer Experte“, „expert buffer“ → „Expertenpuffer“, „gating weight“ → „Gate-Gewicht“.

## chapter_transformers/moe.md#u02 (Codex Astra (medium), 2026-11-26)

- „fine-grained expert segmentation“ → „feingranulare Unterteilung der Experten“.

## chapter_recurrent-modern/seq2seq.md#u01 (Codex Astra (medium), 2026-11-26)

- „string reversal“ → „Umkehrung von Zeichenketten“, im Experiment kurz „Umkehraufgabe“.
- „state width“ → „Zustandsbreite“, „per-letter accuracy“ → „Genauigkeit pro Buchstabe“.

## chapter_recurrent-modern/linear-recurrence.md#u01 (Codex Astra (medium), 2026-11-26)

- „decay“ → „Abklingfaktor“ für den Multiplikator, „Abklingen“ für den Vorgang. „write vector“ → „Schreibvektor“.
- „prefix scan“ → „Präfix-Scan“, „doubling scan“ → „Verdopplungsscan“, „work-efficient scan“ → „arbeitseffizienter Scan“. „combine“ → „Verknüpfung“.

## chapter_recurrent-modern/lstm.md#u01 (Codex Astra (medium), 2026-11-26)

- „gate“ → „Gatter“ für die rekurrenten Zellen, „memory cell“ → „Speicherzelle“, „candidate state“ → „Kandidatenzustand“.
- „delayed recall“ → „verzögerter Abruf“, „sequence mixer“ → „Sequenzmischer“ (der).

## chapter_recurrent-modern/lstm.md#u02 (Codex Astra (medium), 2026-11-26)

- „keep-gate initialization“ → „Initialisierung auf Erhalt“, „backbone“ → „Grundarchitektur“.

## chapter_recurrent-modern/ssm.md#u01 (Codex Astra (medium), 2026-11-26)

- „zero-order hold“ → „Halteglied nullter Ordnung“ (ZOH), „decay rate“ → „Abklingrate“, „per-step decay“ → „Abklingfaktor pro Schritt“.
- „delayed recall“ → „verzögerter Abruf“, „readout weights“ → „Auslesegewichte“, „leaky integrator“ → „verlustbehafteter Integrator“.

## chapter_recurrent-modern/mamba.md#u01 (Codex Astra (medium), 2026-11-26)

- „selective copying“ → „selektives Kopieren“, „filler token“ → „Fülltoken“, „query token“ → „Abfragetoken“.
- „read vector“ → „Lesevektor“, „adjoint scan“ → „adjungierter Scan“, „zero-order hold“ → „Halteglied nullter Ordnung“.

## chapter_recurrent-modern/mamba.md#u02 (Codex Astra (medium), 2026-11-26)

- „spike ratio“ → „Spitzenverhältnis“ für das Verhältnis des größten Beitrags im Symbolfenster zum Beitrag des vorherigen Fülltokens.
- „write“ als additiver Term der Rekurrenz → „Schreibbeitrag“.

## chapter_recurrent-modern/matrix-state.md#u01 (Codex Astra (medium), 2026-11-26)

- „matrix-state recurrence“ → „Rekurrenz mit Matrixzustand“, „retrieval error“ → „Abruffehler“.
- „state space duality“ → „Zustandsraumdualität“, „codebook“ → „Codebuch“.

## chapter_recurrent-modern/further-reading.md#u01 (Codex Astra (medium), 2026-11-26)

- „threshold circuit“ → „Schwellwertschaltung“, „fast weights“ → „schnelle Gewichte“.
- „network memories“ → „neuronale Netze als Speicher“, „log-linear attention“ → „log-lineare Attention“.

## chapter_computational-performance/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „computational performance“ → „Rechenleistung“, „performance model“ → „Leistungsmodell“.
- „host dispatch“ → „Start durch den Host“, bei GPU-Kerneln „Kernel-Start durch den Host“.
- „sharding“ → „Partitionierung“ beim Aufteilen des Trainingszustands auf mehrere GPUs.
- „ring allreduce“ → „Ring-Allreduce“, „DDP bucketing“ → „Bündelung von Gradienten in DDP“.

## chapter_recurrent-modern/deltanet.md#u01 (Codex Astra (medium), 2026-11-26)

- „key rebinding“ → „Neubelegung eines Schlüssels“, „write strength“ → „Schreibstärke“, „signal erosion“ → „Signalverlust“, „recall loss“ → „Abrufverlust“.

## chapter_recurrent-modern/deltanet.md#u02 (Codex Astra (medium), 2026-11-26)

- „cross-moment“ → „Kreuzmoment“, „state tracking“ → „Zustandsverfolgung“, „fast weights“ und „slow weights“ → „schnelle Gewichte“ und „langsame Gewichte“.

## chapter_recurrent-modern/hybrids.md#u01 (Codex Astra (medium), 2026-11-26)

- „Welch's bound“ → „Welch-Schranke“, „multi-query associative recall“ → „assoziativer Abruf mit mehreren Abfragen“. „Abrufgenauigkeit“ bezeichnet den Anteil korrekt abgerufener Werte.

## chapter_computational-performance/hardware.md#u01 (Codex Astra (medium), 2026-11-26)

- „bandwidth-bound“ → „bandbreitenbegrenzt“, „shoreline“ → „Chiprand“.
- „Tensor Cores“, „Streaming Multiprocessors“ und „Warps“ bleiben als Architekturbegriffe erhalten.
- „memory sweep time“ → „Zeit für einen vollständigen Speicherdurchlauf“.

## chapter_computational-performance/hardware.md#u02 (Codex Astra (medium), 2026-11-26)

- „pageable memory“ → „auslagerbarer Speicher“ als Gegenstück zu „auslagerungsgesperrter Speicher“.
- „microscaling“ → „Mikroskalierung“.

## chapter_computational-performance/compilation.md#u01 (Codex Astra (medium), 2026-11-26)

- „graph capture“ → „Grapherfassung“, „graph break“ → „Graphunterbrechung“, „guard“ → „Prüfbedingung“.
- „operation fusion“ und „operator fusion“ → „Operatorfusion“, „kernel replay“ → „Wiedergabe von Kernelstarts“.
- „tracing“ → „Tracing“, „placeholder“ → „Platzhalter“, „bucket“ bei Sequenzlängen → „Längenklasse“.

## chapter_computational-performance/performance-model.md#u01 (Codex Astra (medium), 2026-11-26)

- „bandwidth-bound“ → „bandbreitenbegrenzt“, „bandwidth roof“ → „Bandbreitengrenze“, „compute roof“ → „Grenze der Rechenleistung“.

## chapter_computational-performance/performance-model.md#u02 (Codex Astra (medium), 2026-11-26)

- „overhead-bound“ → „durch die Startkosten begrenzt“, wenn die Startzeit auf dem Host die Laufzeit bestimmt.
- „profiler“ → „der Profiler“, „profile“ → „das Laufzeitprofil“.

## chapter_computational-performance/multiple-gpus.md#u01 (Codex Astra (medium), 2026-11-26)

- „star allreduce“ → „Stern-Allreduce“, „shard“ → „Partition“ eines Minibatchs, „chunk“ → „Block“ eines Gradiententensors.
- „strong scaling“ und „weak scaling“ → „starke Skalierung“ und „schwache Skalierung“.

## chapter_computational-performance/multiple-gpus.md#u02 (Codex Astra (medium), 2026-11-26)

- „gradient fusion“ → „Gradientenfusion“, „bus bandwidth“ → „Busbandbreite“, „speedup“ als Verhältnis von Laufzeiten → „Beschleunigungsfaktor“.

## chapter_computational-performance/multi-gpu-practice.md#u01 (Codex Astra (medium), 2026-11-26)

- „gradient bucket“ → „Gradientenbündel“, kurz „Bündel“, „bucket model“ → „Bündelmodell“.
- „device mesh“ → „Gerätegitter“, „collective operation“ → „kollektive Operation“, „exposed communication time“ → „nicht überdeckte Kommunikationszeit“.
- „weak scaling“ → „schwache Skalierung“, „strong scaling“ → „starke Skalierung“, „SM contention“ → „Konkurrenz um SMs“.
- „allreduce“ → „der Allreduce“, Plural „die Allreduces“.

## chapter_computational-performance/memory-precision.md#u01 (Codex Astra (medium), 2026-11-26)

- „up-projection“ und „down-projection“ → „Aufwärtsprojektion“ und „Abwärtsprojektion“ beim SwiGLU-FFN.
- „peak memory“ → „Spitzenbedarf“ im Zusammenhang mit Speicherbelegung.

## chapter_computational-performance/memory-precision.md#u02 (Codex Astra (medium), 2026-11-26)

- „selective recomputation“ → „selektive Neuberechnung“, „chunked cross-entropy loss“ → „abschnittsweiser Kreuzentropieverlust“.

## chapter_computational-performance/further-reading.md#u01 (Codex Astra (medium), 2026-11-26)

- „preallocation“ → „Vorabreservierung“ bei der Speicherverwaltung.
- „trace“ → „Ablaufprotokoll“ bei der Laufzeitmessung.
- „tensor tile“ → „Tensorkachel“, „tiling“ → „Kachelung“.

## chapter_computational-performance/sharding.md#u01 (Codex Astra (medium), 2026-11-26)

- „shard“ → „Partition“ für den lokal gespeicherten Anteil eines Tensors, „gradient bucket“ → „Gradientenpuffer“.
- „collective“ → „kollektive Operation“, „training state“ → „Trainingszustand“.

## chapter_computational-performance/sharding.md#u02 (Codex Astra (medium), 2026-11-26)

- „prefetching“ → „Vorausladen“, „buffer donation“ → „Freigabe von Puffern zur Wiederverwendung“.

## chapter_computational-performance/fast-transformer.md#u01 (Codex Astra (medium), 2026-11-26)

- „kernel fusion“ → „Kernel-Fusion“, „measurement window“ → „Messfenster“, „rung“ → „Optimierungsstufe“, kurz „Stufe“.

## chapter_computational-performance/fast-transformer.md#u02 (Codex Astra (medium), 2026-11-26)

- „model FLOPs utilization“ → „Auslastung durch Modell-FLOPs“ (MFU).

## chapter_reinforcement-learning/mdp.md#u01 (Codex Astra (medium), 2026-11-26)

- „start-state distribution“ → „Verteilung der Startzustände“, „state augmentation“ → „Zustandserweiterung“.
- „reward shaping“ → „Belohnungsformung“, „verifier“ → „Prüfprogramm“.
- „termination“ → „Terminierung“, „truncation“ → „Abbruch“ im Kontext einer vorzeitig beendeten Aufzeichnung.

## chapter_reinforcement-learning/mdp.md#u02 (Codex Astra (medium), 2026-11-26)

- „belief state“ → „Informationszustand“, „absorbing state“ → „absorbierender Zustand“.

## chapter_reinforcement-learning/mdp.md#u03 (Codex Astra (medium), 2026-11-26)

- „approach bonus“ → „Annäherungsbonus“.

## chapter_reinforcement-learning/mdp.md#u04 (Codex Astra (medium), 2026-11-26)

- „hedge“ → „Ausweichantwort“, wenn eine Antwort alle möglichen Lösungen nennt.

## chapter_reinforcement-learning/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „optimal control“ → „optimale Regelung“, „linear-quadratic regulator“ → „linear-quadratischer Regler“, „receding-horizon model predictive control“ → „modellprädiktive Regelung mit gleitendem Horizont“.
- „action value“ → „Aktionswert“, „value methods“ → „wertbasierte Verfahren“, „policy methods“ → „strategiebasierte Verfahren“.
- „verifier“ → „Prüfprogramm“ bei der automatischen Bewertung einer generierten Antwort.

## chapter_reinforcement-learning/value-iter.md#u01 (Codex Astra (medium), 2026-11-26)

- „backup“ → „das Backup“, „sampled backup“ → „stichprobenbasiertes Backup“.
- „calm ice“ → „griffiges Eis“, „slippery ice“ → „rutschiges Eis“, „cell“ → „Feld“ auf der Karte.
- „approach bonus“ → „Annäherungsbonus“, „Bellman expectation equation“ → „Bellman-Erwartungswertgleichung“.

## chapter_reinforcement-learning/value-iter.md#u02 (Codex Astra (medium), 2026-11-26)

- „sweep“ → „Durchlauf“, „sup norm“ → „Supremumsnorm“, „stopping certificate“ → „beobachtbare Fehlerschranke“ als Abbruchkriterium.
- „uniform policy“ → „Gleichverteilungsstrategie“.

## chapter_reinforcement-learning/value-iter.md#u03 (Codex Astra (medium), 2026-11-26)

- „kernel“ im Übergangsmodell eines MDP → „Übergangskern“.
- „generalized policy iteration“ → „verallgemeinerte Strategieiteration“.

## chapter_reinforcement-learning/value-iter.md#u05 (Codex Astra (medium), 2026-11-26)

- „reward shaping“ → „Belohnungsgestaltung“, „potential-based shaping“ → „potentialbasierte Belohnungsgestaltung“.

## chapter_reinforcement-learning/exploration.md#u01 (Codex Astra (medium), 2026-11-26)

- „pull“ → „Zug“, „pull an arm“ → „an einem Arm ziehen“.
- „annealed rule“ → „Regel mit abklingender Explorationsrate“.
- „Thompson sampling“ → „Thompson-Sampling“.

## chapter_reinforcement-learning/exploration.md#u03 (Codex Astra (medium), 2026-11-26)

- „count-based bonus“ → „Bonus nach Besuchszahl“ für Zustand-Aktion-Paare.

## chapter_reinforcement-learning/qlearning.md#u01 (Codex Astra (medium), 2026-11-26)

- „backup“ → „das Backup“, „sampled backup“ → „Backup aus einem Sample“.
- „semi-gradient step“ → „Semigradientenschritt“, „double-sampling problem“ → „Problem des doppelten Samplings“.
- „maximization bias“ → „Maximierungsverzerrung“, „certainty equivalence“ → „Sicherheitsäquivalenz“.

## chapter_reinforcement-learning/qlearning.md#u02 (Codex Astra (medium), 2026-11-26)

- „seed“ → „Zufallsstartwert“ bei wiederholten Trainingsläufen.

## chapter_reinforcement-learning/imitation.md#u01 (Codex Astra (medium), 2026-11-26)

- „compounding error“ → „Fehlerakkumulation“, „sampling clone“ → „Klon mit Sampling“, „learner“ → „lernender Agent“.

## chapter_reinforcement-learning/imitation.md#u03 (Codex Astra (medium), 2026-11-26)

- „imitation learning“ → „Imitationslernen“, „on-policy distillation“ → „On-Policy-Destillation“.

## chapter_reinforcement-learning/optimal-control.md#u01 (Codex Astra (medium), 2026-11-26)

- „cost-to-go“ → „Restkosten“, „stage cost“ → „Stufenkosten“, „control effort“ → „Stellaufwand“, „control input“ → „Stellgröße“.
- „certainty equivalence“ → „Sicherheitsäquivalenz“, „plant“ → „Regelstrecke“, „double integrator“ → „Doppelintegrator“.
- „linear system“ → „lineares System“, wenn ein dynamisches System gemeint ist.

## chapter_reinforcement-learning/optimal-control.md#u02 (Codex Astra (medium), 2026-11-26)

- „nominal trajectory“ → „nominelle Trajektorie“, „forward rollout“ → „Vorwärtssimulation“, „swing-up“ → „Aufschwingen“ oder „Aufschwung“.
- „open-loop execution“ → „Steuerung ohne Rückführung“, „basin of attraction“ → „Einzugsgebiet“, „energy pumping“ → „gezielte Energiezufuhr“.

## chapter_reinforcement-learning/optimal-control.md#u03 (Codex Astra (medium), 2026-11-26)

- „tracking“ → „Bahnfolgeregelung“ für die Regelung entlang einer vorgegebenen Trajektorie, „warm start“ → „Initialisierung mit dem bisherigen Plan“ im MPC-Kontext.

## chapter_reinforcement-learning/optimal-control.md#u04 (Codex Astra (medium), 2026-11-26)

- „controllability matrix“ → „Steuerbarkeitsmatrix“, „costate“ → „Kozustand“, „adjoint“ → „adjungierte Variable“.

## chapter_reinforcement-learning/policy-gradient.md#u01 (Codex Astra (medium), 2026-11-26)

- „preference“ → „Präferenz“, „log-derivative trick“ → „Trick mit der logarithmischen Ableitung“.

## chapter_reinforcement-learning/policy-gradient.md#u02 (Codex Astra (medium), 2026-11-26)

- „surrogate loss“ → „Ersatzverlust“, „state occupancy“ → „Zustandsbelegung“.

## chapter_reinforcement-learning/baselines.md#u01 (Codex Astra (medium), 2026-11-26)

- „baseline“ → „die Baseline“, „probe policy“ → „Teststrategie“, „score“ → „Score“ für den Gradienten der Log-Wahrscheinlichkeit.

## chapter_reinforcement-learning/baselines.md#u02 (Codex Astra (medium), 2026-11-26)

- „leave-one-out baseline“ → „Leave-one-out-Baseline“, „group centering“ → „Gruppenzentrierung“.

## chapter_reinforcement-learning/baselines.md#u03 (Codex Astra (medium), 2026-11-26)

- „value baseline“ → „Wert-Baseline“, „step length“ → „Schrittlänge“ für die gemessene Länge des Parameterschritts.

## chapter_deep-reinforcement-learning/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „replay buffer“ → „Replay-Puffer“, „world model“ → „Weltmodell“, „policy gradient“ → „Strategiegradient“.
- „rollout“ → „Trajektorie“ für die aufgezeichnete Folge von Interaktionen, „simulierte Trajektorie“ bei Planung im Modell.
- „deadly triad“ → „tödliche Triade“, „twin critics“ → „zwei Kritiker“.

## chapter_reinforcement-learning/deep-rl.md#u01 (Codex Astra (medium), 2026-11-26)

- „tangent kernel“ → „Tangentenkernel“, „policy network“ → „Strategienetz“, „value network“ → „Wertnetz“.

## chapter_reinforcement-learning/deep-rl.md#u02 (Codex Astra (medium), 2026-11-26)

- „Gaussian policy“ → „Gauß-Strategie“, „pathwise gradient“ → „pfadweiser Gradient“, „score-function estimator“ → „Score-Funktionsschätzer“.

## chapter_reinforcement-learning/deep-rl.md#u03 (Codex Astra (medium), 2026-11-26)

- „deadly triad“ → „tödliche Triade“.

## chapter_deep-reinforcement-learning/actor-critic.md#u01 (Codex Astra (medium), 2026-11-26)

- „actor-critic method“ → „Akteur-Kritiker-Verfahren“, „multi-step return“ → „Mehrschrittertrag“, „one-step target“ → „Einschrittziel“.
- „rollout“ → „der Rollout“ für eine mit der Strategie erzeugte Folge von Übergängen.
- „bootstrap“ → „Bootstrap-Wert“ für die eingesetzte Wertschätzung, „Bootstrapping“ für das Verfahren im bestärkenden Lernen.

## chapter_deep-reinforcement-learning/actor-critic.md#u02 (Codex Astra (medium), 2026-11-26)

- „fitted TD“ → „Fitted TD“ für die wiederholte Regression auf neu berechnete TD-Ziele.

## chapter_deep-reinforcement-learning/actor-critic.md#u03 (Codex Astra (medium), 2026-11-26)

- „eligibility trace“ → „Berechtigungsspur“, „group baseline“ → „Gruppen-Baseline“.

## chapter_deep-reinforcement-learning/ppo.md#u01 (Codex Astra (medium), 2026-11-26)

- „surrogate objective“ → „Ersatzzielfunktion“, „performance difference lemma“ → „Lemma zur Ertragsdifferenz“, „clipping“ → „das Clipping“.

## chapter_deep-reinforcement-learning/ppo.md#u03 (Codex Astra (medium), 2026-11-26)

- „clip fraction“ → „Anteil außerhalb des Bands“, „approximate KL“ → „geschätzte KL-Divergenz“, „domain randomization“ → „Domänenrandomisierung“.

## chapter_deep-reinforcement-learning/regularized.md#u01 (Codex Astra (medium), 2026-11-26)

- „gridworld“ → „Gitterwelt“, „hazard lane“ → „Gefahrenstreifen“, „reference policy“ → „Referenzstrategie“.
- „fitted reward“ → „geschätzte Belohnung“, „rms error“ → „RMS-Fehler“.

## chapter_deep-reinforcement-learning/regularized.md#u02 (Codex Astra (medium), 2026-11-26)

- „exponential tilt“ → „exponentielle Umgewichtung“, „reward-divergence frontier“ → „Grenzkurve von Belohnung und Divergenz“.
- „point mass“ → „Punktmasse“, „fixed-reference penalty“ → „Strafterm mit fester Referenz“.

## chapter_deep-reinforcement-learning/regularized.md#u03 (Codex Astra (medium), 2026-11-26)

- „soft backup“ → „weiches Backup“, „soft value iteration“ → „weiche Wertiteration“, „soft value“ → „weicher Wert“.

## chapter_deep-reinforcement-learning/regularized.md#u04 (Codex Astra (medium), 2026-11-26)

- „trajectory divergence“ → „Divergenz von Trajektorienverteilungen“, wenn die KL-Divergenz zwischen den von zwei Strategien erzeugten Verteilungen gemeint ist.

## chapter_deep-reinforcement-learning/dqn.md#u01 (Codex Astra (medium), 2026-11-26)

- „online network“ → „Online-Netz“, „Q-network“ → „Q-Netz“ außerhalb des Modellnamens Deep Q-Network.
- „sampling noise“ → „Stichprobenrauschen“.

## chapter_deep-reinforcement-learning/dqn.md#u03 (Codex Astra (medium), 2026-11-26)

- „single estimator“ → „Einzelschätzer“, „double estimator“ → „Doppelschätzer“.

## chapter_deep-reinforcement-learning/model-based.md#u01 (Codex Astra (medium), 2026-11-26)

- „dynamics model“ → „Dynamikmodell“, „compounding error“ → „Fehlerfortpflanzung“ oder verbal „der Fehler pflanzt sich fort“.
- „random shooting“ → „Random Shooting“ (das), „policy distillation“ → „Strategiedestillation“.

## chapter_deep-reinforcement-learning/model-based.md#u04 (Codex Astra (medium), 2026-11-26)

- „cross-entropy method“ → „Kreuzentropiemethode“.

## chapter_deep-reinforcement-learning/sac.md#u01 (Codex Astra (medium), 2026-11-26)

- „soft value“, „soft policy evaluation“ und „soft policy improvement“ → „entropieregularisierter Wert“, „entropieregularisierte Strategiebewertung“ und „entropieregularisierte Strategieverbesserung“.
- „tilted distribution“ → „exponentiell umgewichtete Verteilung“, „reverse-KL projection“ → „Projektion bezüglich der umgekehrten KL-Divergenz“.

## chapter_deep-reinforcement-learning/sac.md#u02 (Codex Astra (medium), 2026-11-26)

- „squashed Gaussian policy“ → „gestauchte Gauß-Strategie“, „squash“ → „Stauchung“.
- „Jacobian term“ → „Jacobi-Term“ für den Beitrag der Jacobi-Determinante zur Log-Dichte.

## chapter_deep-reinforcement-learning/sac.md#u03 (Codex Astra (medium), 2026-11-26)

- „Polyak averaging“ → „Polyak-Mittelung“, „update-to-data ratio“ → „Verhältnis von Updates zu Daten“, quantitativ „Updates pro Umgebungsschritt“.

## chapter_deep-reinforcement-learning/sac.md#u04 (Codex Astra (medium), 2026-11-26)

- „primacy bias“ → „Übergewichtung früher Erfahrungen“ für das übermäßige Anpassen an die ersten Übergänge.

## chapter_deep-reinforcement-learning/further-reading.md#u01 (Codex Astra (medium), 2026-11-26)

- „prioritized replay“ → „priorisiertes Experience-Replay“, „multi-step target“ → „Mehrschrittziel“.
- „dueling network“ → „Dueling-Netz“, „dueling head“ → „Dueling-Kopf“, „clip band“ → „Clipping-Band“.

## chapter_deep-reinforcement-learning/further-reading.md#u02 (Codex Astra (medium), 2026-11-26)

- „diffusion policy“ → „Diffusionsstrategie“, „process reward“ → „Prozessbelohnung“, „expectile“ → „Expektil“.
- „pseudo-count“ → „Pseudo-Besuchszahl“ für eine aus einem Dichtemodell abgeleitete Besuchszahl.

## chapter_deep-reinforcement-learning/further-reading.md#u03 (Codex Astra (medium), 2026-11-26)

- „leave-one-out weight“ → „Leave-one-out-Gewicht“, analog zu „Leave-one-out-Baseline“.

## chapter_deep-reinforcement-learning/offline-rl.md#u01 (Codex Astra (medium), 2026-11-26)

- „coverage condition“ → „Abdeckungsbedingung“, „support“ bei gezählten Paaren → „Besuchszahl“.
- „stitching“ → „Verknüpfen von Teiltrajektorien“, „narrow log“ → „enges Protokoll“.
- „lower confidence bound“ → „untere Konfidenzschranke“, analog zur oberen Konfidenzschranke.

## chapter_deep-reinforcement-learning/offline-rl.md#u03 (Codex Astra (medium), 2026-11-26)

- „return conditioning“ → „Konditionierung auf den Ertrag“, adjektivisch „ertragskonditioniert“.

## chapter_deep-reinforcement-learning/rl-sequences.md#u01 (Codex Astra (medium), 2026-11-26)

- „response weight“ → „Antwortgewicht“, „token advantage“ → „Tokenvorteil“, „prefix value“ → „Präfixwert“.
- „tilted optimum“ → „exponentiell umgewichtetes Optimum“ für die mit dem exponentiellen Belohnungsfaktor umgewichtete Referenzstrategie.

## chapter_generative-adversarial-networks/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „generative adversarial network“ → „generatives adversariales Netz“ (GAN).
- „non-saturating generator loss“ → „nicht sättigender Generatorverlust“, „zero-centered gradient penalty“ → „um null zentrierter Gradientenstrafterm“.
- „log density ratio“ → „logarithmiertes Dichteverhältnis“, „witness“ bei integralen Wahrscheinlichkeitsmetriken → „Zeugenfunktion“.

## chapter_generative-adversarial-networks/relativistic.md#u01 (Codex Astra (medium), 2026-11-26)

- „relativistic pairing“ → „relativistische Paarbildung“, „pairing critic“ → „Paarkritiker“, „pairing divergence“ → „Paardivergenz“.
- „rank weight“ → „Ranggewicht“, „mode coverage“ → „Modusabdeckung“.

## chapter_generative-adversarial-networks/relativistic.md#u02 (Codex Astra (medium), 2026-11-26)

- „collision minimum“ → „Kollisionsminimum“, „fixed pairing“ → „feste Paarbildung“.

## chapter_generative-adversarial-networks/objectives.md#u01 (Codex Astra (medium), 2026-11-26)

- „Bayes-risk gap“ → „Differenz der Bayes-Risiken“, in Tabellen kurz „Risikodifferenz“.
- „critic link“ → „Linkfunktion des Kritikers“; „payoff“ → „Auszahlung“.
- „integral probability metric“ → „integrale Wahrscheinlichkeitsmetrik“.

## chapter_generative-adversarial-networks/objectives.md#u02 (Codex Astra (medium), 2026-11-26)

- „witness function“ → „Zeugenfunktion“, „sample complexity“ → „Stichprobenkomplexität“.

## chapter_generative-adversarial-networks/gan.md#u01 (Codex Astra (medium), 2026-11-26)

- „generative adversarial network“ → „generatives adversariales Netz“ (GAN).
- „realness logit“ → „Logit für die Echtheit“, „source label“ → „Herkunftslabel“.
- „log density ratio“ → „logarithmiertes Dichteverhältnis“, „log loss“ → „logarithmischer Verlust“.
- „instance noise“ → „Instance Noise“ (das).

## chapter_generative-adversarial-networks/gan.md#u02 (Codex Astra (medium), 2026-11-26)

- „softened reverse KL divergence“ → „abgeschwächte umgekehrte KL-Divergenz“.
- „saturating / non-saturating loss“ → „sättigender / nicht sättigender Verlust“.

## chapter_generative-adversarial-networks/convergence.md#u01 (Codex Astra (medium), 2026-11-26)

- „one-centered penalty“ → „um eins zentrierter Strafterm“, analog zum um null zentrierten Gradientenstrafterm.
- „underdamped“, „critically damped“, „overdamped“ → „unterkritisch gedämpft“, „kritisch gedämpft“, „überkritisch gedämpft“.
- „phase portrait“ → „Phasenporträt“.

## chapter_generative-adversarial-networks/convergence.md#u02 (Codex Astra (medium), 2026-11-26)

- „on-mode fraction“ → „Anteil modusnaher Samples“. Maßgeblich ist der Abstand zum nächsten Moduszentrum.
- „cutoff length scale“ → „Grenzlänge“, „critic steepness“ → „Steilheit des Kritikers“.

## chapter_generative-adversarial-networks/conditional.md#u01 (Codex Astra (medium), 2026-11-26)

- „class alignment“ → „Klassenübereinstimmung“, „condition alignment“ → „Übereinstimmung mit der Vorgabe“ oder verbal „die Vorgabe erfüllen“. Gemeint ist die Einhaltung der Bedingung, kein Alignment von Sequenzen.
- „sample fidelity“ → „Bildtreue“ bei generierten Bildern. „referee classifier“ → „Prüfklassifikator“.
- „projection discriminator“ → „Projektionsdiskriminator“, „projection critic“ → „Projektionskritiker“, „feature modulation“ → „Merkmalsmodulation“.

## chapter_generative-adversarial-networks/conditional.md#u02 (Codex Astra (medium), 2026-11-26)

- „real floor“ → „Vergleichswert für echte Bilder“, der zugehörige Datensatz heißt „echte Vergleichsgruppe“.
- „prototype baseline“ → „Prototypen-Baseline“, „truncation trick“ → „Truncation-Trick“.

## chapter_generative-adversarial-networks/dcgan.md#u01 (Codex Astra (medium), 2026-11-26)

- „checkerboard artifacts“ → „Schachbrettartefakte“, „half-life“ → „Halbwertszeit“.
- „fidelity“ bei erzeugten Bildern → „Bildtreue“.

## chapter_generative-adversarial-networks/dcgan.md#u02 (Codex Astra (medium), 2026-11-26)

- „experimental arm“ → „Versuchsvariante“, im laufenden Vergleich kurz „Variante“.

## chapter_generative-adversarial-networks/dcgan.md#u03 (Codex Astra (medium), 2026-11-26)

- „non-leaking augmentation“ → „Augmentierung, die nicht auf den Generator durchschlägt“.
- „real-versus-real floor“ → „Referenzwert für zwei echte Stichproben“.

## chapter_generative-adversarial-networks/further-reading.md#u01 (Codex Astra (medium), 2026-11-26)

- „extragradient method“ → „Extragradientenverfahren“, „optimistic gradient descent“ → „optimistischer Gradientenabstieg“.
- „alias-free resampling“ → „aliasfreie Neuabtastung“ bei der Verarbeitung von Bildsignalen.
- „diffusion distillation“ → „Diffusionsdestillation“, „few-step distillation“ → „Destillation auf wenige Schritte“.
- „lazy regularization“ → „intervallweise Regularisierung“ für die Berechnung des Strafterms in festen Abständen.

## chapter_diffusion-models/index.md#u01 (Codex Astra (medium), 2026-11-26)

- „score“ einer Dichte → „Score“ (der), im Sinne des Gradienten der logarithmischen Dichte.
- „sampler“ → „Sampler“ (der), „latents“ → „latente Darstellungen“, „log-SNR clock“ → „Zeitskala der Log-SNR“.
- „few-step distillation“ → „Destillation auf wenige Schritte“, „design space“ → „Entwurfsraum“.

## chapter_generative-adversarial-networks/adversarial-losses.md#u01 (Codex Astra (medium), 2026-11-26)

- „perception–distortion trade-off“ → „Zielkonflikt zwischen Wahrnehmungsqualität und Verzerrung“.
- „teacher / student“ → „Lehrer / Schüler“ bei der Destillation, „one-step generator“ → „Einschrittgenerator“.
- „two-band benchmark“ → „Benchmark mit zwei Datenbändern“, „audio frame“ → „Audiozeitrahmen“.

## chapter_generative-adversarial-networks/adversarial-losses.md#u02 (Codex Astra (medium), 2026-11-26)

- „scoring rule“ → „Bewertungsregel“.
