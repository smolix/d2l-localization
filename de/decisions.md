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
