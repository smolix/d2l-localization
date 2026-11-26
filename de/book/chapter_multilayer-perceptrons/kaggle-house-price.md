<!--en:8b5b7d76-->
# Vorhersage von Hauspreisen auf Kaggle

<!--en:86972bcd-->
Bilddatensätze für Benchmarks wie Fashion-MNIST sind sauber aufbereitet. Die Bilder haben ein einheitliches Pixelraster mit Werten in $[0, 1]$, und kein Wert fehlt. Tabellendaten aus der Praxis sind selten so ordentlich. Zum Abschluss des Kapitels nehmen wir uns den Kaggle-Wettbewerb [Hauspreise – fortgeschrittene Regressionsverfahren](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) vor. Die Daten hat :citet:`De-Cock.2011` aus Immobilienverkäufen in Ames im US-Bundesstaat Iowa (2006–2010) zusammengestellt. Für $1{,}460$ Häuser im Trainingsdatensatz kennen wir den Verkaufspreis, für $1{,}459$ Häuser im Testdatensatz ist er unbekannt. Jedes Haus ist durch $79$ unterschiedliche numerische und kategoriale Attribute beschrieben. Ihre Skalen erstrecken sich über fünf Größenordnungen, und in $19$ Spalten fehlen Werte.

<!--en:143bd6dc-->
Auf dem Weg von den Rohtabellen bis zur fertigen Vorhersage verbinden wir Werkzeuge aus :numref:`chap_preliminaries` bis :numref:`chap_perceptrons`. Wir verarbeiten die Daten mit `pandas` ohne Datenleckage (engl. data leakage), logarithmieren die Zielwerte und verwenden eine $K$-fache Kreuzvalidierung (engl. cross-validation) mit gemeinsamen Gruppen. Als Referenz dient die lineare Regression. Damit vergleichen wir MLPs mit He-Initialisierung, die wir durch Gewichtszerfall (engl. weight decay) regularisieren.

<!--en:0b0e4c29-->
## Der Datensatz und die Log-RMSE

<!--en:1ccd247b-->
Mit `d2l.download` laden wir die CSV-Dateien für Training und Test herunter und speichern sie zwischen. Dann lesen wir sie als DataFrames in `pandas` ein (:numref:`sec_pandas`):

<!--en:ea077454-->
Der Aufruf von `read_house_prices()` lädt die beiden Tabellen in `raw_train` und `raw_test`. An ihrer Form sehen wir, dass die Trainingsdaten $1{,}460$ Häuser mit $81$ Spalten (einschließlich `SalePrice`) und die Testdaten $1{,}459$ Häuser mit $80$ Spalten umfassen:

<!--en:7140d33c-->
Betrachten wir die ersten vier Zeilen und einige Spalten am Anfang und Ende von `raw_train`. Dieser Ausschnitt zeigt einen typischen Querschnitt durch die Daten: ganzzahlige Codes (`Id`, `MSSubClass`), Kategorien als Zeichenketten (`MSZoning`, `SaleType`, `SaleCondition`), kontinuierliche Messwerte (`LotFrontage`) und die Zielgröße `SalePrice` in US-Dollar:

<!--en:2f10f892-->
Von den $79$ Attributen, die nach dem Entfernen des nicht aussagekräftigen Zeilenindex `Id` für die Vorhersage bleiben, sind $36$ als Zahlen und $43$ als Zeichenketten gespeichert. Ein fehlender Wert bedeutet oft, dass das Haus ein bestimmtes Merkmal nicht aufweist (etwa `PoolQC = NaN` bei Häusern ohne Schwimmbecken).

<!--en:983b377c-->
Die Hauspreise reichen von $\$34{,}900$ bis $\$755{,}000$. Ein Fehler von $\$50{,}000$ fällt bei einem günstigen ersten Eigenheim schwer ins Gewicht, bei einer Villa dagegen kaum. Kaggle bewertet die Vorhersagen deshalb nach ihrer *relativen* Genauigkeit. Als Maß dient die Wurzel des mittleren quadratischen Fehlers (RMSE) der *Logarithmen* der Preise:

<!--en:7d17b4be-->
Für kleine $\delta = |\log \hat{y} - \log y|$ gilt $\hat{y}/y = e^{\pm \delta} \approx 1 \pm \delta$. Eine Log-RMSE von $0.135$ entspricht also einem typischen prozentualen Fehler von rund $13.5\%$. Trainieren wir unsere Modelle so, dass sie $\log y$ mit möglichst geringem quadratischem Verlust vorhersagen, minimieren wir direkt das Quadrat der Wettbewerbsmetrik.

<!--en:89c3dba3-->
## Vorverarbeitung ohne Datenleckage

<!--en:01737d51-->
Um die rohen DataFrames in Merkmalsmatrizen vom Typ float32 umzuwandeln, müssen wir vier Schritte beachten. Die Funktion `preprocess(train, *others)` weiter unten setzt sie um:

<!--en:9c8615c1-->
1. **Numerische Kategoriecodes in Zeichenketten umwandeln:** In `MSSubClass` bezeichnen Zahlen ($20, 60, 90, \ldots$) unterschiedliche Haustypen (eingeschossig, zweigeschossig, Zweifamilienhaus). Durch die Umwandlung in `str` verhindern wir, dass das Modell Code $90$ als lineares Vielfaches von Code $20$ behandelt.

<!--en:1412fb40-->
2. **Numerische Merkmale ausschließlich mit Kennwerten des *Trainingsdatensatzes* standardisieren:** Wir berechnen die Mittelwerte $\mu$ und Standardabweichungen $s$ nur auf `train`. Die Transformation $(x - \mu)/s$ wenden wir auf die Trainings-, Validierungs- und Testtabellen an. Damit erhalten alle numerischen Merkmale für SGD und Gewichtszerfall eine einheitliche Skala. Zugleich vermeiden wir eine *Datenleckage* aus den Validierungs- oder Testzeilen.

<!--en:0ec41510-->
3. **Fehlende numerische Werte durch $0$ (den Trainingsmittelwert) ersetzen:** Wenn wir nach der Standardisierung alle `NaN`-Einträge mit `0` füllen, ersetzen wir fehlende Messwerte durch den jeweiligen Mittelwert aus dem Trainingsdatensatz (Aufgabe :exref:`ex_kaggle-house-mean-imputation`).

<!--en:a86c491c-->
4. **One-Hot-Kodierung für kategoriale Merkmale verwenden (`pd.get_dummies(..., dummy_na=True)`):** Aus jeder kategorialen Spalte entstehen binäre Indikatoren, darunter ein eigener Indikator für `NaN`. Mit `reindex(..., fill_value=0)` erhalten die Validierungs- und Testtabellen genau dieselben Spalten wie die Trainingstabelle:

<!--en:668e94f2-->
Wenden wir `preprocess(raw_train, raw_test)` auf die vollständigen Tabellen des Wettbewerbs an, entstehen aus den $79$ Rohattributen zwei gleich gegliederte `float32`-Entwurfsmatrizen `X_train` und `X_test` mit je $345$ Merkmalen pro Haus:

<!--en:27200e13-->
Schließlich verbindet `KaggleHouse(d2l.DataModule)` die Funktion `preprocess` mit der Zentrierung der logarithmierten Zielwerte. Dazu ziehen wir den mittleren logarithmierten Preis der Trainingsdaten, `self.offset` ($\approx 12.02$), von $\log(\textrm{SalePrice})$ ab. So beginnt das Netz schon in Schritt 1 auf dem richtigen Preisniveau (Aufgabe :exref:`ex_kaggle-house-scaling`):

<!--en:d2c4dcfb-->
## Modellauswahl durch $K$-fache Kreuzvalidierung

<!--en:ad827284-->
### Gemeinsame Aufteilung in 5 Gruppen und lineares Referenzmodell

<!--en:cf5af55f-->
Bei nur $1{,}460$ Häusern mit bekanntem Preis liefert eine einzelne Validierungsgruppe mit $20\%$ der Daten ($292$ Häuser) für die Log-RMSE einen Standardfehler von mehreren Hundertsteln über die möglichen Gruppen hinweg. Unsere Hilfsfunktion `k_fold_data(raw, k)` teilt `raw` zufällig in $K$ Gruppen auf. Für jede Runde erzeugt sie ein eigenes Datenmodul vom Typ `KaggleHouse`. So gehen die Validierungsgruppen nie in die Merkmalsstatistiken der Trainingsdaten ein:

<!--en:0b94ae91-->
Ausgangspunkt sind die Liste der vorbereiteten Gruppen `folds` und die Funktion `make_model` zum Erzeugen eines Modells. In jeder Runde trainiert `k_fold` ein neues Modell über `max_epochs` Epochen. Um Schwankungen einzelner Epochen zu dämpfen, mittelt die Funktion den quadratischen Verlust der letzten fünf Epochen, bevor sie die Wurzel zieht. Für Training und Validierung gibt sie jeweils die über alle Gruppen gemittelte Log-RMSE aus:

<!--en:09de5faa-->
Wir erzeugen die gemeinsame Aufteilung in 5 Gruppen, `folds`, nur einmal. So bewerten wir im Folgenden alle Modelle mit genau denselben fünf Aufteilungen in Trainings- und Validierungsdaten:

<!--en:002090af-->
Definieren wir zunächst `make_linear` für unser Referenzmodell `d2l.LinearRegression` ($100$ Epochen, Batchgröße $64$, Lernrate $\eta = 0.02$):

<!--en:26abd8a1-->
Mit `k_fold(folds, make_linear)` trainieren wir das unregularisierte lineare Referenzmodell in allen fünf Runden. Die Vektoren mit den Log-RMSE-Werten für Training und Validierung speichern wir in `linear`:

<!--en:93525c5e-->
Bei der 5-fachen Kreuzvalidierung erreicht das lineare Referenzmodell eine Log-RMSE von $\sim 0.15$ auf den Validierungsdaten (typischer Preisfehler $\sim 15\%$) und von $\sim 0.11$ auf den Trainingsdaten.

<!--en:da553caa-->
### Ein MLP mit Gewichtszerfall

<!--en:7d395a05-->
Erstellen wir nun `HouseMLP`, ein MLP mit einer versteckten Schicht aus $64$ ReLU-Einheiten. Das sind $\sim 22{,}000$ Parameter oder $\sim 19$ Parameter pro Haus in den Trainingsdaten einer Runde. Die Gewichte initialisieren wir nach He. Der $\ell_2$-Gewichtszerfall `wd` wirkt ausschließlich auf die Gewichtsmatrizen, die Bias-Terme bleiben ausgenommen:

<!--en:7d0a3ca9-->
Für `wd` probieren wir nun alle Werte aus `wds = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]` mit derselben Aufteilung in 5 Gruppen `folds` aus. Die mittleren RMSE-Werte für Training und Validierung stellen wir denen des linearen Referenzmodells grafisch gegenüber:

<!--en:8fa8c6bb-->
Wir wählen den Gewichtszerfall `best` mit der niedrigsten mittleren Validierungs-RMSE über die 5 Gruppen. Nun können wir auch die gepaarte Differenz `mlp[best][1] - linear[1]` für jede Gruppe betrachten. Dabei heben sich die zufälligen Schwankungen durch die Gruppeneinteilung auf:

<!--en:2889386f-->
Die Kurve der Kreuzvalidierung zeigt die typische U-Form. Bei schwachem Gewichtszerfall (`wd = 0.001`) kommt es zur Überanpassung (engl. overfitting) an die Trainingsgruppen. Die Trainings-RMSE liegt bei $\sim 0.09$, die Validierungs-RMSE bei $\sim 0.15$–$0.18$ und damit höher als bei der linearen Regression. Mittlerer Gewichtszerfall (`wd = 0.03` oder `0.1`) liefert mit einer Validierungs-RMSE von $\sim 0.13$–$0.14$ die besten Ergebnisse und schlägt das lineare Referenzmodell in den meisten Gruppen. Zu starker Gewichtszerfall (`wd = 0.3`) führt zur Unteranpassung (engl. underfitting).

<!--en:6de16f63-->
## Einreichen der Vorhersagen

<!--en:9c921ce6-->
Mit dem Gewichtszerfall `best` aus der 5-fachen Kreuzvalidierung trainieren wir `HouseMLP` erneut, diesmal auf allen $1{,}460$ Häusern mit bekanntem Preis. Alternativ können wir die 5 Modelle der Kreuzvalidierung zu einem Ensemble zusammenfassen (siehe Aufgabe :exref:`ex_kaggle-house-fold-ensemble`). Zu den Vorhersagen addieren wir wieder den mittleren logarithmierten Preis der Trainingsdaten `data.offset`. Durch Exponenzieren erhalten wir die Vorhersagen in US-Dollar und speichern sie in `submission.csv`:

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:a3455337-->
Bei der Regression mit Tabellendaten aus der Praxis kommt es auf eine sorgfältige Vorverarbeitung ohne Datenleckage an. Wir entfernen willkürliche Zeilenkennungen und wandeln numerische Kategoriecodes in Zeichenketten um. Numerische Spalten standardisieren wir mit den Mittelwerten und Standardabweichungen der Trainingsdaten, bevor wir fehlende Einträge mit null füllen. Für kategoriale Merkmale verwenden wir eine One-Hot-Kodierung mit einem eigenen Indikator für fehlende Werte. Die logarithmierten Zielwerte zentrieren wir um den Trainingsmittelwert. Alle Modelle durchlaufen die Kreuzvalidierung mit denselben $K = 5$ Gruppen. In jeder Runde bestimmen wir die Vorverarbeitung neu aus den vier Trainingsgruppen. So heben sich beim direkten Modellvergleich zufällige Schwankungen durch die Gruppeneinteilung auf. Auf dem Datensatz Ames Housing können wir ein ReLU-MLP mit 22 000 Parametern durch Abstimmen des $\ell_2$-Gewichtszerfalls deutlich verbessern. Das MLP überwindet die anfängliche Überanpassung (Log-RMSE $\sim 0.15$–$0.18$), übertrifft das lineare Referenzmodell ($\sim 0.15$) und erreicht eine Log-RMSE von $\sim 0.13$–$0.14$. Bei heterogenen Tabellenmerkmalen bleiben Entscheidungsbäume mit Gradientenboosting allerdings starke Konkurrenten (Aufgabe :exref:`ex_kaggle-house-trees`).

<!--en:2294243a-->
## Aufgaben

<!--en:7d839c80-->
1. [code] **Mittelwertimputation.** `preprocess` ersetzt jeden fehlenden numerischen Wert durch den Mittelwert seiner Spalte im Trainingsdatensatz. Der Aufruf von `pd.get_dummies` mit `dummy_na=True` ergänzt außerdem jedes kategoriale Merkmal um eine Spalte mit einem Indikator für fehlende Werte.
    1. Finden Sie in diesem Datensatz ein numerisches Merkmal, bei dem das Fehlen eines Werts mit dem Verkaufspreis zusammenhängt.
    1. Erklären Sie, welche Verzerrung die Mittelwertimputation allein bei diesem Merkmal verursachen würde. In welche Richtung würden sich die Vorhersagen für die betroffenen Häuser verschieben? Prüfen Sie dann, ob ein lineares Modell diese Verzerrung mithilfe der Indikatorspalten beseitigen kann, die `preprocess` bereits erzeugt.
    1. Ersetzen Sie in einer Variante die Mittelwertimputation durch Medianimputation. Ergänzen Sie in einer anderen Variante jedes Merkmal, dessen fehlende Werte Sie imputieren, um einen binären Indikator für fehlende Werte. Vergleichen Sie die Fehler der drei Varianten in der Kreuzvalidierung.

    *Nach Kaggle Learn, [Maschinelles Lernen für Fortgeschrittene](https://www.kaggle.com/learn/intermediate-machine-learning), Aufgabe 2.*
   :exlabel:`ex_kaggle-house-mean-imputation`

<!--en:0395dfd0-->
1. [code] **Skalierung der Merkmale und der Zielgröße.** Entfernen Sie die Standardisierung aus `preprocess`. Da `fillna(0)` nur bei standardisierten Spalten den Spaltenmittelwert einsetzt, füllen Sie fehlende numerische Werte ausdrücklich mit den jeweiligen Spaltenmittelwerten des Trainingsdatensatzes.
    1. Wiederholen Sie die Kreuzvalidierung des linearen Modells mit der in diesem Abschnitt verwendeten Lernrate von $0.02$. Beschreiben Sie den Verlauf des Trainingsverlusts.
    1. Bestimmen Sie bis auf einen Faktor zehn die größte Lernrate, mit der sich das lineare Modell ohne Standardisierung trainieren lässt, ohne zu divergieren. Vergleichen Sie nach $100$ Epochen seinen Fehler in der Kreuzvalidierung mit dem Fehler des Modells mit standardisierten Daten. Erklären Sie beide Beobachtungen.
    1. Stellen Sie die Standardisierung wieder her. Trainieren Sie das Modell nun auf den logarithmierten Preisen, ohne deren Mittelwert abzuziehen. Vergleichen Sie den Fehler in der Kreuzvalidierung nach $100$ Epochen mit dem Fehler bei zentrierten Zielwerten. Erklären Sie den Unterschied.
    1. Verwenden Sie das Verfahren aus diesem Abschnitt. Protokollieren Sie nach jeder Epoche die Validierungs-RMSE des linearen Modells und den Mittelwert seiner Residuen auf der Validierungsgruppe. Was hat sich in den Epochen verschoben, in denen die RMSE sprunghaft ansteigt? Bei jedem Haus haben genau 44 One-Hot-Merkmale den Wert 1 (45 Eingaben mit dem Wert 1, wenn wir den Bias mitzählen). Erklären Sie, warum ein SGD-Schritt die mittlere Vorhersage stärker verändert als jedes einzelne Gewicht. Prüfen Sie, ob die Sprünge verschwinden, wenn Sie die Lernrate halbieren.
   :exlabel:`ex_kaggle-house-scaling`

<!--en:1aea780b-->
1. [code] **Entscheidungsbäume mit Gradientenboosting.** Bei Tabellendaten wie diesen liefern Ensembles von Entscheidungsbäumen oft bessere Vorhersagen als neuronale Netze :cite:`Grinsztajn.Oyallon.Varoquaux.2022,Shwartz-Ziv.Armon.2022`. Passen Sie ein Baummodell mit Gradientenboosting (etwa `GradientBoostingRegressor` aus scikit-learn oder, falls installiert, XGBoost oder LightGBM) in jeder Runde von `folds` an die zentrierten logarithmierten Preise der Trainingszeilen an. Berechnen Sie für jede Gruppe die Validierungs-RMSE. Vergleichen Sie den Mittelwert mit den Mittelwerten der Modelle aus diesem Abschnitt. Welche Eigenschaften der Daten begünstigen Bäume?
   :exlabel:`ex_kaggle-house-trees`

<!--en:592d3903-->
1. [code] **Datenleckage durch die Zielgröße.** Wenn ein Merkmal erst nach dem vorherzusagenden Ereignis oder aufgrund dieses Ereignisses erfasst wird, fallen die Validierungsergebnisse zu günstig aus. Dem Modell im praktischen Einsatz hilft das Merkmal dabei nicht.
    1. Nennen Sie mindestens ein Rohmerkmal dieses Datensatzes, das plausiblerweise erst nach dem Verkauf eingetragen wird. Erklären Sie, warum die Kreuzvalidierung das Problem nicht aufdeckt.
    1. Überprüfen Sie den Mechanismus empirisch. Erzeugen Sie ein künstliches Merkmal, indem Sie zum logarithmierten Verkaufspreis schwaches gaußsches Rauschen addieren. Wiederholen Sie die Kreuzvalidierung und geben Sie an, wie sich die Validierungs-RMSE verändert.

    *Nach Kaggle Learn, [Maschinelles Lernen für Fortgeschrittene](https://www.kaggle.com/learn/intermediate-machine-learning), Aufgabe 7.*

<!--en:c82b598b-->
1. [code] **Ensemble aus den Modellen der Kreuzvalidierung oder erneutes Training.** In :numref:`subsec_kaggle_submission` trainieren wir das ausgewählte MLP erneut auf allen Häusern. Alternativ könnte jedes der fünf Modelle aus der Kreuzvalidierung die Preise der Testhäuser vorhersagen, jeweils mit der Vorverarbeitung seiner eigenen Runde. Anschließend ließen sich die vorhergesagten logarithmierten Preise mitteln.
    1. Implementieren Sie dieses Ensemble und reichen Sie sowohl seine Vorhersagen als auch die des erneut trainierten Modells ein. Vergleichen Sie die beiden Scores auf der Rangliste miteinander und mit der RMSE aus der Kreuzvalidierung.
    1. Jedes Modell im Ensemble wurde auf vier Fünfteln der Häuser trainiert. Warum könnte das Ensemble dennoch bessere Vorhersagen liefern als das Modell, das erneut auf allen Häusern trainiert wurde?
    1. Erklären Sie, warum der Trainingsfehler des erneut trainierten Modells die Schätzung aus der Kreuzvalidierung nicht ersetzen kann.
   :exlabel:`ex_kaggle-house-fold-ensemble`

<!--en:625aa578-->
1. [code] **Zielwertkodierung.** ● Bei der *Zielwertkodierung* ersetzen wir jede Kategorie einer kategorialen Spalte durch den mittleren logarithmierten Verkaufspreis der Trainingszeilen dieser Kategorie. Dadurch belegt das Merkmal eine einzige Spalte, während `preprocess` für jede Kategorie eine eigene Spalte erzeugt.
    1. Verwenden Sie für jede kategoriale Spalte mit mehr als zehn Kategorien die Zielwertkodierung anstelle der One-Hot-Kodierung. Berechnen Sie die Kategoriemittelwerte in jeder Runde aus den Trainingszeilen. Weisen Sie Kategorien, die dort nicht vorkommen, den Gesamtmittelwert der Trainingszeilen zu. Standardisieren Sie die kodierten Spalten wie die numerischen Merkmale. Vergleichen Sie den Fehler in der Kreuzvalidierung mit dem der One-Hot-Kodierung.
    1. Bei der bisherigen Kodierung fließt das Label jeder Trainingszeile in den Mittelwert ein, mit dem diese Zeile kodiert wird. Kodieren Sie die Trainingszeilen nun gruppenweise mit zurückgehaltenen Daten. Teilen Sie dazu die Trainingszeilen jeder Runde in fünf Teile. Kodieren Sie jeden Teil mit den Kategoriemittelwerten aus den anderen vier Teilen. Vergleichen Sie den Fehler in der Kreuzvalidierung mit dem Fehler der Kodierung, die die jeweilige Zeile einbezieht. Erklären Sie, wie die Größe einer Kategorie den Unterschied zwischen den beiden Kodierungen beeinflusst.

<!--en:8f418750-->
1. **Geometrische Mittelung.** ● Ein Ensemble aus $K$ Modellen, die für ein Haus die logarithmierten Preise $z_1, \ldots, z_K$ vorhersagen, kann diese Vorhersagen vor dem Exponenzieren mitteln. Auf der ursprünglichen Preisskala entspricht das dem geometrischen Mittel.
    1. Vergleichen Sie $\exp\bigl(\frac{1}{K}\sum_{k=1}^K z_k\bigr)$ mit dem Mittelwert auf der ursprünglichen Preisskala $\frac{1}{K}\sum_{k=1}^K e^{z_k}$. Zeigen Sie, welcher der beiden Werte den anderen nie übersteigt, und geben Sie an, wann sie übereinstimmen.
    1. Sei $\log y$ der wahre logarithmierte Preis des Hauses. Zeigen Sie, dass für den Mittelwert auf der logarithmischen Skala folgende Ungleichung gilt:

       $$\Bigl(\log y - \frac{1}{K}\sum_{k=1}^K z_k\Bigr)^2 \leq \frac{1}{K}\sum_{k=1}^K \left(\log y - z_k\right)^2.$$

       Prüfen Sie, ob der Logarithmus des Mittelwerts auf der ursprünglichen Preisskala dieselbe Schranke für alle Vorhersagen $z_1, \ldots, z_K$ einhält.

<!--en:5c2a5e24-->
[Diskussionen](https://d2l.discourse.group/t/106)

<!--en:cdb2dfd6-->
[Diskussionen](https://d2l.discourse.group/t/107)

<!--en:4deadd74-->
[Diskussionen](https://d2l.discourse.group/t/237)

<!--en:80443f51-->
[Diskussionen](https://d2l.discourse.group/t/17988)

<!--en:4de4134e-->
Vorhersage von Hauspreisen auf Kaggle: maschinelles Lernen mit Tabellendaten von Anfang bis Ende<br>**Log-RMSE · Vorverarbeitung ohne Datenleckage · 5-fache Kreuzvalidierung mit gemeinsamen Gruppen · MLP mit Gewichtszerfall · Einreichung**

<!--en:9f443437-->
::: {.slide title="Von einheitlichen Pixelrastern zu uneinheitlichen Tabellendaten"}

<!--en:f17369c6-->
[Die Abschlussaufgabe]{.kicker}

<!--en:47da4bd0-->
- **Wenige, heterogene Daten:** Nur $1{,}460$ Häuser in Ames mit bekanntem Preis ($+1{,}459$ Testhäuser) und $79$ gemischte Spalten: $35$ numerische (Werte von $10^0$ bis $10^5$) und $44$ kategoriale nach der Typumwandlung von `MSSubClass`. `NaN` kommt in $19$ Spalten vor.

<!--en:d987567b-->
- **Relative Bewertungsmetrik:** Die Verkaufspreise reichen von $\$34.9\textrm{k}$ bis $\$755\textrm{k}$ (Faktor $22\times$). Die Bewertung erfolgt anhand der **Log-RMSE**.

<!--en:95acc75a-->
- **Risiko der Überanpassung ($\sim 19$ Parameter pro Haus):** Bei $345$ Merkmalen nach der One-Hot-Kodierung sind Vorverarbeitung ohne Datenleckage, 5-fache Kreuzvalidierung und $\ell_2$-Gewichtszerfall entscheidend.

<!--en:8d47fe5d-->
::: {.slide title="Die Rohtabelle von Ames Housing"}

<!--en:7bfee4f9-->
[Aufbau der Rohdaten]{.kicker}

<!--en:607cfb81-->
Nach dem Laden der CSV-Tabellen mit `read_house_prices()` haben die Trainingsdaten die Form $1{,}460 \times 81$ (`Id`, $79$ Merkmale, `SalePrice`), die Testdaten die Form $1{,}459 \times 80$:

<!--en:69a6a794-->
- **Messwerte und numerische Kategoriecodes:** `LotFrontage` gibt eine Länge in Fuß an (in $259$ Zeilen steht `NaN`). Dagegen kodiert `MSSubClass` den Haustyp (eingeschossig, zweigeschossig, Zweifamilienhaus) durch ganze Zahlen ($20, 60, 90, \ldots$) und muss in `str` umgewandelt werden.

<!--en:bafe2bf8-->
- **Aussagekräftige fehlende Werte:** Bei vielen kategorialen Attributen wie `PoolQC` oder `Alley` bezeichnet `NaN` keinen Erfassungsfehler, sondern das tatsächliche Fehlen eines Schwimmbeckens oder einer Zufahrtsgasse.

<!--en:c2c492a8-->
- **Fallstrick Zeilenkennung:** Die Spalte `Id` ($1, 2, \ldots, 1460$) enthält keine Information für die Vorhersage. Vor dem Training entfernen, damit das Modell die Zeilenindizes nicht auswendig lernt.

<!--en:3d1d6b06-->
::: {.slide title="Warum Kaggle die Log-RMSE statt der RMSE in Dollar verwendet"}

<!--en:802cbaba-->
[Die Wettbewerbsmetrik]{.kicker}

<!--en:db193024-->
- **Deutung als prozentualer Fehler:** Wegen $\log(\hat{y}/y) \approx (\hat{y} - y)/y$ bedeutet eine Log-RMSE von $0.135$, dass die Vorhersagen in allen Preisklassen typischerweise um etwa $e^{\pm 0.135} \approx \pm 14\%$ vom tatsächlichen Preis abweichen.

<!--en:b0cc2d46-->
- **Passender Verlust:** Wer $z_i = \log y_i$ mit dem mittleren quadratischen Fehler als Verlust vorhersagt, minimiert genau das Quadrat der Wettbewerbsmetrik.

<!--en:e396d7cc-->
::: {.slide title="Schritte 1 und 2: Standardisierung und Mittelwertimputation"}

<!--en:e60639a6-->
[Vorverarbeitung ohne Datenleckage]{.kicker}

<!--en:66f52119-->
- **Skalierung auf Varianz eins:** Nach Standardisierung jedes numerischen Merkmals auf Mittelwert null und Varianz eins unterliegen alle Gewichte $w_j$ demselben Strafterm $\frac{\lambda}{2}\|\mathbf{W}\|_{\textrm{F}}^2$ auf gleicher Skala.

<!--en:6ad23529-->
- **Null entspricht dem Mittelwert:** Nach der Standardisierung mit $\hat{\mu}_j, \hat{s}_j$ ersetzt das Auffüllen fehlender Einträge (`fillna(0)`) jeden fehlenden Wert exakt durch den Mittelwert des Trainingsdatensatzes, ohne Datenleckage.

<!--en:ab0607e1-->
::: {.slide title="Schritte 3 und 4: One-Hot-Kodierung mit Indikator für fehlende Werte"}

<!--en:e60639a6-2-->
[Vorverarbeitung ohne Datenleckage]{.kicker}

<!--en:a52d5912-->
- **Aussagekräftige `NaN`-Einträge erhalten:** `pd.get_dummies(df, dummy_na=True)` erzeugt für jede beobachtete Kategorie eine binäre Indikatorspalte und zusätzlich eine eigene `_nan`-Spalte (z. B. `PoolQC_nan = 1` für Häuser ohne Schwimmbecken).

<!--en:728baa2e-->
- **Abgleich der Kategorien (`reindex`):** Kategorien aus dem Testdatensatz, die in `train` fehlen, entfallen. Für Kategorien aus `train`, die in `test` fehlen, erhält die Spalte überall den Wert $0$ (`reindex(columns=train_cols, fill_value=0)`).

<!--en:7361cd56-->
::: {.slide title="Die gesamte Vorverarbeitung ohne Datenleckage"}

<!--en:99e67f6c-->
[Code zur Vorverarbeitung]{.kicker}

<!--en:0a7f1905-->
Alle vier Schritte stützen sich ausschließlich auf `train`. Dieselben Transformationen gelten für `*others` (`val` oder `test`). So entstehen aus $79$ Rohattributen $345$ Merkmale vom Typ `float32`:

<!--en:f09674d1-->
::: {.slide title="Zentrierung der Zielwerte in KaggleHouse"}

<!--en:ce8b981a-->
[Normalisierung der Zielwerte]{.kicker}

<!--en:6aed2a05-->
- **Warum $\log y \approx 12.02$ zentrieren?** Gleich nach der Initialisierung liegen die Ausgaben des Netzes nahe $0$. Liegen die Zielwerte um $12.02$, beträgt der anfängliche MSE $12^2 = 144$. Die ersten Epochen gehen dann für die Verschiebung des Ausgabe-Bias verloren.

<!--en:e4b119f9-->
- **Zielwerte mit Mittelwert null ($y_i - \bar{y}_{\textrm{train}}$):** Nach Abzug von `self.offset` beträgt die RMSE in Schritt 0 nur $\approx 0.40$. Bei der Vorhersage kommt `self.offset` vor dem Exponenzieren wieder hinzu, wie in `np.exp(pred + data.offset)`.

<!--en:980c3863-->
::: {.slide title="5-fache Kreuzvalidierung mit gemeinsamen Gruppen ohne Datenleckage"}

<!--en:984980b5-->
[Verfahren zur Modellauswahl]{.kicker}

<!--en:7b09ca11-->
Bei nur $1{,}460$ Häusern mit bekanntem Preis umfasst eine einzelne Validierungsgruppe mit $20\%$ der Daten $292$ Häuser. Über mögliche Gruppen dieser Größe hinweg beträgt der Standardfehler der Log-RMSE mehrere Hundertstel. Er übersteigt damit den Abstand zwischen konkurrierenden Modellen.

<!--en:525a5238-->
- **Aufteilung in 5 Gruppen (`k_fold_data`):** Die Funktion mischt `raw_train` einmal und teilt die Zeilen in $K = 5$ Gruppen zu je $292$ Häusern auf.

<!--en:afc241b9-->
- **Vorverarbeitung pro Runde:** Jede Runde erzeugt eine eigene Instanz von `KaggleHouse(train_4_folds, val_1_fold)`. Sie bestimmt $\hat{\mu}, \hat{s}$, die Indikatorspalten und `offset` neu, ausschließlich anhand ihrer $1{,}168$ Trainingshäuser.

<!--en:91644d16-->
- **Gemeinsame Gruppen für den gepaarten Vergleich:** Bei der Bewertung aller Modelle auf *genau denselben 5 Gruppen* heben sich die zufälligen Schwankungen durch die Gruppeneinteilung in den paarweisen Differenzen auf.

<!--en:d2db4767-->
::: {.slide title="Lineare Regression als Vergleichsmaßstab"}

<!--en:74421b4b-->
[Referenzmodell]{.kicker}

<!--en:53f34ce9-->
Vor dem Abstimmen des MLP dient `d2l.LinearRegression` als Referenz. Die Auswertung umfasst $5$ Gruppen ($100$ Epochen, Batchgröße $64$, Lernrate $\eta = 0.02$). Der Mittelwert über die letzten $5$ Epochen glättet die Schwankungen einzelner SGD-Epochen:

<!--en:58669af3-->
- **Ergebnis des Referenzmodells:** Log-RMSE von $\sim 0.11$ im Training und **$\sim 0.15$ in der Validierung** (typischer Preisfehler $\sim 15\%$). Die Werte der einzelnen Gruppen reichen von $\sim 0.12$ bis $\sim 0.19$, da eine Gruppe einige Villen als Ausreißer enthält.

<!--en:f385900e-->
::: {.slide title="Aufbau von HouseMLP: 64 ReLU-Einheiten mit Gewichtszerfall"}

<!--en:8cdff7d2-->
[Architektur des regularisierten MLP]{.kicker}

<!--en:aebf22db-->
`HouseMLP` ($345 \to 64 \to 1$, $\sim 22{,}200$ Parameter) mit He-Initialisierung und Gewichtszerfall ausschließlich für die Gewichte:

<!--en:8a9eaca4-->
::: {.slide title="Kreuzvalidierung des Gewichtszerfalls: die U-Kurve"}

<!--en:cb824272-->
[Variation des Hyperparameters]{.kicker}

<!--en:a3775e69-->
- Die Variation von `wd` über $\{0.001, \ldots, 0.3\}$ ergibt auf den $5$ gemeinsamen Gruppen die typische U-Kurve:

<!--en:27e0649d-->
- **Zu schwache Regularisierung (`wd = 0.001`):** Überanpassung an die $1{,}168$ Trainingshäuser (Trainings-RMSE $\sim 0.09$) treibt die Validierungs-RMSE auf $\sim 0.15$–$0.18$. Das ist nicht besser als die lineare Regression.

<!--en:eeeacf0d-->
- **Günstiger Bereich (`wd = 0.03` oder `0.1`):** Die Begrenzung der Gewichtsnormen senkt die Validierungs-RMSE auf **$\sim 0.13$–$0.15$** und schlägt damit das lineare Referenzmodell.

<!--en:1316d320-->
- **Zu starke Regularisierung (`wd = 0.3`):** Die Gewichte schrumpfen zu stark. Die Unteranpassung zeigt sich sowohl in den Trainings- als auch in den Validierungsdaten ($\sim 0.17$).

<!--en:576b81ad-->
::: {.slide title="Kreuzvalidierung des Gewichtszerfalls (Fortsetzung)"}

<!--en:58e1e3ed-->
[Ergebnisse der Versuchsreihe]{.kicker}

<!--en:c349da10-->
Für jeden Gewichtszerfall zeigt die Ausgabe die mittlere Trainings- und Validierungs-RMSE über die $5$ Gruppen sowie die Spannweite der Validierungs-RMSE zwischen den Gruppen:

<!--en:f391de69-->
::: {.slide title="Gepaarter Vergleich mit dem linearen Referenzmodell pro Gruppe"}

<!--en:db51a481-->
[Statistische Überprüfung]{.kicker}

<!--en:b9aa2924-->
Beide Modelle wurden auf genau denselben $5$ Gruppen bewertet. Für jede Gruppe lässt sich daher die gepaarte Differenz $\textrm{RMSE}_{\textrm{MLP}} - \textrm{RMSE}_{\textrm{linear}}$ betrachten:

<!--en:97193fd9-->
- **Varianzreduktion durch Paarung:** Die Validierungs-RMSE schwankt um $\pm 0.04$, je nachdem, in welcher Gruppe ungewöhnliche Luxusverkäufe liegen. Dennoch spricht die gepaarte Differenz in fast jeder Gruppe für das regularisierte MLP. Seine mittlere Validierungs-RMSE liegt um $\sim 0.006$–$0.014$ niedriger.

<!--en:10a37373-->
- **Vergleich auf Tabellendaten:** Das regularisierte MLP mit 64 Einheiten unterbietet die Log-RMSE der linearen Regression im Mittel um $\sim 0.01$. Bei heterogenen Tabellendaten bleiben Entscheidungsbäume mit Gradientenboosting (XGBoost / LightGBM) starke Konkurrenten (siehe Aufgaben).

<!--en:9cdc40e4-->
::: {.slide title="Erneutes Training auf allen 1460 Häusern und Schreiben von submission.csv"}

<!--en:b4b68a78-->
[Einreichung der endgültigen Vorhersagen]{.kicker}

<!--en:224293f9-->
Sobald die 5-fache Kreuzvalidierung den besten Gewichtszerfall `best` liefert, folgt eine neue Anpassung der Vorverarbeitung mit `KaggleHouse(raw_train, raw_test)`. Das erneute Training von `HouseMLP` auf **allen $1{,}460$ Häusern mit bekanntem Preis** dauert $100$ Epochen:

<!--en:0b4e2d0f-->
- **Zentrierung umkehren und exponenzieren:** Erst `data.offset` zu den vorhergesagten zentrierten logarithmierten Preisen addieren, dann `np.exp(...)` anwenden. So entstehen wieder Dollarbeträge für `submission.csv`.

<!--en:6fd28289-->
::: {.slide title="Verständnisfragen zu Datenleckage, Log-RMSE und Selektionsbias"}

<!--en:ffcb46a8-->
[Selbsttest]{.kicker}

<!--en:2859bf3b-->
Fragen zum gesamten Ablauf der Modellierung mit Tabellendaten:

<!--en:2f8f67a5-->
1. **Relativer Charakter der Log-RMSE:** Haus A erzielt $\$100{,}000$ bei einer Vorhersage von $\$125{,}000$, Haus B erzielt $\$400{,}000$ bei einer Vorhersage von $\$500{,}000$. Die quadratischen Fehler für Preise in Dollar und für logarithmierte Preise vergleichen.

<!--en:7d9a3d92-->
2. **Versteckte Datenleckage:** Angenommen, `preprocess(raw_train)` wird einmal auf alle $1{,}460$ Zeilen angewandt, *bevor* die Aufteilung in $5$ Gruppen zur Kreuzvalidierung erfolgt. Welche zwei Arten von Kennwerten gelangen aus der Validierungsgruppe ins Training?

<!--en:6aa8cfd3-->
3. **Fluch des Gewinners bei der Modellauswahl:** Unter $M$ Hyperparametereinstellungen gewinnt diejenige mit der niedrigsten Validierungs-RMSE auf den 5 Gruppen. Ist dieser beste Score ein erwartungstreuer Schätzer der RMSE auf dem Testdatensatz?

<!--en:557b98b6-->
**Lösungen:**

<!--en:775ee339-->
1. **Gleicher Fehler auf der Log-Skala:** Die quadratischen Fehler in Dollar unterscheiden sich um den Faktor $16\times$ ($\$25\textrm{k}^2$ gegenüber $\$100\textrm{k}^2$). Für beide Häuser gilt jedoch $\hat{y}/y = 1.25$. Daher ist der quadratische Fehler der logarithmierten Preise genau gleich: $(\log 1.25)^2 \approx 0.223^2 \approx 0.050$.

<!--en:d9ab4df1-->
2. **Datenleckage bei den Kennwerten:** Die Mittelwerte und Standardabweichungen der Merkmale aus der Validierungsgruppe ($\hat{\mu}_j, \hat{s}_j$ für Skalierung und Mittelwertimputation) sowie deren Kategorien fließen in jede Trainingsgruppe ein.

<!--en:c8a33cbd-->
3. **Selektionsbias nach unten:** Nein, denn $\mathbb{E}[\min_{m=1\dots M} \hat{R}_m] \le \min_m \mathbb{E}[\hat{R}_m]$. Bei $M = 6$ Kandidaten auf $5$ Gruppen ist die optimistische Verzerrung sehr klein. Bei Hunderten von Hyperparameterversuchen braucht es jedoch einen separat zurückgehaltenen Testdatensatz.

<!--en:998426d4-->
::: {.slide title="Zusammenfassung: von den Tabellendaten zur Einreichung auf Kaggle"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:1d474969-->
| Schritt | Umsetzung | Mathematische oder statistische Begründung | Vermiedener Fallstrick |

<!--en:a5818245-->
| **Metrik und Zielwert** | Vorhersage von $z_i = \log y_i - \bar{z}_{\textrm{train}}$ mit MSE als Verlust | $\sqrt{\textrm{MSE}(\log y, \log \hat{y})} = \textrm{log-RMSE}$ ($\approx \%$ Fehler) | MSE in Dollar gewichtet Villen zu stark. Ohne Zentrierung bremst $\log y \approx 12$ das Lernen in Epoche 1. |

<!--en:3f645e4b-->
| **Datentypen** | `Id` entfernen, `MSSubClass` in `str` umwandeln | Willkürliche Zeilenkennungen, numerische Codes für nominale Kategorien | Kein Auswendiglernen der Zeilenfolge, keine rechnerische Deutung des Codes für Zweifamilienhäuser als $90 = 4.5 \times 20$ |

<!--en:392784c1-->
| **Skalierung numerischer Merkmale** | $x_{ij} \leftarrow (x_{ij} - \hat{\mu}_{\textrm{train}}) / \hat{s}_{\textrm{train}}$, dann `fillna(0)` | Gleiche Merkmalsskalen für SGD und $\ell_2$-Strafterm. $0 =$ Trainingsmittelwert | Bei Berechnung von $\hat{\mu}, \hat{s}$ auf der gesamten Tabelle fließen Validierungs- und Testverteilung ins Training ein. |

<!--en:a15be92c-->
| **Kategoriale Merkmale** | `get_dummies(dummy_na=True)` + `reindex` | Binärer Indikator pro Kategorie und eigene `_nan`-Spalte ($345$ Spalten) | Fehlende Werte behalten ihre Aussagekraft (z. B. bedeutet `PoolQC = NaN`, dass kein Schwimmbecken vorhanden ist). |

<!--en:30d66bd0-->
| **Modellauswahl** | 5-fache Kreuzvalidierung mit gemeinsamen Gruppen (`k_fold`) | Mittelung über $5$ Validierungsgruppen (je $292$ Häuser), gepaarte Differenzen | Je nach gewählter Gruppe mit $292$ Häusern schwankt die Log-RMSE um mehrere Hundertstel. Überanpassung beim unregularisierten MLP ($0.16$). |

<!--en:390074be-->
| **Einreichung** | Erneutes Training mit `wd = best` auf allen $1{,}460$ Häusern | $\hat{y} = \exp(\hat{z} + \bar{z}_{\textrm{train}})$ liefert wieder Dollarbeträge | Der Trainingsfehler des erneut trainierten Modells liefert keine gültige Schätzung der Generalisierungsgüte. |
