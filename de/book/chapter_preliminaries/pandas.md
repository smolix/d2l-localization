<!--en:b65ff061-->
# Datenvorverarbeitung

<!--en:ac629468-->
Bislang haben wir mit synthetischen Daten gearbeitet. Sie lagen bereits als Tensoren in der gewünschten Form und mit dem gewünschten Datentyp vor. In der Praxis müssen wir Daten jedoch aus den unterschiedlichsten Dateiformaten auslesen und in saubere numerische Tensoren umwandeln. Der unscheinbare Begriff *Vorverarbeitung* (engl. preprocessing) lässt kaum erahnen, wie viele statistische Entscheidungen nötig sind, um eine Rohtabelle für ein Modell aufzubereiten.

<!--en:ce41a30d-->
Als durchgängiges Beispiel dient uns eine kleine Tabelle mit zehn Hausverkäufen. Sie enthält die Zimmerzahl (`NumRooms`), das Dachmaterial (`RoofType`), die Wohnfläche in Quadratfuß (`Area`) und den Verkaufspreis in Dollar (`Price`). Schon diese kleine Tabelle zeigt drei Hindernisse, die uns daran hindern, tabellarische Rohdaten direkt in ein neuronales Netz einzuspeisen:

<!--en:aff6de83-->
1. **Fehlende Werte:** Bei drei der zehn Häuser fehlt die Zimmerzahl, bei vier die Art der Dacheindeckung.

<!--en:c8d5ad2d-->
2. **Kategoriale Textfelder:** Die Dacheindeckung ist als Wort erfasst (`Slate` oder `Tile`). Neuronale Netze rechnen ausschließlich mit Zahlen.

<!--en:aee723ce-->
3. **Unterschiedliche Skalen:** Der Zahlenwert der Wohnfläche in Quadratfuß ist ungefähr $500$-mal so groß wie die Zimmerzahl.

<!--en:466b0b25-->
Jedes Hindernis verlangt eine statistische Entscheidung. Beim Löschen unvollständiger Zeilen ginge der Großteil der Tabelle verloren. Eine Kodierung der gleichrangigen Dacharten als `0, 1, 2` würde ihnen eine künstliche Reihenfolge geben. Ohne Skalierung würde die Wohnfläche die Abstände und Gradienten dominieren. Die Bibliothek [pandas](https://pandas.pydata.org/) :cite:`McKinney.2010` ist das Standardwerkzeug für tabellarische Daten in Python. Wir gehen die Verarbeitungsschritte in :numref:`fig_pandas_pipeline` durch und erzeugen aus einer CSV-Datei Merkmals- und Zieltensoren mit dem Datentyp `float32`.

<!--en:d1256220-->
![Von der Datei zu Tensoren. `read_csv` lädt die Textdatei in einen DataFrame mit Lücken und Textspalten. Nach Imputation, Kodierung und Skalierung ist jeder Eintrag eine Zahl. Über `to_numpy` gelangt das Ergebnis als Tensoren `X` und `y` ins Framework.](../img/pandas-pipeline.svg)

<!--en:0bec74d9-->
Zuerst importieren wir `pandas` unter dem üblichen Alias `pd` sowie das Python-Modul `os` für Dateipfade. Außerdem importieren wir unser Deep-Learning-Framework. Wir brauchen es im letzten Schritt, um die bereinigte Tabelle in Tensoren umzuwandeln.

<!--en:40c8182f-->
Wie im vorigen Abschnitt importieren wir das Array-Modul von MXNet als `mnp`, sodass `np` weiterhin das Standard-NumPy bezeichnet.

<!--en:6377cb64-->
## Daten laden und untersuchen

<!--en:8de79155-->
### CSV-Dateien

<!--en:85dceda1-->
CSV-Dateien (*Comma-separated values*, durch Kommas getrennte Werte) speichern Tabellendaten als reinen Text. Die erste Zeile enthält die Spaltennamen, jede weitere einen Datensatz mit durch Kommas getrennten Feldern. Wir schreiben die Daten der zehn Häuser in die Datei `../data/house_tiny.csv`. Leere Felder zwischen aufeinanderfolgenden Kommas (etwa in `,Tile,2100,290000`) stehen für nicht erfasste Werte:

<!--en:ac8cad5e-->
Der Aufruf `pd.read_csv` liest die Datei in einen *DataFrame* von pandas ein. Das ist eine zweidimensionale Tabelle aus Spalten mit Namen und Datentypen. Am linken Rand steht der Zeilenindex ($0$ bis $9$):

<!--en:1a5030b3-->
Die Bibliothek `pandas` hat jedes leere Feld und gängige Kennzeichnungen wie `NA` automatisch durch `NaN` (*not a number*) ersetzt. Dieser Wert kennzeichnet einen *fehlenden Wert*.

<!--en:e795d840-->
### Datentypen und statistische Kennzahlen

<!--en:10599656-->
Prüfen Sie bei einem frisch geladenen Datensatz vor jeder Änderung die Anzahl der Zeilen und Spalten, die Anzahl der vorhandenen Einträge und die ermittelten Datentypen mit `data.info()`:

<!--en:a16d9825-->
In allen $10$ Zeilen sind `Area` und `Price` vollständig (`int64`). Für `NumRooms` liegen $7$ Einträge vor (`float64`), für `RoofType` sind es $6$ (`str` oder in älteren pandas-Versionen `object`). Obwohl `NumRooms` ganze Zimmer zählt, hat die Spalte den Datentyp `float64` erhalten. Der Standard IEEE 754 definiert `NaN` als Gleitkomma-Bitmuster. Gewöhnliche ganzzahlige Arrays können fehlende Einträge daher nicht darstellen.

<!--en:d878a275-->
Für numerische Spalten liefert `data.describe()` die Anzahl der beobachteten Werte, ihren Mittelwert, ihre Stichprobenstandardabweichung (`std`), das Minimum, die Quartile und das Maximum:

<!--en:4962ecbc-->
`NumRooms` reicht von $2$ bis $5$ (Mittelwert $3.29$), `Area` von $850$ bis $2{,}600$ Quadratfuß (Mittelwert $1{,}654$) und `Price` von $\$127{,}500$ bis $\$375{,}000$.

<!--en:d133bad8-->
### Trennung von Eingaben und Zielgrößen

<!--en:23501e3d-->
Beim überwachten Lernen sagen wir eine bestimmte *Zielspalte* (oder *Labelspalte*) voraus, hier `Price`. Dazu verwenden wir die übrigen *Eingabespalten* (*Merkmalsspalten*) `NumRooms`, `RoofType` und `Area`. Trennen wir die Merkmale gleich zu Beginn von der Zielgröße, bleibt diese bei der späteren Merkmalstransformation unberührt und fließt auch nicht in die Merkmale ein. Die Positionsindizes für Ausschnitte beginnen bei $0$. Mit `.iloc` wählen wir über `data.iloc[:, :3]` die ersten drei Spalten und über `data.iloc[:, 3]` die vierte:

<!--en:f2898d23-->
Die Auswahl nach Position ist kurz. Fügt man im laufenden Betrieb jedoch eine CSV-Spalte ein oder ändert die Reihenfolge, wählt `data.iloc[:, 3]` unbemerkt die falsche Zielgröße. Die Auswahl nach Namen (`targets = data['Price']` und `inputs = data.drop(columns='Price')`) vermeidet dieses Risiko (Übung 1).

<!--en:6f99219b-->
## Bereinigung und Kodierung

<!--en:5e41f017-->
### Fehlende Werte

<!--en:bfbfb570-->
Fehlende Werte kommen häufig vor, etwa wenn Sensoren ausfallen, Befragte Fragen überspringen oder historische Angaben nie erfasst wurden. Wir verketten `.isna()` mit `.sum()`, um die fehlenden Einträge jeder Spalte zu zählen (`True` zählt als $1$):

<!--en:46ff9ca1-->
Drei Verfahren sind üblich. Jedes bringt einen anderen statistischen Kompromiss mit sich (:numref:`fig_pandas_missing`):

<!--en:1512aa37-->
* Beim **Löschen** entfernen wir jede Zeile (oder Spalte), in der ein Wert fehlt. Das ist leicht umzusetzen, kostet aber Daten. Fehlen die Werte nicht völlig zufällig, ist die verbleibende Stichprobe verzerrt, etwa wenn bei Einzimmerwohnungen die Zimmerzahl besonders häufig fehlt.

<!--en:b41df4a5-->
* Bei der **Imputation** ersetzen wir jedes `NaN` durch einen Schätzwert, etwa den Mittelwert oder Median der Spalte. Alle Zeilen bleiben erhalten, doch die eingesetzten Näherungswerte verringern die beobachtete Streuung der Spalte.

<!--en:34d313dd-->
* **Indikatoren für fehlende Werte** ergänzen die Imputation um eine binäre Spalte ($0/1$). Sie hält fest, *ob* der jeweilige Eintrag ursprünglich fehlte. So kann das Modell das Fehlen selbst als Information nutzen, etwa wenn die Zimmerzahl überwiegend bei Luxusanwesen fehlt. Dafür brauchen wir pro imputierter Spalte ein zusätzliches Merkmal.

<!--en:31a6c28a-->
![Drei Verfahren für eine fehlende Zimmerzahl in den ersten vier Zeilen von `NumRooms`. Beim Löschen entfällt die Zeile. Bei der Imputation setzen wir den Mittelwert der sieben beobachteten Zimmerzahlen ein, $23/7 \approx 3.29$. Beim Verfahren mit Indikator setzen wir denselben Mittelwert ein und halten in einer zusätzlichen Spalte fest, wo die Lücke war.](../img/pandas-missing.svg)

<!--en:39da6f9d-->
Löschen wir in unserer Tabelle mit zehn Häusern die unvollständigen Zeilen (`inputs.dropna()`), bleibt nur wenig übrig:

<!--en:da798ae3-->
Wir verlieren so $6$ unserer $10$ Häuser ($60\%$ des Datensatzes). Deshalb verwenden wir die Imputation. Der Aufruf `inputs.fillna(inputs.mean(numeric_only=True))` ersetzt jeden numerischen `NaN`-Eintrag durch den Mittelwert der beobachteten Einträge seiner Spalte. Verzerren Ausreißer den Mittelwert, ist der Median vorzuziehen. Die Textspalte `RoofType` bleibt für den nächsten Schritt mit kategorialen Daten erhalten:

<!--en:70c74629-->
In den Zeilen $1$, $4$ und $8$ steht jetzt $23/7 \approx 3.285714$, der Mittelwert der sieben beobachteten Zimmerzahlen. Kein Haus hat tatsächlich $3.29$ Zimmer. Der eingesetzte Mittelwert dient als neutraler statistischer Platzhalter.

<!--en:787d4116-->
### Kategoriale Merkmale

<!--en:8bb77bde-->
Zählen wir fehlende Einträge als eigenen Zustand, nimmt `RoofType` drei ungeordnete Kategorien an: `Slate`, `Tile` oder `NaN`. Bei einer Kodierung durch einzelne ganze Zahlen wie $\textrm{Slate} = 0$, $\textrm{Tile} = 1$ und $\textrm{missing} = 2$ müsste jede Linearkombination der Merkmale `Tile` als Mittelwert zwischen `Slate` und einer unbekannten Dacheindeckung behandeln. So entstünden künstliche Abstände und eine künstliche Reihenfolge.

<!--en:8b5673cf-->
Die *One-Hot-Kodierung* (*Kodierung mit Dummyvariablen*) vermeidet das Problem. Sie ersetzt eine Spalte mit $k$ Kategorien durch $k$ binäre Indikatorspalten ($0/1$). Der Eintrag für die zutreffende Kategorie ist $1.0$, die übrigen $k - 1$ Einträge sind $0.0$. In `pandas` kodiert `pd.get_dummies` alle Spalten mit Zeichenketten oder Kategorien und lässt numerische Spalten unverändert. Mit `dummy_na=True` erhalten wir ausdrücklich den Indikator `RoofType_nan` für fehlende Werte. Mit `dtype=float` speichern wir $0.0/1.0$ als Gleitkommazahlen statt als boolesche Werte:

<!--en:cc29cb56-->
In jeder Zeile summieren sich `RoofType_Slate`, `RoofType_Tile` und `RoofType_nan` zu $1.0$. Hat das Modell einen Bias als Achsenabschnitt, ist daher eine Spalte redundant. Ebenso genügt bei zwei möglichen Zuständen ohne fehlende Werte ein einziger $0/1$-Indikator. Die One-Hot-Kodierung eignet sich für eine überschaubare Anzahl von Kategorien. Bei Zehntausenden von Kategorien, etwa Postleitzahlen, Benutzerkennungen oder Wörtern eines Vokabulars, verwenden wir kompakte, lernbare *Einbettungen* (engl. embeddings, siehe :numref:`sec_rnn`).

<!--en:17fe6280-->
## Skalierung und Umwandlung

<!--en:06d1b556-->
### Standardisierung

<!--en:93ec79f7-->
In `inputs` sind jetzt alle Spalten numerisch. Allerdings haben `NumRooms` und `Area` unterschiedliche Maßstäbe. In den ursprünglichen Einheiten ist eine Differenz von $2$ Zimmern zahlenmäßig kleiner als eine Differenz von $3$ Quadratfuß. Ohne Skalierung würde die Wohnfläche die euklidischen Abstände und die Strafterme für Gewichte dominieren. Wie wir in :numref:`sec_linear_scratch` und :numref:`chap_optimization` sehen, ziehen Unterschiede von mehreren Größenordnungen die Verlustfläche zu schmalen Tälern auseinander und bremsen damit den Gradientenabstieg.

<!--en:bae88d00-->
Die *Standardisierung* (*z-Transformation*) bringt stetige Merkmale auf eine gemeinsame Skala. Sie bildet jeden Eintrag $x$ ab auf

<!--en:91bd0bce-->
Dabei ist $\mu$ der Stichprobenmittelwert der Spalte und $\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \mu)^2}$ ihre Stichprobenstandardabweichung (die Bessel-Korrektur mit dem Nenner $n - 1$ leiten wir in :numref:`sec_prob` her). Jede standardisierte Spalte hat den Mittelwert $0$ und die Standardabweichung $1$. Wir messen die Beobachtungen damit in einheitenlosen Standardabweichungen, unabhängig davon, ob die Fläche in Quadratfuß oder Quadratmetern erfasst wurde. Unsere One-Hot-Indikatoren liegen bereits in $\{0, 1\}$. Deshalb standardisieren wir nur `['NumRooms', 'Area']`:

<!--en:6aa7ee4b-->
In Zeile $6$ ($5$ Zimmer, $2{,}600$ Quadratfuß) steht jetzt $(1.89, 1.70)$. Das Haus liegt also in beiden Merkmalen etwa $1.7$ bis $1.9$ Standardabweichungen über dem Mittelwert. Die mit dem Mittelwert imputierten Zimmerzahlen in den Zeilen $1$, $4$ und $8$ liegen bei $0.0$. Einen Nebeneffekt zeigt :numref:`fig_pandas_imputation_spread`. Setzen wir in drei Zeilen den Wert $\mu$ ein, tragen diese Zeilen null zum Zähler von $\sigma$ bei. Gleichzeitig steigt $n$ von $7$ auf $10$. Damit sinkt die Standardabweichung der Zimmerzahl von $1.11$ auf $0.91$. Die beobachteten $z$-Werte werden betragsmäßig größer (von $1.54$ auf $1.89$, siehe Zeile $6$).

<!--en:4b779329-->
![Zimmerzahlen vor und nach der Imputation mit dem Mittelwert. Die drei imputierten Werte (orange) liegen alle beim Mittelwert $\mu$ der sieben beobachteten Werte. Daher streuen die Werte der Spalte weniger als die ursprünglichen Daten. Die Standardabweichung fällt von $1.11$ auf $0.91$.](../img/pandas-imputation-spread.svg)

<!--en:d1b9fb70-->
Wenn wir Daten in einen *Trainingsdatensatz* und einen *Testdatensatz* aufteilen, gilt eine grundlegende Regel: **Alle statistischen Größen der Vorverarbeitung, einschließlich Mittelwerten oder Medianen für die Imputation, Kategorienlisten und Standardisierungsparametern $(\mu, \sigma)$, müssen wir ausschließlich aus dem Trainingsdatensatz bestimmen**. Auf den Testdatensatz wenden wir sie unverändert an (:numref:`fig_pandas_leakage`). Bestimmen wir diese Größen aus dem gesamten Datensatz, gelangen Informationen aus den zurückgehaltenen Testdaten ins Training (*Datenleckage*, engl. data leakage). Dadurch fallen die Testergebnisse zu günstig aus. In :numref:`sec_kaggle_house` wenden wir die Regel auf einen vollständigen Immobilien-Benchmark an (siehe auch Übung 4).

<!--en:ade36895-->
![Reihenfolge von Aufteilung und Schätzung. (a) Schätzen wir die Kenngrößen aus allen Zeilen, fließen die Testdaten schon in die Schätzung ein. Das Testergebnis fällt zu günstig aus. (b) Schätzen wir die Kenngrößen allein aus den Trainingsdaten und wenden sie unverändert auf beide Teildatensätze an, bleiben die Testdaten von der Schätzung ausgeschlossen.](../img/pandas-leakage-split.svg)

<!--en:086d9872-->
### Umwandlung in Tensoren

<!--en:8ece07d9-->
Zum Schluss extrahiert `.to_numpy(dtype='float32')` die bereinigten Spalten als zusammenhängende NumPy-Arrays mit 32-Bit-Gleitkommazahlen. Damit wechseln wir vom pandas-Standardtyp `float64` zu einer geringeren Genauigkeit. Die Arrays können direkt an den Tensorkonstruktor unseres Frameworks gehen (:numref:`sec_ndarray`):

<!--en:d59833f0-->
Die Merkmalsmatrix `X` hat die Form `(10, 5)` ($10$ Häuser, jeweils mit $2$ standardisierten numerischen Merkmalen und $3$ Indikatoren für die Dacheindeckung). Der Zielvektor `y` hat die Form `(10,)`. Beide haben den Datentyp `float32`. Im praktischen Einsatz müssen wir zuvor häufig mehrere relationale Tabellen verknüpfen, fehlerhafte Datensätze erkennen und besondere Kennwerte prüfen, etwa `-9999` für eine unbekannte Koordinate. Als Designmatrix $\mathbf{X}$ stehen die Daten dann für die Operationen der linearen Algebra im nächsten Abschnitt bereit.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:ce69806d-->
Tabellendaten aus der Praxis enthalten fast immer fehlende Werte, kategoriale Textfelder und numerische Spalten mit sehr unterschiedlichen Skalen. Neuronale Netze benötigen dagegen bereinigte `float32`-Tensoren. In `pandas` lesen wir CSV-Dateien mit `read_csv` in einen DataFrame ein. Mit `info()` und `describe()` prüfen wir Spaltentypen, die Anzahl fehlender Werte und die Werteverteilungen. Die Zielspalte trennen wir gleich zu Beginn von den Eingabemerkmalen. Fehlende Werte können wir durch Löschen, durch Imputation (etwa mit Mittelwert oder Median aus den Trainingsdaten) oder mit Indikatorspalten behandeln. Ungeordnete Kategorien wandeln wir durch One-Hot-Kodierung in binäre Indikatorspalten ($0/1$) um (`get_dummies`). So entsteht keine künstliche numerische Ordnung. Stetige numerische Spalten standardisieren wir auf Mittelwert null und Varianz eins ($x \mapsto (x - \mu)/\sigma$). Die dafür nötigen Größen berechnen wir ausschließlich aus den Trainingsdaten. Mit `.to_numpy(dtype='float32')` überführen wir den bereinigten DataFrame direkt in Tensoren unseres Deep-Learning-Frameworks.

<!--en:2294243a-->
## Übungen

<!--en:9dff5489-->
1. [code] **Spalten nach Namen oder Position auswählen.** In diesem Abschnitt haben wir `data` anhand ganzzahliger Positionen mit `data.iloc[:, :3]` und `data.iloc[:, 3]` in Merkmale und Zielgröße aufgeteilt. Angenommen, in einer neuen Version von `house_tiny.csv` steht die zusätzliche Spalte `YearBuilt` zwischen `Area` und `Price`.
    1. Welche Spalten weisen die Positionsauswahlen `data.iloc[:, :3]` und `data.iloc[:, 3]` bei der neuen Datei den Variablen `inputs` und `targets` zu? Löst eine der folgenden Codezeilen eine Ausnahme aus, oder trainiert die Pipeline unbemerkt ein Modell für die falsche Zielgröße?
    1. Schreiben Sie die Aufteilung nach Spaltennamen um (etwa mit `data['Price']` und `data.drop(...)`). Dabei soll `targets` stets `Price` auswählen und `inputs` alle übrigen Spalten behalten, unabhängig von der Reihenfolge und von neu hinzugekommenen Merkmalen. Einzelheiten finden Sie im pandas-Handbuch unter [Indizierung und Datenauswahl](https://pandas.pydata.org/docs/user_guide/indexing.html).

<!--en:cf8fb6ba-->
1. [code] **Imputationsverfahren vergleichen.** Ergänzen Sie die fehlenden Einträge in `NumRooms` mit drei Verfahren: (i) Imputation mit dem Mittelwert, (ii) Imputation mit dem Median und (iii) Imputation mit dem Mittelwert und einer zusätzlichen binären Indikatorspalte `NumRooms_isna`.
    1. Berechnen Sie für die Imputation mit Mittelwert und mit Median jeweils Stichprobenmittelwert und Stichprobenstandardabweichung der vervollständigten Spalte `NumRooms`. Vergleichen Sie die Ergebnisse mit Mittelwert und Standardabweichung der $7$ ursprünglich beobachteten Werte. Erklären Sie die Richtung der Änderung der Standardabweichung.
    1. Angenommen, in einem größeren Immobiliendatensatz fehlt `NumRooms` vorwiegend bei Luxusvillen. Welche Zimmerzahl sieht ein Modell nach der reinen Mittelwertimputation für diese Häuser? Ein lineares Modell sagt den Preis als gewichtete Summe der Eingabespalten zuzüglich eines Bias voraus. Erklären Sie, wie es mithilfe der zusätzlichen Indikatorspalte `NumRooms_isna` die systematische Unterschätzung ausgleichen kann.
    1. Geben Sie für jedes der drei Verfahren (Löschen, reine Mittelwertimputation und Imputation mit Indikator) genau an, welche statistische Annahme es über die Ursache der fehlenden Werte trifft.

    *Nach Kaggle Learn, [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning), Übung 2.*

<!--en:ff4bddec-->
1. **Kategoriale Merkmale mit vielen Ausprägungen.** Die One-Hot-Kodierung wandelt eine Spalte mit $k$ verschiedenen Kategorien in $k$ binäre Spalten um (oder in $k + 1$ mit `dummy_na=True`).
    1. Eine Tabelle mit $n = 10^6$ Häusern enthält eine Spalte `ZipCode` mit $k = 10^4$ verschiedenen Postleitzahlen und ohne fehlende Werte. Berechnen Sie, wie viele Gigabyte die One-Hot-Matrix bei dichter Speicherung als 64-Bit-Gleitkommazahlen (`float64`) belegt. Welcher Anteil ihrer Einträge ist von null verschieden?
    1. Die Tabelle enthalte außerdem die Spalte `ListingID` mit einer eindeutigen Zeichenkette als Kennung für jede Zeile. Vergleichen Sie drei Verfahren für `ListingID`: (i) One-Hot-Kodierung, (ii) Abbildung jeder Kennung auf einen lernbaren Einbettungsvektor und (iii) Entfernen der ganzen Spalte. Was kann ein Modell bei jedem Verfahren aus `ListingID` lernen, das sich auf ein neu angebotenes Haus übertragen lässt, dessen `ListingID` in den Trainingsdaten nicht vorkam? Welches Verfahren sollten Sie wählen?

    *Nach Kaggle Learn, [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning), Übung 3.*

<!--en:25e0610d-->
1. [code] **Datenleckage quantifizieren.** Im Haupttext haben wir fehlende Einträge in `NumRooms` imputiert und `NumRooms` und `Area` standardisiert. Dazu haben wir Stichprobenmittelwert und Standardabweichung aus allen $10$ Zeilen berechnet. Teilen Sie nun `data` in einen Trainingsdatensatz aus den ersten $8$ Zeilen (`data.iloc[:8]`) und einen Testdatensatz aus den letzten $2$ Zeilen (`data.iloc[8:]`) auf.
    1. Wiederholen Sie die Vorverarbeitung. Berechnen Sie den Mittelwert für die Imputation und die Standardisierungsparameter ($\mu$ und $\sigma$) *ausschließlich aus den $8$ Trainingszeilen*. Verwenden Sie die Trainingsparameter unverändert zur Imputation und Standardisierung beider Teildatensätze. Vergleichen Sie die standardisierten Testzeilen mit den Ergebnissen, die Sie mit $\mu$ und $\sigma$ aus allen $10$ Zeilen erhalten hatten.
    1. Erklären Sie genau, welche Informationen über die zurückgehaltenen Testhäuser bei der ursprünglichen Vorverarbeitung der Gesamttabelle in die Trainingsmerkmale eingeflossen sind.
    1. Ein kategoriales oder numerisches Merkmal sei in allen Trainingszeilen konstant (seine Standardabweichung beträgt also $\sigma = 0$), variiere aber in den Testdaten. Was ergibt die Standardisierungsformel $(x - \mu)/\sigma$ mit den Trainingsparametern für die Trainings- und die Testdaten? Wie sollte eine robuste Vorverarbeitung mit Merkmalen der Varianz null umgehen?

<!--en:c257ee4f-->
[Diskussion](https://d2l.discourse.group/t/28)

<!--en:57ef34d9-->
[Diskussion](https://d2l.discourse.group/t/29)

<!--en:0803c89b-->
[Diskussion](https://d2l.discourse.group/t/195)

<!--en:f86cdb85-->
[Diskussion](https://d2l.discourse.group/t/17967)

<!--en:c5c7fd28-->
Von tabellarischen Rohdaten zu sauberen Tensoren<br>**Tabellenschema prüfen · Ursachen fehlender Werte · One-Hot-Kodierung · z-Standardisierung · Datenleckage**

<!--en:23c7040f-->
::: {.slide title="Warum tabellarische Rohdaten für ein neuronales Netz ungeeignet sind"}

<!--en:28312297-->
[Schritte der Vorverarbeitung]{.kicker}

<!--en:50bae54b-->
- **Verschiedene Typen und Lücken:** CSV-Protokolle aus der Praxis enthalten numerische Spalten (`int64`, `float64`), frei eingegebene Kategorien (`"Slate"`, `"Tile"`) und fehlende Werte (`NaN`). Tensor-Kernel benötigen dichte Puffer mit einheitlichem Gleitkommatyp.

<!--en:c02a3a12-->
- **Unterschiedliche Skalen:** In den Hauspreisdaten ist `Area` ($\sim 10^3\text{ sq ft}$) ungefähr $500\times$ so groß wie `NumRooms` ($\sim 10^0$). Ohne Skalierung verzerrt das Abstände und Gradienten.

<!--en:914ad08b-->
- **Statistische Entscheidungen:** Löschen oder Imputieren von `NaN`, Ganzzahl- oder One-Hot-Kodierung, Skalieren anhand der gesamten Tabelle oder der Trainingsdaten: Jede Wahl legt fest, unter welcher statistischen Annahme das Modell lernt.

<!--en:154cefe4-->
::: {.slide title="CSV-Rohdaten in einen DataFrame einlesen" layout="tall-output"}

<!--en:3f392d64-->
[Tabellendaten laden]{.kicker}

<!--en:943e21a3-->
- **Tabellen als Klartext:** CSV-Dateien speichern pro Zeile einen Datensatz mit kommagetrennten Feldern. Leere Felder zwischen aufeinanderfolgenden Kommas (z. B. `,Tile,2100,290000`) kennzeichnen nicht erfasste Messwerte.

<!--en:324be320-->
- **Fehlende Werte automatisch erkennen:** `pd.read_csv` liest die Datei in einen zweidimensionalen `DataFrame` mit benannten, typisierten Spalten ein und ersetzt leere Einträge durch `NaN` (*Not a Number*).

<!--en:c90e1699-->
- **Beispieldatensatz (`house_tiny.csv`):** 10 Hausverkäufe mit Zimmerzahl (`NumRooms`), Dacheindeckung (`RoofType`), Wohnfläche (`Area`, Quadratfuß) und Verkaufspreis (`Price`, $).

<!--en:e8ae1235-->
::: {.slide title="Spaltentypen und statistische Kennzahlen prüfen"}

<!--en:7722b2cd-->
[Erste Datenanalyse]{.kicker}

<!--en:862ac963-->
- **`data.info()` zeigt eine Folge von IEEE 754:** Obwohl Zimmerzahlen ganzzahlig sind, erhält `NumRooms` den Typ `float64`. Gewöhnliche Ganzzahl-Arrays können das Gleitkomma-Bitmuster `NaN` nicht darstellen.

<!--en:7125bf5b-->
| Spalte | Vorhanden | Datentyp | Rolle in der Vorverarbeitung |

<!--en:39fc4178-->
| `NumRooms` | 7 / 10 | `float64` | Stetiges Merkmal (3 Einträge mit `NaN`) |

<!--en:07564b2a-->
| `RoofType` | 6 / 10 | `str` | Kategoriales Merkmal (4 Einträge mit `NaN`) |

<!--en:a7c3cb0d-->
| `Area` | 10 / 10 | `int64` | Stetiges Merkmal (vollständig) |

<!--en:ec418802-->
| `Price` | 10 / 10 | `int64` | Ziellabel $y$ für überwachtes Lernen |

<!--en:a9351fa4-->
- **`data.describe()` zeigt die unterschiedlichen Größenordnungen** der beobachteten Einträge ($\bar{x}_{\text{Area}} = 1{,}654\text{ sq ft}$ gegenüber $\bar{x}_{\text{Rooms}} = 3.29$):

<!--en:f2899e05-->
::: {.slide title="Zuerst Eingabemerkmale (X) von Ziellabels (y) trennen"}

<!--en:16837912-->
[Vorverarbeitung vorbereiten]{.kicker}

<!--en:1be0cf1d-->
- **$y$ vor der Transformation von $X$ abtrennen:** Trennen Sie die Merkmale *vor* Imputation oder Skalierung von der Zielgröße `Price`. So fließen keine Statistiken der Zielgröße in $\mathbf{X}$ ein.

<!--en:1f4bdf0a-->
- **Spalten nach Namen auswählen:** `data.drop(columns='Price')` verträgt Änderungen des Tabellenschemas, bei denen die Positionsauswahl `.iloc[:, :3]` scheitert.

<!--en:f6a66b75-->
::: {.slide title="Fehlende Werte erkennen: Risiken beim Löschen von Zeilen"}

<!--en:029aa6b8-->
[Fehlende Daten]{.kicker}

<!--en:42124db4-->
- **Lücken zählen (`isna().sum()`):** Spaltenweise Summe der booleschen Maske für fehlende Werte (`NumRooms`: 3 fehlen, `RoofType`: 4 fehlen).

<!--en:97a15f12-->
- **Warum `dropna()` in der Praxis scheitert:**
  1. **Verlust von Beispielen:** Nach dem Löschen unvollständiger Zeilen (`inputs.dropna()`) bleiben nur **4 von 10 Häusern**. Damit entfallen $60\%$ der Daten.
  2. **Selektionsverzerrung:** Fehlen Werte *nicht* völlig zufällig (z. B. überproportional oft die Zimmerzahl von Einzimmerwohnungen oder Luxusanwesen), verzerrt das Löschen die verbleibende Stichprobe.

<!--en:1897840c-->
::: {.slide title="Numerische Imputation und Indikatoren für fehlende Werte" layout="tall-output"}

<!--en:4346bc26-->
[Imputationsverfahren]{.kicker}

<!--en:6c7ea474-->
- **Imputation mit Mittelwert oder Median:** Ersetzt numerische `NaN`-Einträge durch $\mu_{\text{Rooms}} \approx 3.29$. Häufen sich imputierte Werte bei $\mu$, sinkt die Streuung der Stichprobe $\sigma$ ($1.11 \to 0.91$).

<!--en:c4fe4c52-->
- **Indikator für fehlende Werte (`is_na`):** Eine zusätzliche binäre $0/1$-Spalte erlaubt es dem linearen Modell, Vorhersagen über ihr Gewicht zu verschieben, wenn das Fehlen selbst Information liefert (z. B. bei Luxusanwesen).

<!--en:cb211df0-->
::: {.slide title="Kategoriale Merkmale als orthogonale One-Hot-Vektoren kodieren"}

<!--en:5d81bef2-->
[Darstellung von Kategorien]{.kicker}

<!--en:66855c8b-->
- **Warum Ganzzahlen für ungeordnete Kategorien ungeeignet sind:** Mit $\textrm{Slate}=0, \textrm{Tile}=1, \textrm{missing}=2$ muss jede lineare Schicht $\mathbf{x}^\top \mathbf{w}$ die Kategorie `Tile` genau zwischen `Slate` und einer unbekannten Dacheindeckung einordnen. So entstehen eine künstliche Reihenfolge und künstliche Abstände.

<!--en:09097672-->
- **One-Hot-Basisvektoren in $\mathbb{R}^K$:** `pd.get_dummies(..., dummy_na=True, dtype=float)` bildet $K$ Kategorien auf $K$ orthogonale $0/1$-Spalten ab (einschließlich `RoofType_nan` als eigener Indikator für fehlende Werte). Alle Kategorien haben paarweise denselben euklidischen Abstand $\sqrt{2}$.

<!--en:82d11788-->
::: {.slide title="Die One-Hot-Tabelle und ihre Grenzen bei vielen Kategorien"}

<!--en:5d81bef2-2-->
[Darstellung von Kategorien]{.kicker}

<!--en:bbab1b72-->
- **Konstante Zeilensumme:** In jeder Zeile gilt `Slate + Tile + nan = 1.0`. Die Kategorien liegen damit auf orthogonalen Eckpunkten von $\mathbb{R}^K$.

<!--en:26c8db4b-->
- **Grenze bei vielen Kategorien:** Bei $K = 50{,}000$ Postleitzahlen oder Token bestehen One-Hot-Vektoren zu $99.99\%$ aus Nullen. Verwenden Sie stattdessen gelernte **Einbettungen** $\mathbf{E} \in \mathbb{R}^{K \times d}$ ($d \ll K$).

<!--en:50b159e5-->
::: {.slide title="Die One-Hot-Tabelle (Fortsetzung)"}

<!--en:78544ca5-->
[One-Hot-Kodierung im Code]{.kicker}

<!--en:812643a6-->
`pd.get_dummies` mit `dummy_na=True` erzeugt für jede Kategorie von `RoofType`, einschließlich `NaN`, eine eigene $0/1$-Spalte:

<!--en:eedb1bec-->
::: {.slide title="Merkmalsskalierung: Wie unterschiedliche Einheiten die Optimierung verzerren"}

<!--en:0cbb6873-->
[Standardisierung]{.kicker}

<!--en:d80d0a60-->
- **Unterschiedliche Einheiten:** In den ursprünglichen Einheiten ist $\Delta\text{Area} = 3\text{ sq ft}$ viel größer als $\Delta\text{NumRooms} = 2\text{ rooms}$. Das verzerrt euklidische Abstände, den $L_2$-Gewichtszerfall und die Gradienten.

<!--en:416262fe-->
- **Schlecht konditionierte Täler:** Ein Skalenunterschied um den Faktor $500\times$ streckt die Niveaukurven der Verlustfunktion, wie die Hesse-Matrix $\nabla^2 \mathcal{L}$ zeigt. Der Gradientenabstieg verläuft im Zickzack und erfordert eine sehr kleine Lernrate.

<!--en:8bdbd096-->
- **z-Standardisierung** bringt jede stetige Spalte auf den **Mittelwert $0$** und die **Varianz eins ($1$)**:
  $$z_i = \frac{x_i - \mu}{\sigma}, \qquad \sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \mu)^2}$$

<!--en:15459e60-->
::: {.slide title="Stetige Merkmale standardisieren und binäre Indikatoren beibehalten"}

<!--en:bb0c84a3-->
[Standardisierung im Code]{.kicker}

<!--en:fdf2f195-->
- **Nur stetige Merkmale skalieren:** Wir standardisieren `['NumRooms', 'Area']`. Die One-Hot-Spalten `RoofType_*` behalten ihre Werte aus $\{0.0, 1.0\}$ und zeigen weiterhin an, ob eine Kategorie vorliegt.

<!--en:51435f5d-->
- **$z$-Werte deuten:** Zeile 6 (5 Zimmer, $2{,}600\text{ sq ft}$) wird auf $(+1.89, +1.70)$ abgebildet. Das Haus liegt in beiden Merkmalen $\approx 1.7\text{--}1.9\sigma$ über dem Mittelwert.

<!--en:c9a33a59-->
- **Imputierte Werte werden zu $0$:** In den Zeilen 1, 4 und 8 haben wir $\mu_{\text{Rooms}}$ eingesetzt. Daher ist ihre standardisierte Zimmerzahl genau $0.0$.

<!--en:2fbd5b98-->
::: {.slide title="Stetige Merkmale standardisieren (Fortsetzung)"}

<!--en:e7dff884-->
[Die standardisierte Tabelle]{.kicker}

<!--en:e427e2e3-->
`NumRooms` und `Area` enthalten nun $z$-Werte. Die Indikatoren `RoofType_*` behalten ihre $0/1$-Werte:

<!--en:74bfef84-->
::: {.slide title="Regel der Vorverarbeitung: aus Trainingsdaten schätzen, auf Testdaten anwenden"}

<!--en:f4309661-->
[Trainings- und Testdaten trennen]{.kicker}

<!--en:f153de7e-->
- **Wie Datenleckage entsteht:** Schätzt man Imputationsmittelwerte, Kategorienschemata oder $(\mu, \sigma)$ vor der Aufteilung aus der ganzen Tabelle, fließt Information über die Testverteilung in die Trainingsmerkmale ein.

<!--en:b2a0f2ac-->
- **Nur aus $\mathcal{D}_{\text{train}}$ schätzen:** $(\mu_{\text{train}}, \sigma_{\text{train}})$ festhalten und unverändert auf $\mathcal{D}_{\text{val}}$ und $\mathcal{D}_{\text{test}}$ anwenden.

<!--en:e553fc66-->
- **Schutz bei Varianz null:** Falls $\sigma_{\text{train}} = 0$ gilt, mit $\sigma \leftarrow \max(\sigma, \epsilon)$ die Division durch null verhindern.

<!--en:8c492ec8-->
::: {.slide title="Verständnisfrage: Datenleckage und Kodierungsfehler erkennen"}

<!--en:929b7671-->
[Verständnis prüfen]{.kicker}

<!--en:ff3db804-->
Finden Sie den statistischen oder technischen Fehler in jedem Codebeispiel:

<!--en:8ae1abc8-->
1. **Beispiel A:** Zuerst den gesamten DataFrame standardisieren: `df[num] = (df[num] - df[num].mean()) / df[num].std()`. Dann aufteilen: `train, test = df.iloc[:800], df.iloc[800:]`.

<!--en:f41ce3e3-->
2. **Beispiel B:** `Neighborhood` (`"Downtown"`, `"Suburb"`, `"Rural"`) als ganze Zahlen `0, 1, 2` kodieren und dem linearen Regressionsmodell $\hat{y} = \mathbf{x}^\top \mathbf{w} + b$ übergeben.

<!--en:ab960894-->
3. **Beispiel C:** `pd.get_dummies(train_df)` und `pd.get_dummies(test_df)` getrennt aufrufen, obwohl `"Slate"` in `train_df` vorkommt, aber nie in `test_df`.

<!--en:557b98b6-->
**Antworten:**

<!--en:357fefc8-->
1. **Datenleckage:** Die Testzeilen $800\ldots N$ haben $\mu$ und $\sigma$ beeinflusst. `mu, sigma` nur aus `train` berechnen, dann `train` und `test` mit diesen Trainingsparametern transformieren.

<!--en:449bb554-->
2. **Künstliche metrische Ordnung:** Ein einziges Gewicht $w_{\text{neigh}}$ erzwingt $\hat{y}(\text{Suburb}) - \hat{y}(\text{Downtown}) = \hat{y}(\text{Rural}) - \hat{y}(\text{Suburb})$. Verwenden Sie die One-Hot-Kodierung.

<!--en:c6f50498-->
3. **Unterschiedliche Merkmalsdimensionen:** In `test_df` fehlt die Spalte `RoofType_Slate`. Bei der Matrixmultiplikation passen die Formen nicht zusammen oder die Gewichte sind falsch zugeordnet. Passen Sie `test_df` an die Spalten von `train_df` an.

<!--en:ee7b98bb-->
::: {.slide title="Bereinigte DataFrames in zusammenhängende float32-Tensoren umwandeln"}

<!--en:c7e41231-->
[Umwandlung in Tensoren]{.kicker}

<!--en:fe14a989-->
- **Typumwandlung `float64` $\to$ `float32`:** pandas verwendet standardmäßig 64-Bit-Gleitkommazahlen (8 B) oder gemischte `bool`/`object`-Spalten. Mit `.to_numpy(dtype='float32')` halbiert sich der Speicherbedarf (4 B), und der Typ passt zu den Schichtgewichten auf der GPU.

<!--en:09027f43-->
::: {.slide title="Zusammenfassung: von Rohtabellen zu Designmatrizen für Modelle"}

<!--en:c5a222f8-->
[Das Wichtigste]{.kicker}

<!--en:54c2f27e-->
| Verarbeitungsschritt | Operation in pandas oder im Framework | Statistischer Zweck | Regel für den Einsatz und Schutz vor Datenleckage |

<!--en:a8e48da4-->
| **Schema und Skalen prüfen** | `df.info()`, `df.describe()` | Typumwandlung durch `NaN`, Textspalten und Skalenunterschiede erkennen | Verteilungen und Kennwerte vor der Modellierung prüfen |

<!--en:29abff42-->
| **Zielgröße abtrennen** | `df.drop(columns='Price')`, `df['Price']` | Designmatrix $\mathbf{X}$ von Ziellabels $\mathbf{y}$ trennen | Nach Spaltennamen auswählen, nie anhand anfälliger `.iloc`-Positionen |

<!--en:92d5bde1-->
| **Fehlende Zahlenwerte** | `fillna(train_mean)` und Indikator `isna()` | Zeilen ohne Selektionsverzerrung erhalten und das Fehlen als Information nutzen | Mittelwert oder Median für die Imputation **nur aus $\mathcal{D}_{\text{train}}$** schätzen |

<!--en:31b2cb67-->
| **Ungeordnete Kategorien** | `pd.get_dummies(..., dummy_na=True)` | $K$ Kategorien auf orthogonale Basisvektoren in $\mathbb{R}^K$ abbilden | Kategorienschema anhand von **$\mathcal{D}_{\text{train}}$** festlegen, bei $K \gg 10^2$ Einbettungen verwenden |

<!--en:eba388f1-->
| **Stetige Merkmale skalieren** | $(x - \mu_{\text{train}}) / \sigma_{\text{train}}$ | Einheiten vereinheitlichen ($\mu=0, \sigma=1$) und die Konditionierung der Verlustfläche verbessern | $(\mu, \sigma)$ **nur aus $\mathcal{D}_{\text{train}}$** schätzen, $0/1$-Indikatoren unskaliert lassen |

<!--en:cdf9598c-->
| **Tensoren erzeugen** | `tensor(df.to_numpy(dtype='float32'))` | Heterogene Spalten in einem zusammenhängenden `float32`-Puffer speichern | Formen `(N, D)` für Merkmale und `(N,)` für Zielgrößen prüfen |
