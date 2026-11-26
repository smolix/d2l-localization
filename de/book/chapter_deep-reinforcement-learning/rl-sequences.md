<!--en:1f5feea4-->
# Bestärkendes Lernen zur Sequenzgenerierung

<!--en:de3f025a-->
Jedes Sprachmodell in diesem Buch hat gelernt, Text nachzuahmen. Das ist Verhaltensklonen (engl. behavior cloning, :numref:`sec_imitation`) am Beispiel der Menschen, die das Korpus verfasst haben. Die heute gebräuchlichen Modelle erhalten im weiteren Training ein Signal, das kein Korpus enthält. Ein Prüfprogramm beurteilt, ob das Endergebnis eines Lösungswegs stimmt, oder ein Leser entscheidet, welche von zwei Antworten besser ist. Der Score kommt nur einmal, nachdem das Modell die gesamte Antwort selbst geschrieben hat. Es gibt kein Zieltoken, das es kopieren könnte. In :numref:`sec_policygradient` und :numref:`sec_baselines` haben wir unter diesen Bedingungen eine Strategie (engl. policy) mit vier Rechenaufgaben als Prompts und einem Prüfprogramm trainiert.

<!--en:4eda153a-->
Die Idee ist älter als diese Modelle. :citet:`Christiano.Leike.Brown.ea.2017` passten eine Belohnungsfunktion an menschliche Vergleiche an und optimierten die erwartete Belohnung mit einem Policy-Gradienten. :citet:`ouyang2022training` nutzten dieses Verfahren für ein großes Sprachmodell und bestraften Abweichungen vom überwacht trainierten Modell. :citet:`Shao.Wang.Zhu.ea.2024` ersetzten das Wertnetz durch eine Gruppe von Samples und :citet:`DeepSeekAI.2025` die erlernte Belohnungsfunktion durch ein Prüfprogramm.

<!--en:7326a04c-->
Was gewinnen wir, wenn wir jedes Token einzeln betrachten, die Belohnung aber nur aus einer Zahl pro Antwort besteht? Wie stark bremst der Strafterm aus :numref:`sec_regularized` eine Strategie, die ein Schlupfloch gefunden hat? Die Antworten zeigen, wie das Update im Post-Training zustande kommt. Es verbindet das Gruppengewicht aus :numref:`sec_baselines` mit dem Wahrscheinlichkeitsverhältnis und dessen Clipping aus :numref:`sec_ppo` sowie mit dem Strafterm.

<!--en:002d967e-->
## Die Antwort als Trajektorie

<!--en:797f7d50-->
### Ein Gradient aus zwei Perspektiven

<!--en:b1b32769-->
Eine Antwort $y = (a_1, \ldots, a_T)$ auf einen Prompt $x$ ist eine Trajektorie, deren Aktionen Token sind (:numref:`tab_rl_sequence_dictionary`). Der Zustand vor Token $t$ ist das Präfix $s_t = (x, a_1, \ldots, a_{t-1})$. Hängen wir $a_t$ an, erreichen wir mit Wahrscheinlichkeit eins den Zustand $s_{t+1}$. Die Wahrscheinlichkeit der Antwort ist also das Produkt der Tokenwahrscheinlichkeiten, $\pi_\theta(y \mid x) = \prod_t \pi_\theta(a_t \mid s_t)$, und es gilt

<!--en:352f1dc5-->
:Autoregressive Textgenerierung als Markov-Entscheidungsprozess.

<!--en:b89d65cb-->
| Sprachmodellierung | Bestärkendes Lernen |

<!--en:0f42cda1-->
| Prompt $x \sim \mu_0$ | Anfangszustand $s_1$ |

<!--en:cf64cfa3-->
| Token $a_t$ | Aktion $a_t$ |

<!--en:9de9c7cd-->
| Präfix $(x, a_1, \ldots, a_{t-1})$ | Zustand $s_t$ |

<!--en:d7e178ad-->
| Vollständige Antwort $y = (a_1, \ldots, a_T)$ | Trajektorie $\tau$ |

<!--en:fa5adcef-->
| Präfix mit EOS am Ende, $(x, y)$ | Endzustand |

<!--en:e7b7f908-->
| Verteilung des nächsten Tokens | Strategie $\pi_\theta(a_t \mid s_t)$ |

<!--en:1cc56631-->
| Score $r(x, y)$ des Prüfprogramms oder Belohnungsmodells | Terminale Belohnung |

<!--en:b4a3d415-->
Die linke Seite von :eqref:`eq_seq_factorization` behandelt die Antwort als eine einzige Aktion, die rechte als $T$ einzelne Tokenaktionen. Bei einer einzigen Belohnung $r(x, y)$ am Ende und ohne Diskontierung beträgt der künftige Ertrag (engl. reward-to-go) für jedes Token $r(x, y)$. Beide Sichtweisen multiplizieren daher den Score jedes Tokens mit demselben *Antwortgewicht* (engl. response weight) $r(x, y) - b(x)$. Das gilt für jede Baseline $b(x)$, die nur vom Prompt abhängt.

<!--en:72d1ec15-->
### Zuordnung des Erfolgs innerhalb einer Antwort

<!--en:71ab46fe-->
Das Antwortgewicht liefert einen unverzerrten Schätzer, behandelt aber alle Token gleich. Dabei kann auch eine falsche Antwort richtige Token enthalten. Auf den Prompt `3*4` antworten wir mit einer Zehnerziffer $a_1$ und einer Einerziffer $a_2$, beide gleichverteilt über $0, \ldots, 9$. Nur `12` ist richtig. Daher gelten $V(x) = 1/100$, $V(x, 1) = 1/10$ und $V(x, a_1) = 0$ für jede andere Zehnerziffer. Die Antwort `13` erzielt den Score $0$. Beide Token erhalten das Gewicht $r - V(x) = -0.01$, sodass die richtige Zehnerziffer an Wahrscheinlichkeit verliert. Die Tokenvorteile (engl. token advantages) aus :numref:`sec_baselines` berücksichtigen dagegen das jeweilige Präfix. Die Zehnerziffer erhält $A(s_1, a_1) = V(x, a_1) - V(x) = +0.09$ und die Einerziffer $A(s_2, a_2) = r - V(x, a_1) = -0.1$. Die Funktion `token_credit` berechnet $\pi(y \mid x)$, das Antwortgewicht und beide Vorteile für alle hundert Antworten $y = 10 a_1 + a_2$:

<!--en:e06d3ea5-->
Für drei Antworten geben wir jeweils die drei Gewichte aus. Wir prüfen, ob Antwortgewicht und Vorteil der Zehnerziffer denselben erwarteten Gradienten für die Zehnerziffer ergeben (ihr Score ist $\mathbf{e}_{a_1} - \mathbf{p}_1$). Dann vergleichen wir die Varianzen der Gewichte, deren Mittelwerte alle null sind:

<!--en:3ae87441-->
Die Varianz des Antwortgewichts ist elfmal so groß wie die des Vorteils der Zehnerziffer. Das Antwortgewicht lässt sich als Teleskopsumme schreiben, $r - V(x) = \big(V(x, a_1) - V(x)\big) + \big(r - V(x, a_1)\big) = A(s_1, a_1) + A(s_2, a_2)$. Die beiden Terme sind unkorreliert, denn für jedes gegebene Präfix hat der zweite den Mittelwert null. Also addieren sich ihre Varianzen zu $0.0099 = 0.0009 + 0.0090$. In den Score jedes Tokens geht das Rauschen aller anderen Token ein. Das ergibt $T$ Terme für $T$ Token.

<!--en:172607a2-->
### Notwendige und optionale Komponenten

<!--en:082db8d0-->
Bei bekannten Übergängen und einer einzigen terminalen Belohnung (engl. terminal reward) entfällt vieles aus diesem Kapitel. Antworten sind endlich, daher gilt $\gamma = 1$. Wir müssen kein Übergangsmodell lernen und können die Generierung bei jedem beliebigen Präfix neu beginnen. Da es kein Off-Policy-Bootstrapping gibt, tritt die tödliche Triade aus :numref:`sec_dqn` nicht auf. Ein Kritiker (engl. critic) ist optional. Das Antwortgewicht liefert einen unverzerrten Schätzer, und der Gruppenmittelwert aus :numref:`sec_baselines` dient als Baseline ohne Netz, wie bei RLOO :cite:`Ahmadian.Cremer.Galle.ea.2024` und GRPO. Mit einem Wertkopf für jedes Präfix :cite:`ouyang2022training` oder mit Fortsetzungen, die wir ausgehend vom Präfix ziehen wie bei VinePPO :cite:`Kazemnejad.Aghajohari.Portelance.ea.2025`, können wir den Erfolg einzelnen Token zuordnen. Weiterhin brauchen wir den Score mit einer Baseline, eine Schranke für jeden Schritt und einen Schutz vor Fehlern in der Belohnung (:numref:`fig_rl_token_mdp`).

<!--en:ad39b937-->
![Der Token-MDP. Oben: Zustände sind Präfixe, Aktionen sind Token. Jeder Übergang hängt das gewählte Token mit Wahrscheinlichkeit eins an. Die Belohnung gibt es nur einmal, wenn EOS in den Endzustand führt. Unten: Eine gezogene Antwort $y \sim \pi_\theta(\cdot \mid x)$ erhält den Score $r(x, y)$. Nach :eqref:`eq_seq_factorization` ergibt sich derselbe Gradient. Rechts: die dadurch überflüssigen Komponenten und der optionale Präfixwert.](../img/mdl-rl-token-mdp.svg)

<!--en:9009337e-->
## Prüfprogramme, Belohnungsmodelle und ihre Schlupflöcher

<!--en:cab4da86-->
Die Belohnung kann von einem Modell stammen, das wir an menschliche Vergleiche angepasst haben (:numref:`sec_regularized`). Am unzuverlässigsten ist es dort, wo keine Vergleiche vorlagen. Alternativ liefert eine Prüfung die Belohnung, etwa eine Testumgebung oder ein Programm, das auf exakte Übereinstimmung prüft. Mit solchem *bestärkenden Lernen aus überprüfbaren Belohnungen* (engl. reinforcement learning from verifiable rewards, RLVR) trainiert man neuere Modelle für schlussfolgerndes Denken :cite:`DeepSeekAI.2025`. Ein Prüfprogramm beurteilt genau die Eigenschaft, die es prüft. Das kann eine andere sein als die beabsichtigte.

<!--en:80aa4186-->
### Der Gruppenschritt mit KL-Strafterm

<!--en:a36c1ba3-->
Bei der Prüfaufgabe aus :numref:`sec_mdp` beantwortet keiner der gewöhnlichen Kandidaten `8*8` oder `9*9` richtig. Der neunte Kandidat, die *Ausweichantwort* „eine der Zahlen 4, 12, 64 oder 81“, hat unter der Referenzstrategie das $e^{-4}$-Fache der Wahrscheinlichkeit jedes gewöhnlichen Kandidaten. `sloppy_checker` akzeptiert jede Antwort, die das richtige Ergebnis enthält, und belohnt daher die Ausweichantwort bei jedem Prompt:

<!--en:ddbe60d4-->
Beim Training mit dem nachsichtigen Prüfprogramm findet die Strategie die Ausweichantwort. Den Strafterm $\beta\, D_{\textrm{KL}}\big(\pi(\cdot \mid x) \Vert \pi_{\textrm{ref}}(\cdot \mid x)\big)$ aus :numref:`sec_regularized` können wir in die Belohnung einbeziehen. Jede gezogene Antwort erhält dann den Score $r(x, y) - \beta \log\big(\pi_\theta(y \mid x) / \pi_{\textrm{ref}}(y \mid x)\big)$. Der REINFORCE-Schätzer bleibt unverzerrt, denn das Ableiten von $\log \pi_\theta$ innerhalb des Scores liefert als zusätzlichen Term $-\beta\, \mathbb{E}[\nabla_\theta \log \pi_\theta] = 0$. Für $T > 1$ Token ist das logarithmische Verhältnis die Summe der logarithmischen Verhältnisse der einzelnen Token. Dadurch wird aus der terminalen Belohnung eine Belohnung an jedem Token. Für das Gewicht jedes Tokens genügen daher die logarithmischen Verhältnisse ab diesem Token (Übung :exref:`ex_rl-sequences-kl-loss`c). `kl_group_step` übergibt die Belohnung einschließlich Strafterm an den Gruppenschritt `d2l.group_step` aus :numref:`sec_baselines`:

<!--en:8fb8053d-->
Wie groß muss $\beta$ sein? Das Optimum ist die mit der Belohnung exponentiell umgewichtete Referenzverteilung, $\pi^\star(y \mid x) = \pi_{\textrm{ref}}(y \mid x)\, e^{r(x, y)/\beta} / Z(x)$ (:numref:`sec_regularized`). Bei `8*8` belohnt das nachsichtige Prüfprogramm nur die Ausweichantwort. Das Verhältnis ihrer Wahrscheinlichkeit zu der eines gewöhnlichen Kandidaten steigt daher von $e^{-4}$ auf $e^{-4} e^{1/\beta}$. Die Ausweichantwort ist genau dann wahrscheinlicher, wenn $\beta < 1/4$ gilt. Bei $\beta = 0.2$ ist sie $e \approx 2.7$-mal so wahrscheinlich. Bei `2+2` und `3*4` erhält auch die richtige Antwort eine Belohnung. Das Verhältnis bleibt bei $e^{-4}$, und es gibt keinen Schwellenwert.

<!--en:5b450b68-->
### Vergleich der Parameterstudie mit der geschlossenen Lösung

<!--en:ba2d7805-->
Wir trainieren mit $K = 32$ über $400$ Updates für fünf Werte von $\beta$ und mitteln die letzten $100$ Strategien. Wir geben den nachsichtigen und den exakten Score sowie die mittlere Wahrscheinlichkeit der Ausweichantwort aus. Daneben stehen dieselben Größen für das exponentiell umgewichtete Optimum (in Präferenzen ausgedrückt: Referenzpräferenzen plus Belohnung geteilt durch $\beta$):

<!--en:8bb2591d-->
Bei $\beta = 0$ erzielt die Strategie den vollen nachsichtigen Score, bei exakter Auswertung aber nur $0.50$. Für `8*8` und `9*9` legt sie die gesamte Wahrscheinlichkeit auf die Ausweichantwort. Diese bleibt bei $\beta = 0.1$ und $0.2$ bevorzugt. Oberhalb von $1/4$ ändert sich das. Bei $\beta = 0.3$ ist sie für `8*8` etwa halb so wahrscheinlich wie jeder gewöhnliche Kandidat, wie $e^{-4} e^{1/0.3} \approx 0.5$ vorhersagt. Das geht auf Kosten des erwünschten Lerneffekts. Bei $\beta = 0.5$ fällt der exakte Score auf etwa ein Viertel. Die Werte aus dem Training weichen in der Tabelle um höchstens etwa $0.02$ vom Optimum ab. Dieser kleine Abstand ist systematisch. Nahe $\pi^\star$ haben die Belohnungen einschließlich Strafterm innerhalb einer Gruppe fast denselben Wert. Die Standardisierung vergrößert den Schritt also gerade dort, wo das Verfahren zur Ruhe kommen sollte. Der Strafterm macht das Schlupfloch bei jedem $\beta$ kostspielig, schließt es aber nie.

<!--en:ff3f7db0-->
## Das Update im Post-Training

<!--en:0a137340-->
### Wahrscheinlichkeitsverhältnisse für veraltete Samples

<!--en:fb052f1e-->
Bisher nutzte jedes Update eine neue Gruppe genau einmal. Beim Post-Training im großen Maßstab erzeugt ein Inferenzsystem die Antworten. Seine Kernel und seine Rechengenauigkeit unterscheiden sich von denen des Trainingssystems, und mehrere Gradientenschritte verwenden denselben Batch. Die Samples stammen von einer älteren Strategie $\pi_{\theta_{\textrm{old}}}$. Deshalb gewichten wir Token $t$ mit dem Verhältnis $\rho_t = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\textrm{old}}}(a_t \mid s_t)$ und wenden Clipping wie in :numref:`sec_ppo` an. Für die ganze Antwort erhalten wir das Verhältnis $W$ und das geometrische Mittel der Tokenverhältnisse $\bar\rho$:

<!--en:f266b8c0-->
Das Produkt $W$ liefert die exakte Korrektur. Sein Logarithmus summiert jedoch $T$ verrauschte Terme, sodass das Produkt bei langen Antworten viele Größenordnungen überspannt. Group Sequence Policy Optimization (GSPO) begrenzt stattdessen $\bar\rho$. Das ist eine Zahl pro Antwort, deren Größenordnung nicht mit der Länge wächst :cite:`Zheng.Liu.Li.ea.2025`.

<!--en:ab5eb999-->
### Asymmetrische Clipping-Intervalle

<!--en:959d12ed-->
Ein Intervall fester Breite für das Verhältnis lässt unterschiedlich große Änderungen der Wahrscheinlichkeit zu. Bei $\epsilon_{\textrm{clip}} = 0.2$ bringt es einem Token mit Wahrscheinlichkeit $0.01$ nichts, über $0.012$ zu steigen. Für ein Token mit Wahrscheinlichkeit $0.6$ liegt diese Grenze bei $0.72$. Umfasst das Vokabular Zehntausende meist seltener Token, nimmt die Wahrscheinlichkeit nützlicher seltener Token nur langsam zu und die Entropie sinkt. Deshalb erhöht DAPO nur die obere Schranke auf $1 + \epsilon_{\textrm{high}}$ mit $\epsilon_{\textrm{high}} = 0.28$ :cite:`Yu.Zhang.Zhu.ea.2025` (Übung :exref:`ex_ppo-asymmetric-bands`).

<!--en:c99c8448-->
### Group Relative Policy Optimization

<!--en:ff81ddba-->
Group Relative Policy Optimization (GRPO) :cite:`Shao.Wang.Zhu.ea.2024` verbindet diese Bausteine. Das Verfahren zieht für jeden Prompt $K$ Antworten aus $\pi_{\theta_{\textrm{old}}}$. Jedes Token der Antwort $j$ erhält das standardisierte Gewicht $\hat{A}_j = (r_j - \bar{r}) / (\sigma_r + 10^{-8})$ aus :numref:`sec_baselines`. Dabei sind $\bar{r}$ und $\sigma_r$ der Mittelwert und die Standardabweichung der Belohnungen in der Gruppe. Über mehrere Epochen mit derselben Gruppe maximiert das Verfahren dann folgenden Ausdruck:

<!--en:e4ad746e-->
Dabei bezeichnet $T_j$ die Länge der Antwort $j$, und $k_3 = \pi_{\textrm{ref}}/\pi_\theta - \log(\pi_{\textrm{ref}}/\pi_\theta) - 1$ wird am gezogenen Token ausgewertet. Sobald das Verhältnis auf der günstigen Seite außerhalb des Intervalls liegt, fällt durch das Minimum der Gradientenbeitrag des Samples weg. Das Clipping vergleicht mit der Strategie, aus der wir die Samples gezogen haben, und beschränkt jedes Update. Der Strafterm vergleicht mit der festgehaltenen Referenzstrategie und verändert das Optimum.

<!--en:808d8e58-->
`grpo_step` setzt diese Zielfunktion für die Prüfaufgabe um, bei der jede Antwort aus einem Token besteht. In jedem der `num_epochs` Durchläufe berechnet die Funktion die Verhältnisse zur Strategie, aus der die Samples stammen. Sie behält die Samples, deren begrenzter Term noch von $\theta$ abhängt (`live`), und macht einen Schritt in Aufstiegsrichtung. Die Gradienten von $\rho\,\hat{A}$ und $-\beta k_3$ ergeben sich als Produkt der jeweiligen Faktoren $\rho\,\hat{A}$ und $-\beta(1 - \pi_{\textrm{ref}}/\pi_\theta)$ mit dem Score $\mathbf{e}_y - \pi_\theta(\cdot \mid x)$:

<!--en:caef0f6b-->
Wir verwenden dieselbe Gruppe von $32$ Antworten pro Prompt über vier Epochen. Für jede Epoche geben wir die Spannweite der Verhältnisse und ihren Anteil außerhalb des Intervalls $[0.8, 1.2]$ aus. Zum Vergleich betrachten wir einen einzigen Schritt mit derselben Gruppe:

<!--en:927e4af8-->
In der ersten Epoche sind alle Verhältnisse gleich $1$, weil sich die Strategie noch nicht geändert hat. Schon nach einem Schritt übersteigen die Verhältnisse für richtige Antworten $1.5$. Das Clipping begrenzt also nicht die Schrittweite, sondern nimmt nur den Anreiz für weiteres Wachstum jenseits der Grenze. Der Anteil außerhalb des Intervalls wächst danach. Bei fast jeder Gruppe ist er in der vierten Epoche größer als in der zweiten. Bei allen vierzig Gruppen war der exakte Score nach vier Epochen höher als nach einer.

<!--en:85a1b9c7-->
Der Term $k_3$ bremst die Strategie anders als der in die Belohnung einbezogene Strafterm. Sein Erwartungswert unter $\pi_\theta$ beträgt $D_{\textrm{KL}}(\pi_\theta \Vert \pi_{\textrm{ref}})$. Leiten wir ihn jedoch bei festgehaltenen Samples ab, ergibt sich im Erwartungswert der Gradient der Vorwärtsdivergenz $D_{\textrm{KL}}(\pi_{\textrm{ref}} \Vert \pi_\theta)$ :cite:`Tang.Munos.2025` (Übung :exref:`ex_rl-sequences-kl-loss`). An einem Schlupfloch unterscheiden sich die beiden Richtungen aus :numref:`subsec_kl_direction` deutlich. Bei `8*8` und $\beta = 0.5$ hält das Optimum der umgekehrten Divergenz die Wahrscheinlichkeit der Ausweichantwort bei $0.017$, das der Vorwärtsdivergenz bei $0.50$. Die Vorwärtsdivergenz bestraft es, wenn die Strategie gewöhnliche Kandidaten der Referenz vernachlässigt. Sie bestraft die Verschiebung von Wahrscheinlichkeit auf einen von der Referenz selten gewählten Kandidaten nicht als solche.

<!--en:dd7307fc-->
Zwei verbreitete Varianten ändern diese Festlegungen. DAPO lässt den Strafterm weg, erhöht die obere Clipping-Grenze und verwirft Gruppen, in denen alle Belohnungen gleich sind :cite:`Yu.Zhang.Zhu.ea.2025`. Dr. GRPO verzichtet auf die Division durch $\sigma_r$ und durch die Antwortlänge (:numref:`subsec_pg_loss_divisor`) :cite:`Liu.Chen.Li.ea.2025`.

<!--en:b67a553d-->
### Direct Preference Optimization

<!--en:3f768aa3-->
Lösen wir die Gleichung für das exponentiell umgewichtete Optimum nach der Belohnung auf, erhalten wir

<!--en:935f1f14-->
Jede Strategie legt also die Belohnung fest, für die sie optimal ist, bis auf einen vom Prompt abhängigen Term, der bei Vergleichen wegfällt. Setzen wir diese Belohnung in die Bradley-Terry-Likelihood ein, wird aus der Anpassung der Belohnungsfunktion ein Klassifikationsverlust (engl. classification loss) für die Strategie selbst. Dieses Verfahren ohne Sampling und ohne Belohnungsmodell heißt *Direct Preference Optimization* (DPO) :cite:`Rafailov.Sharma.Mitchell.ea.2023`. Als Daten dient eine feste Menge von Vergleichen. Damit stellen sich dieselben Fragen wie in :numref:`sec_offline`.

<!--en:1a0e29e1-->
Bisher gab es auf jeden Prompt nur eine Antwort und eine einzige terminale Belohnung. Das macht die Aufgabe zu einem kontextuellen Banditen (engl. contextual bandit), dessen Aktionen ganze Antworten sind (:numref:`sec_exploration`). Bei Werkzeugaufrufen, Rückmeldungen aus einer Umgebung oder Gesprächen über mehrere Runden kommt der Übergangskern wieder ins Spiel. Mit ihm kehren der Kritiker und das Problem der Zuordnung des Erfolgs zurück. Der Buchteil über Sprachmodelle behandelt das Post-Training im großen Maßstab. Weitere Varianten und ein Projekt finden sich in :numref:`sec_deep_rl-further-reading`.

<!--en:9a272665-->
Das Kapitel begann mit einem Pendel, das einige Hunderttausend Schritte benötigte. Bei CartPole sparte der Einschritt-Kritiker keine Schritte ein. Vier PPO-Epochen pro Batch halbierten die Schrittzahl, und DQN brauchte 35 bis 45 tausend Schritte. Bei Pendulum benötigte Soft Actor-Critic einige Tausend, ein erlerntes Modell nur einige Hundert. Der Preis dafür war zusätzlicher Rechenaufwand. Ein Offline-Lernverfahren macht überhaupt keinen neuen Schritt in der Umgebung. Mit denselben Samples kam eine Strategie nach vier Epochen pro Gruppe weiter als nach einer. Jede Einsparung beruht auf einer Schätzung, die falsch sein kann, ob sie von einem Kritiker, einem veralteten Batch, einem Modell, einem Protokoll oder einem Prüfprogramm stammt.

<!--en:30ac03ff-->
## Zusammenfassung

<!--en:3e8fbc4c-->
Eine Antwort ist eine Trajektorie aus Token. Beide Sichtweisen liefern denselben Gradienten. Ein einziger terminaler Score gibt jedem Token das Gewicht $r - b(x)$. Dessen Varianz ist die Summe der Varianzen der Tokenvorteile. Mit Präfixwerten lässt sich der Erfolg jedem Token einzeln zuordnen, allerdings brauchen wir dafür einen Kritiker oder Neustarts. Da das Antwortgewicht einen unverzerrten Schätzer liefert, bleibt der Kritiker optional. Wie teuer ein Schlupfloch im Prüfprogramm durch den KL-Strafterm wird, sagt das exponentiell umgewichtete Optimum einschließlich des Schwellenwerts voraus. GRPO kombiniert das Gruppengewicht, das begrenzte Verhältnis und den Strafterm $k_3$. Dessen Gradient ist der Gradient der Vorwärtsdivergenz, die die Ausweichantwort kaum bremst.

<!--en:2294243a-->
## Übungen

<!--en:ef72f53f-->
1. **Werkzeugausgaben in der Trajektorie.** Ein Modell ruft ein Werkzeug auf. Dazu erzeugt es die Token $y^{(1)}$, empfängt die Werkzeugausgabe $o \sim P_{\textrm{tool}}(\cdot \mid x, y^{(1)})$, die nicht von $\theta$ abhängt, und erzeugt anschließend die Token $y^{(2)}$. Das Prüfprogramm bewertet $\tau = (x, y^{(1)}, o, y^{(2)})$ einmal mit der Belohnung $r(\tau)$.
    1. Geben Sie die Wahrscheinlichkeit von $\tau$ und den Gradienten ihres Logarithmus an. Welche Token tragen Score-Terme bei?
    1. Eine Trainingsschleife, die Log-Wahrscheinlichkeiten für die gesamte Sequenz auswertet, fügt den Term $\big(r(\tau) - b(x)\big)\, \nabla_\theta \log \pi_\theta(o \mid x, y^{(1)})$ hinzu. Dabei verwendet sie die Wahrscheinlichkeit, die das Modell selbst der Werkzeugausgabe zuweist. Zeigen Sie, dass der Erwartungswert des Terms im Allgemeinen selbst bei konstantem Gewicht von null verschieden ist. Welcher Schritt im Beweis, dass ein Score den Mittelwert null hat, gilt hier nicht? Was lernt das Modell durch diesen Term?

<!--en:2ca8e60d-->
1. **Zweistellige Antworten.** Wir ziehen die beiden Ziffern aus :numref:`subsec_rl_sequences_credit` aus den Verteilungen $\mathbf{p}_1$ und $\mathbf{p}_2$. Jede Verteilung ergibt sich aus der Softmax-Funktion ihrer eigenen Präferenzen. Das Prüfprogramm vergibt $1$ nur für die richtige Antwort.
    1. Zeigen Sie, dass das Antwortgewicht $r - V(x)$ und die Tokenvorteile für die Präferenzen jeder Position denselben erwarteten Gradienten liefern. Welcher bedingte Erwartungswert erklärt die Übereinstimmung?
    1. Zeigen Sie, dass $A(s_1, a_1)$ und $A(s_2, a_2)$ unkorreliert sind, und geben Sie das entsprechende Ergebnis für eine Antwort aus $T$ Token an.
    1. Rufen Sie `token_credit` für folgende Strategie auf: Sie weist die Wahrscheinlichkeit $0.9$ der Ziffer $1$ an der ersten Position und der Ziffer $2$ an der zweiten Position zu. Jede andere Ziffer erhält $1/90$. Wie stark sinkt jetzt die Varianz des Updates für die Zehnerziffer, wenn Sie das Antwortgewicht durch $A(s_1, a_1)$ ersetzen? Warum sinkt sie weniger als bei der gleichverteilten Strategie?
   :exlabel:`ex_rl-sequences-two-digit-responses`

<!--en:6c4dd565-->
1. **Zwei Schwellenwerte für die Ausweichantwort.** Sei $\pi^\star$ das exponentiell umgewichtete Optimum für die Belohnung `sloppy` und die Referenzstrategie `task.pi_ref`.
    1. Bei `8*8` und `9*9` ist die Ausweichantwort die einzige Antwort, die `sloppy` belohnt. Geben Sie ihre Wahrscheinlichkeit unter $\pi^\star$ als Funktion von $\beta$ an. Für welches $\beta$ ist sie genauso wahrscheinlich wie jede der acht anderen Antworten? Wann entfällt die Hälfte der Wahrscheinlichkeit auf sie? Welcher der beiden Werte ist der Schwellenwert aus dem Text?
    1. Bei `2+2` und `3*4` werden sowohl die richtige Antwort als auch die Ausweichantwort belohnt. Welche Wahrscheinlichkeit weist $\pi^\star$ der Ausweichantwort für $\beta \to 0$ zu?
    1. Bei $\beta = 0$ legte das Training für `2+2` und `3*4` fast keine Wahrscheinlichkeitsmasse auf die Ausweichantwort, obwohl jede Aufteilung der Wahrscheinlichkeit zwischen den beiden belohnten Antworten die unregularisierte Zielfunktion maximiert. Welche Eigenschaft des Updates in `d2l.group_step` bestimmt die Aufteilung, die das Training erreicht hat?
   :exlabel:`ex_rl-sequences-two-thresholds`

<!--en:890bce52-->
1. [code] **DPO im endlichen Beispiel.** ● Ziehen Sie zwei Antworten auf denselben Prompt aus `task.pi_ref`. Die Präferenz zwischen ihnen folge dem Bradley-Terry-Modell mit der Belohnung `sloppy`. Bezeichnen Sie die bevorzugte Antwort mit $y_w$ und die abgelehnte mit $y_l$. Setzen Sie $h_\theta(y) = \log\big(\pi_\theta(y \mid x) / \pi_{\textrm{ref}}(y \mid x)\big)$.
    1. Setzen Sie :eqref:`eq_dpo_reward` in das Bradley-Terry-Modell ein. Zeigen Sie, dass die negative Log-Likelihood eines Paars gleich $-\log \sigma\big(\beta\, h_\theta(y_w) - \beta\, h_\theta(y_l)\big)$ ist, wobei $\sigma$ die logistische Funktion bezeichnet. Warum kommt $Z(x)$ nicht vor?
    1. Zeigen Sie, dass das exponentiell umgewichtete Optimum bei unendlich vielen Paaren den erwarteten Verlust minimiert. Was folgt daraus für die Schwellenwerte aus Übung :exref:`ex_rl-sequences-two-thresholds`a?
    1. Ziehen Sie $1000$ Paare mit Präferenzlabel pro Prompt. Minimieren Sie für $\beta = 0.5$ den mittleren Verlust durch Gradientenabstieg mit Schrittweite $10$ auf der Präferenztabelle. Beginnen Sie mit `task.theta_ref`. Geben Sie für fünf Zufallsstartwerte die Wahrscheinlichkeit der Ausweichantwort bei jedem Prompt nach $200$, $2000$ und $20000$ Schritten an. Was geschieht bei Prompts, für die die Ausweichantwort nie einen Vergleich verliert? Warum verhindert $\beta$ das nicht?

<!--en:ebea4bff-->
1. **Der KL-Term als Verlust.** ●● GRPO zieht bei jedem gezogenen Token $\beta k_3$ ab und leitet diesen Term bei festgehaltenen Token ab :cite:`Shao.Wang.Zhu.ea.2024`. :citet:`Tang.Munos.2025` untersuchen, was solche Verlustfunktionen optimieren.
    1. Setzen Sie für Antworten aus einem Token $h = \log\big(\pi_\theta(y \mid x) / \pi_{\textrm{ref}}(y \mid x)\big)$ und vergleichen Sie die Verluste $k_1 = h$, $k_2 = h^2 / 2$ und $k_3 = e^{-h} + h - 1$. Zeigen Sie für $y \sim \pi_\theta$, dass der erwartete Gradient von $k_1$ null ist. Zeigen Sie außerdem, dass der erwartete Gradient von $k_2$ gleich $\nabla_\theta D_{\textrm{KL}}(\pi_\theta \Vert \pi_{\textrm{ref}})$ und der von $k_3$ gleich $\nabla_\theta D_{\textrm{KL}}(\pi_{\textrm{ref}} \Vert \pi_\theta)$ ist.
    1. Für den Vorwärtsstrafterm $\beta\, D_{\textrm{KL}}(\pi_{\textrm{ref}} \Vert \pi)$ ist der Maximierer $\pi(y) = \beta\, \pi_{\textrm{ref}}(y) / \big(m - r(y)\big)$. Der Multiplikator $m > \max_y r(y)$ sorgt dafür, dass die Summe der Wahrscheinlichkeiten eins ist (Übung :exref:`ex_regularized-forward-kl-penalty`a). Bestimmen Sie bei `8*8` mit der Belohnung `sloppy` den Wert von $m$ numerisch für $\beta \in \{0.1, 0.25, 0.5\}$. Vergleichen Sie die Wahrscheinlichkeit der Ausweichantwort mit der beim exponentiell umgewichteten Optimum. Wie hängt die Differenz von $\beta$ ab?
    1. Betrachten Sie Antworten aus $T$ Token. Bezeichnen Sie mit $h_t$ das logarithmische Verhältnis für Token $t$. Zeigen Sie mithilfe von :eqref:`eq_seq_factorization`, dass folgende Gleichung gilt:

       $$\nabla_\theta D_{\textrm{KL}}\big(\pi_\theta(\cdot \mid x) \Vert \pi_{\textrm{ref}}(\cdot \mid x)\big) = \mathbb{E}\Big[\sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \sum_{t'=t}^{T} h_{t'}\Big].$$

       Welche Terme fehlen in der Summe der Verluste $k_2$ über die Token? Warum bleiben diese Terme erhalten, wenn man den Strafterm wie in `kl_group_step` in die Belohnung einbezieht?
   :exlabel:`ex_rl-sequences-kl-loss`

<!--en:ca43af9e-->
1. **Asymmetrische Intervalle.** Ein Token hat unter der Strategie, aus der wir ziehen, die Wahrscheinlichkeit $0.01$. Nehmen Sie an, jeder Batch erhöht seine Wahrscheinlichkeit genau um den Faktor $1 + \epsilon_{\textrm{high}}$, ab dem das Clipping weiteres Wachstum nicht mehr belohnt (:numref:`subsec_ppo_asymmetric`). Den nächsten Batch ziehen wir jeweils aus der aktualisierten Strategie.
    1. Wie viele Batches braucht es, um die Wahrscheinlichkeit von $0.01$ auf $0.5$ zu erhöhen, wenn $\epsilon_{\textrm{high}} = 0.2$ gilt? Wie viele bei $\epsilon_{\textrm{high}} = 0.28$?
    1. In einer Aufgabe mit zwei Aktionen hat die andere Aktion anfangs die Wahrscheinlichkeit $0.99$. Berechnen Sie ihr Wahrscheinlichkeitsverhältnis nach einem solchen Batch. Welche Grenze des Intervalls bestimmt bei welcher Aktion, wie schnell die Wahrscheinlichkeit einer selten gezogenen Aktion steigen kann?
    1. Bei zwei Aktionen begrenzt die obere Schranke das Wachstum der wahrscheinlicheren Aktion mit Wahrscheinlichkeit $p$ nur, solange $(1 + \epsilon_{\textrm{high}})\, p < 1$ gilt. Für welche $p$ kann auch die Wahrscheinlichkeit dieser Aktion schneller wachsen, wenn wir $\epsilon_{\textrm{high}}$ von $0.2$ auf $0.28$ erhöhen? Was geschieht dann mit der Entropie? Warum hält dieselbe Änderung bei einem Sprachmodell mit Zehntausenden meist seltener Token die Entropie höher?
   :exlabel:`ex_ppo-asymmetric-bands`

<!--en:01c81796-->
1. **Längennormiertes Verhältnis.** ● GSPO :cite:`Zheng.Liu.Li.ea.2025` ersetzt die Verhältnisse der einzelnen Token sowohl bei der Gewichtung als auch beim Clipping durch das geometrische Mittel $\bar\rho = W^{1/T}$ aus :eqref:`eq_seq_ratio`. Modellieren Sie die logarithmischen Verhältnisse $\log \rho_t$ der einzelnen Token unter der Strategie, aus der wir ziehen, als unabhängig und identisch verteilt mit Mittelwert $-d$ und Varianz $v$. Dabei ist $d$ die Divergenz $D_{\textrm{KL}}(\pi_{\theta_{\textrm{old}}} \Vert \pi_\theta)$ pro Token.
    1. Zeigen Sie, dass $\log W$ den Mittelwert $-d T$ und die Varianz $vT$ hat, während $\log \bar\rho$ den Mittelwert $-d$ und die Varianz $v / T$ hat.
    1. Zeigen Sie, dass $\mathbb{E}[\bar\rho] \leq 1$ gilt und dass $\mathbb{E}[\bar\rho] \to e^{-d}$ für $T \to \infty$ gilt. Ein Wichtigkeitsgewicht $W$ erfüllt $\mathbb{E}_{\pi_{\theta_{\textrm{old}}}}[W f] = \mathbb{E}_{\pi_\theta}[f]$ für jede Funktion $f$ der Antwort. Gilt das auch für $\bar\rho$?
    1. GSPO begrenzt $\bar\rho$ auf $[1 - 3 \cdot 10^{-4}, 1 + 4 \cdot 10^{-4}]$. Das GRPO-Vergleichsverfahren begrenzt die Verhältnisse der einzelnen Token auf $[1 - 0.2, 1 + 0.27]$. Erklären Sie anhand von (a), warum ein Intervall für die ganze Sequenz viel enger sein muss, damit die Begrenzung greift.

<!--en:a5bd10e1-->
[Diskussion](https://d2l.discourse.group/t/36807)

<!--en:ad3fc336-->
Bestärkendes Lernen für Sequenzgenerierung und große Sprachmodelle<br>**Token-MDP · Zerlegung des Erfolgsbeitrags · überprüfbare Belohnungen (RLVR) · GRPO und asymmetrisches Clipping · KL-Strafterm k3 · DPO**

<!--en:e7698634-->
::: {.slide title="Von der Imitation des nächsten Tokens zum Post-Training mit Prüfprogramm"}

<!--en:e02fdbdc-->
[Motivation]{.kicker}

<!--en:b0caa8d9-->
- **Überwachung anhand des Ergebnisses:** Das Modell erzeugt eine vollständige Antwort $y = (a_1, \dots, a_T)$ auf den Prompt $x$. Ein deterministisches **Prüfprogramm** liefert bei EOS einen einzigen skalaren Score $r(x, y)$.

<!--en:c5678402-->
- **Benchmark mit 4 Prompts und Prüfprogramm:** Die Referenzstrategie $\pi_{\textrm{ref}}$ besteht die exakte Prüfung bei $6.2\%$ der Prompts (Obergrenze $50\%$). Eine nachsichtige Teilzeichenkettensuche belohnt die Ausweichantwort bei jedem Prompt.

<!--en:d024bb90-->
::: {.slide title="Autoregressive Textgenerierung als MDP auf Tokenebene"}

<!--en:2c1e963b-->
[Der Token-MDP]{.kicker}

<!--en:80d7102f-->
- Jede Antwort $y = (a_1, \dots, a_T)$ ist eine Trajektorie in einem MDP mit deterministischen Übergängen. Der Zustand $s_t = (x, a_1, \dots, a_{t-1})$ ist das Präfix und die Aktion $a_t$ das nächste Token:
  $$\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(a_t \mid s_t)$$
  $$\nabla_\theta \log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

<!--en:30a955ef-->
- Bei $\gamma = 1$ und einer einzigen terminalen Belohnung $r(x, y)$ bei EOS sind der Banditengradient auf Sequenzebene und der Policy-Gradient des MDP auf Tokenebene **algebraisch identisch**.

<!--en:536671a6-->
::: {.slide title="Erfolgsbeiträge für eine Antwort und ihre beiden Ziffern berechnen"}

<!--en:b3a00e00-->
[Zuordnung des Erfolgs]{.kicker}

<!--en:bfd12af4-->
Zum Vergleich von Antwortgewicht und Erfolgsbeitrag einzelner Token dient `3*4` mit einer zweistelligen Antwort $y = 10 a_1 + a_2$. Nur `12` ($a_1 = 1, a_2 = 2$) erhält die Belohnung $r = 1$:

<!--en:1e39be71-->
- **Wertfunktionen für Präfixe:** Bei einer gleichverteilten Strategie ($p_1 = p_2 = 0.1$) hat der Prompt den Wert $V(x) = 1/100 = 0.01$. Nach der richtigen Zehnerziffer ist der Präfixwert $V(x, a_1{=}1) = 1/10 = 0.10$, andernfalls gilt $V(x, a_1{\neq}1) = 0$.

<!--en:06187769-->
::: {.slide title="Warum Antwortgewichte richtige Token im Präfix bestrafen"}

<!--en:3b8c0458-->
[Erfolgsbeiträge im Vergleich]{.kicker}

<!--en:fb598265-->
Vergleich des Antwortgewichts $r(x, y) - V(x)$ mit den vom Präfix abhängigen Tokenvorteilen $A(s_1, a_1) = V(x, a_1) - V(x)$ und $A(s_2, a_2) = r - V(x, a_1)$. Betrachtete Antworten: `12`, `13` und `52`:

<!--en:67ecb483-->
- Bei `13` (richtige Zehnerziffer `1`, falsche Einerziffer `3`) ergibt das Antwortgewicht **$-0.01$ für beide Ziffern**. Die Wahrscheinlichkeit der richtigen Zehnerziffer *sinkt*. Dagegen belohnt $A(s_1, 1) = +0.09$ die `1` und $A(s_2, 3) = -0.10$ bestraft die `3`.

<!--en:b131babf-->
::: {.slide title="Orthogonale Varianzzerlegung über die Token"}

<!--en:641c6609-->
[Warum der Erfolgsbeitrag einzelner Token zählt]{.kicker}

<!--en:d68b62c3-->
- **Unkorrelierte Zuwächse:** Wegen $\mathbb{E}[A(s_2, a_2) \mid s_1, a_1] = 0$ sind die Tokenvorteile orthogonal. Ihre Varianzen addieren sich: $0.0009 + 0.0090 = 0.0099$.

<!--en:1688c393-->
- **$11\times$ so hohe Varianz:** Mit dem Gewicht der gesamten Antwort geht das Rauschen späterer Token in den Gradienten des ersten Tokens ein. Die Gradientenvarianz ist dadurch $11\times$ so groß.

<!--en:cfe3d4a7-->
::: {.slide title="Was der Token-MDP vereinfacht und was bleibt"}

<!--en:98574918-->
[Vereinfachungen der Architektur]{.kicker}

<!--en:a0a2e01f-->
- **Was beim bestärkenden Lernen mit nur einer LLM-Antwort entfällt:** Die Verkettung von Token ist exakt, daher brauchen wir kein erlerntes Weltmodell. Ein Q-Netz mit Erfahrungswiedergabe entfällt, weil bei $|\mathcal{A}| \sim 10^5$ Token eine starke $\max_a$-Verzerrung entsteht.

<!--en:79154003-->
- **Was wesentlich bleibt:** Gruppen-Baselines (GRPO/RLOO) oder Kritiker für Präfixe senken die Varianz. Clipping des Verhältnisses und KL-Strafterme sichern Updates über mehrere Epochen ab und verhindern Reward-Hacking.

<!--en:5be0e1f2-->
::: {.slide title="Überprüfbare Belohnungen (RLVR) und Schlupflöcher in der Bewertung"}

<!--en:86f8f25e-->
[Prüfprogramme und Schlupflöcher]{.kicker}

<!--en:926baf0f-->
Beim direkten Training mit deterministischen Prüfprogrammen wie Unit-Tests, mathematischen Prüfprogrammen oder Beweisassistenten (**RLVR**, DeepSeek-R1, 2025) entfällt der Approximationsfehler eines erlernten Belohnungsmodells, *es sei denn, die Spezifikation der Prüfung selbst hat ein Schlupfloch*:

<!--en:1279c90f-->
- Bei `VerifierTask` löst keine gewöhnliche Antwort `8*8` oder `9*9` (exakte Obergrenze $0.500$). Dagegen sucht `sloppy_checker` nur nach einer Teilzeichenkette (`answer in resp`) und vergibt $+1$ für die Ausweichantwort *„eine der Zahlen 4, 12, 64 oder 81“* bei jedem Prompt.

<!--en:ac822f81-->
::: {.slide title="KL-Regularisierung in den Gruppenschritt einbeziehen"}

<!--en:6d3a38b1-->
[Gruppenupdate mit KL-Strafterm]{.kicker}

<!--en:01690d03-->
Damit die Strategie nicht ihre ganze Wahrscheinlichkeit auf das Schlupfloch legt, bezieht `kl_group_step` das logarithmische Verhältnis der umgekehrten KL-Divergenz für jedes Sample, $-\beta \log\big(\pi_\theta(y \mid x)/\pi_{\textrm{ref}}(y \mid x)\big)$, direkt in die Belohnung für `d2l.group_step` ein:

<!--en:0c36e786-->
- **Unverzerrter Gradient der umgekehrten KL-Divergenz:** Der Score hat den Erwartungswert null, $\mathbb{E}_{y \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(y \mid x)] = 0$. Die Gewichtung von $\nabla_\theta \log \pi_\theta$ mit $r(x, y) - \beta \log(\pi_\theta/\pi_{\textrm{ref}})$ ergibt daher einen unverzerrten Schätzer für $\nabla_\theta\big(\mathbb{E}[r] - \beta D_{\textrm{KL}}(\pi_\theta \Vert \pi_{\textrm{ref}})\big)$.

<!--en:0693e317-->
::: {.slide title="Schwellenwert für das Schlupfloch aus der geschlossenen Lösung"}

<!--en:6e1b999e-->
[Exakte Analyse des Schwellenwerts]{.kicker}

<!--en:48306428-->
- **Wahrscheinlichkeitsverhältnis unter der Referenz:** Unter $\pi_{\textrm{ref}}$ hat die Ausweichantwort das Logit $-4$, gewöhnliche Antworten haben das Logit $0$ ($e^{-4} \approx 0.018$).

<!--en:af320e7f-->
- **Kritischer Schwellenwert ($\beta = 1/4 = 0.25$):** Bei unlösbaren Prompts ist die Ausweichantwort für $\beta < 0.25$ am wahrscheinlichsten. Für **$\beta > 0.25$** wird sie unterdrückt.

<!--en:0c5a952c-->
::: {.slide title="Beta im Experiment und die exakten umgewichteten Optima"}

<!--en:4908ec83-->
[Theorie und Training im Vergleich]{.kicker}

<!--en:2dedb715-->
Beim Training mit `kl_group_step` ($K = 32$ Samples pro Prompt, 400 Updates) für $\beta \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$ stimmen die Ergebnisse bei jeder Temperatur mit dem geschlossen berechneten, exponentiell umgewichteten Optimum $\pi^\star$ bis auf $0.02$ überein:

<!--en:4c677726-->
**Zielkonflikt bei der Regularisierung:** Bei $\beta = 0$ erzielt die Strategie den vollen nachsichtigen Score ($1.00$). Sie legt $100\%$ ihrer Masse auf die Ausweichantwort für `8*8` und `9*9` (mittlere Wahrscheinlichkeit der Ausweichantwort $0.50$). Eine Erhöhung von $\beta$ auf $0.3 > 0.25$ senkt diese Wahrscheinlichkeit ($0.04$). Ein zu starker Strafterm bei $\beta = 0.5$ senkt auch den exakten Score wieder (von $0.50$ auf etwa $0.25$).

<!--en:1dc6c8c8-->
::: {.slide title="Veraltete Samples, Sequenzverhältnisse und asymmetrisches Clipping"}

<!--en:36e7595e-->
[Besonderheiten großer Systeme]{.kicker}

<!--en:5afa3bd5-->
- **Geometrisches Mittel über die Sequenz (GSPO, Zheng et al., 2025):** Bei Rollouts mit langen Gedankenketten tritt an die Stelle des stark anwachsenden Produkts $W = \prod_{t=1}^T \rho_t$ das geometrische Mittel $\bar{\rho} = W^{1/T} = \exp(\frac{1}{T}\sum_t \log \rho_t)$.

<!--en:5b1beba5-->
- **Asymmetrisches Clipping (`Clip-Higher` in DAPO, Yu et al., 2025):** Das breitere Intervall $[0.80, 1.28]$ lässt die Wahrscheinlichkeit seltener Token mit hohem Vorteil beim Schlussfolgern schneller wachsen, ohne dass die Entropie zusammenbricht.

<!--en:fd3c28d4-->
::: {.slide title="Group Relative Policy Optimization (GRPO)" layout="tight"}

<!--en:ac1abac1-->
[Das vollständige GRPO-Update]{.kicker}

<!--en:d21077ac-->
- **Gruppen-Baseline ohne Kritiker (Shao et al., 2024):** Das Verfahren standardisiert die Belohnungen über $K$ parallel erzeugte Antworten pro Prompt und optimiert die begrenzte Ersatzzielfunktion abzüglich $\beta k_3$.

<!--en:13bc412a-->
::: {.slide title="Group Relative Policy Optimization (GRPO) (Code)" layout="tight"}

<!--en:ac1abac1-2-->
[Das vollständige GRPO-Update]{.kicker}

<!--en:8f969de1-->
::: {.slide title="Dieselbe Gruppe über mehrere GRPO-Epochen verwenden"}

<!--en:d441b4a8-->
[GRPO-Diagnostik]{.kicker}

<!--en:20748ff3-->
Die Wiederverwendung derselben Gruppe von $K = 32$ Antworten pro Prompt über 4 Epochen (`grpo_step`, $\epsilon_{\textrm{clip}} = 0.2$) erhöht den exakten Score des Prüfprogramms von $0.062$ ($\pi_{\textrm{ref}}$) auf $0.111$ (1 Epoche) und schließlich auf **$0.137$** (4 Epochen):

<!--en:bfa1b6a8-->
- In Epoche 1 sind alle Verhältnisse gleich $\rho = 1.00$. Schon nach dem ersten Schritt übersteigen die Verhältnisse richtiger Antworten den Wert $1.50$ ($> 1 + \epsilon_{\textrm{clip}}$). Das Clipping bei PPO und GRPO unterbindet also weitere Gradientenbeiträge, ohne den Parameterschritt selbst zu begrenzen.

<!--en:cd032209-->
::: {.slide title="Der KL-Schätzer k3 und sein Gradient in Vorwärtsrichtung"}

<!--en:ee041448-->
[Die Richtung des KL-Verlusts]{.kicker}

<!--en:fd565608-->
- **Direkter Verlust oder Strafterm in der Belohnung (Tang und Munos, 2025):** Mit $h = \log(\pi_\theta/\pi_{\textrm{ref}})$ ergibt die direkte Ableitung von $k_1 = h$ im Mittel null. Dagegen liefert $k_2 = \frac{1}{2}h^2$ (oder die Einbeziehung von $h$ in die Belohnung) den **Gradienten der umgekehrten KL-Divergenz**. Die Wahrscheinlichkeit der Ausweichantwort sinkt auf $0.017$.

<!--en:d6115dcc-->
- **Vorwärtsdivergenz als Fallstrick bei $k_3 = e^{-h} + h - 1$:** Direktes Ableiten von $k_3$ liefert den **Gradienten der KL-Divergenz in Vorwärtsrichtung** $\nabla_\theta D_{\textrm{KL}}(\pi_{\textrm{ref}} \Vert \pi_\theta)$. Die seltene Ausweichantwort erreicht so ungehindert die Wahrscheinlichkeit $0.500$.

<!--en:ba8778c7-->
::: {.slide title="Direct Preference Optimization kehrt die Umgewichtung um"}

<!--en:0322729a-->
[Offline-Lernen aus Präferenzen]{.kicker}

<!--en:5e2c69d0-->
- **Die Zustandssumme fällt weg (Rafailov et al., 2023):** Beim Einsetzen von $r(x, y)$ in die Bradley-Terry-Differenz $r(x, y_w) - r(x, y_l)$ hebt sich $\beta \log Z(x)$ vollständig auf.

<!--en:1f901682-->
- **Binäre Kreuzentropie im Offline-Training:** Das Verfahren trainiert $\pi_\theta$ direkt mit aufgezeichneten Präferenzpaaren $(y_w \succ y_l)$, ohne separates Belohnungsmodell und ohne Online-Rollouts.

<!--en:616985f8-->
::: {.slide title="Verständnisfragen zu Tokenvorteilen, Clipping und Werkzeugen"}

<!--en:929b7671-->
[Verständniskontrolle]{.kicker}

<!--en:c9c37df1-->
Drei praktische Beispiele zum Post-Training großer Sprachmodelle:

<!--en:7c467c63-->
1. **Tokenvorteile bei falschem Präfix:** Bei der Aufgabe mit zweistelliger Antwort `3*4 = 12` gilt unter der gleichverteilten Strategie $V(x) = 0.01, V(x, 1) = 0.1, V(x, 5) = 0$. Welche Tokenvorteile $A(s_1, 5)$ und $A(s_2, 2)$ ergeben sich für die falsche Antwort `52`? Wie groß ist dagegen das Antwortgewicht $r - V(x)$?

<!--en:d7bb1cee-->
2. **Kopplung durch die Softmax-Funktion nach dem Clipping:** Bei GRPO mit $\epsilon_{\textrm{clip}} = 0.2$ hat eine richtige Antwort den Vorteil $\hat{A} = +2.6$ und das Verhältnis $\rho = 1.9 > 1.2$. Für eine falsche Antwort gelten $\hat{A} = -0.4$ und $\rho = 0.9 \in [0.8, 1.2]$. Kann die Wahrscheinlichkeit der richtigen Antwort in der nächsten Epoche noch steigen?

<!--en:557b98b6-->
**Lösungen:**

<!--en:d62df1c8-->
1. Das Antwortgewicht ergibt **$-0.01$ für beide Ziffern** und bestraft damit zu Unrecht die `2` an der Einerstelle. Die Tokenvorteile ergeben **$A(s_1, 5) = -0.01$** und **$A(s_2, 2) = 0 - 0 = 0.00$**. Sie berücksichtigen, dass mit der Wahl von `5` der Misserfolg bereits feststand.

<!--en:1bc91bb3-->
2. **Ja.** Der begrenzte Term der richtigen Antwort hat zwar den Gradienten null, doch der unbegrenzte Term der falschen Antwort *senkt* deren eigenes Logit. Über den gemeinsamen Nenner der Softmax-Funktion steigt dadurch die Wahrscheinlichkeit der richtigen Antwort.

<!--en:a11c69e0-->
::: {.slide title="Verständnisfragen zu Tokenvorteilen, Clipping und Werkzeugen (Fortsetzung)"}

<!--en:929b7671-2-->
[Verständniskontrolle]{.kicker}

<!--en:c4b54238-->
3. **Ausgaben externer Werkzeuge maskieren:** Ein großes Sprachmodell agiert als Agent. Es erzeugt eine Anfrage $y^{(1)}$, empfängt die Ausgabe $o \sim P_{\textrm{tool}}(\cdot \mid x, y^{(1)})$ eines externen Python-Interpreters und erzeugt eine abschließende Antwort $y^{(2)}$. Warum müssen die Token von $o$ in $\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$ ausmaskiert werden?

<!--en:23db34b8-->
**Lösung:**

<!--en:1b6b6468-->
3. Die Übergangswahrscheinlichkeit $P_{\textrm{tool}}(o \mid x, y^{(1)})$ des Werkzeugs hängt nicht von $\theta$ ab und kürzt sich im Verhältnis der Trajektorienwahrscheinlichkeiten heraus. Bei der Ableitung von $\log \pi_\theta(o \mid x, y^{(1)})$ gilt $\mathbb{E}[\nabla \log \pi] = 0$ für diesen Term nicht mehr. Das Modell lernt so, Werkzeugausgaben zu erfinden.

<!--en:bcc47f13-->
::: {.slide title="Verfahren für das Post-Training von Sprachmodellen im Überblick"}

<!--en:c5a222f8-->
[Das Wichtigste in Kürze]{.kicker}

<!--en:34d4f4bd-->
| Verfahren / Variante | Baseline / Kritiker | Verhältnis und Clipping | KL-Regularisierung | Haupteinsatzgebiet und Abwägung |

<!--en:308136cc-->
| **RLHF mit PPO** | Wertkopf pro Token $\hat{V}_w(s_t)$ + GAE | $\rho_t$ pro Token, begrenzt bei $1 \pm 0.2$ | $-\beta \log(\pi_\theta/\pi_{\textrm{ref}})$ pro Token in der Belohnung | Ordnet den Erfolg einzelnen Token zu, erfordert aber das Training eines separaten großen Sprachmodells als Kritiker |

<!--en:a842cd9b-->
| **GRPO (Shao et al.)** | Standardisierung innerhalb der Gruppe $(r_j - \bar{r})/\sigma_r$ | $\rho_{j,t}$ pro Token, begrenzt bei $1 \pm 0.2$ | $\beta k_3$ direkt als Verlust pro Token (Gradient der KL-Divergenz in Vorwärtsrichtung) | Spart $\sim 50\%$ Speicher durch Verzicht auf den Kritiker. Standard für überprüfbare Lösungswege in Mathematik und Programmierung |

<!--en:e55862f8-->
| **DAPO / Dr. GRPO** | Zentrierung am Gruppenmittelwert $r_j - \bar{r}$ (ohne Division durch $\sigma_r, T_j$) | Asymmetrisch $[0.80, 1.28]$ (`Clip-Higher`) | Bei exakten Prüfprogrammen oft $\beta = 0$ | Verhindert den Zusammenbruch der Entropie bei seltenen Token und beseitigt die längenbedingte Verzerrung |

<!--en:61c82ba7-->
| **GSPO (Zheng et al.)** | Gruppen-Baseline $\hat{A}_j$ | Geometrisches Mittel über die Sequenz $\bar{\rho} = W^{1/T}$ | KL-Strafterm für die ganze Sequenz | Stabilisiert Wichtigkeitsgewichte bei Rollouts mit sehr langen Gedankenketten |

<!--en:827418f5-->
| **DPO (Rafailov et al.)** | Implizit durch den Vergleich von $y_w$ mit $y_l$ | Keines (überwachter Offline-Verlust) | Exakte Parametrisierung über die umgekehrte KL-Divergenz $\beta \log(\pi_\theta/\pi_{\textrm{ref}})$ | Schnelle Offline-Anpassung an Präferenzen ohne Online-Rollouts oder Belohnungsmodell |
