<!--en:8fb4d2dc-->
# Werkzeuge für Deep Learning

<!--en:8b1ec08f-->
Ein Notebook, das auf einem Laptop läuft, kann schon scheitern, wenn das Modell, der Batch oder der Kontext größer wird. Dann lohnt es sich nicht zwangsläufig, den Beschleuniger mit dem höchsten beworbenen Durchsatz zu kaufen. Prüfen Sie zuerst, ob der Speicher für die Arbeitslast ausreicht und ob die Rechenleistung oder der Datentransport die Ausführung begrenzt. Ermitteln Sie auch, was ein Experiment bis zum Abschluss kostet. Training und Inferenzbetrieb stellen unterschiedliche Anforderungen. Beim verteilten Training verursacht die Kommunikation zwischen den Beschleunigern Kosten. Im Inferenzbetrieb muss das System zudem die Vorgaben für Latenz und Durchsatz einhalten und den Zustand von Modell und Cache dauerhaft im Speicher halten.

<!--en:4bf478ec-->
In diesem Teil zeigen wir, wie Sie diese Entscheidungen treffen, vom einzelnen Notebook bis zum bereitgestellten System. In der folgenden Tabelle finden Sie den Abschnitt zu Ihrer Fragestellung.

<!--en:4b99a004-->
| Aufgabe | Abschnitt | Leitfrage |

<!--en:73f42508-->
| Die Notebooks des Buches lokal ausführen und untersuchen | :numref:`sec_interactive_development` | Wie organisieren wir Code, Zustand und Ausgaben, damit sich jeder Arbeitsschritt reproduzieren lässt? |

<!--en:2303e278-->
| Ohne lokale Installation beginnen | :numref:`sec_hosted_notebooks` | Welche gehostete Umgebung bietet das benötigte Framework und den passenden Beschleuniger? |

<!--en:2f8f4438-->
| Rechenkapazität mieten | :numref:`sec_cloud_instances` | Was kostet ein abgeschlossener Lauf insgesamt, einschließlich Einrichtung und Datenübertragung? |

<!--en:39e3be02-->
| Hardware auswählen | :numref:`sec_hardware_buyers` | Reicht der Speicher für die Arbeitslast aus? Wo liegt der Engpass, bei der Rechenleistung, der Speicherbandbreite oder der Verbindung zwischen den Komponenten? |

<!--en:0f6cb96f-->
| Modelle, Datensätze und Belege finden | :numref:`sec_software_ecosystem` | Welche Artefakte, Implementierungen oder Benchmarks sind reproduzierbar und vertrauenswürdig? |

<!--en:25872e60-->
| Training skalieren | :numref:`sec_training_systems` | Wie verteilen wir Parameter, Aktivierungen, Daten und Kommunikation? |

<!--en:3b9484b4-->
| Ein Modell für Anfragen bereitstellen | :numref:`sec_model_serving` | Wie hängen Batching, Latenz, Durchsatz und Cache-Speicher zusammen? |

<!--en:68228702-->
| Am Buch mitarbeiten | :numref:`sec_developers_guide` | Wie testen wir Änderungen und halten sie über alle vier Frameworks hinweg konsistent? |

<!--en:c6cbd04f-->
| Hilfsklassen und Hilfsfunktionen des Buches nachschlagen | :numref:`sec_utils` und :numref:`sec_d2l` | Wo ist eine Klasse oder Funktion definiert und dokumentiert? |

<!--en:d3bb345c-->
Wenn Sie ein System aufbauen, können Sie die Tabelle von oben nach unten durchgehen. Bei einer konkreten Frage zum Betrieb können Sie direkt den passenden Abschnitt aufschlagen. Hardwarebezeichnungen, Preise, Cloud-Schnittstellen und Bibliotheksversionen ändern sich rasch. Deshalb geben die verlinkten Abschnitte an, von wann solche Angaben stammen, und legen den Schwerpunkt auf Messungen, die auch bei Produktänderungen aussagekräftig bleiben. Die Seiten zu den Hilfsklassen und Hilfsfunktionen sowie zur `d2l`-API dienen als durchsuchbare Nachschlagewerke und stehen außerhalb der regulären Kapitelfolge.
