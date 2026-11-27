<!--en:19a28d32-->
# Ältere Bibliothek für mehrere GPUs (nur für den Build, nicht im Inhaltsverzeichnis)

<!--en:c483bfe9-->
**`split_batch` gibt es nur für TensorFlow und MXNet. PyTorch und JAX haben keine entsprechende Funktion.**

<!--en:cd45a68c-->
**`resnet18` (nur für TensorFlow und MXNet). In `chapter_computational-performance/multi-gpu-practice.md` ist die Version für PyTorch und JAX mit `#@save` gekennzeichnet.**

<!--en:afb13205-->
**`evaluate_accuracy_gpus` (nur für MXNet) berechnet die Genauigkeit über mehrere GPUs hinweg. Die MXNet-Notebooks `image-augmentation.md` und `kaggle-cifar10.md` in Kapitel 19 rufen die Hilfsfunktion über `d2l.evaluate_accuracy_gpus` auf. Die Varianten für PyTorch, TensorFlow und JAX berechnen die Genauigkeit direkt im Notebook.**
