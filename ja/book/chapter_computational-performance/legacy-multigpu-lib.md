<!--en:19a28d32-->
# レガシーマルチ GPU ライブラリ（ビルド専用、非掲載）

<!--en:c483bfe9-->
**`split_batch`（tensorflow、mxnet のみ。PyTorch と JAX にはない）。**

<!--en:cd45a68c-->
**`resnet18`（tensorflow、mxnet のみ。PyTorch・JAX 版は `chapter_computational-performance/multi-gpu-practice.md` の `#@save` である）。**

<!--en:afb13205-->
**`evaluate_accuracy_gpus`（mxnet のみ。第 19 章の mxnet ノートブック `image-augmentation.md` と `kaggle-cifar10.md` が `d2l.evaluate_accuracy_gpus` 経由で呼び出すマルチ GPU 正解率ヘルパであり、PyTorch・TF・JAX ではインラインで評価する）。**
