<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=tr rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## Alıştırmalar

<!--en:6f4ddec6 kind=exercise-->
1. [code] **Hand-designed kernels.** Design some kernels manually.
    1. Given a directional vector $\mathbf{v} = (v_1, v_2)$, derive an
       edge-detection kernel that detects edges orthogonal to $\mathbf{v}$,
       i.e., edges in the direction $(v_2, -v_1)$.
    1. Derive a finite difference operator for the second derivative.
       Which structures in images respond most strongly to it?
    1. What is the minimum size of a kernel to obtain a derivative of
       order $d$?
    1. How would you design a blur kernel? Why might you want to use such
       a kernel?
    1. Load a photograph, convert it to one channel, and downsample it to
       about $100 \times 100$ pixels (`corr2d` loops in Python). Apply the
       edge detector, the blur kernel, and the sharpening kernel "twice
       the identity minus the blur", and display each result next to the
       original. What does the edge map of a photograph show that the
       edge map of the black band could not?

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6f4ddec6-->
1. [code] **Elle tasarlanan çekirdekler.** Birkaç çekirdeği elle tasarlayın.
    1. Bir yön vektörü $\mathbf{v} = (v_1, v_2)$ verildiğinde, $\mathbf{v}$ vektörüne dik kenarları, yani $(v_2, -v_1)$ yönündeki kenarları algılayan bir kenar algılama çekirdeği türetin.
    1. İkinci türev için bir sonlu fark operatörü türetin. Görüntülerdeki hangi yapılar buna en güçlü tepkiyi verir?
    1. Mertebesi $d$ olan bir türev elde etmek için bir çekirdeğin en küçük boyutu nedir?
    1. Bir bulanıklaştırma çekirdeğini nasıl tasarlardınız? Böyle bir çekirdeği neden kullanmak isteyebilirsiniz?
    1. Bir fotoğraf yükleyin, tek kanala dönüştürün ve yaklaşık $100 \times 100$ piksele küçültün (`corr2d` Python'da döngüyle çalışır). Kenar dedektörünü, bulanıklaştırma çekirdeğini ve "birim çekirdeğin iki katı eksi bulanıklaştırma" biçimindeki keskinleştirme çekirdeğini uygulayın ve her sonucu özgün görüntünün yanında gösterin. Bir fotoğrafın kenar haritası, siyah bandın kenar haritasının gösteremediği neyi gösterir?

    *Stanford CS231n, [Ödev 2](https://cs231n.github.io/assignments2024/assignment2/), ConvolutionalNetworks not defterinden uyarlanmıştır.*

<!--en:ec8621f8 kind=exercise-->
2. [code] **Convolution as a banded matrix.** The patch matrix of
   :numref:`subsec_conv_matmul` unfolds the input. The opposite
   construction unrolls the kernel. Let $\mathbf{X}$ be a $4 \times 4$
   input, $\mathbf{K}$ a $3 \times 3$ kernel, and $\textrm{vec}(\cdot)$
   the row-by-row flattening.
    1. Construct the $4 \times 16$ matrix $\mathbf{W}$ with
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       where $\mathbf{Y}$ is the cross-correlation of $\mathbf{X}$ with
       $\mathbf{K}$. How many nonzero entries does each row have, and how
       many distinct values appear in $\mathbf{W}$?
    1. Build $\mathbf{W}$ in code from `K` and confirm that the product
       reproduces `corr2d(X, K)`.
    1. For an $n \times n$ input and a $k \times k$ kernel, count the
       entries of $\mathbf{W}$ stored as a dense array, the nonzero
       entries of $\mathbf{W}$, and the entries of the patch matrix. How
       does each count grow with $n$?
    1. Keep the sparsity pattern of $\mathbf{W}$ but give every row its
       own $k^2$ values. The result is a locally connected layer: the
       fourth-order tensor $\mathsf{V}$ of :numref:`sec_why-conv` in
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restricted to a $k \times k$ window of offsets $(a, b)$ before
       translation equivariance ties the weights. For an $n \times n$
       input, count the parameters of this layer, of a fully connected
       layer with the same input and output shapes, and of the
       convolution.

    *Adapted from CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*
<!--tr:ec8621f8-->
2. [code] **Bant matris olarak evrişim.** :numref:`subsec_conv_matmul` içindeki yama matrisi girdiyi açar. Karşıt kurgu ise çekirdeği açar. $\mathbf{X}$, $4 \times 4$ boyutlu bir girdi; $\mathbf{K}$, $3 \times 3$ boyutlu bir çekirdek; $\textrm{vec}(\cdot)$ ise satır satır düzleştirme olsun.
    1. $4 \times 16$ boyutlu $\mathbf{W}$ matrisini $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ olacak biçimde kurun; burada $\mathbf{Y}$, $\mathbf{X}$ ile $\mathbf{K}$ arasındaki çapraz korelasyondur. Her satırda sıfırdan farklı kaç eleman vardır ve $\mathbf{W}$ içinde kaç farklı değer görünür?
    1. $\mathbf{W}$ matrisini kodda `K` üzerinden kurun ve çarpımın `corr2d(X, K)` sonucunu yeniden ürettiğini doğrulayın.
    1. $n \times n$ boyutlu bir girdi ve $k \times k$ boyutlu bir çekirdek için, yoğun bir dizi olarak saklanan $\mathbf{W}$ matrisinin elemanlarını, $\mathbf{W}$ matrisinin sıfırdan farklı elemanlarını ve yama matrisinin elemanlarını sayın. Her sayı $n$ ile nasıl büyür?
    1. $\mathbf{W}$ matrisinin seyreklik örüntüsünü koruyun, ama her satıra kendi $k^2$ değerini verin. Sonuç yerel bağlı bir katmandır: dördüncü mertebeden $\mathsf{V}$ tensörünün (:numref:`sec_why-conv`, $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ ifadesinde), ötelemeye eşdeğişim ağırlıkları birbirine bağlamadan önce, $k \times k$ boyutlu bir $(a, b)$ kayma penceresiyle sınırlandırılmış halidir. $n \times n$ boyutlu bir girdi için bu katmanın, aynı girdi ve çıktı şekillerine sahip tam bağlı bir katmanın ve evrişimin parametrelerini sayın.

    *CMU 11-785, [Ödev 2, Kısım 1](https://deeplearning.cs.cmu.edu/S21/index.html) kaynağından uyarlanmıştır.*

<!--en:54bcee26 kind=exercise-->
3. [code] **Effective receptive field.** ● The theoretical receptive
   field $r$ of this section bounds which inputs *can* influence an
   output; the effective receptive field measures how much
   they actually do :cite:`Luo.Li.Urtasun.ea.2016`. Build a stack of $L$
   single-channel $3 \times 3$ convolutional layers with no
   padding, no bias, and no nonlinearity on a $32 \times 32$ input, using
   the built-in layer of :numref:`subsec_learning_kernel`. Draw every
   kernel entry independently from a standard normal distribution.
    1. For $L = 3$ and a single draw of the kernels, backpropagate a unit
       gradient from an output unit near the center of the last layer to
       the input and plot the magnitude of the input gradient. Compare its
       support with the $7 \times 7$ theoretical receptive field, and say
       where the magnitude is largest.
    1. Average the squared input gradient over 500 draws of the kernels,
       for $L = 3$ and for $L = 10$. Sum each averaged map over its rows
       and normalize the resulting column profile to sum to one. How does
       the ratio of the profile's standard deviation to the theoretical
       side length $2L + 1$ change with depth?
    1. :numref:`fig_conv_rf_paths` counts the paths for $L = 3$ in one
       dimension. Show that for general $L$ the expected squared input
       gradient, summed over rows, is proportional to the convolution of
       $L$ copies of the vector $(1, 1, 1)$. Derive how the standard
       deviation of the normalized profile grows with $L$.
<!--tr:54bcee26-->
3. [code] **Etkin alıcı alan.** ● Bu bölümdeki kuramsal alıcı alan $r$, hangi girdilerin bir çıktıyı *etkileyebileceğini* sınırlar; etkin alıcı alan ise bunların gerçekte ne kadar etkilediğini ölçer :cite:`Luo.Li.Urtasun.ea.2016`. Tek kanallı $L$ adet $3 \times 3$ evrişim katmanını dolgusuz, ek girdisiz ve doğrusal olmayan dönüşümsüz olarak $32 \times 32$ boyutlu bir girdi üzerinde üst üste dizin; bunun için :numref:`subsec_learning_kernel` içindeki yerleşik katmanı kullanın. Her çekirdek elemanını standart normal dağılımdan bağımsız olarak çekin.
    1. $L = 3$ ve çekirdeklerin tek bir çekilişi için, son katmanın merkezine yakın bir çıktı biriminden girdiye doğru birim gradyanı geri yayın ve girdi gradyanının büyüklüğünü çizin. Desteğini $7 \times 7$ boyutlu kuramsal alıcı alanla karşılaştırın ve büyüklüğün nerede en yüksek olduğunu belirtin.
    1. Girdi gradyanının karesinin, çekirdeklerin 500 çekilişi üzerinden ortalamasını $L = 3$ ve $L = 10$ için alın. Ortalaması alınmış her haritayı satırları boyunca toplayın ve ortaya çıkan sütun profilini toplamı bir olacak biçimde normalleştirin. Profilin standart sapmasının kuramsal kenar uzunluğu $2L + 1$ değerine oranı derinlikle nasıl değişir?
    1. :numref:`fig_conv_rf_paths` yolları $L = 3$ için tek boyutta sayar. Genel $L$ için, satırlar üzerinden toplanan beklenen girdi gradyanı karesinin, $L$ adet $(1, 1, 1)$ vektörünün evrişimiyle orantılı olduğunu gösterin. Normalleştirilmiş profilin standart sapmasının $L$ ile nasıl büyüdüğünü türetin.

<!--en:6ab5b67c kind=exercise-->
4. [code] **Backward pass of a convolution.** ● Let $\mathbf{Y}$ be the
   single-channel cross-correlation of an input $\mathbf{X}$ with a kernel
   $\mathbf{K}$, as computed by `corr2d`, and let $\ell$ be a scalar loss
   that depends on $\mathbf{X}$ and $\mathbf{K}$ only through $\mathbf{Y}$.
    1. Derive $\partial \ell/\partial \mathbf{K}$ and
       $\partial \ell/\partial \mathbf{X}$ in terms of
       $\partial \ell/\partial \mathbf{Y}$, in the style of the
       backpropagation equations of :numref:`sec_backprop`.
    1. Implement your expression for $\partial \ell/\partial \mathbf{K}$
       without automatic differentiation for the loss
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, where $\mathbf{G}$ is a fixed
       random matrix of the shape of $\mathbf{Y}$. Compare it with a
       centered finite-difference gradient with step size $10^{-2}$ on a
       small random `X` and `K`, and report the largest relative error.
    1. Check your expression for $\partial \ell/\partial \mathbf{K}$
       against the kernel gradient that automatic differentiation
       computes in the experiment of :numref:`subsec_learning_kernel`.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chapter 6, and Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6ab5b67c-->
4. [code] **Bir evrişimin geri geçişi.** ● $\mathbf{Y}$, bir $\mathbf{X}$ girdisinin bir $\mathbf{K}$ çekirdeğiyle tek kanallı çapraz korelasyonu (`corr2d` ile hesaplandığı biçimiyle), $\ell$ ise $\mathbf{X}$ ile $\mathbf{K}$ değişkenlerine yalnızca $\mathbf{Y}$ üzerinden bağlı olan skaler bir kayıp olsun.
    1. $\partial \ell/\partial \mathbf{K}$ ile $\partial \ell/\partial \mathbf{X}$ türevlerini $\partial \ell/\partial \mathbf{Y}$ cinsinden, :numref:`sec_backprop` içindeki geri yayılım denklemlerinin tarzında türetin.
    1. $\partial \ell/\partial \mathbf{K}$ için bulduğunuz ifadeyi, $\ell = \sum_{i,j} G_{ij} Y_{ij}$ kaybı için otomatik türev alma kullanmadan kodlayın; burada $\mathbf{G}$, $\mathbf{Y}$ ile aynı şekle sahip sabit bir rastgele matristir. Sonucu, adım büyüklüğü $10^{-2}$ olan merkezi sonlu fark gradyanıyla küçük, rastgele bir `X` ve `K` üzerinde karşılaştırın ve en büyük bağıl hatayı bildirin.
    1. $\partial \ell/\partial \mathbf{K}$ için bulduğunuz ifadeyi, otomatik türev almanın :numref:`subsec_learning_kernel` içindeki deneyde hesapladığı çekirdek gradyanıyla karşılaştırarak sınayın.

    *Michael Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html), 6. Bölüm ile Stanford CS231n, [Ödev 2](https://cs231n.github.io/assignments2024/assignment2/), ConvolutionalNetworks not defterinden uyarlanmıştır.*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[Tartışmalar](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[Tartışmalar](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[Tartışmalar](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[Tartışmalar](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- Exercises: the `[code]` tag and the "●" marker are copied unchanged; the bold exercise title ends with a full stop as in the English; sub-items keep the "1." numbering and the four-space indent.
- Attribution lines: "Adapted from X" → "X … uyarlanmıştır"; course and book names stay in English (Stanford CS231n, CMU 11-785, *Neural Networks and Deep Learning*); "Assignment" / "Homework" → "Ödev"; "Part" → "Kısım"; "Chapter 6" → "6. Bölüm"; "notebook" → "not defteri".
- "Discussions" (forum link) → "Tartışmalar".
- "blur kernel" → "bulanıklaştırma çekirdeği"; "sharpening kernel" → "keskinleştirme çekirdeği"; "the identity" (kernel) → "birim çekirdek"; "downsample" → "küçültmek".
- "derivative of order d" → "mertebesi $d$ olan türev"; "fourth-order tensor" → "dördüncü mertebeden tensör"; "operator" → "operatör".
- "locally connected layer" → "yerel bağlı katman"; "ties the weights" → "ağırlıkları birbirine bağlamak"; "dense array" → "yoğun dizi".
- "standard normal distribution" → "standart normal dağılım" (fixed name; the glossary's "Gauss dağılımı" is kept for "Gaussian distribution"); "a draw" → "çekiliş"; "support" (of a function) → "destek"; "magnitude" → "büyüklük"; "relative error" → "bağıl hata"; "centered finite difference" → "merkezi sonlu fark"; "normalize" → "normalleştirmek"; "expected" → "beklenen"; "proportional" → "orantılı".

=== NOTES ===
- 54bcee26: "no nonlinearity" → "doğrusal olmayan dönüşümsüz" (see the decision from unit 2); "standard normal distribution" → "standart normal dağılım" rather than the glossary's "Gauss dağılımı".
- 54bcee26: "Average …" is the verb "ortalamasını almak" (glossary "ortalama"); "the convolution of $L$ copies of the vector" is ordinary discrete convolution, "evrişim".
- 6f4ddec6, ec8621f8, 54bcee26, 6ab5b67c: each list item is written on one line; the indentation, the blank line and the italic attribution line are kept as in the English.
- ec8621f8: "how many distinct values" is "kaç farklı değer" (ordinary "value", not the attention term).
