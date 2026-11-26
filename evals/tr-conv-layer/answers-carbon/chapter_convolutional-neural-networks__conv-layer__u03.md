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
1. [code] **Elle tasarlanmış çekirdekler.** Bazı çekirdekleri elle tasarlayın.
    1. Bir $\mathbf{v} = (v_1, v_2)$ yön vektörü verildiğinde, $\mathbf{v}$ vektörüne dik olan, yani $(v_2, -v_1)$ yönündeki kenarları tespit eden bir kenar tespiti çekirdeği türetin.
    1. İkinci türev için bir sonlu fark operatörü türetin. Görüntülerdeki hangi yapılar buna en güçlü tepkiyi verir?
    1. Mertebesi $d$ olan bir türev elde etmek için bir çekirdeğin en küçük boyutu nedir?
    1. Bir bulanıklaştırma çekirdeğini nasıl tasarlardınız? Böyle bir çekirdeği neden kullanmak isteyebilirsiniz?
    1. Bir fotoğraf yükleyin, tek kanala dönüştürün ve yaklaşık $100 \times 100$ piksele alt örnekleyerek küçültün (`corr2d` Python'da döngü kurar). Kenar tespit edicisini, bulanıklaştırma çekirdeğini ve "birim işlecin iki katı eksi bulanıklaştırma" keskinleştirme çekirdeğini uygulayın ve her sonucu özgün görüntünün yanında görüntüleyin. Bir fotoğrafın kenar haritası, siyah bandın kenar haritasının gösteremediği neyi gösterir?

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
2. [code] **Bantlı matris olarak evrişim.** :numref:`subsec_conv_matmul` içindeki yama matrisi girdiyi açar. Bunun karşıtı olan kurulum ise çekirdeği açar. $\mathbf{X}$ matrisi $4 \times 4$ boyutlu bir girdi, $\mathbf{K}$ matrisi $3 \times 3$ boyutlu bir çekirdek ve $\textrm{vec}(\cdot)$ satır satır düzleştirme işlemi olsun.
    1. $4 \times 16$ boyutlu $\mathbf{W}$ matrisini, $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ eşitliğini sağlayacak biçimde oluşturun; burada $\mathbf{Y}$, $\mathbf{X}$ ile $\mathbf{K}$ matrislerinin çapraz korelasyonudur. Her satırda sıfırdan farklı kaç eleman vardır ve $\mathbf{W}$ içinde kaç farklı değer görünür?
    1. Kodda $\mathbf{W}$ matrisini `K` üzerinden oluşturun ve çarpımın `corr2d(X, K)` sonucunu birebir ürettiğini doğrulayın.
    1. Boyutu $n \times n$ olan bir girdi ve $k \times k$ boyutlu bir çekirdek için yoğun bir dizi olarak saklanan $\mathbf{W}$ matrisinin eleman sayısını, $\mathbf{W}$ matrisinin sıfırdan farklı eleman sayısını ve yama matrisinin eleman sayısını hesaplayın. Her bir sayı $n$ ile nasıl büyür?
    1. $\mathbf{W}$ matrisinin seyreklik örüntüsünü koruyun, ancak her satıra kendi $k^2$ değerini verin. Sonuç yerel bağlı bir katmandır: bu, dördüncü mertebeden $\mathsf{V}$ tensörünün (:numref:`sec_why-conv` kapsamındaki $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ bağıntısında), ötelemeye eşdeğişim ağırlıkları birbirine bağlamadan önce $k \times k$ boyutlu bir $(a, b)$ kayma penceresiyle kısıtlanmış halidir. Boyutu $n \times n$ olan bir girdi için bu katmanın, aynı girdi ve çıktı şekillerine sahip tam bağlı bir katmanın ve evrişimin parametre sayılarını hesaplayın.

    *CMU 11-785, [Ödev 2 Bölüm 1](https://deeplearning.cs.cmu.edu/S21/index.html) kaynağından uyarlanmıştır.*

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
3. [code] **Etkin alıcı alan.** ● Bu bölümdeki teorik alıcı alan $r$, hangi girdilerin bir çıktıyı *etkileyebileceğini* sınırlar; etkin alıcı alan ise gerçekte ne kadar etkilediklerini ölçer :cite:`Luo.Li.Urtasun.ea.2016`. $L$ adet tek kanallı, dolgu, ek girdi ve doğrusal olmama içermeyen $3 \times 3$ evrişim katmanından oluşan bir yığını $32 \times 32$ boyutlu bir girdi üzerinde :numref:`subsec_learning_kernel` kapsamındaki yerleşik katmanı kullanarak kurun. Her çekirdek elemanını birbirinden bağımsız olarak standart normal dağılımdan (Gauss dağılımından) çekin.
    1. $L = 3$ ve çekirdeklerin tek bir örneklemi için, son katmanın merkezine yakın bir çıktı biriminden girdiye birim gradyanı geri yayılımla iletin ve girdi gradyanının büyüklüğünü çizdirin. Desteğini $7 \times 7$ teorik alıcı alanla karşılaştırın ve büyüklüğün nerede en yüksek olduğunu belirtin.
    1. Karesi alınmış girdi gradyanının 500 çekirdek örneklemi üzerinden ortalamasını $L = 3$ ve $L = 10$ için hesaplayın. Ortalaması alınmış her haritayı satırları üzerinden toplayın ve elde edilen sütun profilini toplamı bir olacak şekilde normalize edin. Profilin standart sapmasının teorik kenar uzunluğu $2L + 1$ değerine oranı derinlikle nasıl değişir?
    1. :numref:`fig_conv_rf_paths`, tek boyutta $L = 3$ için yolları saymaktadır. Genel $L$ için satırlar üzerinden toplanan beklenen karesel girdi gradyanının, $L$ adet $(1, 1, 1)$ vektörünün evrişimiyle orantılı olduğunu gösterin. Normalize edilmiş profilin standart sapmasının $L$ ile nasıl büyüdüğünü türetin.

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
4. [code] **Bir evrişimin geri geçişi.** ● $\mathbf{Y}$, bir $\mathbf{X}$ girdisi ile bir $\mathbf{K}$ çekirdeğinin `corr2d` tarafından hesaplanan tek kanallı çapraz korelasyonu; $\ell$ ise $\mathbf{X}$ ve $\mathbf{K}$ matrislerine yalnızca $\mathbf{Y}$ üzerinden bağlı olan skaler bir kayıp olsun.
    1. $\partial \ell/\partial \mathbf{K}$ ve $\partial \ell/\partial \mathbf{X}$ ifadelerini, $\partial \ell/\partial \mathbf{Y}$ cinsinden, :numref:`sec_backprop` kapsamındaki geri yayılım denklemleri tarzında türetin.
    1. $\partial \ell/\partial \mathbf{K}$ ifadenizi, otomatik türev alma kullanmadan $\ell = \sum_{i,j} G_{ij} Y_{ij}$ kaybı için gerçekleyin; burada $\mathbf{G}$, $\mathbf{Y}$ ile aynı şekilde sabit bir rastgele matristir. Bunu adım büyüklüğü $10^{-2}$ olan merkezi sonlu fark gradyanıyla küçük rastgele `X` ve `K` üzerinde karşılaştırın ve en büyük bağıl hatayı raporlayın.
    1. $\partial \ell/\partial \mathbf{K}$ ifadenizi, :numref:`subsec_learning_kernel` deneyinde otomatik türev almanın hesapladığı çekirdek gradyanıyla karşılaştırarak kontrol edin.

    *Michael Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html), Bölüm 6 ve Stanford CS231n, [Ödev 2](https://cs231n.github.io/assignments2024/assignment2/), ConvolutionalNetworks not defterinden uyarlanmıştır.*

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
- Register & voice: Use inclusive first-person plural ("biz", e.g., "türettik", "gerçekliyoruz", "inceleyeceğiz") in explanatory prose, second-person plural imperative ("-in/-iniz", e.g., "tasarlayın", "türetin", "karşılaştırın") in instructions/exercises, and concise noun-phrase headings without trailing periods.
- Numbers & references: Keep `.` as decimal separator in prose numbers (`0.01`, `0.02`) to match math/code literals; keep dimensional shorthands (`2D`, `4D`) unchanged; write prose percentages with leading `%` (`%45`) per standard Turkish convention; place postpositions/nouns (`kapsamında`, `örneğinde`, `içindeki`, `eşitliğini`) after `:numref:`/`:eqref:`/`§{...}` tokens instead of attaching apostrophe suffixes directly to reference tokens.
- Parenthetical English on first mention: Include English in parentheses on the first occurrence of newly coined or domain-specific terms (`*kenar haritası* (edge map)`, `*pencere* (window)`, `*öznitelik haritası* (feature map)`, `*alıcı alan* (receptive field)`, `*etkin* alıcı alan (effective receptive field)`), and include standard English acronyms alongside Turkish expansions (`çok katmanlı algılayıcı (MLP)`, `evrişimli sinir ağı (CNN)`, `stokastik gradyan inişi (SGD)`).
- "locality" → "yerellik"
- "edge detection / edge detector / edge map" → "kenar tespiti / kenar tespit edici / kenar haritası"
- "implement / implementation" → "gerçeklemek / gerçekleme" (distinguishes code implementation from "apply" → "uygulamak")
- "sliding window" → "kayan pencere"; "patch / patch matrix" → "yama / yama matrisi"; "elementwise" → "eleman bazında"
- "dense layer" → "yoğun katman"; "locally connected layer" → "yerel bağlı katman"
- "initialize" → "ilklendirmek"
- "framework" → "çatı"; "library" → "kütüphane"
- "transpose / transposed" → "devrik"
- "flatten / flattened / flattening" → "düzleştirmek / düzleştirilmiş / düzleştirme"; "unfold / unroll" → "açmak"
- "banded matrix / sparse banded matrix" → "bantlı matris / seyrek bantlı matris"
- "theoretical receptive field / effective receptive field" → "teorik alıcı alan / etkin alıcı alan"
- "sensitivity" → "hassasiyet"
- "trade-off" → "ödünleşim"
- "blur kernel / sharpening kernel" → "bulanıklaştırma çekirdeği / keskinleştirme çekirdeği"
- "downsample" → "alt örnekleyerek küçültmek"
- "support" (of a function/gradient) → "destek"
- "Discussions" → "Tartışmalar"
