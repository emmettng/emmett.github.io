
Images, sound clips and many other similar kinds of data have an intrinsic structure. More formally, they share these important properties

Channel ⇒ different viewpoint ?
One axis, called the channel axis, is used to access different views of the data (e.g., the red, green and blue channels of a color image, or the left and right channels of a stereo audio track).


# Convolution
`convolution in *artificial neural network*`
**This is an intuitive explination**


## Basic Assumption
All discussion in this section is based upon the belief of the following assumption:

> **_Complexity is the composition of limited set of simplicities_**.

More specifically, simplicity comes first, then composition of sufficient amount of different simplicities is the necessary condition for the emerge of complexity *(What is simplicity and Complexity? How could they be measured ? These will be discussed later)*.

- Lines and curves to figures
- Letters to literatures
- Syllables to languages  

  > It is possible that different language contains different set of syllables, some syllables in one language is not existed in another language, but each of these languese are all composed with limited set of syllables.


I will use **Elementary Feature** and **Higher Level Feature** instead of **Simplicity** and **Complexity** in the following discussion. Generally speaking, 'Feature Level' is a relative concept.

VGG 16 feature map .

## The binding problem
Firstly, I will give a intuitive disucssion about a problem of the Fully Connected network (usually being referred as FC network).

Assuming that we have somehow trained this network successfully, so that in layer N of this network, there are invariant representations of all elementary features, and when being activated, neuron $\alpha_1$  is the only representation of a vertical bar in the input layer `I` and neuron $\alpha_2$ is the only representation of a horizontal bar in the input layer I.  In other words, if there is any vertical bar shows up in the input layer, no neuro other than $\alpha_1$ will be activated, so the same as $\alpha_2$ to a horizontal bar.

![binding_1](../imgs/binding_1.png)

> Thans for Dr Simon Stringer's [talk](https://www.youtube.com/watch?v=j4kjGso_Ze0&t=183s) and the example of letters `'T'` and `'L'`.

If our input image contains only one English letter, then it is possible that ,in layer `N+`, there is a neuron $\beta_1$ which is the only representation of a letter `‘T’` and its activation status depends on both $\alpha_1$ and $\alpha_2$.

Now the question is : In layer `N+`, could there be another neuron $\beta_2$ whose activation status depends on neurons $\alpha_1$ and $\alpha_2$ and it is the only representation of letter `‘L’` in input layer I.

It is apparently that the representation of letter ‘L’ cannot depends on  $\alpha_1$ and $\alpha_2$ , otherwise it will bring ambiguity.

The simplest solution is to have one more neuron that represent either vertical bar or horizontal bar.  Such as in figure below. In this case, neuron $\alpha_2$ and $\alpha_3$ both represent horizontal bar, furthermore they also carry composition information that indicate the role of horizontal bar in different context (`'L'` or `'T'`) repectively.

![binding_2](../imgs/binding_2.png)



Of course, it it a common sense that different objects (higher level feature) have different parts, but as the feature level goes higher and higher, more and more basic elementary features will take part in the composition of more and more higher level features. For instance, as shown in figure below, the composition information seems to be not necessary. Because the Optimus Prime has a [Barrage cannon](https://tfwiki.net/wiki/Barrage_cannon) while a truck does not. In addition, a truck carries a cargo, while the Optimus Prime does not. However, it is highly likely that the barrage cannon and the trailer share a lot elementray features. Thus, introducing composition information will be inevitable in a deep architecture.

![binding_3](../imgs/binding_3.jpg)

Now we know, in a fully connected neuron network, duplicated representations of same elementary features are necessary for expressing composition information (context information). But how much trouble it would bring into the network?

![combinatorial explosing](../imgs/binding_combinatorial_1.png)
Intuitively, the matrix above shows relation between higher level features in a certain layer and elementary features one layer below.
$O_1$ to $O_i$ are different features, $i$ is the index. If two features share exactly same set of elementary features, then there will be at least one replicated representation of an elementary features to provide extra composition information.

For example, $O_1$ and $O_2$ are presented by same set of elementary features, then there will be one extra elementary feature that carrying composition information. $O_1$ and $O_3$ might depends on not exactly the same set of elementary features, then it is not necessary to have an extra replicated feature.

![combinatorial explosing 2](../imgs/binding_combinatorial_2.png)
Assume, there is a new feature $O_{i+1}$ that need to be represented by neurons in this layer `M` and features $O_{i+1}$, $O_i$, $O_2$ are all composed with same set of features, therefore the minimun number of replicated features in previous layer after introducing $O_{i+1}$ is as shown in figure above.

This is just the relation between two adjacent layers, the need for extra features will be augmented as the layer goes down. For example, feature $O_{i+1}$ is in layer `M`, the matrix indicates that we need at least two replicated representation in layer `M-1`, and each of these `M-1` features might need a certain number of replicated represent of their own in layer `M-2`.

So, for a artifical neuron network with deep architecture:   
- From top to buttom, The necessary replication of different features will increase dramatically.
- From buttom to Top, the search time of certain higher level feature will be increase as well.

It would be easily lead to combinatorial explosion on both space and time complexity.
![conbinatorial_explosion](../imgs/combinatorial-explosion.jpeg)

The consequence is: It would be extremmly hard to train. This is part of the reason there are all different kinds of feature map, but none of them is trained on a `'Fully Connected'` network.

This section is just an **Intuitive Explanation** of the binding problem in `Fuclly Connected` **Deep Neural Network**. In real world the case would be much more complicated:

- Features could be represented by a set of neurons.
- Other contrains will limited the amount of distinguishable feature representations, such as regularization.

## Convolution ! What is it good for?
Convolution is one solution which would so the problem above.

> There are intrinsic strucutre in input space.

This is a redundent assumption, it means the same as

> **_Complexity is the composition of limited set of simplicities_**.

## convolution
Each channel is a detector of a lower level feature.  And the composition information is preserved as it is in the original input.

## arithematic

[doc](https://arxiv.org/pdf/1603.07285.pdf)

## deconvolution
we assume the input the a result of down-sample of the output, and now we would like to retrive output (doing up-sampling) ...

## Intuitively & Computationally

persumablelly we will not only being able to increase the dimensionallity but also bring sparsity in the representation .
## details

## affine transformation
- [checkerboard artif]





## **Others**

Images, sound clips and many other similar kinds of data have an intrinsic structure. More formally, they share these important properties

Channel ⇒ different viewpoint ?
One axis, called the channel axis, is used to access different views of the data (e.g., the red, green and blue channels of a color image, or the left and right channels of a stereo audio track).

[^fn1]:http://www.quotedb.com/quotes/2112

polychronization it is not the only solution.
