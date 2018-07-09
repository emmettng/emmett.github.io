
Images, sound clips and many other similar kinds of data have an intrinsic structure. More formally, they share these important properties

Channel ⇒ different viewpoint ?
One axis, called the channel axis, is used to access different views of the data (e.g., the red, green and blue channels of a color image, or the left and right channels of a stereo audio track).


# Convolution in *artificial neural network*
`... don't speak too soon  for the wheel's still in spin... - Dylan`

## Basic Assumption
All discussion in this section is based upon the belief of the following assumption:

> **_Complexity is the composition of limited set of simplicities_**.

More specifically, simplicity comes first, then enough composition of different simplicities is the necessary condition for the emerge of complexity *(What is simplicity and Complexity? How could they be measured ? These will be discussed later)*.

- Lines and curves to figures
- Letters to literatures
- Syllables to languages  

  > It is possible that different language contains different set of syllables, some syllables in one language is not existed in another language, but each of these languese are all composed with limited set of syllables.

I will use **Elementary Feature** and **Higher Level Feature** instead of **Simplicity** and **Complexity** in the following discussion.

## Convolution ! What is it good for?

Firstly, I will give a intuitive disucssion about a problem of the Fully Connected network (usually being referred as FC network).
### The binding problem
Assuming that we have somehow trained this network successfully, so that in layer N of this network, there are invariant representations of all elementary features, and when being activated, neuron $\alpha_1$  is the only representation of a vertical bar in the input layer I and neuron $\alpha_2$ is the only representation of a horizontal bar in the input layer I.  In other words, if there is any vertical bar shows up in the input layer, no neuro other than $\alpha_1$ will be activated, so the same as $\alpha_2$ to a horizontal bar.

![binding_1](../imgs/binding_1.png)


If our input image contains only one English letter, then it is possible that ,in layer N+, there is a neuron $\beta_1$ which is the only representation of a letter ‘T’, and of course its activation status depends on both $\alpha_1$ and $\alpha_2$.

Now the question is : In layer N+, could there be another neuron $\beta_2$ whose activation status depends on neurons $\alpha_1$ and $\alpha_2$ and it is the only representation of letter ‘L’ in input layer I.

It is apparently that if a higher level feature depends on a set of lower level features, then without other information, the same set of lower level features can only contribute to the representations of the same higher level features. In this example, the representation of letter ‘L’ cannot depends on  $\alpha_1$ and $\alpha_2$ , otherwise it will bring ambiguity.

A simple solution is to have one more neuron with represent either vertical bar or horizontal bar.  Such as in figure below. Now we have two neurons $\alpha_2$ and $\alpha_3$ that represent horizontal bar in different context.

![binding_2](../imgs/binding_2.png)

Now we know, due to the lack of composition information, no context information,  the reuse rate of lower level feature is really poor.

In the example above, we assume that T and L will be presented separately and there is features will be represented by only one neuron,  what if T and L could be presented same time maybe multiply times in the input layer, and $\alpha_1$ is actually a set of neuron,


Two object does not share same set of features will be find,
Furthermore,  as the higher level feature being more and more complicated, almost all elementary features will be contribute the higher.  And as the network growing deeper and deeper.


The necessary would increase dramatically and lead to Combinatorial explosion.


this is an Intuitive explanation of binding problem ,
the coding refers to this [blog](../Intelligence/intelligence.md)


we need to replicated elementary feature so that different higher level composition could be possible representated.

combinatorial explosion

what's the point ?

## convolution
Each channel is a detector of a lower level feature.  And the composition information is preserved as it is in the original input. 

## arithematic

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
