Download Link: https://assignmentchef.com/product/solved-ee270-large-scale-matrix-computation-optimization-and-learning-hw-4
<br>



<ol>

 <li><strong>Randomized Gauss-Newton Algorithm for Training Neural Networks  </strong>In this problem, you will implement randomized Gauss-Newton (GN) algorithm to train a neural network. We will consider a single layer architecture and a squared loss training objective. This special case is also referred to as a nonlinear least squares problem. Consider a set of training data, where <em>x<sub>i </sub></em>∈R<em><sup>d </sup></em>is the <em>i<sup>th </sup></em>data sample and <em>y<sub>i </sub></em>∈{0<em>,</em>1} is the</li>

</ol>

corresponding label. Then, use the following nonlinear function

to fit the given data labels. In order to estimate <em>w</em>, we can minimize the sum of squares as follows

<em>n</em>

<em>w</em><sup>∗ </sup>=argmin<sup>X</sup>(<em>σ</em>(<em>w<sup>T</sup>x<sub>i</sub></em>) − <em>y<sub>i</sub></em>)<sup>2</sup>

<em>w</em>

<em>i</em>=1

= argmin

where <em>w</em><sup>∗ </sup>denotes the optimal parameter vector, <em>y </em>is the vector containing labels <em>y</em><sub>1</sub><em>,…,y<sub>n</sub></em>, and the vector output function <em>f</em>(<em>w</em>) is defined as

<em> .</em>

The above problem is <u>non-convex </u>unlike the ordinary least squares and logistic regression problems.

Download the provided MNIST dataset (mnist all.mat), where you will use only the samples belonging to digit 0 (train0) and digit 1(train1). Next, we will apply the Gauss-Newton (GN) heuristic to approximately solve the non-convex optimization roblem. In the GN algorithm, you first need to linearize the nonlinear vector output function <em>f</em>(<em>w</em>). The first order expansion of <em>f</em>(<em>w</em>) around the current estimate, <em>w<sub>k </sub></em>is given by

<em>f</em>(<em>w</em>) ≈ <em>f</em>(<em>w<sub>k</sub></em>) + <em>J<sup>k</sup></em>(<em>w </em>− <em>w<sub>k</sub></em>)                                                                            (1)

where <em>J<sup>k </sup></em>∈R<em><sup>n</sup></em><sup>×<em>d </em></sup>is the Jacobian matrix at the iteration <em>k </em>whose <em>ij<sup>th </sup></em>entry is defined as

<em>.</em>

Then, Gauss-Newton algorithm performs the update

<em>w<sup>k</sup></em><sup>+1 </sup>=argmin<em> .            </em>(2) <em>w</em>

<ul>

 <li>Find the Gauss-Newton update (2) explicitly for this problem by deriving the Jacobian. Describe how the sub-problems can be solved. What is the computational complexity per iteration for <em>n </em>× <em>d </em>data matrices?</li>

</ul>

One can incorporate a step-size <em>µ </em>∈ (0<em>,</em>1] in the above update as follows.

<em>w<sup>k</sup></em><sup>+1 </sup>= (1 − <em>µ</em>)<em>w<sup>k </sup></em>+ <em>µd<sup>k</sup></em>

where <em>d<sup>k </sup></em>:= argmin<em> .                </em>(3) <em>w</em>

This is referred to as damped Gauss-Newton algorithm.

<ul>

 <li>Implement the damped Gauss Newton algorithm on the MNIST dataset to classify digits 0 and 1. Find a reasonable step-size for fast convergence by trial and error. Plot the training error, i.e., , as a function of the iteration index.</li>

 <li>You will now apply sampling to reduce computational complexity of the GN algorithm. Apply uniform sampling to the rows of the Least-Squares sub-problem (3) at each iteration. Plot the training error, i.e., , as a function of the iteration index. Repeat this procedure for row norm scores sampling.</li>

</ul>

<h1>Randomized Kaczmarz Algorithm</h1>

In this question, you will implement the Randomized Kaczmarz Algorithm <a href="https://web.stanford.edu/class/ee270/Lecture16.pdf">(see Lecture</a> <a href="https://web.stanford.edu/class/ee270/Lecture16.pdf">16 </a><a href="https://web.stanford.edu/class/ee270/Lecture16.pdf">slides)</a> on the YearPredictionMSD dataset, which can be downloaded from <a href="https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD">https://archive. </a><a href="https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD">ics.uci.edu/ml/datasets/YearPredictionMSD</a><a href="https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD">.</a> In this file, there are 463,715 training and 51,630 test samples (songs) with 90 audio features and your goal is to predict the release year of each song using least squares regression. After downloading the file, you can import and partition the data using the following command in Python

import numpy as np file_name=’YearPredictionMSD.txt’ file = open(file_name, ’r’)

data=[] for line in file.readlines():

fname = line.rstrip().split(’,’) data.append(fname)

data_arr=np.asarray(data).astype(’float’) A=data_arr

b=(data_arr[:,0]-np.min(data_arr[:,0]))/(np.max(data_arr[:,0])-np.min(data_arr[:,0])) You may also normalize the data matrix as follows

meanA=np.mean(A,axis=0) stdA=np.std(A,axis=0) A=(A-meanA)/stdA

<ul>

 <li>Plot the training and test cost vs iteration curves for Randomized Kaczmarz Algorithm (i.e., the one with optimal sampling distribution) and the randomized algorithm with uniform sampling distribution on the same figure. For the uniformly sampled version, you should use an explicit learning rate and tune this parameter by trial and error.</li>

 <li>Note that the analysis of the Randomized Kaczmarz Algorithm assumes that the linear system <em>Ax </em>= <em>b </em>is consistent, which is may not be valid in our case. Show that an optimal solution of the least squares problem min can be found via solving</li>

</ul>

the augmented linear system .

<ul>

 <li>Since the augmented linear system in part (b) is consistent, we may apply Randomized Kaczmarz Algorithm. Repeat part (a) using Randomized Kaczmarz algorithm on the augmented linear system.</li>

 <li>Repeat part (a) using SGD with diminishing step-sizes (see page 11 of Lecture 16 slides) by tuning the initial step size.</li>

</ul>




<h1>Randomized Low-Rank Approximation and Randomized SVD</h1>

In this problem, you will implement a randomized approach in order to obtain a low-rank approximation for a given data matrix. Assume that you are given a data matrix <em>A </em>∈

R<em><sup>n</sup></em><sup>×<em>d </em></sup>with <em>U</em>Σ<em>V <sup>T </sup></em>as its SVD. The best rank-k approximation for the data matrix is <em>A<sub>k </sub></em>=

. However, since this is computationally expensive, you need to use

another approach to reduce the complexity.

First, you will generate a 5000×1000 data matrix with decaying singular values. To construct such a matrix, you can first generate a random Gaussian matrix with zero mean and identity covariance. Then, compute the SVD of this matrix as <em>U</em>Σ<em>V <sup>T</sup></em>. Now, replace Σ with a diagonal matrix Σ with decaying entries,<sup>ˆ </sup>Σ<sup>ˆ</sup><em><sub>ii </sub></em>= <em>i</em><sup>−2</sup>. Then, you can construct the data matrix with decaying singular values as <em>A </em>= <em>U</em>Σ<sup>ˆ</sup><em>V <sup>T</sup></em>.

<ul>

 <li>One approach to obtain a rank-k approximation is as follows.

  <ul>

   <li>Obtain a random approximation for the data matrix <em>A </em>as <em>C </em>by uniformly sampling <em>k </em>columns and scaling appropriately.</li>

   <li>Verify the approximation by computing the relative error k<em>AA<sup>T </sup></em>−<em>CC<sup>T</sup></em>k<em><sub>F</sub>/</em>k<em>AA<sup>T</sup></em>k<em><sub>F</sub></em>.</li>

   <li>Compute the randomized rank-k approximation defined as <em>A</em>˜<em><sub>k </sub></em>= <em>CC</em><sup>†</sup><em>A</em>.</li>

  </ul></li>

</ul>

Plot the approximation error, i.e., , as a function of the rank <em>k</em>. Verify the error bound we have in <a href="https://web.stanford.edu/class/ee270/Lecture19.pdf">Lecture 19</a> slides, i.e., , and find the numerical value of . Repeat this procedure for uniform sampling, column norm score sampling, Gaussian sketch, +1<em>/ </em>− 1 i.i.d. sketch, and randomized Hadamard (FJLT) sketch.

<ul>

 <li>Now we use the low rank approximation in part (a) to produce an approximate Singular Value Decomposition of <em>A</em>. The Randomized SVD algorithm is as follows.

  <ul>

   <li>Generate a sketching matrix <em>S</em>.</li>

   <li>Compute <em>C </em>= <em>AS</em></li>

   <li>Calculate QR decomposition: <em>C </em>= <em>AS </em>= <em>QR</em></li>

   <li>Calculate the SVD of <em>Q<sup>T</sup>A</em>, i.e., <em>Q<sup>T</sup>A </em>= <em>U</em>Σ<em>V <sup>T</sup></em></li>

   <li>Approximate SVD of <em>A </em>is given by <em>A </em>≈ (<em>QU</em>)Σ<em>V <sup>T</sup></em>.</li>

   <li>Note that the randomized rank-k approximation <em>A</em><sup>˜</sup><em><sub>k </sub></em>:= (<em>QU</em>)Σ<em>V <sup>T </sup></em>= <em>QQ<sup>T</sup>A </em>= <em>CC</em><sup>†</sup><em>A </em>as in part (a)</li>

  </ul></li>

</ul>

Plot the approximation error, i.e., , as a function of the rank <em>k</em>. To compare with the exact SVD, plot , where <em>A<sub>k </sub></em>is the best rank <em>k </em>approximation of <em>A </em>using the SVD of <em>A</em>. Repeat the whole procedure for uniform sampling, column norm score sampling, Gaussian sketch, +1<em>/</em>−1 i.i.d. sketch, and randomized Hadamard (FJLT) sketch.

<h1>CUR decomposition</h1>

CUR decomposition is a dimensionality reduction and low-rank approximation method in a similar spirit to Singular Value Decomposition (SVD). One particular difficulty of SVD is that the left and right singular vectors are difficult to interpret since they lack any direct meaning in terms of the original data. On the other hand, CUR decomposition is interpretable since it involves small subsets of the rows/columns of the original data.

Suppose that <em>A </em>is an m × n matrix of approximate rank k, and that we have two column/row subsampled approximations

<table width="336">

 <tbody>

  <tr>

   <td width="317"><em>C </em>:= <em>A</em>(:<em>,J<sub>S</sub></em>)</td>

   <td width="19">(4)</td>

  </tr>

  <tr>

   <td width="317"><em>R </em>:= <em>A</em>(<em>I<sub>S</sub>,</em>&#x1f642;</td>

   <td width="19">(5)</td>

  </tr>

 </tbody>

</table>

where <em>I<sub>S</sub>,J<sub>S </sub></em>are appropriate subsets. Then we can approximate the matrix <em>A </em>via

<h1>                                                           <em>A </em>≈ <em>CC</em><sup>†</sup><em>AR</em><sup>†</sup><em>R </em>= <em>CUR,                                                                                 </em>(6)</h1>

where <em>U </em>:= <em>C</em><sup>†</sup><em>AR</em><sup>† </sup>and the superscript † denotes the pseudoinverse operation. We choose set of columns <em>C </em>and a set of rows <em>R</em>, which play the role of <em>U </em>and <em>V </em>in SVD. We may pick any number of rows and columns. Therefore, this factorization provides an interpretable alternative to the Singular Value Decomposition. Furthermore, CUR has computational advantages especially for sparse matrices. Particularly, since CUR directly select random rows and columns <em>C </em>and <em>R </em>are sparse for a sparse matrix <em>A</em>, whereas <em>U </em>and <em>V </em>in SVD can still be dense matrices.

You will be using the movie ratings dataset from the <a href="https://movielens.org/">https://movielens.org/</a> website provided at <a href="https://grouplens.org/datasets/movielens/100k/">https://grouplens.org/datasets/movielens/100k/</a><a href="https://grouplens.org/datasets/movielens/100k/">.</a> The dataset contains 100,000 ratings from 943 users on 1682 movies. The ’usermovies’ matrix of dimension 943 x 1682 is very sparse.

<ul>

 <li>For <em>m </em>= 1<em>,…,</em>10, apply uniform row/column subsampling with sample size <em>m </em>to obtain <em>C </em>and <em>R </em>matrices of rank <em>m </em>and solve <em>U </em>to obtain the CUR decomposition. Plot the approximation error in Frobenius norm and spectral norm as a function of <em>m</em>.</li>

 <li>Repeat part (a) using <em>`</em><sub>2 </sub>row/column norm scores for row/column subsampling respectively.</li>

 <li>Repeat part (a) using leverage scores of <em>A </em>and <em>A<sup>T </sup></em>for row/column subsampling respectively.</li>

</ul>