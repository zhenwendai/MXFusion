# Reviewer 1

In the paper, it is claimed that many other distribution can be used as the auxiliary distribution beside normal distribution , but no example of them given neither on synthetic nor on real data experiments. That would be interesting to see how this new proposed model behave given data are generated from latent variables which have multi-modal and/or overdispresed distributions. Also it is interesting to discuss under what condition we can use multiple non-invertible transformation (like NF) and how expensive they are computationally since it will be auxiliary distribution inference at each transformation step.

- it makes the paper easier to read if derivation for equation 6 and 7 in the paper be added to appendix

(not doable)

- Should K be used in equation 3 instead of M?

- In Experiments section, many details of experiments set ups have been eliminated these details needs to be discussed specially on how auxiliary distribution parameters have been set up and learned, also it will be great to explain the competing algorithms parameters set ups.

(promise to add experimental details in the supplemental material)

- As suggested in quality section of this review, it is desired to see how this new proposed transformation will infer the posteriori if data are generated from latent variables with multi-modal and/or overdispresed distributions.

- It should be clear under which condition the comparison on VAE experiments being made between algorithms (e.g. set ups for the SIVI-VAE number of samples and network architecture)

- It is interesting to see what would be advantage and limitation of using other auxiliary variables and a discussion about the outcome on certain example

# Reviewer 2

In the "Variaitonal Autoencoder" part (line 223), it is not clear to me if the results for the other models (Yin and Zhou 2018, Titsias and Ruiz 2019) were obtained by running the models or if they were extracted from the papers. It is also not clear to me if the same architecture (decoder network, dimension of the latent space) was used for all models or not. Also, in order for a reader to be able to reproduce the results, the training parameters should be included: learning rate, minibatches, number of epochs, etc. Many of these things could be added in the supplementary material of the paper.

Quality and significance: The work is technically sound. While the algorithm proposed is new and original, I feel that some aspects of the results are not completely compelling.
1- In the "imputation with VAE" part of the results (line 258), the method is compared only against mean field variational inference. Looking at the figure, the proposed method seems to work better. However, a mean field approximation with a Gaussian distribution is one of the simplest possible variational distributions one could use, and it is known to have issues when latent variables are correlated. I think that comparing against other flow-based approaches in this experiment should be done.
2- In the "BNN regression" part of the results (line 273) it can be observed that the proposed method does not perform considerably better than mean field VI (the intervals [mean +- std] for each method have a considerable overlap). Again, in this part it would be interesting to see how the method compares agains other widely used flow-based approaches (ie, Normalizing flows, IAF, Real-NVP).

Despite the fact that results could be compared against more related methods, I think that the use of non-invertible transformations is an interesting idea that is worth exploring.

Since the paper presents a new flow-based approach to build flexible variational distributions, I think a comparison against other flow-based methods should be included. For example, Normalizing flows, Inverse Autorregressive flows, and/or Real NVP.

A few typos and a missing reference:
- lines 22-23: "A popular approach is to address the problem of the integral is to approximate..." The middle part of that sentence should be removed.
- Equation 1: Wrong sign, the right hand side of that equation should be multiplied by -1.
- line 71: The ELBO is maximized, not minimized.
- The following paper should be cited: "A family of non-parametric density estimation algorithms", by Tabak and Turner. This introduced the idea of flows, before the "Variational Inference with Normalizing Flows" paper.


# Reviewer 3

My main concerns are
1. In Eq.9, the Gaussian distribution approximates the delta function with a small alpha. Was alpha optimized in the experiments since the authors used the word "initialize"? If so, how was it optimized? If not, how was it chosen?

The work would be complete if the authors can discuss the effect of the magnitude of alpha theoretically or empirically.

2. The similar question to the beta in \tilde{q}(z|x).

- Explain the intuition like approximate a log with a quadratic function. (depends on the progress of the optimization, different alpha and beta give good approximation at different locations)

3. What is the size of the Monte Carlo samples mentioned in Line 157. The authors claim their approach does not require a large sample size. Please clarify if it indicates the data or Monte Carlo. It would be more convincing if the authors provide quantitative comparisons regarding to this claim.

Show the number of samples. Find a way to compare the lower bound between different approaches.

4. The work would be more complete if the authors show how fast the learning is.

...
