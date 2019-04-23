

class ISMarginalLikelihood(VariationalInference):
    """
    The class of the Stochastic Variational Inference (SVI) algorithm.

    :param num_samples: the number of samples used in estimating the variational lower bound
    :type num_samples: int
    :param model: the definition of the probabilistic model
    :type model: Model
    :param posterior: the definition of the variational posterior of the probabilistic model
    :param posterior: Posterior
    :param observed: A list of observed variables
    :type observed: [Variable]
    """
    def __init__(self, num_samples, model, posterior, observed):
        super(ISMarginalLikelihood, self).__init__(
            model=model, posterior=posterior, observed=observed)
        self.num_samples = num_samples

    def compute(self, F, variables):
        """
        Compute the inference algorithm

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param variables: the set of MXNet arrays that holds the values of
        variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :returns: the outcome of the inference algorithm
        :rtype: mxnet.ndarray.ndarray.NDArray or mxnet.symbol.symbol.Symbol
        """
        samples = self.posterior.draw_samples(
            F=F, variables=variables, num_samples=self.num_samples)
        variables.update(samples)
        logL = self.model.log_pdf(F=F, variables=variables)
        logL = logL - self.posterior.log_pdf(F=F, variables=variables)
        return -logL, -logL
