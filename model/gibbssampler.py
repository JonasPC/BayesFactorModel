
class GibbsSampler(self, data, """hyperparameters, starting_values"""):

    """
    Samples parameters of posterior given data and hyper parametersself.

    ===
    - Parameters -
    data: Must be a df of dimension (N x p)
    B: Hyper parameter of the var/covar matrix (scalar)
    v: Hyper parameter of the var/covar matrix (scalar)
    m: Hyper parameter of Lambda (scalar)
    H: Hyper parameter of Lambda (scalar or matrix)
    lambda_0: starting value of lambda (scalar or matrix)
    F_0: starting value of F (scalar or vector of dimension m)

    """

    def __init__(self):
        self.data = data
        self.n = len(data)
        self.p = "something"
