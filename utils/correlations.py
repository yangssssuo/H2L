'''
Pearson’s correlation coefficient: r = sum((x - mean(x)) * (y - mean(y))) / sqrt(sum((x - mean(x)) ** 2) * sum((y - mean(y)) ** 2))
Spearman’s rank correlation coefficient: rho = 1 - 6 * sum((rank(x) - rank(y)) ** 2) / (n * (n ** 2 - 1))
Kendall’s rank correlation coefficient: tau = (concordant_pairs - discordant_pairs) / (n * (n - 1) / 2)
'''


def pearsonr(x, y):
    x_mean = x.mean()
    y_mean = y.mean()
    return ((x - x_mean) * (y - y_mean)).sum() / (((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum()).sqrt()

def spearmanr(x, y):
    x_rank = x.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()
    n = x.numel()
    return 1 - 6 * ((x_rank - y_rank) ** 2).sum() / (n * (n ** 2 - 1))

def kendalltau(x, y):
    n = x.numel()
    x_rank = x.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()
    concordant_pairs = ((x_rank - y_rank) < 0).sum()
    discordant_pairs = ((x_rank - y_rank) > 0).sum()
    return (concordant_pairs - discordant_pairs) / (n * (n - 1) / 2)

def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    """Compute correlation between 2 1-D vectors
    Args:
        x: Shape (N, )
        y: Shape (N, )
    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)


class Spearman_loss(_Loss):
    def __init__(self) -> None:
        super(Spearman_loss,self).__init__()

    def _spearmanr(self,x, y):
        x_rank = x.argsort().float()
        y_rank = y.argsort().float()
        n = x.shape[-1]
        return 1-  6 * ((x_rank - y_rank) ** 2).sum(dim=2) / (n * (n ** 2 - 1))
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        aaa =  self._spearmanr(input,target).requires_grad_()
        return self._spearmanr(input,target)