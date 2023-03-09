import numpy as np
import torch
import torch.nn.functional as F


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


# It is adapted from https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/mixture.py
def discretized_mix_logistic_loss(y_hat, y, num_classes=65536,
                                  log_scale_min=None, reduce=True):
    """Discretized mixture of logistic distributions loss
        Note that it is assumed that input is scaled to [-1, 1].
        Args:
            y_hat (Tensor): Predicted output (B x C x T)
            y (Tensor): Target (B x T x 1).
            num_classes (int): Number of classes
            log_scale_min (float): Log scale minimum value
            reduce (bool): If True, the losses are averaged or summed for each
              minibatch.
        Returns
            Tensor: loss
        """
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    y_hat = y_hat.permute(0,2,1)
    assert y_hat.dim() == 3
    assert y_hat.size(1) % 3 == 0 # 为何第1个维度是3的整数倍？
    nr_mix = y_hat.size(1) // 3

    # (B x T x C)
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(F.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - F.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()

    inner_inner_out = inner_inner_cond * \
        torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
        (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (y > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (y < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -torch.mean(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)

# https://github.com/Rayhane-mamah/Tacotron-2/issues/155
# cumulative distribution function, 概率密度函数的积分
def sample_from_discretized_mix_logistic(y, log_scale_min=None):
    """
    Sample from discretized mixture of logistic distributions
    从多个逻辑分布的中取值
    选取哪一个需要一个参数， 然后每一个logistic distribution 的参数有两个分别为mean, bias。
    得到这个logistic distribution 之后，就可以从中sample出具体音频数值。
    采取这种方式，音频值不用做mu_law encoding。
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    assert y.size(1) % 3 == 0
    # number of mixtures is 10
    # number of mixture of logistic distributions 10。 每一个logsitic 有其均值和方差， 同时还有一个权重矩阵，所以y.size(1)为30
    nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]

    # sample mixture indicator from softmax
    # apply a softmax-like sampling to select which logistic distribution to use
    # 每一个里面加一个噪声，然后取argmax就模拟了softmax sample的过程
    # https://phimos.github.io/2020/02/13/RN-Categorical-Reparameterization-with-Gumbel-Softmax/
    # 这里均匀分布的值做了限制，不能太小或太大
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1) # argmax不能做梯度反转，也不需要，最终是要选择一个logistic regression, 让它的参数做梯度计算即可。

    # (B, T) -> (B, T, nr_mix)
    one_hot = F.one_hot(argmax, nr_mix).float()
    # select logistic parameters
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.clamp(torch.sum(
        y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1), min=log_scale_min)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    # logistic 分布产生sample, 通过inverse的方式来产生
    # 首先计算得到u,和delta。然后得到其cdf，cdf的inverse
    u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5) # 这个值均匀分布产生的值，做了一定的限制, 这里采用mol是因为其cdf是可逆的
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u)) # 这个是与均匀分布对应logistic regression的sample 高斯分布pdf不能inverse,要想其他方法。

    x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

    return x

'''
def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot'''
