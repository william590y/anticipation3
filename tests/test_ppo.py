import math

import torch

from ppo import approximate_kl, compute_gae, discounted_returns, policy_loss


def test_discounted_returns_are_independent_of_value_bootstrap():
    rewards = torch.tensor([[1.0, -1.0, 2.0]])
    valid = torch.ones_like(rewards)
    values = torch.tensor([[10.0, -4.0, 7.0]])

    monte_carlo = discounted_returns(rewards, valid, gamma=1.0)
    _, lambda_returns = compute_gae(
        rewards, values, valid, gamma=1.0, lam=1.0
    )

    expected = torch.tensor([[2.0, 1.0, 2.0]])
    torch.testing.assert_close(monte_carlo, expected)
    torch.testing.assert_close(lambda_returns, expected)


def test_approximate_kl_uses_new_over_old_log_ratio():
    old_logprob = torch.tensor([[math.log(0.5)]])
    new_logprob = torch.tensor([[math.log(0.25)]])
    mask = torch.ones_like(old_logprob)

    # k3 = r - log(r) - 1 for r = pi_new / pi_old = 0.5.
    expected = 0.5 - math.log(0.5) - 1.0
    torch.testing.assert_close(
        approximate_kl(new_logprob, old_logprob, mask),
        torch.tensor(expected),
    )


def test_policy_ratio_is_identity_before_an_update():
    logprob = torch.tensor([[-0.2, -1.0, -2.0]])
    advantages = torch.tensor([[1.0, -1.0, 0.5]])
    mask = torch.ones_like(logprob)

    _, clipped, kl = policy_loss(
        logprob, logprob, advantages, mask, clip_eps=0.2
    )

    assert clipped.item() == 0
    assert kl.item() == 0
