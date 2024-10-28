import numpy as np
import torch
from torch.distributions import Categorical
from PolicyNetwork import PolicyNetwork
from ValueNetwork import ValueNetwork
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
class TRPOAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lam=0.97, kl_threshold=0.01):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.gamma = gamma
        self.lam = lam
        self.kl_threshold = kl_threshold

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.policy(state)
        distribution = Categorical(action_probs)

        print(f"State: {state.numpy()}, Action Probabilities: {action_probs.numpy()}")
        action = distribution.sample().item()
        return action, distribution.log_prob(torch.tensor(action)).item()

    def compute_advantages(self, rewards, values, masks):
        values = np.append(values, 0)
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * masks[t] - values[t]
            gae = delta + self.gamma * self.lam * masks[t] * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns

    def conjugate_gradient(self, fisher_vector_product, b, nsteps=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rsold = torch.dot(r, r)

        for i in range(nsteps):
            Ap = fisher_vector_product(p)
            alpha = rsold / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            rsnew = torch.dot(r, r)
            if torch.sqrt(rsnew) < residual_tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return x

    def fisher_vector_product(self, states, p):
        p = p.detach()
        kl = self.compute_kl(states)
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        grad_kl_p = torch.dot(flat_grad_kl, p)
        grads = torch.autograd.grad(grad_kl_p, self.policy.parameters())
        flat_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

        return flat_grad_kl + 0.1 * p

    def compute_kl(self, states):
        action_probs = self.policy(states)
        old_action_probs = action_probs.detach()

        kl = (old_action_probs * (torch.log(old_action_probs) - torch.log(action_probs))).sum(-1).mean()
        return kl

    def update_policy(self, states, actions, old_log_probs, advantages):
        # Compute the surrogate loss
        action_probs = self.policy(states)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate_loss = (ratio * advantages).mean()

        # Compute gradients of surrogate loss
        loss_grad = torch.autograd.grad(surrogate_loss, self.policy.parameters(), retain_graph=True)
        loss_grad = torch.cat([grad.view(-1) for grad in loss_grad]).detach()

        # Compute the Fisher vector product
        def fisher_vector_product(p):
            return self.fisher_vector_product(states, p)

        # Use Conjugate Gradient to find step direction
        step_dir = self.conjugate_gradient(fisher_vector_product, loss_grad)

        # Step size calculation with backtracking line search
        step_size = torch.sqrt(2 * self.kl_threshold / (torch.dot(step_dir, fisher_vector_product(step_dir)) + 1e-8))

        # Update the parameters of the policy
        new_params = [param + step_size * step for param, step in zip(self.policy.parameters(), self.vector_to_parameters(step_dir))]
        self.update_parameters(new_params)

    def vector_to_parameters(self, vec):
        """Helper to convert a vector to model parameters"""
        index = 0
        params = []
        for param in self.policy.parameters():
            size = param.numel()
            params.append(vec[index:index + size].view(param.size()))
            index += size
        return params

    def update_parameters(self, new_params):
        """Helper to update model parameters"""
        for param, new_param in zip(self.policy.parameters(), new_params):
            param.data.copy_(new_param.data)

    def update_value_net(self, states, returns):
        optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)
        for _ in range(80):
            optimizer.zero_grad()
            values = self.value_net(states).squeeze()
            loss = nn.MSELoss()(values, returns)
            loss.backward()
            optimizer.step()
