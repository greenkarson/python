import torch, gym, random
from torch import nn
from torch import optim


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.sequential(x)


class Game:
    def __init__(self, exp_pool_size, explore=0.9):
        self.env = gym.make("CartPole-v1")

        self.exp_pool = []
        self.exp_pool_size = exp_pool_size

        self.q_net = QNet()

        self.explore = explore

        self.loss_fn = nn.MSELoss()

        self.opt = optim.Adam(self.q_net.parameters())

    def __call__(self):
        is_render = False
        avg = 0
        while True:
            # 数据采样
            state = self.env.reset()
            R = 0
            while True:
                if is_render: self.env.render()
                if len(self.exp_pool) >= self.exp_pool_size:
                    # print(".............2")
                    self.exp_pool.pop(0)
                    self.explore += 0.00001
                    if random.random() > self.explore:
                        # print(".............3")
                        action = self.env.action_space.sample()
                    else:
                        # print(".............4")
                        _state = torch.tensor(state).float()
                        Qs = self.q_net(_state[None,...])
                        action = Qs.argmax(dim=1)[0].item()
                else:
                    # print(".............1")
                    action = self.env.action_space.sample()

                next_state, reward, done, _ = self.env.step(action)
                R += reward
                self.exp_pool.append([state, reward, action, next_state, done])
                state = next_state

                if done:
                    avg = 0.95 * avg + 0.05 * R
                    # 训练值大概在475
                    print(avg,R,self.env.spec.reward_threshold)
                    if avg > 200:
                        is_render = True
                    break
            # print(self.exp_pool)

            # 训练
            if len(self.exp_pool) >= self.exp_pool_size:
                exps = random.choices(self.exp_pool, k=100)
                _state = torch.tensor([exp[0].tolist() for exp in exps ])
                _reward = torch.tensor([[exp[1]] for exp in exps])
                _action = torch.tensor([[exp[2]] for exp in exps])
                _next_state = torch.tensor([exp[3].tolist() for exp in exps])
                _done = torch.tensor([[int(exp[4])] for exp in exps])

                _Qs = self.q_net(_state)
                _Q = torch.gather(_Qs,1,_action)

                _next_Qs = self.q_net(_next_state)
                _max_Q = _next_Qs.max(dim=1, keepdim=True)[0]
                _target_Q = _reward + (1 - _done) * 0.9 * _max_Q

                loss = self.loss_fn(_Q,_target_Q.detach())

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                print(loss)


if __name__ == '__main__':
    game = Game(1000)
    game()
