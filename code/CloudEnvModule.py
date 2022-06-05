import gym
from gym.spaces import Box, MultiDiscrete, Dict
from collections import OrderedDict
import numpy as np

class CpuMem():
    def __init__(self, cpu, mem):
        self.cpu = cpu
        self.mem = mem
        self._scpu = cpu
        self._smem = mem
    @classmethod
    def fromCpuMem(cls, cpumem):
        return cls(cpumem[0], cpumem[1])
    def get_lcc(self):
        return self.cpu / self._scpu
    def get_lmm(self):
        return self.mem / self._smem
    def get_arr(self):
        return [self.cpu, self.mem]
    def __le__(self, other):
        return self.cpu <= other.cpu and self.mem <= other.mem
    def __isub__(self, other):
        self.cpu -= other.cpu
        self.mem -= other.mem
        return self

STOP = 2048

def calculate_evenness(servers):
    lc = [server.get_lcc() for server in servers]
    lm = [server.get_lmm() for server in servers]
    D = lambda v: np.std(v)
    return (1 / (D(lc) + 1) + 1 / (D(lm) + 1)) / 2


class CloudEnv(gym.Env):
    def __init__(self, n=300, m=30, dtype=np.float64):
        self.n = n
        self.m = m
        self.dtype = dtype
        self.reset()
        self.action_space = MultiDiscrete([n, m])
        self.observation_space = Dict(
            servers=Box(low=0, high=STOP, shape=(m, 2), dtype=self.dtype),
            vms=Box(low=0, high=STOP, shape=(n, 2), dtype=self.dtype),
        )

    def get_state(self):
        vms = np.zeros((self.n, 2), dtype=self.dtype)
        for i in range(self.n):
            vms[i] = self.vms[i].get_arr()
        servers = np.zeros((self.m, 2), dtype=self.dtype)
        for i in range(self.m):
            servers[i] = self.servers[i].get_arr()
        return OrderedDict([
            ('servers', servers),
            ('vms', vms),
        ])
    
    def update_score(self, score):
        reward = score - self.was
        self.was = score
        return reward

    def get_reward(self):
        ans = 0.0
        for i in range(self.n):
            if self.vms[i].cpu != STOP:
                ans -= 1
        ans += calculate_evenness(self.servers)
        return self.update_score(ans)
    
    def can_move(self, vm_index, server_index):
        vm = self.vms[vm_index]
        server = self.servers[server_index]
        if vm.cpu == STOP:
            return False
        if not vm <= server:
            return False
        return True

    def is_done(self):
        for vm in self.vms:
            if vm.cpu == STOP:
                continue
            for server in self.servers:
                if vm <= server:
                    return False
        return True

    def render(self):
        pass

    def step(self, action):
        vm_index, server_index = action[0], action[1]
        if self.can_move(vm_index, server_index):
            self.servers[server_index] -= self.vms[vm_index]
            self.vms[vm_index] = CpuMem(STOP, STOP)
        return self.get_state(), self.get_reward(), self.is_done(), {}

    def reset(self):
        self.vms = []
        self.servers = []
        for i in range(self.n):
            self.vms.append(CpuMem(1, 2))
        for i in range(self.m):
            self.servers.append(CpuMem(16, 32))
        self.was = 0
        return self.get_state()