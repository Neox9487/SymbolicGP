import torch
import sympy

class LGPEngine:
    def __init__(self, pop_size, num_regs=6, max_instr=40, device='cuda'):
        self.pop_size = pop_size
        self.num_regs = num_regs
        self.max_instr = max_instr
        self.device = device
        self.pop = torch.randint(0, num_regs, (pop_size, max_instr, 4), device=device)
        self.pop[:, :, 1] = torch.randint(0, 3, (pop_size, max_instr), device=device)

    def evaluate(self, x):
        n_pts = x.shape[0]
        regs = torch.zeros((self.pop_size, self.num_regs, n_pts), device=self.device)
        regs[:, 0, :] = x.expand(self.pop_size, n_pts)
        
        for i in range(1, self.num_regs):
            const_val = (i - self.num_regs//2) * 5.0
            regs[:, i, :] = torch.full((self.pop_size, n_pts), const_val, device=self.device)

        for i in range(self.max_instr):
            instr = self.pop[:, i, :]
            dst, op, s1, s2 = instr[:, 0], instr[:, 1], instr[:, 2], instr[:, 3]
            
            v1 = regs[torch.arange(self.pop_size), s1]
            v2 = regs[torch.arange(self.pop_size), s2]
            
            res_add = v1 + v2
            res_sub = v1 - v2
            res_mul = v1 * v2
            
            ops_stack = torch.stack([res_add, res_sub, res_mul], dim=1)
            regs[torch.arange(self.pop_size), dst] = ops_stack[torch.arange(self.pop_size), op]
            
        return torch.clamp(regs[:, 0, :], -1e6, 1e6)

def evolve(engine, fitness):
    pop_size, max_instr, _ = engine.pop.shape
    idx = torch.argsort(fitness)
    
    elites = engine.pop[idx[:pop_size // 10]]
    
    p1 = elites[torch.randint(0, len(elites), (pop_size,))]
    p2 = elites[torch.randint(0, len(elites), (pop_size,))]
    
    cp = torch.randint(0, max_instr, (pop_size, 1, 1), device=engine.device)
    mask = torch.arange(max_instr, device=engine.device).view(1, -1, 1) < cp
    offspring = torch.where(mask, p1, p2)
    
    mut_mask = torch.rand(offspring.shape, device=engine.device) < 0.08
    mut_vals = torch.randint(0, engine.num_regs, offspring[mut_mask].shape, device=engine.device)
    offspring[mut_mask] = mut_vals
    offspring[:, :, 1] %= 3
    
    offspring[:len(elites)] = elites
    engine.pop = offspring

def get_final_expression(best_individual, num_regs=6):
    x = sympy.Symbol('x')
    regs = [0.0] * num_regs
    regs[0] = x
    for i in range(1, num_regs):
        regs[i] = float((i - num_regs//2) * 5.0)
    
    for i in range(best_individual.shape[0]):
        dst, op, s1, s2 = map(int, best_individual[i])
        v1, v2 = regs[s1], regs[s2]
        if op == 0: res = v1 + v2
        elif op == 1: res = v1 - v2
        elif op == 2: res = v1 * v2
        regs[dst] = res

    final_expr = regs[0]
    try:
        return sympy.expand(final_expr)
    except:
        return final_expr