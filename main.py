import torch
import matplotlib.pyplot as plt
from gp_engine import LGPEngine, evolve, get_final_expression

def target_func(x):
    return 3*(x**3) - 5*(x**2) + 20*x - 30

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
POP_SIZE = 8192
MAX_GENS = 2000
NUM_REGS = 6
MAX_INSTR = 50

print(f"Running LGP on {DEVICE}")

x_train = torch.linspace(-5, 5, 200, device=DEVICE)
y_train = target_func(x_train)

engine = LGPEngine(POP_SIZE, NUM_REGS, MAX_INSTR, DEVICE)

for gen in range(MAX_GENS):
    preds = engine.evaluate(x_train)
    mse = torch.mean((preds - y_train)**2, dim=1)
    mse = torch.nan_to_num(mse, nan=1e12, posinf=1e12)
    
    best_val, best_idx = torch.min(mse, 0)
    
    if gen % 100 == 0:
        print(f"Gen {gen:04d} | Best MSE: {best_val.item():.4f}")
    
    if best_val < 0.5:
        break
        
    evolve(engine, mse)

best_pred = engine.evaluate(x_train)[best_idx].detach().cpu().numpy()
x_plot = x_train.cpu().numpy()
y_true = y_train.cpu().numpy()

best_ind = engine.pop[best_idx].cpu().numpy()
final_expr = get_final_expression(best_ind, NUM_REGS)

plt.figure(figsize=(12, 7))
plt.plot(x_plot, y_true, color='black', linestyle='--', linewidth=3, alpha=0.7, label="Target", zorder=10)
plt.plot(x_plot, best_pred, color='red', linestyle='-', linewidth=2, label="LGP GPU prediction", zorder=5)
plt.scatter(x_plot, y_true, color='blue', s=15, alpha=0.4, label="Data", zorder=1)
plt.title(f"Formula: {final_expr}")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('result.png', dpi=300, bbox_inches='tight')
plt.show()

print("-" * 30)
print(f"Final Formula: {final_expr}")