import numpy as np
import matplotlib.pyplot as plt
from gp_engine import generate_random_tree, mutate, crossover, final_simplify

def target_function(x):
    return 3*(x**3) - 5*(x**2) + 20*x - 30

X_SCOPE = 7
POP_SIZE = 1200
MAX_DEPTH = 10
LOSS_THRESHOLD = 10.0
MAX_GENS = 2000
ELITE_SIZE = int(POP_SIZE * 0.05)
STAGNATION_LIMIT = 30

x_train = np.linspace(-X_SCOPE, X_SCOPE, X_SCOPE*10)
y_train = target_function(x_train) + np.random.normal(0, 0.5, size=x_train.shape)

population = [generate_random_tree(depth=4) for _ in range(POP_SIZE)]
best_overall_tree = None
min_fitness = float('inf')
gen = 0
stagnation_counter = 0

while min_fitness > LOSS_THRESHOLD and gen < MAX_GENS:
    fitness_scores = []
    for tree in population:
        pred = tree.evaluate(x_train)
        mse = np.mean((pred - y_train)**2)
        complexity = str(tree).count('(') * 0.4
        depth_penalty = max(0, tree.get_depth() - MAX_DEPTH) * 100
        score = mse + complexity + depth_penalty
        fitness_scores.append(score if np.isfinite(score) else 1e9)
    
    fitness_scores = np.array(fitness_scores)
    sorted_indices = np.argsort(fitness_scores)
    current_best_score = fitness_scores[sorted_indices[0]]

    if current_best_score < min_fitness - 0.05:
        min_fitness = current_best_score
        best_overall_tree = population[sorted_indices[0]].copy()
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    if gen % 10 == 0:
        print(f"Gen {gen:03d} | Loss: {min_fitness:.2f} | Stagnation: {stagnation_counter}")

    next_gen = [population[i].copy() for i in sorted_indices[:ELITE_SIZE]]
    
    if stagnation_counter > STAGNATION_LIMIT:
        while len(next_gen) < POP_SIZE:
            r = np.random.rand()
            if r < 0.6:
                next_gen.append(generate_random_tree(depth=np.random.randint(3, 7)))
            else:
                p = population[np.random.choice(sorted_indices[:10])].copy()
                next_gen.append(mutate(mutate(p)))
        stagnation_counter = 0
    else:
        while len(next_gen) < POP_SIZE:
            r = np.random.rand()
            if r < 0.7:
                p1 = population[np.random.choice(sorted_indices[:50])]
                p2 = population[np.random.choice(sorted_indices[:50])]
                next_gen.append(crossover(p1, p2))
            elif r < 0.95:
                p = population[np.random.choice(sorted_indices[:100])].copy()
                next_gen.append(mutate(p))
            else:
                next_gen.append(generate_random_tree(depth=3))
    
    population = next_gen
    gen += 1

x_test = np.linspace(-X_SCOPE*1.5, X_SCOPE*1.5, 200)
y_true = target_function(x_test)
y_pred = best_overall_tree.evaluate(x_test)
final_expr = final_simplify(best_overall_tree)

print("-" * 30)
print(f"Finished at Gen {gen} | Final Loss: {min_fitness:.2f}")
print(f"Final Expression: {final_expr}")

plt.figure(figsize=(10, 6))
plt.plot(x_test, y_true, color='black', linestyle='--', linewidth=3, alpha=0.7, label="Target", zorder=5)
plt.plot(x_test, y_pred, 'r-', linewidth=2, label="GP prediction", zorder=4)
plt.scatter(x_train, y_train, color='blue', s=10, alpha=0.5, label="Data")
plt.title(f"Predicted Formula - {final_expr}")
plt.legend()
plt.grid(True)
plt.show()