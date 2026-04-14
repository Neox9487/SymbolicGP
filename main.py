import numpy as np
import matplotlib.pyplot as plt
from gp_engine import (
    generate_random_tree, mutate, crossover, 
    final_simplify
)

def target_function(x):
    return 6*(x**3) - 20*(x**2)

X_SCOPE = 5
POP_SIZE = 1000
MAX_DEPTH = 10
LOSS_THRESHOLD = 15.0
MAX_GENS = 1000

x_train = np.linspace(-X_SCOPE, X_SCOPE, 50)
y_train = target_function(x_train) + np.random.normal(0, 0.5, size=x_train.shape)

population = [generate_random_tree(depth=3) for _ in range(POP_SIZE)]
best_overall_tree = None
min_fitness = float('inf')
gen = 0
stagnation_counter = 0

while min_fitness > LOSS_THRESHOLD and gen < MAX_GENS:
    fitness_scores = []
    for tree in population:
        pred = tree.evaluate(x_train)
        mse = np.mean((pred - y_train)**2)
        complexity = str(tree).count('(') * 1.2
        depth_penalty = max(0, tree.get_depth() - MAX_DEPTH) * 100
        score = mse + complexity + depth_penalty
        fitness_scores.append(score if np.isfinite(score) else 1e10)
    
    fitness_scores = np.array(fitness_scores)
    best_idx = np.argmin(fitness_scores)
    
    if fitness_scores[best_idx] < min_fitness:
        if min_fitness - fitness_scores[best_idx] > 0.1:
            stagnation_counter = 0
        min_fitness = fitness_scores[best_idx]
        best_overall_tree = population[best_idx].copy()
    else:
        stagnation_counter += 1

    if gen % 10 == 0:
        print(f"Gen {gen:03d} | Loss: {min_fitness:.2f} | Stagnation: {stagnation_counter}")

    def select_parent():
        idx = np.random.choice(POP_SIZE, 5)
        return population[idx[np.argmin(fitness_scores[idx])]]

    next_gen = [best_overall_tree.copy()]
    
    if stagnation_counter > 25:
        while len(next_gen) < POP_SIZE * 0.7:
            next_gen.append(mutate(select_parent().copy()))
        while len(next_gen) < POP_SIZE:
            next_gen.append(generate_random_tree(depth=4))
        stagnation_counter = 0
    else:
        while len(next_gen) < POP_SIZE:
            r = np.random.rand()
            if r < 0.5:
                next_gen.append(crossover(select_parent(), select_parent()))
            elif r < 0.9:
                next_gen.append(mutate(select_parent().copy()))
            else:
                next_gen.append(generate_random_tree(depth=3))
    
    population = next_gen
    gen += 1

x_test = np.linspace(-X_SCOPE*2, X_SCOPE*2, 200)
y_true = target_function(x_test)
y_pred = best_overall_tree.evaluate(x_test)
final_expr = final_simplify(best_overall_tree)

print("-" * 30)
print(f"Finished at Gen {gen} | Final Loss: {min_fitness:.2f}")
print(f"Final Expression: {final_expr}")

plt.figure(figsize=(10, 6))
plt.plot(x_test, y_true, 'g--', label="Target")
plt.plot(x_test, y_pred, 'r-', linewidth=2, label="GP prediction")
plt.scatter(x_train, y_train, color='blue', s=10, alpha=0.5, label="Data")
plt.title(f"Formula: {final_expr}")
plt.legend()
plt.grid(True)
plt.show()