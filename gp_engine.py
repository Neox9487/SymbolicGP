import numpy as np
import sympy

operators = {
    'add': np.add,
    'sub': np.subtract,
    'mul': np.multiply,
    'pow': lambda x, y: np.power(np.abs(x), y)
}

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        if self.value == 'x': 
            return x
        if isinstance(self.value, (int, float)): 
            return np.full_like(x, self.value)
        
        l_val = self.left.evaluate(x)
        r_val = self.right.evaluate(x)
        
        try:
            if self.value == 'pow':
                r_val = np.round(np.clip(r_val, -3, 3))
            res = operators[self.value](l_val, r_val)
            return np.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
        except:
            return np.zeros_like(x)

    def copy(self):
        left_copy = self.left.copy() if self.left else None
        right_copy = self.right.copy() if self.right else None
        return Node(self.value, left_copy, right_copy)

    def __str__(self):
        if self.value == 'x': return "x"
        if isinstance(self.value, (int, float)): return f"{self.value:.2f}"
        if self.value == 'pow':
            exponent = self.right.evaluate(np.array([1.0]))[0]
            return f"({self.left}**{int(np.round(exponent))})"
        return f"({self.left} {self.value} {self.right})"

    def get_depth(self):
        if not self.left and not self.right: return 1
        l = self.left.get_depth() if self.left else 0
        r = self.right.get_depth() if self.right else 0
        return 1 + max(l, r)

def generate_random_tree(depth=3):
    if depth <= 1 or np.random.rand() < 0.2:
        return Node('x') if np.random.rand() > 0.4 else Node(np.random.randint(-10, 10))
    
    op = np.random.choice(list(operators.keys()))
    if op == 'pow':
        return Node(op, generate_random_tree(depth-1), Node(float(np.random.randint(0, 4))))
    return Node(op, generate_random_tree(depth-1), generate_random_tree(depth-1))

def mutate(tree):
    if np.random.rand() < 0.3 and isinstance(tree.value, (int, float)):
        tree.value += np.random.normal(0, 2.0)
        return tree
    
    if tree.value == 'pow':
        if np.random.rand() < 0.5:
            tree.left = mutate(tree.left)
        else:
            tree.right.value = float(np.random.randint(0, 4))
        return tree
    
    r = np.random.rand()
    if r < 0.2:
        return generate_random_tree(depth=2)
    
    if tree.left and np.random.rand() < 0.5:
        tree.left = mutate(tree.left)
    elif tree.right:
        tree.right = mutate(tree.right)
    return tree

def crossover(parent1, parent2):
    child = parent1.copy()
    if child.left and parent2.left:
        child.left = parent2.left.copy()
    elif child.right and parent2.right:
        if child.value != 'pow':
            child.right = parent2.right.copy()
    return child

def final_simplify(node):
    x = sympy.Symbol('x')
    raw_str = str(node).replace('add', '+').replace('sub', '-').replace('mul', '*').replace('pow', '**')
    try:
        expr = sympy.sympify(raw_str)
        return sympy.expand(expr).n(4) 
    except:
        return raw_str