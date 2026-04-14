import numpy as np
import sympy

operators = {
    'add': np.add,
    'sub': np.subtract,
    'mul': np.multiply,
    'pow': lambda x, y: np.power(np.abs(x), np.clip(y, -4, 4))
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
            res = operators[self.value](l_val, r_val)
            return np.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
        except:
            return np.zeros_like(x)

    def copy(self):
        return Node(self.value, self.left.copy() if self.left else None, self.right.copy() if self.right else None)

    def __str__(self):
        if self.value == 'x': return "x"
        if isinstance(self.value, (int, float)): return f"{self.value:.2f}"
        if self.value == 'pow':
            try:
                exp = self.right.evaluate(np.array([1.0]))[0]
                return f"({self.left}**{int(np.round(exp))})"
            except:
                return f"({self.left}**0)"
        return f"({self.left} {self.value} {self.right})"

    def get_depth(self):
        if not self.left and not self.right: return 1
        return 1 + max(self.left.get_depth() if self.left else 0, self.right.get_depth() if self.right else 0)

    def get_all_nodes(self):
        nodes = [self]
        if self.left: nodes.extend(self.left.get_all_nodes())
        if self.right: nodes.extend(self.right.get_all_nodes())
        return nodes

def generate_random_tree(depth=3):
    if depth <= 1 or np.random.rand() < 0.2:
        return Node('x') if np.random.rand() > 0.4 else Node(float(np.random.randint(-15, 16)))
    op = np.random.choice(list(operators.keys()))
    if op == 'pow':
        return Node(op, generate_random_tree(depth-1), Node(float(np.random.randint(0, 4))))
    return Node(op, generate_random_tree(depth-1), generate_random_tree(depth-1))

def mutate(tree):
    r = np.random.rand()
    if isinstance(tree.value, (int, float)):
        if r < 0.6:
            tree.value += np.random.normal(0, 5.0)
            return tree
    
    if r < 0.2:
        return generate_random_tree(depth=3)
    
    if tree.left and np.random.rand() < 0.5:
        tree.left = mutate(tree.left)
    elif tree.right:
        tree.right = mutate(tree.right)
    return tree

def crossover(p1, p2):
    c = p1.copy()
    nodes_c = c.get_all_nodes()
    nodes_p2 = p2.get_all_nodes()
    target = np.random.choice(nodes_c)
    replacement = np.random.choice(nodes_p2).copy()
    target.value, target.left, target.right = replacement.value, replacement.left, replacement.right
    return c

def final_simplify(node):
    x = sympy.Symbol('x')
    raw = str(node).replace('add', '+').replace('sub', '-').replace('mul', '*').replace('pow', '**')
    try:
        return sympy.expand(sympy.sympify(raw)).n(4)
    except:
        return raw