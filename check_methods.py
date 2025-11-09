import ast

with open('lgd_mapping/matching/exact_matcher.py', 'r', encoding='utf-8') as f:
    tree = ast.parse(f.read())

cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ExactMatcher'][0]
methods = [m.name for m in cls.body if isinstance(m, ast.FunctionDef)]

print('Methods in ExactMatcher class:')
for m in methods:
    print(f'  {m}')

print(f'\nTotal methods: {len(methods)}')
print(f'\n_should_use_hierarchical_matching in methods: {"_should_use_hierarchical_matching" in methods}')
print(f'_create_hierarchical_mapping_safe in methods: {"_create_hierarchical_mapping_safe" in methods}')
print(f'_process_entities_hierarchical in methods: {"_process_entities_hierarchical" in methods}')
