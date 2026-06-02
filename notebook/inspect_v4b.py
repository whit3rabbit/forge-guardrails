import json

nb = json.load(open('notebook/toolcall_verifier_training_production_colab_v4.ipynb'))

# Print cell 27 source
cell27 = nb['cells'][27]
print("=== Cell 27 full source ===")
print(''.join(cell27['source']))

# Print cell 45 source
cell45 = nb['cells'][45]
print("\n=== Cell 45 full source ===")
print(''.join(cell45['source']))