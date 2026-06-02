import json

nb = json.load(open('notebook/toolcall_verifier_training_production_colab_v4.ipynb'))
print("v4 cells:", len(nb['cells']))

# Check cell indices for key sections
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if '## 8. Train classifier' in src:
        print(f"Cell {i}: '## 8. Train classifier'")
    if '## 13. Recommended ablation' in src:
        print(f"Cell {i}: '## 13. Recommended ablation'")
    if 'Read the latest T4 run' in src:
        print(f"Cell {i}: 'Read the latest T4 run'")
    if 'Recent T4 results' in src:
        print(f"Cell {i}: 'Recent T4 results'")

# Now find and print the relevant sections to update
# Section 1: intro (cell 0)
cell0 = nb['cells'][0]
src0 = cell0['source']
for i,line in enumerate(src0):
    if 'Read the latest T4 run' in line:
        print(f"\nCell 0 intro section (lines {i}-{i+12}):")
        for j in range(i, min(i+12, len(src0))):
            print(f"  {j}: {repr(src0[j])}")
        break

# Section 2: cell 27 (## 8)
cell27 = nb['cells'][27]
src27 = ''.join(cell27['source'])
if '## 8. Train classifier' in src27:
    idx = src27.find('## 8. Train classifier')
    print(f"\nCell 27 section (chars {idx}-{idx+500}):")
    print(src27[idx:idx+500])
    print("---")

# Section 3: cell 45 (## 13)
cell45 = nb['cells'][45]
src45 = ''.join(cell45['source'])
if '## 13. Recommended ablation' in src45:
    idx = src45.find('## 13. Recommended ablation')
    print(f"\nCell 45 section (chars {idx}-{idx+500}):")
    print(src45[idx:idx+500])
    print("---")