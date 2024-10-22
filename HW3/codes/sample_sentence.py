arr = [920, 2406, 1961, 3667, 107, 7, 4903, 655, 4870, 2456]

file1 = 'data/test.txt'
with open(file1, 'r') as f:
    s1s = f.readlines()

file2 = 'output_pt_random.txt'
with open(file2, 'r') as f:
    s2s = f.readlines()

file3 = 'output_ft_random.txt'
with open(file3, 'r') as f:
    s3s = f.readlines()

print("Ground_truth:")
print("=" * 50)
count = 0
for i in arr:
    print(s1s[i], end="")
    count += 1
    if count != 10:
        print("-" * 50)
print("=" * 50)

print()

print("Tfmr_scratch:")
print("=" * 50)
count = 0
for i in arr:
    print(s2s[i], end="")
    count += 1
    if count != 10:
        print("-" * 50)
print("=" * 50)

print()

print("Tfmr_finetune:")
print("=" * 50)
count = 0
for i in arr:
    print(s3s[i], end="")
    count += 1
    if count != 10:
        print("-" * 50)
print("=" * 50)