a = [1, 2, 3]
b = a

for elem in a:
    print(id(elem))


print(id(a))
print(id(b))
print(id(a) == id(b))  # True
