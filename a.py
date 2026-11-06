a = [1, 2]
b = [1, 2]
c = a

print(a is b)  # False
print(a == b)  # True

print(a is c)  # True
print(a == c)  # True

print(id(a))  # 2116236119360
print(id(b))  # 2116236989312
print(id(c))  # 2116236119360
