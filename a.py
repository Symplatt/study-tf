original = [[1, 2], [3, 4]]  # 假设ori还是指向4000
shallow = original[:]  # sha也依然指向5000

original[0][0] = 99
original[0][1] = 88

print(original)  # [[99, 2], [3, 4]]
print(shallow)  # [[99, 2], [3, 4]] —— 浅拷贝也修改了
