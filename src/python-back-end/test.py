from collections import namedtuple

# NT4 = namedtuple("NT4", "a b c", defaults=(1, 2, None))

PTZ = namedtuple("PTZ", ["pan", 'tilt', "zoom", 'lrf'], defaults=(0, 0, 0, None))

# n1 = NT4(5)
# n2 = NT4()

# print(f'n1 = {n1}')
# print(f'n2 = {n2}')

p1 = PTZ(1, 2, 3)
print(f'p1 = {p1}')