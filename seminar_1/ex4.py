import numpy
speed = [32,111,138,28,59,77,97]

x=numpy.mean(speed)
y=numpy.var(speed)

for i in len(speed):
    a = speed[i] - x
    print(a)

