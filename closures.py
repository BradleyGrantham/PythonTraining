def multiplier_of(x):
    def multiplywith(y):
        return x*y
    return multiplywith



multiplywith5 = multiplier_of(5)
multiplywith4 = multiplier_of(4)
print(multiplywith4(9))
print(multiplywith5(9))