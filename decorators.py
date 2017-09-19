def type_check(correct_type):
    def check_type(old_func):
        def new_func(*args, **kwargs):
            if type(args[0]) == correct_type:
                return old_func(args[0])
            else:
                return "Bad type"
        return new_func
    return check_type

@type_check(int)
def times2(num):
    return num*2

print(times2(2))
print(times2('Not A Number'))

@type_check(str)
def first_letter(word):
    return word[0]

print(first_letter('Hello World'))
print(first_letter(['Not', 'A', 'String']))