# create a function that squares elements
def square(num):
    return num * num


# list initialization
my_list = [i for i in range(10)]
my_list_squared = [square(i) for i in my_list]
print('My squared List:-', my_list_squared)


def cal_sum(some_list):
    if len(some_list) == 0:
        return 0
    else:
        return some_list[0] + cal_sum(some_list[1:])


list_sum = sum(my_list)
print('Sum with python inbuilt sum() is: ', list_sum)
list_recursion = cal_sum(my_list)
print('sum with recursive calculation: ', list_recursion)
