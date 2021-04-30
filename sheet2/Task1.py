# creates a list contining values from 1-20
my_list = [(x + 1) for x in range(20)]
print('The list containing values from 1-20 is :- ', my_list)
# list comprehension that squares all odd values of list comprehension
my_list_odd_squared = [x*x if x % 2 else x for x in my_list]
print('The above list with odd values squared is :- ', my_list_odd_squared)
# request a number from user
number = []
for i in range(4):
    number.append(int(input('Enter number {}: '.format(i + 1))))
number.sort()
print('Sorted list of numbers entered is :- ', number)

