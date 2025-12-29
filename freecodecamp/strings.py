# python strings

# string methods

my_string = ' this is my string '
upper_case_string = my_string.upper()
lower_case_string = my_string.lower()
striped_string = my_string.strip()
print(striped_string)

my_str = 'hello world'
changer = my_str.replace('hello', 'goodbye')
is_all_lower = my_str.islower()
print(is_all_lower, changer)

#f strings

name = 'Cyper'
age = 41
print(f'My name is {name} and I am {age} years old.')


# slicing strings
slice_string = '0123hello456'
print(slice_string[4:9])
#slicing does not effect the original string
print(slice_string)








