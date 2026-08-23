def split_eve_odd(numbers: tuple[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    even = []
    odd = []

    for nos in numbers:
        if nos % 2 == 0:
            even.append(nos)
        else:
            odd.append(nos)

    t_even ,t_odd = tuple(even), tuple(odd)
    print(f"The values inside the odd tuples are ->{t_odd}\nand The values inside the even tuples are -> {t_even}") 

def count_values(values: tuple, target: int):
    cnt = values.count(target)
    print(f"The value {target} is in tuple {cnt} times.")

def find_postion(values: tuple, target: int):
    idx = values.index(target)
    print(f"The index of {target} is {idx}")



values = (10, 20, 10, 30, 10, 40)
numbers = (1, 2, 3, 4, 5, 6, 7, 8, 9)
split_eve_odd(numbers)
count_values(values, 10)
find_postion()