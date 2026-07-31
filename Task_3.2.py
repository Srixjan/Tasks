
def highest_dat(steps: list[int]) -> int:
    return max(steps)

def lowest_day(steps: list[int]) -> int:
    return min(steps)

def rank_days(steps: list[int]) -> list[int]:
    rank = sorted(steps, reverse=True)
    return rank

def add_days(steps: list[int], new_value: int) -> list[int]:
    steps.append(new_value)
    return steps

new_value=12333
steps = [8200, 10500, 6300, 12000, 9800, 7100, 11200]
print(add_days(steps, new_value))