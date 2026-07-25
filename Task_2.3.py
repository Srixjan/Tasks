# C -> F
def celsius_to_farenheit(celsius: float) -> float:
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

try:
    c = input("Enter the degree in °C: ")
    c = float(c)
    if (c < 0 or c > 100):
        print("°C must be between 0 and 100")
        exit()

except ValueError:
    print("°C must be numeric!!")

else:
    f = celsius_to_farenheit(c)
    print(f"{c}°C is {f} in °F")


# F -> C
def fahrenheit_to_celsius(far: float) -> float:
    cel = (far - 32) * 5/9
    return cel

try:
    f = input("Enter the degree in °F: ")
    f = float(f)
    if (f < 32 or f > 212):
        print("°F must be between 32 and 212")
        exit()

except ValueError:
    print("°F must be numeric!")

else:
    c = fahrenheit_to_celsius(f)
    print(f"{f}°F is {c} in °C")

# Doubled Value
dr = input("Enter the value to be doubled: ")
dr = int(dr)
double_reading = lambda d_r: d_r * 2
ans = double_reading(dr)
print(f"{dr} value of doubled is {ans}")

value = input("Enter the degree is abover or below freezing!")
value = float(value)

is_above_freezing = lambda v: "Not Freezing" if v > 0 else "Freezing"
print(f" -> {is_above_freezing(value)}")
