try:
    temp = input(str("Enter a temperature(C): "))
    temp = float(temp)

except ValueError:
    print("Enter a valid temperature")

else:
    if temp < -50 or temp > 61:
        print("Temp value must be between -50 and 60!")
        print("Please try again!")
        exit()


try:
    humidity = input(str("Enter humidity(%): "))
    humidity = float(humidity)

except ValueError:
    print("Enter a valid humidity")

else:
    if humidity < 0 or humidity > 100:
        print("Humidity value range should be between 0 to 100%")
        print("Please try again!")
        exit()


try: 
    pressure = input(str("Enter pressure(hPa): "))
    pressure = float(pressure)

except ValueError:
    print("Enter a valid pressure value!")

else: 
    if pressure < 950 or pressure > 1050:
        print("Pressure range must lie between 950 - 1050")
        print("Please try again!")
        exit()


print(f"========== SENSOR READING ==========")
print(f"Temperature:    {temp}C")
print(f"Humidity:       {humidity}%")
print(f"Pressure:       {pressure}hpa")
print("Status: ✓ All readings valid")
print("====================================")