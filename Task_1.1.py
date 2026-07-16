try:
    expense_1 = str(input("Enter expense 1:$")).strip().replace("$", "").replace(",", "")
    expense_1 = float(expense_1)
    
except ValueError:
    print("Not a number, please enter a valid number (e.g. 100, 10.50)")

else:
    if expense_1 < 0:
        print("Number must be postive")
        exit()

try:
    expense_2 = str(input("Enter expense 2:$")).strip().replace("$", "").replace(",", "")
    expense_2 = float(expense_2)
    
except ValueError:
    print("Not a number, please enter a valid number (e.g. 100, 10.50)")

else:
    if expense_2 < 0:
        print("Number must be postive")
        exit()

try:
    expense_3 = str(input("Enter expense 3:$")).strip().replace("$", "").replace(",", "")
    expense_3 = float(expense_3)
    
except ValueError:
    print("Not a number, please enter a valid number (e.g. 100, 10.50)")

else:
    if expense_3 < 0:
        print("Number must be postive")
        exit()

total  = expense_1 + expense_2 + expense_3
average = total/3

print("=========Summary========")
print(f"Expense 1:${expense_1:.2f}")
print(f"Expense 2:${expense_2:.2f}")
print(f"Expense 3:${expense_3:.2f}")
print(f"Total    :${total}")
print(f"Average  :${average}")
print("========================")