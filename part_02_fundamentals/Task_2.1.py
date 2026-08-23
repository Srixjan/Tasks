class InvalidScoreError(Exception):
    """Marks should be between 0 and 100 bitch"""
    pass

try:
    marks = input("Enter the marks needed: ")
    marks = int(marks)
    if(marks < 0 or marks > 100):
        raise InvalidScoreError("Marks should lie within 0 and 100") # this raise wont work, if i commit out the except invalid block because raise searches for  the except block.

except ValueError:
    print("Value must be in numbers!!")

except InvalidScoreError as e:
    print(e)
else:
    if marks >= 90 and marks <= 100:
        print("A")
    elif marks >= 75 and marks <= 89:
        print("B")
    elif marks >= 50 and marks <=74:
        print("C")
    else:
        print("F")