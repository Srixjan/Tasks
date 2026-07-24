count = 1
true_code = 17102004

while(count <= 5):
    try:
        code = input("Enter the callibration code: ")
        code = int(code)

        if (code != true_code):
            print(f"Invalid code. Attempt {count} out of 5")
        else:
            print("Callibration Succesful!!")
            break # DONT USE EXIT() <- THIS KILLS THE ENTIRE SCRIPT USE BREAK TO CLOSE APP.

        if (count == 3):
            print("Warning!! 2 attempts remaining.")
        elif(count == 5):
            print("Too many attempts! System locked!")


    except ValueError:
        print(f"Invalid Input! must be a number. Attempt {count} of 5")

    count = count + 1
