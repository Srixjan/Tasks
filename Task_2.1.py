class InvalidScoreError(Exception):
    """Value is Invalid as it doenst confines between 0 and 100"""
    pass

try:
    grade_score = input("Enter the grade score: ")
    grade_score = int(grade_score)

except ValueError:
    print("Value must contain numbers!")

else:
    if (grade_score < 0 or grade_score > 100):
        raise InvalidScoreError(f"ERROR")
    exit()
    
print(f"{grade_score}")