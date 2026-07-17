principal = str(input("Enter the Principal Amount($):")).strip().replace("$", "").replace(",", "")
principal = float(principal)
if principal < 0:
    print("Error, Please try again")
    exit()

intrest_rate = str(input("Enter the intrest rate(%): ")).strip().replace("%", "")
intrest_rate = float(intrest_rate)
if 0 < intrest_rate or intrest_rate > 30:
    print("Error, Please try again")
    exit()
    
loan_term = str(input("Enter the tenure(years): "))
loan_term = int(loan_term)
if 1 < loan_term or loan_term > 50:
    print("Error, Please try again")
    exit()

(monthly_rate) = (intrest_rate / 100) / 12
num_payments = loan_term * 12
monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** num_payments)/((1 + monthly_rate) ** num_payments - 1)
total_intrest = (monthly_payment * num_payments) - principal

print(f"========Loan Calculation========")
print(f"Principal:     ${principal}")
print(f"Annual Rate:    {intrest_rate}%")
print(f"Term/Tenure:    {loan_term} yrs")
print()
print(f"Monthly Payment: ${abs(monthly_payment):.2f}")
print(f"Total Intrest:   ${abs(total_intrest):.2f}")
print(f"Total Repaid:   ${abs(total_intrest+principal):.2f}")
print(f"================================")
