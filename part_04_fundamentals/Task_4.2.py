class Vehicle:
    def __init__(self, brand, model, mileage=0): 
        self.brand = brand
        self.model = model
        self.mileage = mileage
        self.rented = False
    
    def rent_out(self):
        self.rented = True

    def return_vehicle(self):
        self.rented = False

    def add_mileage(self, accumulated_mileage):
        self.mileage += accumulated_mileage

    def display_info(self):
        print(f"Brand: {self.brand}")
        print(f"Model: {self.model}")
        print(f"Mileage: {self.mileage}")
        print(f"Rented: {self.rented}")


v1 = Vehicle("Ferrari", "Daytona SP3")
v1.display_info()
v1.rent_out()
v1.add_mileage(150)
v1.display_info()

print()

v2 = Vehicle("Porsche", "911 GT3 RS")
v2.display_info()