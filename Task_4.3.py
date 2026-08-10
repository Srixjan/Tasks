class Product:
    store_name = "Bright Mart"
    total_products = 0
    def __init__(self, name, price):
        self.name = name
        self.price = price
        Product.total_products += 1 

    def display_info(self):
        print(f"Store: {Product.store_name}")
        print(f"Product: {self.name}")
        print(f"Price: ₹{self.price}")
        print(f"Total Products Created: {Product.total_products}")

p1 = Product("NoteBook", 50)
p1.store_name="Solo man"
p1.display_info()

print()
p2 = Product("Pencil Bag", 250)
p2.display_info()
print()
p3 = Product("Geometrry Box", 500)
p3.display_info()