class OrderError(Exception):
    pass

def check_order(order: dict) -> bool:
    item = order.get("item")
    if not item:
        raise OrderError("item is missing!")

    quantity = order.get("quantity")
    if quantity is None:
        raise OrderError("quantity is missing!")
    if not isinstance(quantity, int) or  isinstance(quantity, bool): # needs to be changed.
        raise OrderError("quanity must be in whole numbers")
    if quantity < 1:
        raise OrderError("quantity must be greater than 0!")

    return True

order1 = {"item": "pen", "quantity": 5}
order2 = {"item": "", "quantity": 5}
order3 = {"item": "pen", "quantity": 0}

orders = [order1, order2, order3]

for order in orders:
    try:
        check_order(order)
        print(f"{order}: valid")
    except OrderError as e:
        print(f"{order}: invalid - {e}")