class Book:
    def set_details(self, title, author, genre):
        self.title = title
        self.author = author
        self.genre = genre
        self.status = "Unread"

    def mark_as_read(self):
        self.status = "Read"


    def display_info(self):
        print(f"Title: {self.title}")
        print(f"Author: {self.author}")
        print(f"Genre: {self.genre}")
        print(f"Status: {self.status}")
        print("\n")

b1 = Book()
b1.set_details("The Adventures of Feluda","Satyajit Ray", "Detective")
b1.mark_as_read()
b1.display_info()

b2 = Book()
b2.set_details("Sherlock Holmes", "Arthur Conan Doyle", "Detective")
b2.display_info()