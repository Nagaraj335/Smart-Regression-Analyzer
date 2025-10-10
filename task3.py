import getpass
credentials = {"username": "admin", "password": "1234"}  


for attempt in range(3):
    input_username = input("Enter username: ")
    input_password = getpass.getpass("Enter password: ")
    if input_username == credentials["username"] and input_password == credentials["password"]:
        print("Login successful")
        break
    else:
        print(f"Invalid credentials. Attempts left: {2 - attempt}")
else:
    print("Login failed, too many attempts")
