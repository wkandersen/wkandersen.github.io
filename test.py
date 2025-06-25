import time
import os

text = "hej mit navn er william, kom og tag"
string = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.?! "

empty = ""
for i in range(len(text)):
    for j in range(len(string)):
        os.system('cls' if os.name == 'nt' else None)  # Clear screen
        print(empty + string[j])
        time.sleep(0.01)
        if text[i] == string[j]:
            empty += string[j]
            break
