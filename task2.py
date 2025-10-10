student = {"name":"","age":0,"marks":[]}
student["name"] = input("Enter name:")
student["age"] = int(input("Enter age:"))
for i in range(3):
    marks = int(input("Enter marks:"))
    student["marks"].append(marks)

def average(marks):
    return sum(marks)/len(marks)
average_marks = average(student["marks"])
print("Average marks:", average_marks)
if average_marks>85:
    print("Excellent Performance")

elif average_marks>=60:
    print("Good Performance")

else:
    print("Needs Improvement")
        
for key in student:
            print(f"{key}:{student[key]}")

