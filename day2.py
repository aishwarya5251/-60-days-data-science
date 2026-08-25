# Day 2: Logic Building
# Program to calculate student average and classify the result

# Function to calculate average marks
def calculate_average(marks):
    total = 0

    # Loop through all marks
    for mark in marks:
        total += mark

    # Calculate average
    average = total / len(marks)
    return average


# Function to classify the student's result
def classify_grade(average):
    if average >= 75:
        return "Distinction"
    elif average >= 40:
        return "Pass"
    else:
        return "Fail"


# Take student name
name = input("Enter student name: ")

# Take number of subjects
n = int(input("Enter number of subjects: "))

marks = []

# Take marks using a loop
for i in range(n):
    mark = float(input(f"Enter marks for subject {i + 1}: "))
    marks.append(mark)

# Calculate average using the function
average = calculate_average(marks)

# Classify result using the function
result = classify_grade(average)

# Display the result
print("\n----- Student Result -----")
print("Student Name:", name)
print("Marks:", marks)
print("Average Marks:", round(average, 2))
print("Result:", result)