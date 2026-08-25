import csv

# Open the students.csv file in read mode
with open("students.csv", "r") as file:

    # Read the CSV file
    reader = csv.DictReader(file)

    # Dictionary to store grade frequency
    grade_count = {}

    # Read each student's data
    for student in reader:
        grade = student["Grade"]

        # Count the grade
        if grade in grade_count:
            grade_count[grade] += 1
        else:
            grade_count[grade] = 1


# Display the grade frequency
print("Grade Frequency:")
for grade, count in grade_count.items():
    print(grade, ":", count)


# Write the summary into a new file
with open("summary.txt", "w") as file:

    file.write("Student Grade Summary\n")
    file.write("=====================\n")

    for grade, count in grade_count.items():
        file.write(f"Grade {grade}: {count} students\n")

print("\nSummary saved to summary.txt")