# Day 1 - Python Basics
# 60 Days of Data Science

# Taking user input
name = input("Enter your name: ")
age = int(input("Enter your age: "))
monthly_salary = float(input("Enter your monthly salary: "))

# Calculate yearly salary
yearly_salary = monthly_salary * 12

# Store the data in a dictionary
person = {
    "name": name,
    "age": age,
    "monthly_salary": monthly_salary,
    "yearly_salary": yearly_salary
}

# Display the data
print("\n--- Personal Details ---")
print("Name:", person["name"])
print("Age:", person["age"])
print("Monthly Salary:", person["monthly_salary"])
print("Yearly Salary:", person["yearly_salary"])