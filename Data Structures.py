#Assigning the Elements to the Lists:

# intializing the list
list = []
# assigning the elements
no_of_ele = int(input("Enter the number of elemnts to append"))
for i in range(no_of_ele):
    x = input("Enter the number :")
    list.append(x)

#printing the final output
print(list)


#Accessing the Elements from the Tuple:

# intializing the tuple
tuple = (1,2,3,4,5)
no_of_ele = int(input("Enter the number of elements to Access from tuple: "))
if no_of_ele <= len(tuple):
    for i in range(no_of_ele):
        index = int(input("Enter the index of the element: "))
        if index <= 4:
            print(tuple[index])
        else:
            print("Index is not in range")
else:
    print("Enter the number less the length of tuple")
    
    
#Deleting the Elements from Dictionaries:

# Python code to demonstrate removal of dict. pair using del 

# Initializing dictionary 
test_dict = {"Arushi" : 22, "Anuradha" : 21, "Mani" : 21, "Haritha" : 21} 

# Printing dictionary before removal 
print ("The dictionary before performing remove is : " + str(test_dict)) 

# Using del to remove a dict elements
x = input("Enter the Element to remove :")
del test_dict[x] 

# Printing dictionary after removal 
print ("The dictionary after remove is : " + str(test_dict)) 
