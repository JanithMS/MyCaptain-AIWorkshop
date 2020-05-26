noe = int(input("Enter the number of Elements: "))
mylist = [0 for i in range (0,noe)]
for i in range (0,noe):
    mylist[i] = int(input("Enter the number :"))
print("Input List: "+str(mylist))
i = 0
x = 0
while x<noe:
    if mylist[i]<0:
        del mylist[i]
        x += 1
    else:
        i += 1
        x +=1
print("Output List: "+str(mylist))
