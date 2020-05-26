a = [0,0,1,1,2,3,5,8,13,21,34,55,89,144]#13
b = [0 for  i in a]
i = 0
for z in b:
    if i<2:
        b[i] = 0
    elif i==2:
        b[i] = 1
    else:
        b[i] = a[i-1] + a[i-2]
    i += 1
if a == b:
    print("Yes, it is a Fibonacci Number")
else:
    print("Sorry, the given number is not a Fibonacci Number")
