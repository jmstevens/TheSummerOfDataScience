def ex33(number, itor):
    i = 0
    numbers = []

    while i < number:
        print "At the top i is %d" % i
        numbers.append(i)

        i += itor
        print "Numbers now: ", numbers
        print "At the bottom i is %d" % i

    print "The numbers: "

    for num in numbers:
        print num

def ex33_for(number, itor):
    i = 0
    numbers = []

    for i in range(0,number,itor):
        print "At the top i is %d" % i
        numbers.append(i)

        print "Numbers now: ", numbers
        print "At the bottom i is %d" % i

    print "The numbers: "

    for num in numbers:
        print num
ex33(1000, 500)
ex33_for(1000,500)
