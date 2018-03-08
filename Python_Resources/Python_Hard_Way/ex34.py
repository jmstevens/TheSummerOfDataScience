animals = ['bear','python','peacock','kangaroo','whale','platypus']
for i in range(0,len(animals)):
    print "The animal at %r is a %s" % (i, animals[i])
    if i == 0:
        print "The %rst is a %s" % (i + 1, animals[i])
    elif i == 1:
        print "The %rnd is a %s" % (i + 1, animals[i])
    elif i == 2:
        print "The %rrd is a %s" % (i + 1, animals[i])
    else:
        print "The %rth is a %s" % (i + 1, animals[i])
