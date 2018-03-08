def cheese_and_crackers(cheese_count, boxes_of_crackers):
    print "You have %d cheeses" % cheese_count
    print "You have %d boxes of crackers!" % boxes_of_crackers
    print "Man that's enough for a party!"
    print "Get a blanket.\n"

print "We can just give the functions numbers directly:"
cheese_and_crackers(20,30)

print "OR, we can use variables from our script:"
amount_of_cheese = 10
amount_of_crackers = 50

cheese_and_crackers(amount_of_cheese,amount_of_crackers)

print "We can even do math inside too:"
cheese_and_crackers(10 + 20, 5 + 6)

print "And we can combine the two, variables and math:"
cheese_and_crackers(amount_of_cheese + 100, amount_of_crackers + 1000)

def beer_and_brats(beer_count, brats_count):
    print "You have %d gallons of beer" % beer_count
    print "You have %d pounds of brats" % brats_count
    print "Man thats enough to get LIT!!!!"
    print "BRUUUUUHHHHHHHH"

beer_and_brats(1000,10000)
beer_and_brats(amount_of_cheese * 10000, amount_of_crackers / 2)
