the_count = [1, 2, 3, 4, 5]
fruits = ['apples', 'oranges', 'pears', 'apricots']
change = [1, 'pennies', 2, 'dimes', 3, 'quarters']

# this first kind of for-loop goes through a list
for number in the_count:
    print "This is count %d" % number

# same as above
for fruit in fruits:
    print "A fruit of type: %s" % fruit

# also we can go through mixed lists too
# notice we have to use %r since we don't know what's in it
for i in change:
    print "I got %r" % i

# we can also build lists, first start with an empty one
elements = []

# then use the range function to do 0 to 5 counts
for i in range(0, 6):
    print "Adding %d to the list." % i
    # append is a function that lists understand
    elements.append(i)

# now we can print them out too
for i in elements:
    print "Element was: %d" % i

# list.append(elem) -- adds a single element to the end of the list. Common error: does not return the new list, just modifies the original.
# list.insert(index, elem) -- inserts the element at the given index, shifting elements to the right.
# list.extend(list2) adds the elements in list2 to the end of the list. Using + or += on a list is similar to using extend().
# list.index(elem) -- searches for the given element from the start of the list and returns its index. Throws a ValueError if the element does not appear (use "in" to check without a ValueError).
# list.remove(elem) -- searches for the first instance of the given element and removes it (throws ValueError if not present)
# list.sort() -- sorts the list in place (does not return it). (The sorted() function shown below is preferred.)
# list.reverse() -- reverses the list in place (does not return it)
# list.pop(index) -- removes and returns the element at the given index. Returns the rightmost element if index is omitted (roughly the opposite of append()).
