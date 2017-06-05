from sys import argv

script, first, second, third = argv

print "The script is called:", script
print "Your first variable is:", first
print "Your second variable is:", second
print "Your third variable is:", third

response = raw_input("Do you understand this? ")
question = raw_input("Do you have a question? ")

print 'So you said %r to my first question and %r to my second question' %(response, question)
