name = 'Joel M. Stevens'
age = 26 # my age
height = 72 # in inches
weight = 214 # in lbs
eyes = 'Blue'
teeth = 'White'
hair = 'Brown'

print "Let's talk about %s." % name
print "He's %d inches tall." % height
print "He's %d pounds heavy." % weight
print "Actually that's not too heavy."
print "He's got %s eyes and %s hair." % (eyes, hair)
print "His teeth are usually %s depending on the coffee." % teeth

# this line is tricky, try to get it exactly right
print "If I add %d, %d, and %d I get %d." % (
    age, height, weight, age + height + weight)
# converting to metric
metric_height = round(height * 2.54)
metric_weight = round(weight * 0.453592)
print "If you were to convert his height to cm he is %s tall" % metric_height
print "If you were to convert his weight to kg he weighs %s " % metric_weight
