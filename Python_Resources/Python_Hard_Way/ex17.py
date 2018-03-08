from sys import argv
from os.path import exists

script, from_file, to_file = argv

print "Copying from %s to %s" % (from_file, to_file)

# we could do these two on one line, how?

indata = open(from_file).read()
# indata = in_file.read()

print "The input file is %d bytes long" % len(indata)

print "Does the output file exist? %r" % exists(to_file)
print "Read, hit RETURN to continue, CTRL-C to abort."

raw_input()

out_file = open(to_file,'w')
out_file.write(indata)

print "Alright, all done."

out_file.close()
# in_file.close()

# Shorter Version of the Script 
# from sys import argv
# from os.path import exists
#
# script, from_file, to_file = argv
# indata = open(from_file).read()
#
# if exists(to_file):
#     print "Read, hit RETURN to continue, CTRL-C to abort."
#     raw_input()
#     open(to_file,'w').write(indata)
#     print ".....done"
#     # to_file.close()s
