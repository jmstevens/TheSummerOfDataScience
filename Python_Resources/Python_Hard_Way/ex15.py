from sys import argv # Imports argv from the sys package

script, filename = argv # Sets the arguments for argv to the two parameters

txt = open(filename) # Makes a file object from the given filename

print "Here's your file %r:" % filename # Prints a file name based on the user input
print txt.read() # for the file object txt read its contents

print "Type the filename again:" # Print that file name again
file_again = raw_input("> ") # User input with the prompt "> "

txt_again = open(file_again) # Create the file object txt_again

print txt_again.read() # Print out that file object

txt_again.close()
txt.close()
