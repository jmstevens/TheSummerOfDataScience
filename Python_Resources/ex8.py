formatter = "%r %r %r %r" # Formatter for the text to print

print formatter % (1, 2, 3, 4) # Prints the variable formatter
# which takes four arguments in numeric form
print formatter % ("one", "two", "three", "four")
# Same as line three in text form
print formatter % (True, False, False, True)
# Same as a line 3 and 6 prints the formatter
print formatter % (formatter, formatter, formatter, formatter)
# Prints the formatter in teh formatter
print formatter % (
    "I had this thing.",
    "That you could type up right.",
    "But it didnt sing.",
    "So I said goodnight."
)
# Should I use %s or %r for formatting?
# You should use %s and only use %r for
# getting debugging information about something.
# The %r will give you the "raw programmer's" version of variable,
# also known as the "representation."
