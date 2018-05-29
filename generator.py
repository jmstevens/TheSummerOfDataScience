def gen_squares(max_root):
    root = 0
    while root < max_root:
        yield root**2
        root += 1
squares = gen_squares(5)
print(type(squares))
print(list(squares))

def words_in_test(path):
    BUFFER_SIZE = 2**20
    def read(): return handle.read(BUFFER_SIZE)
    def normalize(chunk): return chunk.lower().rstrip(',!.\n')
    with open(path) as handle:
        buffer = read()
        start, end = 0, -1
        while True:
            for match in re.finiter(r'[])

def house_records(path):
    with open(path) as lines:
        record = {}
        for line in lines:
            if line == '\n':
                yield record
                record = {}
