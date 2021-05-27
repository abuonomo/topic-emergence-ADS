

with open('input.txt', 'r') as inp:
    lines = inp.readlines()

result = []
for l in lines:
    result.append(l.split(':')[0])

with open ('out.txt', 'w') as outp:
    for r in result:
        outp.write(f"{r}\n")
