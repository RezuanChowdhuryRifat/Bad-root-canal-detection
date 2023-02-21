import os

dirname = ""
for txt_in in os.listdir(dirname):
    with open(os.path.join(dirname, txt_in), 'r') as f:
        # Don't read entire file since we
        # are looping line by line
        #infile = f.read()# Read contents of file
        result = []
        for line in f:  # changed to file handle
            line = line.rstrip() # remove trailing '\n'
            # Only split once since you're only check the first word
            words = line.split(" ", maxsplit = 1)
            word = words[0]  # word 0 may change
            if word == "15":
                word = word.replace('15', '0')
            elif word=="16":
                word = word.replace('16', '1')
            elif word == "17":
                word = word.replace('17', '2')
            elif word == "18":
                word = word.replace('18', '3')
            else:
                pass
            # Update the word you modified
            words[0] = word  # update word 0
            # save new line into results
            # after converting back to string
            result.append(" ".join(words))

    with open(os.path.join(dirname, txt_in), 'w') as f:
        # Convert result list to string and write to file
        outfile = '\n'.join(result)
        f.write(outfile)
