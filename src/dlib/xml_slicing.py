import re

# regex that identify the part section of the xml
REG_PART = re.compile("part name='[0-9]+'")

# regex that identify all the numbers (name, x, y) inside the part section
REG_NUM = re.compile("[0-9]+")


def slice_xml(in_path, out_path, parts):
    '''creates a new xml file stored at [out_path] with the desired landmark-points.
    The input xml [in_path] must be structured like the ibug annotation xml.'''
    file = open(in_path, "r")
    out = open(out_path, "w")
    pointSet = set(parts)

    for line in file.readlines():
        finds = re.findall(REG_PART, line)

        # find the part section
        if len(finds) <= 0:
            out.write(line)
        else:
            # we are inside the part section
            # so we can find the part name and the landmark x, y coordinates
            name, x, y = re.findall(REG_NUM, line)

            # if is one of the point i'm looking for, write in the output file
            if int(name) in pointSet:
                out.write(f"      <part name='{int(name)}' x='{x}' y='{y}'/>\n")

    out.close()
