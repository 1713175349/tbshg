
def readconfig(fn:str):
    with open(fn,"r") as fp:
        buf=""
        out={}
        while 1:
            line = fp.readline()
            if line == "":
                break
            if line.strip() == "" or line.strip()[0]=="#":
                continue
            if "#" in line:
                line=line[:line.index("#")]

            buf += line.strip()
            if line.strip()[-1]=="\\":
                buf=buf[:-1]
                continue
            k,v=[j.strip() for j in buf.split(":")]
            out[k]=v
            buf=""
    return out
            