
import array as arr
sFile  = open("worldcities.csv", 'r')

instrs = sFile.readlines()
i = len(instrs)
print(instrs[1])
lat = []
lng = []
country = []
target = []
for k in range(i):
    #begin
    if(instrs[k] != '\n'):
        #begin
        instrTemp = instrs[k].replace(',', ' ')
        instrTemp = instrTemp.replace('\n', '')
        #instrTemp = re.sub('\s+',' ',instrTemp)
        words = instrTemp.split(' ')
        lat.append(words[0])# = words[0]
        lng.append(words[1])
        country.append(words[2])
        target.append(words[3])
        #end
    #end