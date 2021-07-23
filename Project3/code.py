import re
s = "as224 "
print(re.findall('\b?[a-zA-Z]{2}[0-9]{3}$',s))
# import numpy as np
# a = [[1,2,3,4,5,6,7,8,9,0],[11,12,13,14,15,16,17,18,19,10],[91,2,3,4,6,4,15,10,9,8]]
# x = np.array(a)
# print(x[-3:-1,0])


