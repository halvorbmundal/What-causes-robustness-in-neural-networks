import matplotlib.pyplot as plt

a = "10 µm – 1971\n\
6 µm – 1974\n\
3 µm – 1977\n\
1.5 µm – 1981\n\
1 µm – 1984\n\
800 nm – 1987\n\
600 nm – 1990\n\
350 nm – 1993\n\
250 nm – 1996\n\
180 nm – 1999\n\
130 nm – 2001\n\
90 nm – 2003\n\
65 nm – 2005\n\
45 nm – 2007\n\
32 nm – 2009\n\
22 nm – 2012\n\
14 nm – 2014\n\
10 nm – 2016\n\
7 nm – 2018\n\
5 nm – ~2020\n\
3 nm – ~2021\n\
2 nm – ~2024"

x=[]
y=[]
post = {"µm":1000, "nm":1}
for i in a.split("\n"):
    hm = i.split(" ")
    y.append(float(hm[0])*post[hm[1]])
    x.append(hm[3])

plt.plot(x, y)
plt.yscale("log")
plt.show()

