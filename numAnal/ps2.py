import math

def q1f(x):
	return x**3  * (x-1) - 10

def q1findRoot(a, b):
	eps = 10**-10

	mid = ((a+b)/2)
	eval = q1f(mid)

	iter = 0
	while abs(eval) >= eps:
		if eval > 0:
			b = mid
		elif eval < 0:
			a = mid
		else:
			return mid

		mid = ((a+b)/2)
		eval = q1f(mid)
		print(iter, mid, eval)
		iter+=1
	print("total number of iterations: " + str(iter))
	return mid

def q3comp():
	x = 0.1
	for i in range(4):
		print('{:.20f}'.format(x))
		x = -0.5*x**2


def q5comp():
	p3 = lambda x: x - x**3/6
	p5 = lambda x: x - x**3/6 + x**5/120
	n = 2
	for i in range(3):
		x = math.pi/n
		print('$\\pm \\frac{\\pi}{' + str(n) + "}$" + ' & $\\pm ' + str(x) + '$ & $\\pm ' + str(p3(x)) +  '$ & $\\pm ' + str(p5(x)) + '$ & $\\pm ' + str(math.sin(x)) + "$\\\\")
		n += 1
def main():
	q1findRoot(2,3)
	q3comp()
	q5comp()

if __name__ == "__main__":
	main()
