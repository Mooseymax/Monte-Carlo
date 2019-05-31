import numpy as np
import csv

'''
Geometric Brownian Motion
Describe as dS = uS dt + oS dW(t)

Where:
S is the stock price
u is the drift coefficient
o is the diffusion coefficient
W(t) is the Brownian Motion

"drift coefficient" represents the mean of returns over X period
"diffusion coefficient" represents the standard deviation of those returns

W(t) is the random portion of the equation.
Each Brownian Increment - W(i) - is computed as:

Standard random variable - z(i) - from a normal distribution - N(0,1) - With mean 0 and standard deviation 1
multiplied by square root of the time increment - SQR[delta t(i)]
'''

def Brownian(seed, N):
    np.random.seed(seed)
    dt = 1. / N
    b = np.random.normal(0., 1., int(N)) * np.sqrt(dt)
    W = np.cumsum(b)
    return W, b

def GBM(So, mu, sigma, W, T, N):
    t = np.linspace(0., 1., int(N) + 1)
    S = []
    S.append(So)
    for i in range(1,int(N+1)):
        drift = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i-1]
        S_temp = So*np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t

def daily_return(adj_close):
    returns = []
    for i in range(0, len(adj_close)-1):
        today = adj_close[i+1]
        yesterday = adj_close[i]
        daily_return = (today - yesterday)/yesterday
        returns.append(daily_return)
    return returns

def EM(So, mu, sigma, b, T, N, M):
    dt = M * (1/N)  # EM step size
    L = N / M
    wi = [So]
    for i in range(0,int(L)):
        Winc = np.sum(b[(M*(i-1)+M):(M*i + M)])
        w_i_new = wi[i]+mu*wi[i]*dt+sigma*wi[i]*Winc
        wi.append(w_i_new)
    return wi, dt

'''
Closed-form solution of GMB (using stocks)
S(t) = So e^(mu-[1/2]o^2) t + oW(t)

So:     initial stock price
mu:     returns (drift coefficient)
sigma:  volatility (diffusion coefficient)
W:      brownian motion
T:      time period
N:      number of increments

'''

seed = 22
N = 2.**8
T = 1.

#start = '2017-01-01'
#end = '2017-12-31'
#df = quandl.get('WIKI/AMZN', start_date = start, end_date = end)
#adj_close = df['Adj. Close']

adv_perf = []
mod_perf = []
cau_perf = []

new = True
with open('Performance.csv','r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        if(new == True):
            new = False
        else:
            adv_perf.append(float(row[0]))
            mod_perf.append(float(row[1]))
            cau_perf.append(float(row[2]))

So_a = adv_perf[0]
So_m = mod_perf[0]
So_c = cau_perf[0]

returns_a = daily_return(adv_perf)
returns_b = daily_return(mod_perf)
returns_c = daily_return(cau_perf)

#252 trade dates in one year usually


monte = []
'''for i in range(1,2):
    seed = i
    M = 1
    L = N / M
    mu = np.mean(returns_c) * 414.            # drift coefficient
    sig = np.std(returns_c) * np.sqrt(414.)   # diffusion coefficient
    W = Brownian(seed, N)[0]                # brownian movement
    b = Brownian(seed, N)[1]                # brownian increment
    output = EM(So_c, mu, sig, b, T, N, M)[0]
    monte.append(output)
'''
# Monte carlo simulation
for i in range(1,1000):
    seed = i
    M = 1
    mu = np.mean(returns_c) * 414.            # drift coefficient
    sig = np.std(returns_c) * np.sqrt(414.)   # diffusion coefficient
    W = Brownian(seed, N)[0]                # brownian movement
    b = Brownian(seed, N)[1]                # brownian increment
    output = GBM(So_c, mu, sig, W, T, N)[0]
    monte.append(output)

'''
j = 0
printable = []
for i in range(0,len(monte[0])):
    empty = ''
    for x in monte:
        empty += str(x[j]) + ','
    printable.append(empty)
    j += 1

for x in printable:
    print(x)

'''

# Output, final figure
for i in monte:
    print(i[len(i)-1])