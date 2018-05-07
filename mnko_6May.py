__author__ = 'Бригада 7 (Меденцій, Сніжко, Токовенко, Чаус), КА-41, 2018'

from numpy import zeros, array
from numpy.linalg import matrix_rank

def print_list(lst, name):
    print('%s:' % name)
    for el in lst:
        print('  %s' % str(el))
    print()

class MNKO (object):
    def __init__ (self, X, y_measured):
        self.X = X
        self.n = X.shape[0]
        self.m = X.shape[1]
        self.X_T_X = self.X.T.dot(self.X)
        self.y_measured = y_measured.reshape((self.n, 1))
        self.X_T_y = self.X.T.dot(self.y_measured)

        self.theta_hat = []
        self.RSS_arr = []
        self.Cp_arr = []
        self.FPE_arr = []
        self.launch()

    def launch(self):
        self.H_inv = array([1 / self.X_T_X[0,0]]).reshape((1, 1))
        self.theta_hat.append(array([self.X_T_y[0,0] / self.X_T_X[0,0]]).reshape((1,1)))
        self.theta_hand_hat = []
        self.beta = []
        self.y_est = [self.X[:, :1].dot(self.theta_hat[0])]

        for i in range(1, self.m):
            self.eta = self.X_T_X[i,i]
            self.h = self.X_T_X[0:i, i].reshape((i, 1))
            self.a = self.H_inv.dot(self.h).reshape((i, 1))
            self.beta.append(self.eta - float(self.h.T.dot(self.a)))
            self.gamma = self.X_T_y[i, 0]
            self.theta_hand_hat.append((self.gamma - float(self.h.T.dot(self.theta_hat[i-1]))) / (self.beta[-1]))
            self.theta_star_hat = self.theta_hat[i-1] - self.theta_hand_hat[-1] * self.a
            self.H_inv = self.get_H_inv(i)
            self.theta_hat.append(self.get_theta_hat(i))
            self.y_est.append(self.X[:, :(i+1)].dot(self.theta_hat[i]))

    def get_H_inv(self, i):
        res = zeros((i+1, i+1))
        res[:i, :i] = self.H_inv + (self.a.dot(self.a.T)) / self.beta[i-1]
        res[:i, i:] = -(self.a) / self.beta[i-1]
        res[i:, :i] = -(self.a.T) / self.beta[i-1]
        res[i:, i:] = 1 / self.beta[i-1]
        return res

    def get_theta_hat(self, i):
        res = zeros((i+1, 1))
        res[:i, 0] = self.theta_star_hat[:, 0]
        res[i, 0] = self.theta_hand_hat[i-1]
        return res

    def print_input_data(self):
        for arr in [(self.X, 'X'), (self.y_measured, 'y_measured'), (self.X_T_X, 'X_T_X'),
                    (self.X_T_y, 'X_T_y')]:
            print_list([arr[0][i, :] for i in range(arr[0].shape[0])], arr[1])

    def print_results(self):
        for s in range(self.m):
            print('s = %i' % (s+1))
            print('Theta:')
            print(self.theta_hat[s])
            print('Output:')
            print(self.y_est[s])
            print('RSS(%i) = %lf' % (s+1, self.RSS(s+1)))
            print('Cp(%i) = %lf' % (s+1, self.Cp(s+1)))
            print('FPE(%i) = %lf' % (s+1, self.FPE(s+1)))
            print()

    def RSS (self, s):
        if s < 1:
            return None
        elif s == 1:
            return float(self.y_measured.T.dot(self.y_measured) - self.y_measured.T.dot(self.y_est[0]))
        else:
            return self.RSS(s-1) - pow(self.theta_hand_hat[s-2], 2) * self.beta[s-2]

    def Cp(self, s):
        return self.RSS(s) + 2*s

    def FPE(self, s):
        return self.RSS(s) * (self.n + s) / (self.n - s)

    def MSE(self):
        return ((self.X.dot(self.theta_hat) - self.y_measured)**2).sum() / (2 * self.n)

def demonstrate():
    X = array([[3, 1, 2], [4, -3, -2.5], [10, 0, -4], [-5, 1.2, 3]])
    y_measured = array([10.3, -9.21, -1.9, 6.48])
    obj1 = MNKO(X, y_measured)
    obj1.print_input_data()
    obj1.print_results()

if __name__ == '__main__':
    demonstrate()
