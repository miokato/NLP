import numpy as np
import sys
import time
import os


class LifeGame:
    def __init__(self, L=50, p=0.3, survive=(2, 3), birth=(3,)):
        self.L = L
        self.survive = survive
        self.birth = birth
        lattice = np.random.random((self.L+2, self.L+2))
        self.lattice = lattice < p
        self.lattice[0, :] = False
        self.lattice[-1, :] = False
        self.lattice[:, 0] = False
        self.lattice[:, -1] = False

    def update(self):
        os.system('clear')
        print('\n')
        l = ''
        for y in range(1, self.L+1):
            for x in range(1, self.L+1):
                if self.lattice[x, y]:
                    l += '%️️'
                else:
                    l += '#️'
            l += '\n'
        print(l)
        time.sleep(0.1)

    def progress(self):
        L = self.L
        Tmax = 2000
        t = 0
        while t < Tmax:
            self.update()
            next_sites = []

            # 端に到達した時の挙動
            self.lattice[0, 0] = self.lattice[L, L]
            self.lattice[0, self.L+1] = self.lattice[L, 1]
            self.lattice[L+1, 0] = self.lattice[1, L]
            self.lattice[L+1, L+1] = self.lattice[1, 1]
            for m in range(1, self.L+1):
                self.lattice[m, self.L+1] = self.lattice[m, 1]
                self.lattice[m, 0] = self.lattice[m, self.L]
                self.lattice[0, m] = self.lattice[self.L, m]
                self.lattice[self.L+1, m] = self.lattice[1, m]

            # 近隣の活動状況で自分の活動を変更する
            for m in range(1, L+1):
                for n in range(1, L+1):
                    if self.lattice[m, n]:
                        neighber = np.sum(self.lattice[m-1:m+2, n-1:n+2])-1
                        if neighber in self.survive:
                            next_sites.append((m, n))
                    else:
                        neighber = np.sum(self.lattice[m-1:m+2, n-1:n+2])
                        if neighber in self.birth:
                            next_sites.append((m, n))
            self.lattice[:] = False
            for next_site in next_sites:
                self.lattice[next_site] = True


if __name__ == '__main__':
    lg = LifeGame()
    lg.progress()