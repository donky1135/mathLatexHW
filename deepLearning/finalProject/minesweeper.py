from copy import deepcopy
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import math
import time
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def printBoard(board):

    h = len(board)
    for i in range(h):
        for j in range(len(board[0])):
            if j != len(board[0]) - 1 and j != 0:
                print(" %2d," % board[i][j], end='')
            else:
                if j == len(board[0])-1:
                    print(" %2d]\n" % board[i][j], end='')
                else:
                    print("[%2d," % board[i][j], end='')

class game_environment():
    board = [[]]
    anotatedBoard = [[]]
    W = 0
    H = 0
    def showboard(self):
        printBoard(self.board)
        print()
        printBoard(self.anotatedBoard)
    def __init__(self, H,W,numMines, seed=None):
        self.W = W
        self.H = H
        self.board = [ [0]*W for _ in range(H) ]
        self.anotatedBoard = [ [0]*W for _ in range(H) ]
        if seed != None:
            random.seed(seed)

        mines = 0
        while (mines < numMines):
            x = random.randint(0,W-1)
            y = random.randint(0,H-1)
            if self.board[y][x] == 0:
                self.board[y][x] = 1
                mines += 1

        for i in range(H):
            for j in range(W):
                if self.board[i][j] == 1:
                    self.anotatedBoard[i][j] = -1
                else:
                    mineCount = 0
                    for k in range(3):
                        for l in range(3):
                            ni = i + k - 1
                            nj = j + l - 1
                            if ni < H and ni >= 0 and nj < W and nj >= 0:
                                if self.board[ni][nj] == 1:
                                    mineCount += 1
                    self.anotatedBoard[i][j] = mineCount


    def query(self, x,y,inside):
        if x < 0 or x >= self.W or y < 0 or y >= self.H:
            return {}

        if self.anotatedBoard[y][x] != 0:
            return {(x,y): self.anotatedBoard[y][x]}
        else:
            inside.update({(x,y):0})
            for i in range(3):
                for j in range(3):
                    if (x + i - 1,y + j - 1) not in inside:
                        inside.update(self.query(x + i - 1,y + j - 1, inside))
            return inside
    def firstQuery(self):
        c = 0
        while True:
            x = random.randint(0,self.W-1)
            y = random.randint(0, self.H - 1)
            if self.anotatedBoard[y][x] == 0:
                    return (x,y),self.query(x,y,{})
            c += 1

        return (0,0),self.query(0,0,{})

class logicBot():
    curBoard = None
    cells_remaining = set()
    inferred_safe = set()
    inferred_mine = set()
    clue_number = {}
    H = 0
    W = 0
    numMines = 0
    def __init__(self, h, w, numMines, seed=None, hasBoard=False):
        self.numMines = numMines
        self.H = h
        self.W = w
        if seed != None:
            random.seed(seed)
        if not hasBoard:
            self.curBoard = game_environment(h, w, numMines)
        else:
            self.curBoard = hasBoard
        for y in range(h):
            for x in range(w):
                self.cells_remaining.update({(x,y)})

    def GameState(self):
        curGameState = deepcopy(self.curBoard.anotatedBoard)
        for y in range(self.H):
            for x in range(self.W):
                if (x,y) not in self.clue_number:
                    curGameState[y][x] = -2
        return curGameState

    def showGameState(self):
        printBoard(self.GameState())

    def playGame(self, quiet=True, otherBoard = None):
        # printBoard(self.curBoard.anotatedBoard)
        cell = (0,0)
        for y in range(self.H):
            for x in range(self.W):
                self.cells_remaining.update({(x,y)})
        self.inferred_safe = set()
        self.inferred_mine = set()
        self.clue_number = {}

        firstMove = True
        if otherBoard != None:
            self.curBoard = otherBoard
        while self.numMines > len(self.inferred_mine) or len(self.cells_remaining) != 0 or len(self.inferred_safe) != 0:
            endCon = self.numMines > len(self.inferred_mine) or len(self.cells_remaining) != 0 or len(self.inferred_safe) != 0
            # print(self.inferred_safe)
            # print(self.inferred_mine)
            if not quiet:
                self.showGameState()

                print(self.inferred_mine)

            if firstMove:
                cell = self.playMove(firstMove)
                firstMove = False
            else:
                cell = self.playMove()
            if not quiet:
                print("choose to click:",cell)

            if -1 in self.clue_number.values():
                if not quiet:
                    print("hit a mine")
                return 0
        return 1




    def playMove(self, first=False):
        # print(self.cells_remaining)
        if first:
            cell, self.clue_number = self.curBoard.firstQuery()
            self.cells_remaining = self.cells_remaining - set(self.clue_number.keys())
            # print(self.clue_number)

        else:
            if not self.inferred_safe:
                # print(list(self.cells_remaining)[random.randint(0, len(self.cells_remaining) - 1)])
                # print("nothing safe ", self.cells_remaining)
                if len(self.cells_remaining) > 1:
                    self.inferred_safe.update({list(self.cells_remaining)[random.randint(0, len(self.cells_remaining) - 1)]})
                elif self.cells_remaining:
                    self.inferred_safe.update({next(iter(self.cells_remaining))})
                else:
                    return (0,0)
            # print(self.inferred_safe)
            cell = self.inferred_safe.pop()
            # print(cell)
            # print("final call",cell)
            self.clue_number.update(self.curBoard.query(cell[0], cell[1], {}))
            self.cells_remaining = self.cells_remaining - set(self.clue_number.keys())


        for e in self.clue_number:
            clueNum = self.clue_number[e]
            unkownNeighNum = 0
            unkownNeigh = set()
            minedNeigh = 0
            safeNeigh = 0
            neigh = 0
            for i in range(3):
                for j in range(3):
                    if not(i == 1 and j == 1):
                        # testPair = tuple(map(lambda x, y: x + y, e, (i,j))) #weird line
                        testPair = (e[0] + i - 1, e[1] + j - 1)
                        if 0 <= testPair[0] and testPair[0] < self.W and 0 <= testPair[1] and testPair[1] < self.H :
                            if (testPair in self.cells_remaining):
                                unkownNeigh.update({testPair})
                                unkownNeighNum += 1
                            if (testPair in self.inferred_mine):
                                minedNeigh += 1
                            if (testPair in self.inferred_safe or testPair in self.clue_number) and (testPair not in self.cells_remaining):
                                safeNeigh += 1

                            neigh += 1
            # print(self.inferred_mine)
            if clueNum - minedNeigh == unkownNeighNum and len(self.inferred_mine) != self.numMines:
                self.inferred_mine.update(unkownNeigh)
                self.cells_remaining = self.cells_remaining - unkownNeigh
            if (neigh-clueNum) - safeNeigh == unkownNeighNum:
                self.inferred_safe.update(unkownNeigh)
                self.cells_remaining = self.cells_remaining - unkownNeigh

        return cell

    # class NetworkBot(nn.Module):
    #     def __init__(self):
    #         super(NetworkBot, self).__init__()
    #         self.embed = nn.Embedding(11,11)
    #         self.layer1 = nn.Conv2d


testbot = logicBot(5,5,3)
# testbot.curBoard.board =  [[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, -1, 0],[0, -1, 0, 0, 0],[-1, 0, 0, 0, 0]]
# testbot.curBoard.anotatedBoard = [[0, 0, 0, 0, 0],[0, 0, 1, 1, 1],[1, 1, 2, -1, 1],[2, -1, 2, 1, 1],[-1, 2, 1, 0, 0]]
# testbot.curBoard.board =  [[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 1, 0, 1],[0, 1, 0, 0, 0],[0, 0, 0, 0, 0]]
# testbot.curBoard.anotatedBoard = [[0, 0, 0, 0, 0],[0, 1, 1, 2, 1],[1, 2, -1, 2, -1],[1, -1, 2, 2, 1],[1, 1, 1, 0, 0]]
# testbot.curBoard.anotatedBoard =  [[0, 1, -1, 2, 1],[0, 1, 3, -1, 2],[0, 0, 2, -1, 2],[0, 0, 1, 1, 1],[0, 0, 0, 0, 0]]
# testbot.curBoard.board = [[0, 0, 1, 0, 0],[0, 0, 0, 1, 0],[0, 0, 0, 1, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]
# [ 0,  0,  1, -1, -1]
# [ 1,  1,  1,  2,  2]
# [-1,  1,  0,  0,  0]
# [ 1,  1,  0,  0,  0]
# [ 0,  0,  0,  0,  0]


test = testbot.curBoard


# for i in range(5):
#     for j in range(5):
#         # print(test.anotatedBoard[j][i], end='')
#         print(i,j,test.query(i,j,{}))
#     # print()

# printBoard(testbot.curBoard.anotatedBoard)
# print()
# print(testbot.playGame(False))
# testbot.showGameState()
# print(testbot.inferred_mine)
count = 0
bots = []
boards = []
h = 16
w = 16
m = 40

bot = logicBot(h,w,m,True)
print("experiment")
testnum = 1000


for a in range(testnum):
    boards.append(game_environment(h,w,m,a))
    # boards[a].showboard()
    # print()

print("testing")
for z in range(testnum):
    val = bot.playGame(True, boards[z])
    # if val == 0:
    #     printBoard(boards[z].anotatedBoard)
    #     print()
    #     print(z)
    #     bot.playGame(False, boards[z])
    #     print(val)
    count += val
print(count)
print(count/testnum)
