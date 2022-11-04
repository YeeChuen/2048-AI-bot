import sys
import os
import numpy as np
import random
import copy
import time

class State2048:
    def __init__(self, board=None, score=0, prev=None, boardSize=4):
        self.score = score
        self.prev = prev
        self.board = board
        self.boardSize = boardSize

        if self.board is None:
            self.board = np.zeros((self.boardSize,self.boardSize))
            self.spawn()
            self.spawn()
        else:
            self.spawn()

    #Spawn random tile, 90% chance it's a 2 and 10% chance a 4
    def spawn(self):
        x = random.randint(0,self.boardSize-1)
        y = random.randint(0,self.boardSize-1)
        while self.board[y][x] != 0:
            x = random.randint(0,self.boardSize-1)
            y = random.randint(0,self.boardSize-1)

        value = random.randint(1,10)
        if value < 10:
            value = 2
        else:
            value = 4

        self.board[y][x] = value

    #array of values of legal moves, up:0, right:1, down:2, left:3 More efficient to just try a move and check if result is None
    def actions(self):
        res = []
        for i in range(4):
            if self.move(i) is not None:
                res.append(i)

        return res

    #up:0, right:1, down:2, left:3
    def move(self, direction):
        if direction == 0:
            res = self.moveUp(self.board)
            if np.array_equal(self.board, res[0]):
                return None
            return State2048(res[0], self.score + res[1], self, self.boardSize)

        if direction == 1:
            res = self.moveRight(self.board)
            if np.array_equal(self.board, res[0]):
                return None
            return State2048(res[0], self.score + res[1], self, self.boardSize)

        if direction == 2:
            res = self.moveDown(self.board)
            if np.array_equal(self.board, res[0]):
                return None
            return State2048(res[0], self.score + res[1], self, self.boardSize)

        if direction == 3:
            res = self.moveLeft(self.board)
            if np.array_equal(self.board, res[0]):
                return None
            return State2048(res[0], self.score + res[1], self, self.boardSize)


    def moveLeft(self, board):
        newBoard = np.zeros((self.boardSize,self.boardSize))
        score = 0

        for y in range(self.boardSize):
            firstEmpty = 0
            lastMerged = -1
            for x in range(self.boardSize):

                val = board[y][x]

                if val != 0:

                    if firstEmpty > 0 and lastMerged < firstEmpty-1 and newBoard[y][firstEmpty - 1] == val:
                        newBoard[y][firstEmpty-1] = val*2
                        score = score + val*2
                        lastMerged = firstEmpty - 1

                    else:
                        newBoard[y][firstEmpty] = val
                        firstEmpty = firstEmpty + 1

        return [newBoard, score]

    def moveRight(self, board):
        toFlip = copy.deepcopy(board)
        flipped = np.fliplr(toFlip)

        res = self.moveLeft(flipped)
        newBoard = np.fliplr(res[0])
        return [newBoard, res[1]]

    def moveUp(self, board):
        toRot = copy.deepcopy(board)
        rotated = np.rot90(toRot)

        res = self.moveLeft(rotated)
        newBoard = np.rot90(res[0], 3)
        return [newBoard, res[1]]

    def moveDown(self, board):
        toRot = copy.deepcopy(board)
        rotated = np.rot90(toRot)

        res = self.moveRight(rotated)
        newBoard = np.rot90(res[0], 3)
        return [newBoard, res[1]]

    def getScore(self):
        return self.score

    def getParent(self):
        return self.prev


    def print(self):
        for row in self.board:
            line = ""
            for val in row:
                line = line + str(int(val)) + "  "
            print(line)

    def checkGameOver(self):
        if 0 in self.board:
            return False

        for d in range(4):
            if self.move(d) is not None:
                return False

        return True

    #All rotations hash to the same value
    def __hash__(self):
        h = 0
        for r in range(4):
            h = h + hash(tuple(np.rot90(self.board, r).reshape(np.prod(self.board.shape))))
        return h % sys.maxsize

#Output to use in machine learning
def getStateToInput(state):
    out = np.zeros((self.boardSize,self.boardSize,17))

    for y in range(self.boardSize):
        for x in range(self.boardSize):

            if state.board[y][x] == 0:
                out[y][x][0] = 1
            else:
                out[y][x][int(np.log2(state.board[y][x]))] = 1
    return out

#Random playthrough of the game with output to console
def randomPlay():
    state = State2048(boardSize=4)
    while state.checkGameOver() is False:
        time.sleep(.5)
        os.system('cls||clear')
        state.print()

        newState = state.move(random.randint(0,3))
        if newState is not None:
            state = newState

    state.print()
    print("Game Over, you scored")
    print(state.score)


#For user to play 2048/test implementation
def main():
    state = State2048(boardSize=4)
    while state.checkGameOver() is False:
        #os.system('cls||clear')
        state.print()
        print("enter 'w', 'd', 's', or 'a' to move up, right, down, or left respectively")
        m = str(input())
        d = None
        if m == "w":
            d = 0
        if m == "d":
            d = 1
        if m == "s":
            d = 2
        if m == "a":
            d = 3

        if d is not None:
            newState = state.move(d)

            if newState is not None:
                state = newState

    state.print()
    print("Game Over, you scored")
    print(state.score)

if __name__ == "__main__":
    #randomPlay()
    main()
