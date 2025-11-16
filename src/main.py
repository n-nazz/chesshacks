from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import torch

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

def FindVal(ctx,searchDepth,fanOut):

    if (searchDepth == 0):
        pass
        #positionEncoding  = encode(board)

        #return magnus(positionEncoding)


    legal_moves = list(ctx.board.generate_legal_moves())

    if not legal_moves:
        ctx.logProbabilities({})
        if (ctx.board.turn):
            return -3300
        else:
            return 3300
    
    #rating each possible move
    PredMoveValues = []
    for move in legal_moves:
        ctx.board.push(move)

        #positionEncoding  = encode(board)

        #PredMoveValues.append(magnus(positionEncoding))

        ctx.board.pop()

    PredMoveValues_Tens = torch.Tensor(PredMoveValues)
    WhiteMoveProbs_tens = torch.nn.functional.softmax(PredMoveValues_Tens)
    WhiteMoveProbs = WhiteMoveProbs_tens.tolist()
    # Normalize so probabilities sum to 1

    if (ctx.board.turn):
        move_probs = {
            move: WhiteMoveProb
            for move, WhiteMoveProb in zip(legal_moves, WhiteMoveProbs)
        }
    else:
        move_probs = {
            move: 1 - WhiteMoveProb 
            for move, WhiteMoveProb in zip(legal_moves, WhiteMoveProbs)
        }
    chosenMoves = random.choices(legal_moves, weights=move_probs, k=fanOut)

    nextPositionValues = []

    for move in chosenMoves:
        ctx.board.push(move)

        nextPositionValues.append(FindVal(ctx,searchDepth-1,fanOut))

        ctx.board.pop()
    
    nextPositionValues = torch.Tensor(nextPositionValues)
    
    if (ctx.board.turn):
        return nextPositionValues[torch.argmax(nextPositionValues)]
    else:
        return nextPositionValues[torch.argmax(nextPositionValues)]


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")
    
    searchDepth = 3
    fanOut = 5

    PredMoveValues = []
    for move in legal_moves:
        ctx.board.push(move)

        #positionEncoding  = encode(board)

        #PredMoveValues.append(FindVal(ctx,positionEncoding,searchDepth,fanOut))

    total_value = sum(PredMoveValues)
    # Normalize so probabilities sum to 1
    move_probs = {
        move: value / total_value
        for move, value in zip(legal_moves, PredMoveValues)
    }
    ctx.logProbabilities(move_probs)

    return random.choices(legal_moves, weights=total_value, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
